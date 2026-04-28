"""Private shared helpers for SQL-backed index implementations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from stilt.receptor import Receptor

from .protocol import IndexCounts, OutputSummary
from .updates import IndexUpdate

REQUIRED_SIMULATION_COLUMNS = {
    "sim_id",
    "scene_id",
    "receptor",
    "trajectory_status",
    "traj_present",
    "error_traj_present",
    "log_present",
    "footprints",
    "error",
    "footprint_targets",
    "created_at",
    "updated_at",
}


@dataclass(frozen=True, slots=True)
class SqlIndexPredicates:
    """Rendered durable-index predicates for one SQL dialect."""

    completed: str
    pending: str
    prune_eligible: str


@dataclass(frozen=True, slots=True)
class SqlPredicateDialect:
    """Dialect fragments needed to render shared simulation-state predicates."""

    true_sql: str
    false_sql: str
    target_rows_sql: str
    footprint_status_sql: str


def build_index_predicates(dialect: SqlPredicateDialect) -> SqlIndexPredicates:
    """Render completion/pending/prune predicates for one SQL dialect."""
    completed = f"""
COALESCE(s.traj_present, {dialect.false_sql}) = {dialect.true_sql}
AND NOT EXISTS (
    SELECT 1
    FROM {dialect.target_rows_sql}
    WHERE COALESCE({dialect.footprint_status_sql}, '') NOT IN ('complete', 'complete-empty')
)
"""
    pending = f"""
COALESCE(s.trajectory_status, 'pending') NOT IN ('running', 'failed')
AND NOT (
    {completed}
)
"""
    prune_eligible = f"""
s.trajectory_status = 'failed'
OR (
    {completed}
)
"""
    return SqlIndexPredicates(
        completed=completed,
        pending=pending,
        prune_eligible=prune_eligible,
    )


def validate_required_columns(found: set[str], *, backend_name: str) -> None:
    """Raise a clear error when one SQL index schema is missing required columns."""
    missing = sorted(REQUIRED_SIMULATION_COLUMNS - found)
    if missing:
        raise RuntimeError(
            f"{backend_name} index schema is incompatible with the current alpha "
            f"implementation; missing simulations columns: {missing}. "
            "Rebuild or replace the index schema."
        )


def output_summary_from_row(row: Any) -> OutputSummary:
    """Build one output summary directly from one SQL row."""
    footprints = row["footprints"] or {}
    if isinstance(footprints, str):
        footprints = json.loads(footprints)
    return OutputSummary(
        traj_present=bool(row["traj_present"]),
        error_traj_present=bool(row["error_traj_present"]),
        log_present=bool(row["log_present"]),
        footprints=footprints,
    )


def normalize_footprint_targets(raw: Any) -> list[str]:
    """Normalize a SQL JSON/JSONB footprint-target field to a list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return list(json.loads(raw))
    return list(raw)


def load_json_blob(raw: Any) -> Any:
    """Normalize a SQL JSON/JSONB field to a Python object."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def receptor_mapping(rows: Iterable[Any]) -> dict[str, Receptor]:
    """Build a ``sim_id -> Receptor`` mapping from SQL rows."""
    return {
        row["sim_id"]: Receptor.from_dict(load_json_blob(row["receptor"]))
        for row in rows
    }


def output_summary_mapping(rows: Iterable[Any]) -> dict[str, OutputSummary]:
    """Build a ``sim_id -> OutputSummary`` mapping from SQL rows."""
    return {row["sim_id"]: output_summary_from_row(row) for row in rows}


def registration_rows(
    pairs: list[tuple[str, Receptor]],
    *,
    footprint_names: list[str] | None = None,
    scene_id: str | None = None,
) -> list[tuple[str, str | None, str, str]]:
    """Serialize one-or-many registrations into SQL-ready row tuples."""
    targets_json = json.dumps(sorted(set(footprint_names or [])))
    return [
        (
            sim_id,
            scene_id,
            json.dumps(receptor.to_dict()),
            targets_json,
        )
        for sim_id, receptor in pairs
    ]


def _update_counts(
    bucket: IndexCounts,
    *,
    completed: bool,
    running: bool,
    failed: bool,
) -> IndexCounts:
    return IndexCounts(
        total=bucket.total + 1,
        completed=bucket.completed + int(completed),
        running=bucket.running + int(running),
        pending=bucket.pending + int(not completed and not running and not failed),
        failed=bucket.failed + int(failed),
    )


def count_rows(rows: Iterable[Any]) -> IndexCounts:
    """Aggregate one iterable of SQL rows into durable index counts."""
    counts = IndexCounts()
    for row in rows:
        summary = output_summary_from_row(row)
        completed = summary.outputs_complete(
            normalize_footprint_targets(row["footprint_targets"])
        )
        running = row["trajectory_status"] == "running"
        failed = row["trajectory_status"] == "failed" and not completed and not running
        counts = _update_counts(
            counts,
            completed=completed,
            running=running,
            failed=failed,
        )
    return counts


def group_counts(rows: Iterable[Any]) -> dict[str, IndexCounts]:
    """Aggregate SQL rows into per-scene durable counts."""
    grouped: dict[str, IndexCounts] = {}
    for row in rows:
        scene_id = row["scene_id"]
        if scene_id is None:
            continue
        summary = output_summary_from_row(row)
        completed = summary.outputs_complete(
            normalize_footprint_targets(row["footprint_targets"])
        )
        running = row["trajectory_status"] == "running"
        failed = row["trajectory_status"] == "failed" and not completed and not running
        grouped[scene_id] = _update_counts(
            grouped.get(scene_id, IndexCounts()),
            completed=completed,
            running=running,
            failed=failed,
        )
    return grouped


def index_counts(
    *,
    total: int,
    completed: int,
    pending: int,
    running: int,
) -> IndexCounts:
    """Build a consistent count bundle from aggregate SQL query results."""
    return IndexCounts(
        total=total,
        completed=completed,
        running=running,
        pending=pending,
        failed=max(total - completed - pending - running, 0),
    )


def count_where(conn: Any, where_sql: str) -> int:
    """Return the number of simulation rows matching one WHERE clause."""
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM simulations AS s WHERE {where_sql}"
    ).fetchone()
    return int(row["n"]) if row is not None else 0


def ordered_sim_ids(conn: Any, where_sql: str | None = None) -> list[str]:
    """Return sorted simulation ids, optionally filtered by one WHERE clause."""
    sql = "SELECT s.sim_id FROM simulations AS s"
    if where_sql:
        sql += f" WHERE {where_sql}"
    sql += " ORDER BY s.sim_id"
    rows = conn.execute(sql).fetchall()
    return [row["sim_id"] for row in rows]


def total_rows(conn: Any) -> int:
    """Return the total number of simulations in one index."""
    row = conn.execute("SELECT COUNT(*) AS n FROM simulations").fetchone()
    return int(row["n"]) if row is not None else 0


def existing_sim_ids(conn: Any) -> set[str]:
    """Return the current simulation ids present in one SQL-backed index."""
    return {row["sim_id"] for row in conn.execute("SELECT sim_id FROM simulations")}


def update_values(
    update: IndexUpdate,
    *,
    encode_bool: Any = bool,
) -> tuple[Any, ...]:
    """Serialize one index update into SQL parameter values."""
    return (
        update.trajectory_status,
        update.error,
        encode_bool(update.summary.traj_present),
        encode_bool(update.summary.error_traj_present),
        encode_bool(update.summary.log_present),
        json.dumps(update.summary.footprints),
    )


def reset_for_rebuild(
    conn: Any,
    *,
    now_sql: str,
    false_sql: str,
    empty_footprints_sql: str,
) -> None:
    """Reset one index to a clean pending state before applying a rebuild scan."""
    conn.execute(
        "UPDATE simulations SET trajectory_status = 'pending', "
        f"updated_at = {now_sql} "
        "WHERE trajectory_status = 'running'"
    )
    conn.execute(
        "UPDATE simulations SET trajectory_status = 'pending', error = NULL, "
        f"traj_present = {false_sql}, error_traj_present = {false_sql}, "
        f"log_present = {false_sql}, footprints = {empty_footprints_sql}, "
        f"updated_at = {now_sql}"
    )


def chunked(values: list[str], size: int = 500) -> Iterator[list[str]]:
    """Yield fixed-size chunks from one list of simulation ids."""
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def apply_rebuild_records(
    records: Iterable[Any],
    *,
    existing_ids: set[str],
    register_missing: Any,
    store_update: Any,
) -> None:
    """Apply scanned durable outputs to one mutable SQL-backed index."""
    for record in records:
        if record.sim_id not in existing_ids:
            if record.receptor is None:
                continue
            register_missing(record.sim_id, record.receptor)
            existing_ids.add(record.sim_id)
        store_update(record.sim_id, record.summary)
