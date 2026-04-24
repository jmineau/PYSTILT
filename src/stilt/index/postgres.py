"""PostgreSQL durable index backend."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from .base import _SqlIndex
from .rebuild import scan_durable_simulations

if TYPE_CHECKING:
    from stilt.execution import SimulationResult

_COMPLETED_WHERE_SQL = """
COALESCE(s.traj_present, FALSE) = TRUE
AND NOT EXISTS (
    SELECT 1
    FROM jsonb_array_elements_text(COALESCE(s.footprint_targets, '[]'::jsonb)) AS target(name)
    WHERE COALESCE(COALESCE(s.footprints, '{}'::jsonb) ->> target.name, '')
        NOT IN ('complete', 'complete-empty')
)
"""

_PENDING_WHERE_SQL = f"""
COALESCE(s.trajectory_status, 'pending') NOT IN ('running', 'failed')
AND NOT (
    {_COMPLETED_WHERE_SQL}
)
"""

_PRUNE_ELIGIBLE_WHERE_SQL = f"""
s.trajectory_status = 'failed'
OR (
    {_COMPLETED_WHERE_SQL}
)
"""

POSTGRES_PENDING_SIMULATIONS_SQL = f"""
SELECT COUNT(*)
FROM simulations AS s
WHERE {_PENDING_WHERE_SQL}
"""


def _connect_postgres(db_url: str) -> Any:
    """Open one psycopg connection configured for dict-style rows."""
    try:
        import psycopg
        import psycopg.rows
    except ImportError as exc:
        raise ImportError(
            "PostgresIndex requires psycopg. "
            "Install it with: pip install 'pystilt[cloud]'"
        ) from exc
    return psycopg.connect(db_url, row_factory=psycopg.rows.dict_row)  # pyright: ignore[reportArgumentType]


@dataclass(slots=True)
class PostgresClaim:
    """Transactional claim bundle returned by PostgreSQL pull workers."""

    sim_id: str
    _index: PostgresIndex
    _conn: Any
    _released: bool = False

    def release(self) -> None:
        """Mark this unit of work for rollback instead of commit."""
        self._released = True

    @property
    def released(self) -> bool:
        return self._released

    def record(self, result: SimulationResult) -> None:
        """Persist one worker result inside the claim transaction."""
        self._index._record_result_conn(self._conn, result)


class PostgresIndex(_SqlIndex):
    """Durable PostgreSQL simulation index for shared-service deployments."""

    _ph: ClassVar[str] = "%s"
    _now_sql: ClassVar[str] = "NOW()"
    _true_sql: ClassVar[str] = "TRUE"
    _false_sql: ClassVar[str] = "FALSE"
    _empty_footprints_sql: ClassVar[str] = "'{}'::jsonb"
    _jsonb_cast: ClassVar[str] = "::jsonb"
    _bool_encode: ClassVar[Any] = bool
    _backend_name: ClassVar[str] = "PostgreSQL"
    _completed_where: ClassVar[str] = _COMPLETED_WHERE_SQL
    _pending_where: ClassVar[str] = _PENDING_WHERE_SQL
    _prune_eligible_where: ClassVar[str] = _PRUNE_ELIGIBLE_WHERE_SQL

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS simulations (
            sim_id             TEXT        NOT NULL PRIMARY KEY,
            scene_id           TEXT,
            receptor           JSONB       NOT NULL,
            trajectory_status  TEXT        NOT NULL DEFAULT 'pending',
            traj_present       BOOLEAN     NOT NULL DEFAULT FALSE,
            error_traj_present BOOLEAN     NOT NULL DEFAULT FALSE,
            log_present        BOOLEAN     NOT NULL DEFAULT FALSE,
            footprints         JSONB       NOT NULL DEFAULT '{}'::jsonb,
            error              TEXT,
            footprint_targets  JSONB       NOT NULL DEFAULT '[]'::jsonb,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """

    def __init__(
        self,
        db_url: str,
        output_root: str | Path | None = None,
        max_rows: int | None = None,
    ) -> None:
        self._db_url = db_url
        self._output_root = output_root
        self._max_rows = max_rows
        self._ensure_schema()

    def _connect(self) -> Any:
        return _connect_postgres(self._db_url)

    def _table_columns(self, conn: Any, table: str) -> set[str]:
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s",
            (table,),
        ).fetchall()
        return {row["column_name"] for row in rows}

    def _execute_match_ids(
        self,
        conn: Any,
        *,
        prefix: str,
        sim_ids: list[str],
        suffix: str = "",
        prefix_params: tuple = (),
        fetch: bool = False,
    ) -> list[Any]:
        sql = f"{prefix} WHERE sim_id = ANY(%s)"
        if suffix:
            sql += f" {suffix}"
        cursor = conn.execute(sql, (*prefix_params, sim_ids))
        return cursor.fetchall() if fetch else []

    @property
    def db_url(self) -> str:
        return self._db_url

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(self._SCHEMA)
        self._validate_schema()

    @contextmanager
    def claim_one(self) -> Iterator[PostgresClaim | None]:
        """Atomically claim one pending simulation for pull-mode execution."""
        with self._connect() as conn:
            try:
                row = conn.execute(
                    "SELECT s.sim_id "
                    "FROM simulations AS s "
                    f"WHERE {self._pending_where} "
                    "ORDER BY s.sim_id "
                    "LIMIT 1 "
                    "FOR UPDATE OF s SKIP LOCKED"
                ).fetchone()
                if row is None:
                    conn.rollback()
                    yield None
                    return

                claim = PostgresClaim(
                    sim_id=row["sim_id"],
                    _index=self,
                    _conn=conn,
                )
                yield claim
                if claim.released:
                    conn.rollback()
                else:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise

    def rebuild(self) -> None:
        if self._output_root is None:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE simulations SET trajectory_status = 'pending', updated_at = NOW() "
                    "WHERE trajectory_status = 'running'"
                )
            return

        records = scan_durable_simulations(self._output_root)
        with self._connect() as conn:
            self._rebuild_apply(conn, records)


__all__ = ["PostgresClaim", "PostgresIndex", "POSTGRES_PENDING_SIMULATIONS_SQL"]
