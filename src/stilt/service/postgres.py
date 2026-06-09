"""
Postgres-backed work queue for distributed (pull/serve) execution.

The queue distributes work to claim-mode workers: simulations are enqueued
``pending``, atomically claimed (``FOR UPDATE SKIP LOCKED``), run, then marked
``done``/``failed``. It tracks **work status only** — whether outputs exist is
decided by key from the store (see :mod:`stilt.completion`), never here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stilt.receptors import Receptor

if TYPE_CHECKING:
    from stilt.execution import SimulationResult

POSTGRES_PENDING_SIMULATIONS_SQL = "SELECT COUNT(*) FROM queue WHERE status = 'pending'"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
    sim_id     TEXT        NOT NULL PRIMARY KEY,
    scene      TEXT,
    receptor   JSONB       NOT NULL,
    status     TEXT        NOT NULL DEFAULT 'pending',
    error      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _connect(db_url: str) -> Any:
    """Open one psycopg connection configured for dict-style rows."""
    try:
        import psycopg
        import psycopg.rows
    except ImportError as exc:  # pragma: no cover - optional cloud dependency
        raise ImportError(
            "The Postgres queue requires psycopg. "
            "Install it with: pip install 'pystilt[cloud]'"
        ) from exc
    return psycopg.connect(db_url, row_factory=psycopg.rows.dict_row)  # pyright: ignore[reportArgumentType]


def _status_for(result: SimulationResult) -> str:
    """Map a worker result onto a queue status."""
    if result.status == "interrupted":
        return "pending"
    if result.status in ("complete", "complete-empty"):
        return "done"
    return "failed"


@dataclass(slots=True)
class PostgresClaim:
    """One claimed work item, recorded inside its claim transaction."""

    sim_id: str
    _conn: Any
    _released: bool = False

    def release(self) -> None:
        """Mark this claim for rollback instead of commit."""
        self._released = True

    @property
    def released(self) -> bool:
        return self._released

    def record(self, result: SimulationResult) -> None:
        """Persist the work status for this claim inside its transaction."""
        self._conn.execute(
            "UPDATE queue SET status = %s, error = %s, updated_at = NOW() "
            "WHERE sim_id = %s",
            (_status_for(result), result.error, str(result.sim_id)),
        )


class PostgresQueue:
    """A Postgres work queue: enqueue → claim → done/failed."""

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        with _connect(db_url) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    @property
    def db_url(self) -> str:
        return self._db_url

    def register(
        self,
        pairs: Iterable[tuple[str, Receptor]],
        *,
        scene_id: str | None = None,
    ) -> None:
        """Enqueue one or many simulations as pending work (idempotent)."""
        rows = [
            (sim_id, scene_id, json.dumps(receptor.to_dict()))
            for sim_id, receptor in pairs
        ]
        if not rows:
            return
        with _connect(self._db_url) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO queue (sim_id, scene, receptor) "
                    "VALUES (%s, %s, %s::jsonb) "
                    "ON CONFLICT (sim_id) DO UPDATE SET "
                    "scene = COALESCE(EXCLUDED.scene, queue.scene), "
                    "receptor = EXCLUDED.receptor, "
                    "status = 'pending', updated_at = NOW()",
                    rows,
                )
            conn.commit()

    @contextmanager
    def claim_one(self) -> Iterator[PostgresClaim | None]:
        """Atomically claim one pending simulation for pull-mode execution."""
        with _connect(self._db_url) as conn:
            try:
                row = conn.execute(
                    "SELECT sim_id FROM queue WHERE status = 'pending' "
                    "ORDER BY sim_id LIMIT 1 FOR UPDATE SKIP LOCKED"
                ).fetchone()
                if row is None:
                    conn.rollback()
                    yield None
                    return
                conn.execute(
                    "UPDATE queue SET status = 'running', updated_at = NOW() "
                    "WHERE sim_id = %s",
                    (row["sim_id"],),
                )
                claim = PostgresClaim(sim_id=row["sim_id"], _conn=conn)
                yield claim
                if claim.released:
                    conn.rollback()
                else:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise


__all__ = ["PostgresClaim", "PostgresQueue", "POSTGRES_PENDING_SIMULATIONS_SQL"]
