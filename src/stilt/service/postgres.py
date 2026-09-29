"""
PostgreSQL work queue for pull workers.

Receptors are added as ``pending``. A worker claims one with
``FOR UPDATE SKIP LOCKED``, so no other worker can take it, runs every
variant of it, and marks it ``done`` or ``failed``. The queue records only
this status. Whether a simulation is complete is decided by its outputs in
the project.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stilt.execution import ReceptorResult

POSTGRES_PENDING_SIMULATIONS_SQL = "SELECT COUNT(*) FROM queue WHERE status = 'pending'"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
    receptor_id TEXT        NOT NULL PRIMARY KEY,
    status     TEXT        NOT NULL DEFAULT 'pending',
    error      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _connect(db_url: str) -> Any:
    """Open a psycopg connection that returns rows as dicts."""
    try:
        import psycopg
        import psycopg.rows
    except ImportError as exc:  # pragma: no cover - optional cloud dependency
        raise ImportError(
            "The Postgres queue requires psycopg. "
            "Install it with: pip install 'pystilt[cloud]'"
        ) from exc
    return psycopg.connect(db_url, row_factory=psycopg.rows.dict_row)  # pyright: ignore[reportArgumentType]


def _status_for(result: ReceptorResult) -> str:
    """Return the queue status for a receptor result. An interrupted receptor goes back to pending."""
    if result.status == "interrupted":
        return "pending"
    return "done" if result.status == "complete" else "failed"


@dataclass(slots=True)
class PostgresClaim:
    """
    One claimed receptor.

    The claim holds a database transaction open while the receptor runs. The
    transaction commits when the claim ends, or rolls back if the worker
    fails, which puts the receptor back in the queue.
    """

    receptor_id: str
    _conn: Any
    _released: bool = False

    def release(self) -> None:
        """Roll back the claim when it ends, leaving the receptor pending."""
        self._released = True

    @property
    def released(self) -> bool:
        """Whether :meth:`release` has been called."""
        return self._released

    def record(self, result: ReceptorResult) -> None:
        """Set the receptor's queue status from its result."""
        self._conn.execute(
            "UPDATE queue SET status = %s, error = %s, updated_at = NOW() "
            "WHERE receptor_id = %s",
            (_status_for(result), result.error, self.receptor_id),
        )


class PostgresQueue:
    """
    Work queue of receptors in a PostgreSQL database.

    Creates the ``queue`` table if it does not exist.

    Parameters
    ----------
    db_url : str
        PostgreSQL connection URL.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        with _connect(db_url) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def register(self, receptor_ids: Iterable[str]) -> None:
        """Add receptors to the queue as pending, resetting any that are already there."""
        rows = [(str(rid),) for rid in receptor_ids]
        if not rows:
            return
        with _connect(self._db_url) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO queue (receptor_id) VALUES (%s) "
                    "ON CONFLICT (receptor_id) DO UPDATE SET "
                    "status = 'pending', error = NULL, updated_at = NOW()",
                    rows,
                )
            conn.commit()

    @contextmanager
    def claim_one(self) -> Iterator[PostgresClaim | None]:
        """
        Context manager that claims one pending receptor.

        Yields a :class:`PostgresClaim`, or ``None`` when no receptor is
        pending. The claim's status update is committed when the block ends.
        """
        with _connect(self._db_url) as conn:
            try:
                row = conn.execute(
                    "SELECT receptor_id FROM queue WHERE status = 'pending' "
                    "ORDER BY receptor_id LIMIT 1 FOR UPDATE SKIP LOCKED"
                ).fetchone()
                if row is None:
                    conn.rollback()
                    yield None
                    return
                conn.execute(
                    "UPDATE queue SET status = 'running', updated_at = NOW() "
                    "WHERE receptor_id = %s",
                    (row["receptor_id"],),
                )
                claim = PostgresClaim(receptor_id=row["receptor_id"], _conn=conn)
                yield claim
                if claim.released:
                    conn.rollback()
                else:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise


__all__ = ["PostgresClaim", "PostgresQueue", "POSTGRES_PENDING_SIMULATIONS_SQL"]
