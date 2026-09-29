"""
Finding the work queue for a model.

A project has a work queue only when ``PYSTILT_DB_URL`` is set. Without one,
receptors run in the calling process or through push workers, and whether a
simulation is complete is decided by the outputs in the project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stilt.config import RuntimeSettings

    from .postgres import PostgresQueue


def resolve_queue(runtime: RuntimeSettings) -> PostgresQueue | None:
    """Return the PostgreSQL queue when ``runtime.db_url`` is set, else ``None``."""
    if not runtime.db_url:
        return None
    from .postgres import PostgresQueue

    return PostgresQueue(runtime.db_url)
