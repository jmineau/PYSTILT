"""
Work-queue resolution.

The only persistent backend is the Postgres work queue, present when a DB URL is
configured. Local projects have no queue: work runs inline or via push workers,
the registry is the manifest, and completion is computed by key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stilt.config import RuntimeSettings

    from .postgres import PostgresQueue


def resolve_queue(runtime: RuntimeSettings) -> PostgresQueue | None:
    """Return the Postgres queue when a DB URL is configured, else ``None``."""
    if not runtime.db_url:
        return None
    from .postgres import PostgresQueue

    return PostgresQueue(runtime.db_url)
