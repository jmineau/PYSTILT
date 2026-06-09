"""
Work-queue backend resolution.

The only persistent index backend is the Postgres work queue, present when a DB
URL is configured. Local projects have no index: the registry is the manifest
(``stilt.manifest``) and completion is computed by key (``stilt.completion``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from stilt.config import RuntimeSettings

from .postgres import PostgresIndex
from .protocol import SimulationIndex


def resolve_index(
    index: SimulationIndex | None,
    *,
    output_root: str | Path,
    runtime: RuntimeSettings,
    builtin_backend: Literal["postgres"] = "postgres",
) -> SimulationIndex | None:
    """Return the Postgres work queue when a DB URL is configured, else ``None``."""
    if index is not None:
        return index
    if runtime.db_url:
        return PostgresIndex(
            runtime.db_url,
            output_root=output_root,
            max_rows=runtime.max_rows,
        )
    return None
