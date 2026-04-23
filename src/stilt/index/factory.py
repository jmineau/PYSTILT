"""Index backend resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from stilt.config import RuntimeSettings
from stilt.errors import ConfigValidationError

from .postgres import PostgresIndex
from .protocol import SimulationIndex
from .sqlite import SqliteIndex


def resolve_index(
    index: SimulationIndex | None,
    *,
    output_root: str | Path,
    runtime: RuntimeSettings,
    builtin_backend: Literal["sqlite", "postgres"],
) -> SimulationIndex:
    """Resolve the durable simulation index backend for one model instance."""
    if index is not None:
        return index
    if runtime.db_url:
        return PostgresIndex(
            runtime.db_url,
            output_root=output_root,
            max_rows=runtime.max_rows,
        )
    if builtin_backend == "postgres":
        db_url = runtime.db_url
        if not db_url:
            raise ConfigValidationError(
                "PYSTILT_DB_URL env var or RuntimeSettings.db_url is required "
                "for cloud output projects"
            )
        return PostgresIndex(
            db_url,
            output_root=output_root,
            max_rows=runtime.max_rows,
        )
    return SqliteIndex(Path(output_root), max_rows=runtime.max_rows)
