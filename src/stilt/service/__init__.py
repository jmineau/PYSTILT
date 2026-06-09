"""Service-layer runtime: the optional Postgres-backed work queue."""

from __future__ import annotations

from .factory import resolve_queue
from .postgres import PostgresQueue

__all__ = [
    "PostgresQueue",
    "resolve_queue",
]
