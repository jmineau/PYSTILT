"""SQLite durable index backend."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, ClassVar, Literal

from stilt.storage import ProjectFiles

from .base import _SqlIndex
from .rebuild import scan_durable_simulations
from .sql import chunked

_SCHEMA = """
CREATE TABLE IF NOT EXISTS simulations (
    sim_id TEXT PRIMARY KEY,
    scene_id TEXT,
    receptor TEXT NOT NULL DEFAULT '{}',
    trajectory_status TEXT NOT NULL DEFAULT 'pending',
    traj_present INTEGER NOT NULL DEFAULT 0,
    error_traj_present INTEGER NOT NULL DEFAULT 0,
    log_present INTEGER NOT NULL DEFAULT 0,
    footprints TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    footprint_targets TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_COMPLETED_WHERE_SQL = """
COALESCE(s.traj_present, 0) = 1
AND NOT EXISTS (
    SELECT 1
    FROM json_each(COALESCE(s.footprint_targets, '[]')) AS target
    LEFT JOIN json_each(COALESCE(s.footprints, '{}')) AS fp
        ON fp.key = target.value
    WHERE COALESCE(fp.value, '') NOT IN ('complete', 'complete-empty')
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


class _ManagedConnection(sqlite3.Connection):
    """SQLite connection whose context manager also closes the handle."""

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return False


class SqliteIndex(_SqlIndex):
    """SQLite-backed durable simulation index for a STILT project directory."""

    _ph: ClassVar[str] = "?"
    _now_sql: ClassVar[str] = "datetime('now')"
    _true_sql: ClassVar[str] = "1"
    _false_sql: ClassVar[str] = "0"
    _empty_footprints_sql: ClassVar[str] = "'{}'"
    _jsonb_cast: ClassVar[str] = ""
    _bool_encode: ClassVar[Any] = int
    _backend_name: ClassVar[str] = "SQLite"
    _completed_where: ClassVar[str] = _COMPLETED_WHERE_SQL
    _pending_where: ClassVar[str] = _PENDING_WHERE_SQL
    _prune_eligible_where: ClassVar[str] = _PRUNE_ELIGIBLE_WHERE_SQL

    _project_dir: Path

    def __init__(
        self,
        project_dir: Path,
        *,
        db_path: str | Path | None = None,
        uri: bool = False,
        max_rows: int | None = None,
    ):
        self._project_dir = project_dir
        self._db_path: str | Path = (
            db_path if db_path is not None else ProjectFiles(project_dir).index_db_path
        )
        self._uri = uri
        self._max_rows = max_rows
        self._keepalive: sqlite3.Connection | None = None
        if self._is_memory_db:
            self._keepalive = self._connect()
        self._init_db()

    @property
    def _is_memory_db(self) -> bool:
        return (
            self._uri
            and isinstance(self._db_path, str)
            and "mode=memory" in self._db_path
        )

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=timeout,
            uri=self._uri,
            factory=_ManagedConnection,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _table_columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        return {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _execute_match_ids(
        self,
        conn: sqlite3.Connection,
        *,
        prefix: str,
        sim_ids: list[str],
        suffix: str = "",
        prefix_params: tuple = (),
        fetch: bool = False,
    ) -> list[Any]:
        accumulated: list[Any] = []
        for chunk in chunked(sim_ids):
            placeholders = ",".join("?" for _ in chunk)
            sql = f"{prefix} WHERE sim_id IN ({placeholders})"
            if suffix:
                sql += f" {suffix}"
            cursor = conn.execute(sql, (*prefix_params, *chunk))
            if fetch:
                accumulated.extend(cursor.fetchall())
        return accumulated

    def _init_db(self) -> None:
        if not self._is_memory_db:
            assert isinstance(self._db_path, Path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            if not self._is_memory_db:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
        self._validate_schema()

    def rebuild(self) -> None:
        records = scan_durable_simulations(self._project_dir)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._rebuild_apply(conn, records)

    def close(self) -> None:
        if self._keepalive is not None:
            self._keepalive.close()
            self._keepalive = None

    def __del__(self) -> None:
        self.close()


__all__ = ["SqliteIndex"]
