"""Unit tests for index backend resolution."""

from pathlib import Path

import pytest

from stilt.config import RuntimeSettings
from stilt.errors import ConfigValidationError
from stilt.index.factory import resolve_index


def test_resolve_index_returns_explicit_index_unchanged(tmp_path):
    explicit = object()

    resolved = resolve_index(
        explicit,
        output_root=tmp_path,
        runtime=RuntimeSettings(),
        builtin_backend="sqlite",
    )

    assert resolved is explicit


def test_resolve_index_prefers_postgres_when_runtime_db_url_present(monkeypatch):
    captured: list[tuple[str, str, int | None]] = []

    class _FakePostgresIndex:
        def __init__(self, db_url: str, *, output_root: str, max_rows: int | None):
            captured.append((db_url, output_root, max_rows))

    monkeypatch.setattr("stilt.index.factory.PostgresIndex", _FakePostgresIndex)

    resolved = resolve_index(
        None,
        output_root="s3://bucket/project",
        runtime=RuntimeSettings(db_url="postgresql://example", max_rows=25),
        builtin_backend="sqlite",
    )

    assert isinstance(resolved, _FakePostgresIndex)
    assert captured == [("postgresql://example", "s3://bucket/project", 25)]


def test_resolve_index_requires_db_url_for_postgres_builtin(tmp_path):
    with pytest.raises(ConfigValidationError, match="PYSTILT_DB_URL"):
        resolve_index(
            None,
            output_root=tmp_path,
            runtime=RuntimeSettings(),
            builtin_backend="postgres",
        )


def test_resolve_index_uses_sqlite_when_runtime_has_no_db_url(monkeypatch, tmp_path):
    captured: list[tuple[Path, int | None]] = []

    class _FakeSqliteIndex:
        def __init__(self, root: Path, *, max_rows: int | None):
            captured.append((root, max_rows))

    monkeypatch.setattr("stilt.index.factory.SqliteIndex", _FakeSqliteIndex)

    resolved = resolve_index(
        None,
        output_root=tmp_path,
        runtime=RuntimeSettings(max_rows=10),
        builtin_backend="sqlite",
    )

    assert isinstance(resolved, _FakeSqliteIndex)
    assert captured == [(tmp_path, 10)]
