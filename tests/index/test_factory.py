"""Unit tests for work-queue backend resolution."""

from stilt.config import RuntimeSettings
from stilt.index.factory import resolve_index


def test_resolve_index_returns_explicit_index_unchanged(tmp_path):
    explicit = object()

    resolved = resolve_index(
        explicit,
        output_root=tmp_path,
        runtime=RuntimeSettings(),
        builtin_backend="postgres",
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
        builtin_backend="postgres",
    )

    assert isinstance(resolved, _FakePostgresIndex)
    assert captured == [("postgresql://example", "s3://bucket/project", 25)]


def test_resolve_index_returns_none_without_db_url(tmp_path):
    """No DB URL → no queue. Local projects use the manifest + by-key completion."""
    assert (
        resolve_index(
            None,
            output_root=tmp_path,
            runtime=RuntimeSettings(),
            builtin_backend="postgres",
        )
        is None
    )
