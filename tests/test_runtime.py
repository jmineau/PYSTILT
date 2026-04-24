"""Tests for runtime-only deployment settings."""

import pytest
from pydantic import ValidationError

from stilt.config import RuntimeSettings, resolve_runtime_settings


def test_runtime_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSTILT_DB_URL", "postgresql://user:pass@db/pystilt")
    monkeypatch.setenv("PYSTILT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PYSTILT_COMPUTE_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("PYSTILT_MAX_ROWS", "25")

    runtime = RuntimeSettings.from_env()

    assert runtime.db_url == "postgresql://user:pass@db/pystilt"
    assert runtime.cache_dir == tmp_path / "cache"
    assert runtime.compute_root == tmp_path / "scratch"
    assert runtime.max_rows == 25


def test_resolve_runtime_settings_prefers_explicit_instance(tmp_path):
    runtime = RuntimeSettings(
        db_url="postgresql://explicit",
        cache_dir=tmp_path / "cache",
        max_rows=10,
    )

    resolved = resolve_runtime_settings(runtime)

    assert resolved is runtime


def test_runtime_settings_reject_non_positive_max_rows():
    with pytest.raises(ValidationError):
        RuntimeSettings(max_rows=0)
