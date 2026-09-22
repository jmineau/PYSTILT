"""Tests for runtime-only deployment settings."""

from stilt.config import RuntimeSettings


def test_runtime_settings_read_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSTILT_DB_URL", "postgresql://user:pass@db/pystilt")
    monkeypatch.setenv("PYSTILT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PYSTILT_COMPUTE_ROOT", str(tmp_path / "scratch"))

    runtime = RuntimeSettings()

    assert runtime.db_url == "postgresql://user:pass@db/pystilt"
    assert runtime.cache_dir == tmp_path / "cache"
    assert runtime.compute_root == tmp_path / "scratch"


def test_runtime_settings_default_to_none(monkeypatch):
    for name in ("PYSTILT_DB_URL", "PYSTILT_CACHE_DIR", "PYSTILT_COMPUTE_ROOT"):
        monkeypatch.delenv(name, raising=False)

    runtime = RuntimeSettings()

    assert runtime.db_url is None
    assert runtime.cache_dir is None
    assert runtime.compute_root is None


def test_explicit_values_override_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSTILT_DB_URL", "postgresql://from-env")

    runtime = RuntimeSettings(db_url="postgresql://explicit", cache_dir=tmp_path)

    assert runtime.db_url == "postgresql://explicit"
    assert runtime.cache_dir == tmp_path


def test_unknown_environment_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("PYSTILT_MAX_ROWS", "25")

    runtime = RuntimeSettings()

    assert not hasattr(runtime, "max_rows")
