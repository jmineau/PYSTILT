"""Tests for runtime-only deployment settings."""

from stilt.runtime import RuntimeSettings, resolve_runtime_settings


def test_runtime_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSTILT_DB_URL", "postgresql://user:pass@db/pystilt")
    monkeypatch.setenv("STILT_MET_ARCHIVE", str(tmp_path / "met"))
    monkeypatch.setenv("PYSTILT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PYSTILT_COMPUTE_ROOT", str(tmp_path / "scratch"))

    runtime = RuntimeSettings.from_env()

    assert runtime.db_url == "postgresql://user:pass@db/pystilt"
    assert runtime.met_archive == tmp_path / "met"
    assert runtime.cache_dir == tmp_path / "cache"
    assert runtime.compute_root == tmp_path / "scratch"


def test_resolve_runtime_settings_prefers_explicit_instance(tmp_path):
    runtime = RuntimeSettings(
        db_url="postgresql://explicit",
        cache_dir=tmp_path / "cache",
    )

    resolved = resolve_runtime_settings(runtime)

    assert resolved is runtime
