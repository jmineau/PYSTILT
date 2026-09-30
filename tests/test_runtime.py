"""Tests for the settings read from the environment."""

from stilt.config import RuntimeSettings


def test_runtime_settings_read_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSTILT_COMPUTE_ROOT", str(tmp_path / "scratch"))

    assert RuntimeSettings().compute_root == tmp_path / "scratch"


def test_runtime_settings_default_to_none(monkeypatch):
    monkeypatch.delenv("PYSTILT_COMPUTE_ROOT", raising=False)

    assert RuntimeSettings().compute_root is None


def test_explicit_values_override_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSTILT_COMPUTE_ROOT", str(tmp_path / "from-env"))

    assert RuntimeSettings(compute_root=tmp_path).compute_root == tmp_path


def test_unknown_environment_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("PYSTILT_MAX_ROWS", "25")

    assert not hasattr(RuntimeSettings(), "max_rows")
