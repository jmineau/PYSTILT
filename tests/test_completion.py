"""Tests for the by-key completion spine (stilt.completion)."""

from dataclasses import dataclass

from stilt.completion import (
    ERROR_TRAJECTORY,
    TRAJECTORY,
    expected_artifacts,
    expected_for_config,
    footprint_artifact,
    is_complete,
)
from stilt.storage import LocalStore, ProjectFiles, Storage


@dataclass
class _Cfg:
    """Minimal stand-in for the config surface completion reads."""

    footprints: dict
    error_enabled: bool


def _storage(tmp_path):
    return Storage(
        project_dir=tmp_path, output_dir=tmp_path, store=LocalStore(tmp_path)
    )


def _touch(tmp_path, sim_id, kind, name=""):
    files = ProjectFiles(tmp_path).simulation(sim_id)
    path = {
        "traj": files.trajectory_path,
        "error": files.error_trajectory_path,
        "foot": files.footprint_path(name),
        "empty": files.empty_footprint_path(name),
    }[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


# -- expected_artifacts ------------------------------------------------------


def test_expected_without_error():
    assert expected_artifacts(["default"], error_enabled=False) == frozenset(
        {TRAJECTORY, footprint_artifact("default")}
    )


def test_expected_with_error_includes_error_trajectory():
    exp = expected_artifacts(["default"], error_enabled=True)
    assert exp == frozenset(
        {TRAJECTORY, ERROR_TRAJECTORY, footprint_artifact("default")}
    )


def test_expected_never_requires_error_footprint():
    """error_enabled gates the error trajectory, never an error footprint."""
    exp = expected_artifacts(["default"], error_enabled=True)
    assert footprint_artifact("default_error") not in exp


# -- is_complete (by key over the store) -------------------------------------


def test_complete_when_error_not_expected(tmp_path):
    _touch(tmp_path, "s1", "traj")
    _touch(tmp_path, "s1", "foot", "default")
    cfg = _Cfg(footprints={"default": None}, error_enabled=False)
    assert is_complete("s1", expected_for_config(cfg), _storage(tmp_path)) is True


def test_incomplete_when_error_expected_but_missing(tmp_path):
    _touch(tmp_path, "s1", "traj")
    _touch(tmp_path, "s1", "foot", "default")
    cfg = _Cfg(footprints={"default": None}, error_enabled=True)
    assert is_complete("s1", expected_for_config(cfg), _storage(tmp_path)) is False


def test_complete_when_error_present(tmp_path):
    _touch(tmp_path, "s1", "traj")
    _touch(tmp_path, "s1", "error")
    _touch(tmp_path, "s1", "foot", "default")
    cfg = _Cfg(footprints={"default": None}, error_enabled=True)
    assert is_complete("s1", expected_for_config(cfg), _storage(tmp_path)) is True


def test_empty_footprint_marker_counts_complete(tmp_path):
    _touch(tmp_path, "s1", "traj")
    _touch(tmp_path, "s1", "empty", "default")
    cfg = _Cfg(footprints={"default": None}, error_enabled=False)
    assert is_complete("s1", expected_for_config(cfg), _storage(tmp_path)) is True


def test_incomplete_when_trajectory_missing(tmp_path):
    _touch(tmp_path, "s1", "foot", "default")
    cfg = _Cfg(footprints={"default": None}, error_enabled=False)
    assert is_complete("s1", expected_for_config(cfg), _storage(tmp_path)) is False
