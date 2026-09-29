"""Tests for stilt.project (project root, layout helpers, and input persistence)."""

from pathlib import Path

import pytest

from stilt.project import (
    CONFIG_KEY,
    RECEPTORS_KEY,
    SIMULATIONS_PREFIX,
    Project,
    project_slug,
    resolve_directory,
    simulation_prefix,
)
from stilt.store import FsspecStore, LocalStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def local_project(tmp_path) -> Project:
    return Project(tmp_path / "proj")


@pytest.fixture
def memory_project(tmp_path) -> Project:
    return Project(f"memory://proj-{tmp_path.name}", cache_dir=tmp_path / "cache")


@pytest.fixture(params=["local", "memory"])
def any_project(request, local_project, memory_project) -> Project:
    return local_project if request.param == "local" else memory_project


# ---------------------------------------------------------------------------
# resolve_directory (ported from the removed tests/storage/test_files.py)
# ---------------------------------------------------------------------------


def test_resolve_directory_none_returns_tempdir():
    p = resolve_directory(None)
    assert p.exists()
    assert p.is_dir()


def test_resolve_directory_absolute_path_unchanged(tmp_path):
    p = resolve_directory(tmp_path)
    assert p == tmp_path


def test_resolve_directory_relative_name_resolves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = resolve_directory("subdir")
    assert p.is_absolute()
    assert p.name == "subdir"


def test_resolve_directory_nested_relative_path_resolves(tmp_path, monkeypatch):
    """runs/a used to stay relative, which broke workers started elsewhere (#58)."""
    monkeypatch.chdir(tmp_path)
    assert resolve_directory("runs/a") == tmp_path.resolve() / "runs" / "a"


def test_resolve_directory_string_input(tmp_path):
    p = resolve_directory(str(tmp_path))
    assert p == tmp_path


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_simulation_prefix_uses_by_id_layout():
    assert SIMULATIONS_PREFIX == "simulations/by-id"
    assert simulation_prefix("hrrr_202301011200_site") == (
        "simulations/by-id/hrrr_202301011200_site"
    )


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        ("/data/projects/My_Project", "my-project"),
        ("/data/projects/My_Project/", "my-project"),
        ("s3://bucket/path/slv_hrrr", "slv-hrrr"),
        ("gs://bucket/Weird  Name!!", "weird-name"),
        ("memory://only-bucket", "only-bucket"),
        ("", "project"),
    ],
)
def test_project_slug(root, expected):
    assert project_slug(root) == expected


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------


def test_project_relative_name_resolves_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = Project("myproj")
    assert Path(project.root).is_absolute()
    assert Path(project.root) == tmp_path / "myproj"
    assert not project.is_cloud


def test_project_none_makes_temp_dir():
    project = Project(None)
    assert not project.is_cloud
    assert Path(project.root).is_dir()


def test_project_local_root_is_string_and_uses_local_store(local_project, tmp_path):
    assert isinstance(local_project.root, str)
    assert local_project.root == str(tmp_path / "proj")
    assert isinstance(local_project.store, LocalStore)
    assert local_project.store.root == (tmp_path / "proj").resolve()


def test_project_cloud_root(memory_project):
    assert memory_project.is_cloud
    assert memory_project.root.startswith("memory://")
    assert not memory_project.root.endswith("/")
    assert isinstance(memory_project.store, FsspecStore)


def test_project_cloud_root_strips_trailing_slash(tmp_path):
    project = Project(f"memory://trail-{tmp_path.name}/")
    assert project.root == f"memory://trail-{tmp_path.name}"


# ---------------------------------------------------------------------------
# name / directory / str / fspath
# ---------------------------------------------------------------------------


def test_project_name_local_is_basename(local_project):
    assert local_project.name == "proj"


def test_project_name_cloud_is_slug():
    project = Project("s3://bucket/path/SLV_hrrr")
    assert project.name == "slv-hrrr"


def test_project_local_directories(local_project, tmp_path):
    root = tmp_path / "proj"
    assert local_project.directory == root
    assert local_project.simulations_dir == root / "simulations" / "by-id"


def test_project_cloud_directories_raise_type_error(memory_project):
    with pytest.raises(TypeError):
        _ = memory_project.directory
    with pytest.raises(TypeError):
        _ = memory_project.simulations_dir


def test_project_str_is_root(any_project):
    assert str(any_project) == any_project.root


def test_project_repr_includes_root(any_project):
    assert any_project.root in repr(any_project)


# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------


def test_project_has_config_false_and_load_raises_when_absent(any_project):
    assert not any_project.has_config
    with pytest.raises(FileNotFoundError):
        any_project.load_config()


def test_project_config_round_trip(any_project, model_config):
    any_project.save_config(model_config)

    assert any_project.has_config
    assert any_project.store.exists(CONFIG_KEY)
    loaded = any_project.load_config()
    # The saved file always declares its variants; everything else is as given.
    assert loaded.model_dump(exclude={"variants"}) == model_config.model_dump(
        exclude={"variants"}
    )
    assert loaded.resolve_variants() == model_config.resolve_variants()
    assert set(loaded.mets) == {"hrrr"}
    assert loaded.n_hours == -24


def test_project_save_config_writes_config_yaml_in_local_root(
    local_project, model_config
):
    local_project.save_config(model_config)
    assert (local_project.directory / "config.yaml").is_file()


# ---------------------------------------------------------------------------
# receptors.csv
# ---------------------------------------------------------------------------


def test_project_load_receptors_none_when_absent(any_project):
    assert not any_project.has_receptors
    assert any_project.load_receptors() is None


def test_project_receptors_round_trip_points(any_project, point_receptor):
    assert any_project.add_receptors([point_receptor]) == [point_receptor]

    assert any_project.has_receptors
    assert any_project.store.exists(RECEPTORS_KEY)
    assert any_project.load_receptors() == [point_receptor]


def test_project_receptors_round_trip_preserves_groups(
    any_project, point_receptor, column_receptor, multipoint_receptor
):
    """r_idx grouping survives, so column/multipoint receptors come back intact."""
    receptors = [point_receptor, column_receptor, multipoint_receptor]
    any_project.add_receptors(receptors)

    loaded = any_project.load_receptors()
    assert loaded == receptors
    assert [type(r) for r in loaded] == [type(r) for r in receptors]
    assert [len(r) for r in loaded] == [1, 2, 3]


def test_project_add_receptors_appends_only_new_ones(
    any_project, point_receptor, column_receptor
):
    any_project.add_receptors([point_receptor])
    assert any_project.add_receptors([point_receptor, column_receptor]) == [
        column_receptor
    ]
    assert any_project.load_receptors() == [point_receptor, column_receptor]


def test_project_add_receptors_appends_in_the_files_own_columns(
    any_project, tmp_path, point_receptor, column_receptor
):
    """A hand-written file keeps its columns, its r_idx values, and extra columns."""
    text = (
        "r_idx,time,lati,long,zagl,scene\n"
        "1155,2023-01-01 12:00:00,40.77,-111.85,5.0,A\n"
    )
    any_project.store.write_bytes(RECEPTORS_KEY, text.encode())

    assert any_project.add_receptors([point_receptor, column_receptor]) == [
        column_receptor
    ]

    stored = any_project.store.read_bytes(RECEPTORS_KEY).decode()
    assert stored.startswith(text)  # the original bytes are untouched
    lines = stored.splitlines()
    assert lines[1].split(",")[0] == "1155"
    assert lines[2].split(",")[0] == "1156" and lines[2].endswith(",")
    assert any_project.load_receptors() == [point_receptor, column_receptor]


def test_project_add_receptors_refuses_a_group_without_r_idx(
    any_project, column_receptor
):
    any_project.store.write_bytes(RECEPTORS_KEY, b"time,lati,long,zagl\n")
    with pytest.raises(ValueError, match="r_idx"):
        any_project.add_receptors([column_receptor])
