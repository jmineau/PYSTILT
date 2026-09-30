"""Tests for stilt.project: the project directory and its input files."""

from pathlib import Path

import pytest

from stilt.config import ModelConfig
from stilt.project import (
    CONFIG_KEY,
    RECEPTORS_KEY,
    Project,
    project_slug,
    resolve_directory,
)

# ---------------------------------------------------------------------------
# resolve_directory
# ---------------------------------------------------------------------------


def test_resolve_directory_none_returns_tempdir():
    p = resolve_directory(None)
    assert p.exists()
    assert p.is_dir()


def test_resolve_directory_absolute_path_unchanged(tmp_path):
    assert resolve_directory(tmp_path) == tmp_path


def test_resolve_directory_relative_paths_resolve(tmp_path, monkeypatch):
    """runs/a used to stay relative, which broke workers started elsewhere (#58)."""
    monkeypatch.chdir(tmp_path)
    assert resolve_directory("subdir") == tmp_path.resolve() / "subdir"
    assert resolve_directory("runs/a") == tmp_path.resolve() / "runs" / "a"


def test_resolve_directory_expands_user_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSTILT_TEST_DIR", str(tmp_path))
    assert resolve_directory("$PYSTILT_TEST_DIR/x") == tmp_path.resolve() / "x"


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        ("/data/projects/My_Project", "my-project"),
        ("/data/projects/My_Project/", "my-project"),
        ("/data/Weird  Name!!", "weird-name"),
        ("", "project"),
    ],
)
def test_project_slug(root, expected):
    assert project_slug(root) == expected


# ---------------------------------------------------------------------------
# The directory
# ---------------------------------------------------------------------------


def test_project_relative_name_resolves_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = Project("myproj")
    assert project.directory == tmp_path.resolve() / "myproj"
    assert project.root == str(project.directory)
    assert project.name == "myproj"


def test_project_none_makes_temp_dir():
    project = Project(None)
    assert project.directory.is_dir()


def test_project_str_and_repr(tmp_path):
    project = Project(tmp_path / "proj")
    assert str(project) == str(tmp_path / "proj")
    assert str(tmp_path / "proj") in repr(project)


def test_output_path_is_relative_to_the_project_unless_absolute(tmp_path, model_config):
    project = Project(tmp_path / "proj")
    assert project.output_path(model_config) == tmp_path / "proj" / "output"
    shared = model_config.model_copy(update={"output": str(tmp_path / "shared")})
    assert project.output_path(shared) == tmp_path / "shared"
    nested = model_config.model_copy(update={"output": "../results"})
    assert project.output_path(nested) == (tmp_path / "results").resolve()


# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------


def test_project_has_config_false_and_load_raises_when_absent(tmp_path):
    project = Project(tmp_path / "proj")
    assert not project.has_config
    with pytest.raises(FileNotFoundError):
        project.load_config()


def test_project_config_round_trip(tmp_path, model_config):
    project = Project(tmp_path / "proj")
    project.save_config(model_config)

    assert project.has_config
    assert project.config_path == tmp_path / "proj" / CONFIG_KEY
    assert project.config_path.is_file()
    loaded = project.load_config()
    assert isinstance(loaded, ModelConfig)
    # The saved file always declares its variants; everything else is as given.
    assert loaded.model_dump(exclude={"variants"}) == model_config.model_dump(
        exclude={"variants"}
    )
    assert loaded.resolve_variants() == model_config.resolve_variants()


# ---------------------------------------------------------------------------
# receptors.csv
# ---------------------------------------------------------------------------


def test_project_load_receptors_none_when_absent(tmp_path):
    project = Project(tmp_path / "proj")
    assert not project.has_receptors
    assert project.load_receptors() is None


def test_project_receptors_round_trip_points(tmp_path, point_receptor):
    project = Project(tmp_path / "proj")
    assert project.add_receptors([point_receptor]) == [point_receptor]
    assert project.receptors_path == tmp_path / "proj" / RECEPTORS_KEY
    assert project.has_receptors
    assert project.load_receptors() == [point_receptor]


def test_project_receptors_round_trip_preserves_groups(
    tmp_path, point_receptor, column_receptor, multipoint_receptor
):
    """r_idx grouping survives, so column/multipoint receptors come back intact."""
    project = Project(tmp_path / "proj")
    receptors = [point_receptor, column_receptor, multipoint_receptor]
    project.add_receptors(receptors)

    loaded = project.load_receptors()
    assert loaded == receptors
    assert [type(r) for r in loaded] == [type(r) for r in receptors]
    assert [len(r.coords()) for r in loaded] == [1, 2, 3]


def test_project_add_receptors_appends_only_new_ones(
    tmp_path, point_receptor, column_receptor
):
    project = Project(tmp_path / "proj")
    project.add_receptors([point_receptor])
    assert project.add_receptors([point_receptor, column_receptor]) == [column_receptor]
    assert project.load_receptors() == [point_receptor, column_receptor]


def test_project_add_receptors_appends_in_the_files_own_columns(
    tmp_path, point_receptor, column_receptor
):
    """A hand-written file keeps its columns, its r_idx values, and extra columns."""
    project = Project(tmp_path / "proj")
    project.directory.mkdir()
    text = (
        "r_idx,time,lati,long,zagl,scene\n"
        "1155,2023-01-01 12:00:00,40.77,-111.85,5.0,A\n"
    )
    project.receptors_path.write_text(text)

    assert project.add_receptors([point_receptor, column_receptor]) == [column_receptor]

    stored = project.receptors_path.read_text()
    assert stored.startswith(text)  # the original bytes are untouched
    lines = stored.splitlines()
    assert lines[1].split(",")[0] == "1155"
    assert lines[2].split(",")[0] == "1156" and lines[2].endswith(",")
    assert project.load_receptors() == [point_receptor, column_receptor]


def test_project_add_receptors_refuses_a_group_without_r_idx(tmp_path, column_receptor):
    project = Project(tmp_path / "proj")
    project.directory.mkdir()
    project.receptors_path.write_text("time,lati,long,zagl\n")
    with pytest.raises(ValueError, match="r_idx"):
        project.add_receptors([column_receptor])


def test_project_paths_are_paths(tmp_path):
    project = Project(tmp_path / "proj")
    assert isinstance(project.directory, Path)
