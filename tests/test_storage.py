"""Tests for local artifact layout helpers in stilt.artifacts."""

from stilt.artifacts import (
    error_trajectory_path,
    footprint_index_path,
    footprint_path,
    particle_index_path,
    project_config_path,
    project_receptors_path,
    resolve_directory,
    simulation_dir_path,
    simulation_index_path,
    simulation_log_path,
    simulation_met_path,
    simulations_path,
    trajectory_path,
)


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


def test_resolve_directory_string_input(tmp_path):
    p = resolve_directory(str(tmp_path))
    assert p == tmp_path


def test_simulation_helpers_derive_from_directory(tmp_path):
    sim_id = "hrrr_202301011200_-111.85_40.77_5"
    sim_dir = tmp_path / sim_id
    sim_dir.mkdir()
    assert simulation_log_path(sim_dir) == sim_dir / "stilt.log"
    assert simulation_met_path(sim_dir) == sim_dir / "met"
    assert trajectory_path(sim_dir).suffix == ".parquet"
    assert "traj" in trajectory_path(sim_dir).name
    assert error_trajectory_path(sim_dir).name.endswith("_error.parquet")


def test_footprint_path_with_name(tmp_path):
    sim_id = "hrrr_202301011200_-111.85_40.77_5"
    sim_dir = tmp_path / sim_id
    sim_dir.mkdir()
    fp = footprint_path(sim_dir, "slv")
    assert fp.name.endswith("_slv_foot.nc")
    assert fp.parent == sim_dir


def test_footprint_path_no_name(tmp_path):
    sim_id = "hrrr_202301011200_-111.85_40.77_5"
    sim_dir = tmp_path / sim_id
    sim_dir.mkdir()
    fp = footprint_path(sim_dir, "")
    assert "_foot.nc" in fp.name
    assert not fp.name.startswith("_")


def test_project_layout_helpers(tmp_path):
    assert project_config_path(tmp_path) == tmp_path / "config.yaml"
    assert project_receptors_path(tmp_path) == tmp_path / "receptors.csv"
    assert simulations_path(tmp_path) == tmp_path / "simulations"
    assert simulation_index_path(tmp_path) == tmp_path / "simulations" / "by-id"
    assert particle_index_path(tmp_path, "traj.parquet") == (
        tmp_path / "simulations" / "particles" / "traj.parquet"
    )
    assert footprint_index_path(tmp_path, "foot.nc") == (
        tmp_path / "simulations" / "footprints" / "foot.nc"
    )


def test_simulation_dir_path(tmp_path):
    sid = "hrrr_202301011200_-111.85_40.77_5"
    assert simulation_dir_path(tmp_path, sid) == tmp_path / "simulations" / "by-id" / sid
