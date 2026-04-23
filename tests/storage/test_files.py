"""Tests for storage file layout helpers."""

from stilt.storage import (
    ProjectFiles,
    SimulationFiles,
    resolve_directory,
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
    files = SimulationFiles(sim_dir, sim_id)
    assert files.log_path == sim_dir / "stilt.log"
    assert files.met_dir == sim_dir / "met"
    assert files.trajectory_path.suffix == ".parquet"
    assert "traj" in files.trajectory_path.name
    assert files.error_trajectory_path.name.endswith("_error.parquet")


def test_footprint_path_with_name(tmp_path):
    sim_id = "hrrr_202301011200_-111.85_40.77_5"
    sim_dir = tmp_path / sim_id
    sim_dir.mkdir()
    fp = SimulationFiles(sim_dir, sim_id).footprint_path("slv")
    assert fp.name.endswith("_slv_foot.nc")
    assert fp.parent == sim_dir


def test_footprint_path_no_name(tmp_path):
    sim_id = "hrrr_202301011200_-111.85_40.77_5"
    sim_dir = tmp_path / sim_id
    sim_dir.mkdir()
    fp = SimulationFiles(sim_dir, sim_id).footprint_path("")
    assert "_foot.nc" in fp.name
    assert not fp.name.startswith("_")


def test_project_layout_helpers(tmp_path):
    files = ProjectFiles(tmp_path)
    assert files.config_path == tmp_path / "config.yaml"
    assert files.receptors_path == tmp_path / "receptors.csv"
    assert files.simulations_dir == tmp_path / "simulations"
    assert files.by_id_dir == tmp_path / "simulations" / "by-id"
    assert files.particle_index_dir / "traj.parquet" == (
        tmp_path / "simulations" / "particles" / "traj.parquet"
    )
    assert files.footprint_index_dir / "foot.nc" == (
        tmp_path / "simulations" / "footprints" / "foot.nc"
    )


def test_simulation_dir_path(tmp_path):
    sid = "hrrr_202301011200_-111.85_40.77_5"
    assert ProjectFiles(tmp_path).simulation(sid).directory == (
        tmp_path / "simulations" / "by-id" / sid
    )
