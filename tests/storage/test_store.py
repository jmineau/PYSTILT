"""Tests for storage store backends and durable key layout."""

from types import SimpleNamespace

from stilt.storage import (
    FsspecStore,
    ProjectFiles,
    SimulationFiles,
    Store,
)


def test_fsspec_store_is_runtime_checkable(tmp_path):
    store = FsspecStore(tmp_path / "output")
    assert isinstance(store, Store)


def test_store_key_helpers_return_canonical_layout():
    assert ProjectFiles.config_key() == "config.yaml"
    assert ProjectFiles.receptors_key() == "receptors.csv"
    assert SimulationFiles.key_prefix_for("sim-1") == "simulations/by-id/sim-1"
    assert SimulationFiles.key_for("sim-1", "traj.parquet") == (
        "simulations/by-id/sim-1/traj.parquet"
    )
    assert ProjectFiles.particle_index_key("traj.parquet") == (
        "simulations/particles/traj.parquet"
    )
    assert ProjectFiles.footprint_index_key("foot.nc") == (
        "simulations/footprints/foot.nc"
    )


def test_simulation_index_db_path_uses_simulations_subdir(tmp_path):
    assert (
        ProjectFiles(tmp_path).index_db_path
        == tmp_path / "simulations" / "index.sqlite"
    )


def test_fsspec_store_file_round_trip(tmp_path):
    store = FsspecStore(tmp_path / "output")
    store.write_bytes("data/test.bin", b"hello")

    assert store.read_bytes("data/test.bin") == b"hello"
    assert store.exists("data/test.bin")
    assert store.list_prefix("data") == ["data/test.bin"]
    assert store.local_path("data/test.bin") == (
        tmp_path / "output" / "data" / "test.bin"
    )


def test_fsspec_store_memory_round_trip(tmp_path):
    store = FsspecStore("memory://artifact-store-tests", cache_dir=tmp_path / "cache")
    store.write_bytes("nested/file.txt", b"hello memory")

    assert store.read_bytes("nested/file.txt") == b"hello memory"
    assert store.exists("nested/file.txt")
    assert store.list_prefix("nested") == ["nested/file.txt"]

    local = store.local_path("nested/file.txt")
    assert local.read_bytes() == b"hello memory"
    assert local.is_file()


def test_fsspec_store_publish_simulation_to_file_store(tmp_path):
    sim_id = "hrrr_202301011200_site1"
    sim_dir = tmp_path / "compute" / sim_id
    sim_dir.mkdir(parents=True)
    files = SimulationFiles(sim_dir, sim_id)
    log_path = files.log_path
    traj_path = files.trajectory_path
    error_path = files.error_trajectory_path
    foot_path = files.footprint_path("slv")

    log_path.write_text("log")
    traj_path.write_bytes(b"traj")
    error_path.write_bytes(b"error")
    foot_path.write_bytes(b"foot")

    sim = SimpleNamespace(
        id=sim_id,
        directory=sim_dir,
        log_path=log_path,
        trajectories_path=traj_path,
        error_trajectories_path=error_path,
    )
    store = FsspecStore(tmp_path / "output")
    store.publish_simulation(sim)

    output = tmp_path / "output" / "simulations"
    assert (output / "by-id" / sim_id / log_path.name).read_text() == "log"
    assert (output / "by-id" / sim_id / traj_path.name).read_bytes() == b"traj"
    assert (output / "particles" / traj_path.name).read_bytes() == b"traj"
    assert (output / "by-id" / sim_id / error_path.name).read_bytes() == b"error"
    assert (output / "by-id" / sim_id / foot_path.name).read_bytes() == b"foot"
    assert (output / "footprints" / foot_path.name).read_bytes() == b"foot"


def test_fsspec_store_publish_simulation_to_memory_store(tmp_path):
    sim_id = "hrrr_202301011200_site2"
    sim_dir = tmp_path / "compute" / sim_id
    sim_dir.mkdir(parents=True)
    files = SimulationFiles(sim_dir, sim_id)
    log_path = files.log_path
    traj_path = files.trajectory_path

    log_path.write_text("log")
    traj_path.write_bytes(b"traj")

    sim = SimpleNamespace(
        id=sim_id,
        directory=sim_dir,
        log_path=log_path,
        trajectories_path=traj_path,
        error_trajectories_path=files.error_trajectory_path,
    )
    store = FsspecStore("memory://publish-tests", cache_dir=tmp_path / "cache")
    store.publish_simulation(sim)

    assert store.read_bytes(f"simulations/by-id/{sim_id}/{log_path.name}") == b"log"
    assert store.read_bytes(f"simulations/by-id/{sim_id}/{traj_path.name}") == b"traj"
    assert store.read_bytes(f"simulations/particles/{traj_path.name}") == b"traj"
