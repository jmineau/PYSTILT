from types import SimpleNamespace

from stilt.artifacts import (
    ArtifactStore,
    FsspecArtifactStore,
    error_trajectory_path,
    footprint_index_key,
    footprint_path,
    particle_index_key,
    project_config_key,
    project_receptors_key,
    simulation_artifact_key,
    simulation_artifact_prefix,
    simulation_log_path,
    simulation_state_db_path,
    trajectory_path,
)


def test_fsspec_artifact_store_is_runtime_checkable(tmp_path):
    store = FsspecArtifactStore(tmp_path / "output")
    assert isinstance(store, ArtifactStore)


def test_artifact_key_helpers_return_canonical_layout():
    assert project_config_key() == "config.yaml"
    assert project_receptors_key() == "receptors.csv"
    assert simulation_artifact_prefix("sim-1") == "simulations/by-id/sim-1"
    assert simulation_artifact_key("sim-1", "traj.parquet") == (
        "simulations/by-id/sim-1/traj.parquet"
    )
    assert particle_index_key("traj.parquet") == "simulations/particles/traj.parquet"
    assert footprint_index_key("foot.nc") == "simulations/footprints/foot.nc"


def test_simulation_state_db_path_uses_simulations_subdir(tmp_path):
    assert (
        simulation_state_db_path(tmp_path) == tmp_path / "simulations" / "state.sqlite"
    )


def test_fsspec_artifact_store_file_round_trip(tmp_path):
    store = FsspecArtifactStore(tmp_path / "output")
    store.write_bytes("data/test.bin", b"hello")

    assert store.read_bytes("data/test.bin") == b"hello"
    assert store.exists("data/test.bin")
    assert store.list_prefix("data") == ["data/test.bin"]
    assert store.local_path("data/test.bin") == (
        tmp_path / "output" / "data" / "test.bin"
    )


def test_fsspec_artifact_store_memory_round_trip(tmp_path):
    store = FsspecArtifactStore(
        "memory://artifact-store-tests", cache_dir=tmp_path / "cache"
    )
    store.write_bytes("nested/file.txt", b"hello memory")

    assert store.read_bytes("nested/file.txt") == b"hello memory"
    assert store.exists("nested/file.txt")
    assert store.list_prefix("nested") == ["nested/file.txt"]

    local = store.local_path("nested/file.txt")
    assert local.read_bytes() == b"hello memory"
    assert local.is_file()


def test_fsspec_artifact_store_publish_simulation_to_file_store(tmp_path):
    sim_id = "hrrr_202301011200_site1"
    sim_dir = tmp_path / "compute" / sim_id
    sim_dir.mkdir(parents=True)
    log_path = simulation_log_path(sim_dir, sim_id)
    traj_path = trajectory_path(sim_dir, sim_id)
    error_path = error_trajectory_path(sim_dir, sim_id)
    foot_path = footprint_path(sim_dir, "slv", sim_id=sim_id)

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
    store = FsspecArtifactStore(tmp_path / "output")
    store.publish_simulation(sim)

    output = tmp_path / "output" / "simulations"
    assert (output / "by-id" / sim_id / log_path.name).read_text() == "log"
    assert (output / "by-id" / sim_id / traj_path.name).read_bytes() == b"traj"
    assert (output / "particles" / traj_path.name).read_bytes() == b"traj"
    assert (output / "by-id" / sim_id / error_path.name).read_bytes() == b"error"
    assert (output / "by-id" / sim_id / foot_path.name).read_bytes() == b"foot"
    assert (output / "footprints" / foot_path.name).read_bytes() == b"foot"


def test_fsspec_artifact_store_publish_simulation_to_memory_store(tmp_path):
    sim_id = "hrrr_202301011200_site2"
    sim_dir = tmp_path / "compute" / sim_id
    sim_dir.mkdir(parents=True)
    log_path = simulation_log_path(sim_dir, sim_id)
    traj_path = trajectory_path(sim_dir, sim_id)

    log_path.write_text("log")
    traj_path.write_bytes(b"traj")

    sim = SimpleNamespace(
        id=sim_id,
        directory=sim_dir,
        log_path=log_path,
        trajectories_path=traj_path,
        error_trajectories_path=error_trajectory_path(sim_dir, sim_id),
    )
    store = FsspecArtifactStore("memory://publish-tests", cache_dir=tmp_path / "cache")
    store.publish_simulation(sim)

    assert store.read_bytes(f"simulations/by-id/{sim_id}/{log_path.name}") == b"log"
    assert store.read_bytes(f"simulations/by-id/{sim_id}/{traj_path.name}") == b"traj"
    assert store.read_bytes(f"simulations/particles/{traj_path.name}") == b"traj"
