"""Tests for storage store backends and output key layout."""

import shutil
from types import SimpleNamespace

import pytest

from stilt.storage import (
    FsspecStore,
    LocalStore,
    ProjectFiles,
    SimulationFiles,
    Store,
    make_store,
)

# ---------------------------------------------------------------------------
# Protocol / factory
# ---------------------------------------------------------------------------


def test_local_store_is_runtime_checkable(tmp_path):
    store = LocalStore(tmp_path / "output")
    assert isinstance(store, Store)


def test_remote_store_is_runtime_checkable(tmp_path):
    store = FsspecStore(f"memory://proto-check-{tmp_path.name}")
    assert isinstance(store, Store)


def test_make_store_returns_local_store_for_local_path(tmp_path):
    store = make_store(tmp_path / "output")
    assert isinstance(store, LocalStore)


def test_make_store_returns_fsspec_store_for_remote_uri(tmp_path):
    store = make_store(f"memory://make-store-{tmp_path.name}")
    assert isinstance(store, FsspecStore)


# ---------------------------------------------------------------------------
# Key layout helpers (no store involved)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Contract tests — run against both LocalStore and FsspecStore(memory://)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["local", "memory"])
def any_store(request, tmp_path):
    if request.param == "local":
        return make_store(tmp_path / "output")
    return make_store(
        f"memory://contract-{tmp_path.name}", cache_dir=tmp_path / "cache"
    )


def test_store_contract_read_write_round_trip(any_store):
    any_store.write_bytes("nested/file.txt", b"hello")
    assert any_store.read_bytes("nested/file.txt") == b"hello"
    assert any_store.exists("nested/file.txt")
    assert any_store.list_prefix("nested") == ["nested/file.txt"]
    local = any_store.local_path("nested/file.txt")
    assert local.read_bytes() == b"hello"


def test_store_contract_nonexistent_key(any_store):
    assert not any_store.exists("does/not/exist.txt")
    assert any_store.list_prefix("does/not") == []


# ---------------------------------------------------------------------------
# LocalStore-specific tests
# ---------------------------------------------------------------------------


def test_local_store_file_round_trip(tmp_path):
    store = LocalStore(tmp_path / "output")
    store.write_bytes("data/test.bin", b"hello")

    assert store.read_bytes("data/test.bin") == b"hello"
    assert store.exists("data/test.bin")
    assert store.list_prefix("data") == ["data/test.bin"]
    assert store.local_path("data/test.bin") == (
        tmp_path / "output" / "data" / "test.bin"
    )


def _assert_relative_symlink(alias, target):
    assert alias.is_symlink()
    link_target = alias.readlink()
    assert not link_target.is_absolute()
    assert (alias.parent / link_target).resolve() == target.resolve()


def test_local_store_publish_simulation(tmp_path):
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
    store = LocalStore(tmp_path / "output")
    store.publish_simulation(sim)

    output = tmp_path / "output" / "simulations"
    canonical_traj = output / "by-id" / sim_id / traj_path.name
    canonical_foot = output / "by-id" / sim_id / foot_path.name
    particle_alias = output / "particles" / traj_path.name
    footprint_alias = output / "footprints" / foot_path.name

    assert (output / "by-id" / sim_id / log_path.name).read_text() == "log"
    assert canonical_traj.read_bytes() == b"traj"
    assert (output / "by-id" / sim_id / error_path.name).read_bytes() == b"error"
    assert canonical_foot.read_bytes() == b"foot"
    assert particle_alias.read_bytes() == b"traj"
    assert footprint_alias.read_bytes() == b"foot"
    _assert_relative_symlink(particle_alias, canonical_traj)
    _assert_relative_symlink(footprint_alias, canonical_foot)


def test_publish_file_is_atomic_via_tmp_then_replace(tmp_path, monkeypatch):
    """A failure mid-copy must leave the canonical key untouched."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"final-payload")
    store = LocalStore(tmp_path / "output")
    target = tmp_path / "output" / "data" / "out.bin"

    # Pre-populate the canonical key so we can verify it isn't clobbered on failure.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"old-payload")

    real_copy = shutil.copy2

    def _failing_copy(s, dst):
        # Simulate a writer that crashes after creating the .tmp.
        real_copy(s, dst)
        raise RuntimeError("boom")

    monkeypatch.setattr("stilt.storage.store.shutil.copy2", _failing_copy)

    with pytest.raises(RuntimeError, match="boom"):
        store.publish_file(src, "data/out.bin")

    # Canonical key is unchanged; tmp sibling cleaned up.
    assert target.read_bytes() == b"old-payload"
    assert not (target.parent / "out.bin.tmp").exists()


def test_publish_file_replaces_existing_local_target(tmp_path):
    src_v1 = tmp_path / "src1.bin"
    src_v2 = tmp_path / "src2.bin"
    src_v1.write_bytes(b"v1")
    src_v2.write_bytes(b"v2")
    store = LocalStore(tmp_path / "output")

    store.publish_file(src_v1, "data/out.bin")
    store.publish_file(src_v2, "data/out.bin")

    target = tmp_path / "output" / "data" / "out.bin"
    assert target.read_bytes() == b"v2"
    assert not (target.parent / "out.bin.tmp").exists()


def test_publish_simulation_index_alias_is_relative_symlink(tmp_path):
    """Index aliases are local-only relative symlink views of canonical files."""
    sim_id = "hrrr_202301011200_alias"
    sim_dir = tmp_path / "compute" / sim_id
    sim_dir.mkdir(parents=True)
    files = SimulationFiles(sim_dir, sim_id)
    traj_path = files.trajectory_path
    traj_path.write_bytes(b"traj-v1")

    sim = SimpleNamespace(
        id=sim_id,
        directory=sim_dir,
        log_path=files.log_path,
        trajectories_path=traj_path,
        error_trajectories_path=files.error_trajectory_path,
    )
    store = LocalStore(tmp_path / "output")
    store.publish_simulation(sim)

    canonical = tmp_path / "output" / "simulations" / "by-id" / sim_id / traj_path.name
    alias = tmp_path / "output" / "simulations" / "particles" / traj_path.name
    assert alias.is_file()
    assert alias.read_bytes() == b"traj-v1"
    _assert_relative_symlink(alias, canonical)


def test_publish_simulation_alias_atomic_on_republish(tmp_path):
    """Republished alias must never go through an unlinked window."""
    sim_id = "hrrr_202301011200_relink"
    sim_dir = tmp_path / "compute" / sim_id
    sim_dir.mkdir(parents=True)
    files = SimulationFiles(sim_dir, sim_id)
    traj_path = files.trajectory_path
    traj_path.write_bytes(b"traj-v1")

    sim = SimpleNamespace(
        id=sim_id,
        directory=sim_dir,
        log_path=files.log_path,
        trajectories_path=traj_path,
        error_trajectories_path=files.error_trajectory_path,
    )
    store = LocalStore(tmp_path / "output")
    store.publish_simulation(sim)

    alias = tmp_path / "output" / "simulations" / "particles" / traj_path.name
    assert alias.read_bytes() == b"traj-v1"

    # Republish with new content; alias should continue to point at the
    # canonical path as that file is atomically replaced.
    traj_path.write_bytes(b"traj-v2")
    store.publish_simulation(sim)
    assert alias.is_file()
    assert alias.is_symlink()
    assert alias.read_bytes() == b"traj-v2"
    _assert_relative_symlink(alias, tmp_path / "output" / files.key(traj_path))
    assert not (alias.parent / f"{traj_path.name}.tmp").exists()


# ---------------------------------------------------------------------------
# FsspecStore (remote / memory://) tests
# ---------------------------------------------------------------------------


def test_remote_store_memory_round_trip(tmp_path):
    store = FsspecStore(
        f"memory://artifact-store-{tmp_path.name}", cache_dir=tmp_path / "cache"
    )
    store.write_bytes("nested/file.txt", b"hello memory")

    assert store.read_bytes("nested/file.txt") == b"hello memory"
    assert store.exists("nested/file.txt")
    assert store.list_prefix("nested") == ["nested/file.txt"]

    local = store.local_path("nested/file.txt")
    assert local.read_bytes() == b"hello memory"
    assert local.is_file()


def test_remote_store_publish_simulation(tmp_path):
    sim_id = "hrrr_202301011200_site2"
    sim_dir = tmp_path / "compute" / sim_id
    sim_dir.mkdir(parents=True)
    files = SimulationFiles(sim_dir, sim_id)
    log_path = files.log_path
    traj_path = files.trajectory_path
    foot_path = files.footprint_path("slv")

    log_path.write_text("log")
    traj_path.write_bytes(b"traj")
    foot_path.write_bytes(b"foot")

    sim = SimpleNamespace(
        id=sim_id,
        directory=sim_dir,
        log_path=log_path,
        trajectories_path=traj_path,
        error_trajectories_path=files.error_trajectory_path,
    )
    store = FsspecStore(
        f"memory://publish-tests-{tmp_path.name}", cache_dir=tmp_path / "cache"
    )
    store.publish_simulation(sim)

    assert store.read_bytes(f"simulations/by-id/{sim_id}/{log_path.name}") == b"log"
    assert store.read_bytes(f"simulations/by-id/{sim_id}/{traj_path.name}") == b"traj"
    assert store.read_bytes(f"simulations/by-id/{sim_id}/{foot_path.name}") == b"foot"
    assert not store.exists(f"simulations/particles/{traj_path.name}")
    assert not store.exists(f"simulations/footprints/{foot_path.name}")
