"""Tests for stilt.workers — SimulationTask, run_worker, and _run_batch."""

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from stilt.artifacts import FsspecArtifactStore
from stilt.config import STILTParams
from stilt.errors import SimulationError
from stilt.meteorology import MetStream
from stilt.receptor import Receptor
from stilt.repositories import ArtifactSummary, SimulationClaim, SQLiteRepository
from stilt.simulation import SimID
from stilt.workers import SimulationTask, _run_batch, run_worker

InMemoryRepository = SQLiteRepository.in_memory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def receptor():
    return Receptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


@pytest.fixture
def sim_id(receptor):
    return SimID.from_parts("hrrr", receptor)


@pytest.fixture
def met(tmp_path):
    return MetStream(
        "hrrr",
        directory=tmp_path / "met",
        file_format="%Y%m%d_%H",
        file_tres="1h",
    )


@pytest.fixture
def params():
    return STILTParams(n_hours=-24, numpar=10)


@pytest.fixture
def repository(tmp_path, sim_id, receptor):
    repo = InMemoryRepository(tmp_path)
    repo.register_many([(str(sim_id), receptor)])
    repo.claim_pending()  # mark as running so mark_*() calls don't no-op
    return repo


@pytest.fixture
def artifact_store(tmp_path):
    return FsspecArtifactStore(tmp_path)


@pytest.fixture
def run_args(tmp_path, sim_id, receptor, met, params, artifact_store, repository):
    return SimulationTask(
        compute_root=tmp_path,
        sim_id=sim_id,
        meteorology=met,
        receptor=receptor,
        params=params,
        artifact_store=artifact_store,
        repository=repository,
    )


def _fake_simulation(sim_id: SimID, traj_path: Path, succeed: bool = True):
    """Build a MagicMock that mimics a Simulation instance."""
    sim = MagicMock()
    sim.id = sim_id
    sim.directory = traj_path.parent
    sim.trajectories_path = traj_path
    sim.log_path = traj_path.parent / "stilt.log"
    sim.error_trajectories_path = traj_path.parent / "error.parquet"
    sim.footprint_path.side_effect = (
        lambda name="": traj_path.parent / f"{name or 'foot'}.nc"
    )
    sim._artifact_path.side_effect = lambda path: path if Path(path).exists() else None
    if succeed:
        sim.run_trajectories.return_value = None
    else:
        sim.run_trajectories.side_effect = SimulationError("HYSPLIT failed")
    return sim


# ---------------------------------------------------------------------------
# SimulationTask validation
# ---------------------------------------------------------------------------


def test_run_args_requires_artifact_store_and_repository(
    tmp_path, sim_id, receptor, met, params
):
    with pytest.raises(ValidationError):
        SimulationTask(
            compute_root=tmp_path,
            sim_id=sim_id,
            meteorology=met,
            receptor=receptor,
            params=params,
            # artifact_store and repository omitted
        )


def test_run_args_foot_configs_defaults_to_none(run_args):
    assert run_args.foot_configs is None


# ---------------------------------------------------------------------------
# run_worker — trajectory-only path, success
# ---------------------------------------------------------------------------


def test_run_worker_success_returns_complete(tmp_path, run_args, monkeypatch):
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()
    log_path = tmp_path / "stilt.log"
    log_path.write_text("ok")
    error_path = tmp_path / "sim_error.parquet"
    error_path.touch()
    fake_sim = _fake_simulation(run_args.sim_id, traj_path, succeed=True)
    fake_sim.log_path = log_path
    fake_sim.error_trajectories_path = error_path
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(run_args)

    assert result.status == "complete"
    assert result.wrote_traj is True
    assert result.sim_id == run_args.sim_id
    assert result.log_path == log_path
    assert result.error_traj_path == error_path
    assert run_args.repository.artifact_summary(
        str(run_args.sim_id)
    ) == ArtifactSummary(
        traj_present=True,
        error_traj_present=True,
        log_present=True,
        footprints={},
    )
    attempts = run_args.repository.list_attempts(str(run_args.sim_id))
    assert len(attempts) == 1
    assert attempts[0].outcome == "complete"


def test_run_worker_success_marks_trajectory_complete(tmp_path, run_args, monkeypatch):
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()
    fake_sim = _fake_simulation(run_args.sim_id, traj_path, succeed=True)
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    run_worker(run_args)

    assert run_args.repository.traj_status(str(run_args.sim_id)) == "complete"


def test_run_worker_success_creates_traj_symlink(tmp_path, run_args, monkeypatch):
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()
    fake_sim = _fake_simulation(run_args.sim_id, traj_path, succeed=True)
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    run_worker(run_args)

    particles_dir = tmp_path / "simulations" / "particles"
    links = list(particles_dir.iterdir())
    assert len(links) == 1
    assert links[0].is_symlink()


def test_run_worker_publishes_outputs_via_artifact_store(
    tmp_path, run_args, monkeypatch
):
    """Durable artifacts are always published through the artifact store."""
    compute_dir = tmp_path / "compute" / str(run_args.sim_id)
    compute_dir.mkdir(parents=True)
    traj_path = compute_dir / f"{run_args.sim_id}_traj.parquet"
    traj_path.write_bytes(b"traj")
    log_path = compute_dir / "stilt.log"
    log_path.write_text("ok")
    error_path = compute_dir / f"{run_args.sim_id}_error.parquet"
    error_path.write_bytes(b"error")

    fake_sim = _fake_simulation(run_args.sim_id, traj_path, succeed=True)
    fake_sim.log_path = log_path
    fake_sim.error_trajectories_path = error_path
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    store = FsspecArtifactStore(tmp_path / "output")
    args = run_args.model_copy(
        update={
            "compute_root": compute_dir.parent,
            "artifact_store": store,
        }
    )

    result = run_worker(args)

    assert result.status == "complete"
    assert (
        store.read_bytes(f"simulations/by-id/{run_args.sim_id}/{traj_path.name}")
        == b"traj"
    )
    assert (
        store.read_bytes(f"simulations/by-id/{run_args.sim_id}/{log_path.name}")
        == b"ok"
    )
    assert (
        store.read_bytes(f"simulations/by-id/{run_args.sim_id}/{error_path.name}")
        == b"error"
    )


# ---------------------------------------------------------------------------
# run_worker — trajectory-only path, failure
# ---------------------------------------------------------------------------


def test_run_worker_simulation_error_returns_failed(tmp_path, run_args, monkeypatch):
    traj_path = tmp_path / "sim_traj.parquet"
    fake_sim = _fake_simulation(run_args.sim_id, traj_path, succeed=False)
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(run_args)

    assert result.status == "failed"
    assert result.wrote_traj is False
    assert result.error is not None


def test_run_worker_failure_marks_trajectory_failed(tmp_path, run_args, monkeypatch):
    traj_path = tmp_path / "sim_traj.parquet"
    fake_sim = _fake_simulation(run_args.sim_id, traj_path, succeed=False)
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    run_worker(run_args)

    assert run_args.repository.traj_status(str(run_args.sim_id)) == "failed"
    attempts = run_args.repository.list_attempts(str(run_args.sim_id))
    assert len(attempts) == 1
    assert attempts[0].outcome == "failed"


def test_run_worker_simulation_error_appends_pystilt_log(
    tmp_path, run_args, monkeypatch
):
    traj_path = tmp_path / "sim_traj.parquet"
    fake_sim = _fake_simulation(run_args.sim_id, traj_path, succeed=False)
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    run_worker(run_args)

    text = fake_sim.log_path.read_text()
    assert "=== PYSTILT ERROR ===" in text
    assert "Phase: trajectory" in text
    assert "Message: HYSPLIT failed" in text


def test_run_worker_unexpected_error_returns_error_status(
    tmp_path, run_args, monkeypatch
):
    traj_path = tmp_path / "sim_traj.parquet"
    fake_sim = _fake_simulation(run_args.sim_id, traj_path, succeed=True)
    fake_sim.run_trajectories.side_effect = RuntimeError("unexpected")
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(run_args)

    assert result.status == "error"
    assert result.wrote_traj is False
    assert run_args.repository.traj_status(str(run_args.sim_id)) == "failed"
    assert str(run_args.sim_id) not in run_args.repository.pending_trajectories()
    attempts = run_args.repository.list_attempts(str(run_args.sim_id))
    assert len(attempts) == 1
    assert attempts[0].outcome == "error"
    assert attempts[0].terminal is True


# ---------------------------------------------------------------------------
# _run_batch — sequential
# ---------------------------------------------------------------------------


def test_run_batch_empty_chunk_returns_empty():
    assert _run_batch([], n_cores=1) == []


def test_run_batch_sequential_returns_all_results(tmp_path, run_args, monkeypatch):
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()
    fake_sim = _fake_simulation(run_args.sim_id, traj_path, succeed=True)
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    results = _run_batch([run_args], n_cores=1)

    assert len(results) == 1
    assert results[0].status == "complete"


def test_run_batch_sigterm_marks_failed_and_stops(tmp_path, run_args, monkeypatch):
    """KeyboardInterrupt (from SIGTERM handler) stops batch and leaves the sim retryable."""
    call_count = {"n": 0}

    def fake_run_worker(args):
        call_count["n"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr("stilt.workers.run_worker", fake_run_worker)

    # Pass two copies — only one should execute before break
    results = _run_batch([run_args, run_args], n_cores=1)

    assert call_count["n"] == 1
    assert results == []  # interrupted before any success was appended
    assert run_args.repository.traj_status(str(run_args.sim_id)) == "pending"


# ---------------------------------------------------------------------------
# run_worker — footprint path
# ---------------------------------------------------------------------------


def _foot_run_args(base_args, foot_config):
    """Return a SimulationTask copy with foot_configs set."""
    return base_args.model_copy(update={"foot_configs": {"slv": foot_config}})


@pytest.fixture
def foot_config():
    from stilt.config import FootprintConfig, Grid

    return FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)
    )


def test_run_worker_footprint_success_marks_footprint_complete(
    tmp_path, run_args, foot_config, monkeypatch
):
    args = _foot_run_args(run_args, foot_config)
    traj_path = tmp_path / "sim_traj.parquet"
    log_path = tmp_path / "stilt.log"
    log_path.write_text("ok")
    fake_sim = MagicMock()
    fake_sim.id = run_args.sim_id
    fake_sim.directory = tmp_path
    fake_sim.trajectories_path = traj_path
    fake_sim.log_path = log_path
    fake_sim.error_trajectories_path = tmp_path / "error.parquet"
    fake_sim.footprint_path.return_value = tmp_path / "foot.nc"
    fake_sim._artifact_path.side_effect = (
        lambda path: path if Path(path).exists() else None
    )
    fake_foot = MagicMock()
    fake_sim.generate_footprint.return_value = fake_foot
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(args)

    assert result.status == "complete"
    assert len(result.foot_paths) == 1
    assert result.footprint_statuses == {"slv": "complete"}
    assert result.log_path == log_path
    assert args.repository.footprint_completed(str(args.sim_id), "slv")


def test_run_worker_footprint_returns_empty_when_no_paths(
    tmp_path, run_args, foot_config, monkeypatch
):
    """If generate_footprint returns None the result status is 'complete-empty'."""
    args = _foot_run_args(run_args, foot_config)
    traj_path = tmp_path / "sim_traj.parquet"
    fake_sim = MagicMock()
    fake_sim.id = run_args.sim_id
    fake_sim.directory = tmp_path
    fake_sim.trajectories_path = traj_path
    fake_sim.footprint_path.return_value = tmp_path / "foot.nc"
    fake_sim.log_path = tmp_path / "stilt.log"
    fake_sim.error_trajectories_path = tmp_path / "error.parquet"
    fake_sim._artifact_path.side_effect = (
        lambda path: path if Path(path).exists() else None
    )
    fake_sim.generate_footprint.return_value = None
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(args)

    assert result.status == "complete-empty"
    assert result.foot_paths == []
    assert result.empty_footprints == ["slv"]
    assert result.footprint_statuses == {"slv": "complete-empty"}
    assert args.repository.footprint_completed(str(args.sim_id), "slv")
    assert args.repository.artifact_summary(str(args.sim_id)).footprints == {
        "slv": "complete-empty"
    }


def test_run_worker_footprint_error_variant_empty_records_metadata(
    tmp_path, run_args, foot_config, monkeypatch
):
    """Error-footprint empty outputs are marked as complete-empty."""
    cfg = foot_config.model_copy(update={"error": True})
    args = _foot_run_args(run_args, cfg)
    traj_path = tmp_path / "sim_traj.parquet"

    fake_sim = MagicMock()
    fake_sim.id = run_args.sim_id
    fake_sim.directory = tmp_path
    fake_sim.trajectories_path = traj_path
    fake_sim.footprint_path.return_value = tmp_path / "foot.nc"
    fake_sim.log_path = tmp_path / "stilt.log"
    fake_sim.error_trajectories_path = tmp_path / "error.parquet"
    fake_sim._artifact_path.side_effect = (
        lambda path: path if Path(path).exists() else None
    )

    def _generate(name, fc, write=False, error=False):
        return None if error else MagicMock()

    fake_sim.generate_footprint.side_effect = _generate
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(args)

    assert result.status == "complete"
    assert result.footprint_statuses == {
        "slv": "complete",
        "slv_error": "complete-empty",
    }
    assert "slv_error" in result.empty_footprints
    assert args.repository.footprint_completed(str(args.sim_id), "slv")
    assert args.repository.footprint_completed(str(args.sim_id), "slv_error")


def test_run_worker_footprint_failure_marks_footprint_failed(
    tmp_path, run_args, foot_config, monkeypatch
):
    args = _foot_run_args(run_args, foot_config)
    traj_path = tmp_path / "sim_traj.parquet"
    fake_sim = MagicMock()
    fake_sim.id = run_args.sim_id
    fake_sim.directory = tmp_path
    fake_sim.trajectories_path = traj_path
    fake_sim.log_path = tmp_path / "stilt.log"
    fake_sim.error_trajectories_path = tmp_path / "error.parquet"
    fake_sim._artifact_path.side_effect = (
        lambda path: path if Path(path).exists() else None
    )
    fake_sim.generate_footprint.side_effect = RuntimeError("footprint failed")
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(args)

    assert result.status == "error"
    assert result.error is not None
    assert args.repository.traj_status(str(args.sim_id)) == "failed"
    assert str(args.sim_id) not in args.repository.pending_trajectories()
    assert not args.repository.footprint_completed(str(args.sim_id), "slv")
    attempts = args.repository.list_attempts(str(args.sim_id))
    assert len(attempts) == 1
    assert attempts[0].outcome == "error"
    assert attempts[0].terminal is True


def test_run_worker_footprint_failure_appends_to_existing_log(
    tmp_path, run_args, foot_config, monkeypatch
):
    args = _foot_run_args(run_args, foot_config)
    traj_path = tmp_path / "sim_traj.parquet"
    log_path = tmp_path / "stilt.log"
    log_path.write_text("Complete Hysplit\n")
    fake_sim = MagicMock()
    fake_sim.id = run_args.sim_id
    fake_sim.directory = tmp_path
    fake_sim.trajectories_path = traj_path
    fake_sim.log_path = log_path
    fake_sim.error_trajectories_path = tmp_path / "error.parquet"
    fake_sim._artifact_path.side_effect = (
        lambda path: path if Path(path).exists() else None
    )
    fake_sim.generate_footprint.side_effect = RuntimeError("footprint failed")
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    run_worker(args)

    text = log_path.read_text()
    assert "Complete Hysplit" in text
    assert "=== PYSTILT ERROR ===" in text
    assert "Phase: footprint:slv" in text
    assert "Message: footprint failed" in text


def test_run_worker_footprint_traj_written_marks_traj_complete(
    tmp_path, run_args, foot_config, monkeypatch
):
    """mark_trajectory_complete is called when traj didn't exist and then appears."""
    args = _foot_run_args(run_args, foot_config)
    traj_path = tmp_path / "sim_traj.parquet"
    fake_sim = MagicMock()
    fake_sim.id = run_args.sim_id
    fake_sim.directory = tmp_path
    fake_sim.trajectories_path = traj_path
    fake_sim.log_path = tmp_path / "stilt.log"
    fake_sim.error_trajectories_path = tmp_path / "error.parquet"
    fake_sim._artifact_path.side_effect = (
        lambda path: path if Path(path).exists() else None
    )

    def _generate(name, fc, write=False, error=False):
        traj_path.touch()
        return MagicMock()

    fake_sim.generate_footprint.side_effect = _generate
    fake_sim.footprint_path.return_value = tmp_path / "foot.nc"
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(args)

    assert result.wrote_traj is True
    assert args.repository.traj_status(str(args.sim_id)) == "complete"


def test_run_worker_footprint_existing_traj_marks_traj_complete(
    tmp_path, run_args, foot_config, monkeypatch
):
    """A pre-existing trajectory is restored to complete during footprint-only runs."""
    args = _foot_run_args(run_args, foot_config)
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()

    fake_sim = MagicMock()
    fake_sim.id = run_args.sim_id
    fake_sim.directory = tmp_path
    fake_sim.trajectories_path = traj_path
    fake_sim.log_path = tmp_path / "stilt.log"
    fake_sim.error_trajectories_path = tmp_path / "error.parquet"
    fake_sim.generate_footprint.return_value = MagicMock()
    fake_sim.footprint_path.return_value = tmp_path / "foot.nc"
    fake_sim._artifact_path.side_effect = (
        lambda path: path if Path(path).exists() else None
    )
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(args)

    assert result.status == "complete"
    assert result.wrote_traj is False
    assert args.repository.traj_status(str(args.sim_id)) == "complete"


def test_run_worker_footprint_remote_existing_traj_marks_traj_complete(
    tmp_path, run_args, foot_config, monkeypatch
):
    """Storage-backed trajectories count as existing for footprint-only runs."""
    args = _foot_run_args(run_args, foot_config)
    traj_path = tmp_path / "missing_local_traj.parquet"
    remote_traj = tmp_path / "remote_traj.parquet"
    remote_traj.touch()

    fake_sim = MagicMock()
    fake_sim.id = run_args.sim_id
    fake_sim.directory = tmp_path
    fake_sim.trajectories_path = traj_path
    fake_sim.log_path = tmp_path / "stilt.log"
    fake_sim.error_trajectories_path = tmp_path / "error.parquet"
    fake_sim.footprint_path.return_value = tmp_path / "foot.nc"
    fake_sim._artifact_path.side_effect = (
        lambda path: remote_traj if Path(path) == traj_path else None
    )
    fake_sim.generate_footprint.return_value = MagicMock()
    monkeypatch.setattr("stilt.workers.Simulation", lambda **kw: fake_sim)

    result = run_worker(args)

    assert result.status == "complete"
    assert result.wrote_traj is False
    assert args.repository.traj_status(str(args.sim_id)) == "complete"


# ---------------------------------------------------------------------------
# pull_worker_loop
# ---------------------------------------------------------------------------


class TestPullWorkerLoop:
    """Tests for pull_worker_loop using a fake model with shared-memory SQLite."""

    def _make_model(self, tmp_path, n_sims=3):
        """Build a fake model with a shared-memory SQLite repo and n_sims pending."""
        from stilt.workers import pull_worker_loop

        repo = InMemoryRepository(tmp_path)
        pairs = []
        for i in range(n_sims):
            r = Receptor(
                time=f"2023010112{i:02d}",
                longitude=-111.85,
                latitude=40.77,
                altitude=5.0,
            )
            sid = f"hrrr_2023010112{i:02d}_-111.85_40.77_5"
            pairs.append((sid, r))
        repo.register_many(pairs)

        model = MagicMock()
        model.repository = repo

        return model, [sid for sid, _ in pairs], pull_worker_loop

    def test_drains_queue(self, tmp_path, monkeypatch):
        """pull_worker_loop drains all pending sims and exits."""
        model, sids, pull_worker_loop = self._make_model(tmp_path, n_sims=3)

        run_args_stub = MagicMock(spec=SimulationTask)
        model._build_run_args.return_value = run_args_stub

        chunk_calls = []
        monkeypatch.setattr(
            "stilt.workers._run_batch",
            lambda chunk, n_cores=1: chunk_calls.append(len(chunk)) or [],
        )

        pull_worker_loop(model, n_cores=3, follow=False)

        assert model._build_run_args.call_count == 3
        assert sum(chunk_calls) == 3
        assert model.repository.claim_pending_claims(1) == []  # queue is drained

    def test_releases_skipped_claims(self, tmp_path, monkeypatch):
        """pull_worker_loop releases claims when _build_run_args returns None."""
        model, sids, pull_worker_loop = self._make_model(tmp_path, n_sims=2)

        run_args_stub = MagicMock(spec=SimulationTask)

        def build_side_effect(sid):
            if sid == sids[0]:
                return None
            return run_args_stub

        model._build_run_args.side_effect = build_side_effect

        chunk_calls = []
        monkeypatch.setattr(
            "stilt.workers._run_batch",
            lambda chunk, n_cores=1: chunk_calls.append(len(chunk)) or [],
        )

        pull_worker_loop(model, n_cores=2, follow=False)

        # The skipped sim should be back to pending
        assert model.repository.traj_status(sids[0]) == "pending"
        assert sum(chunk_calls) == 1

    def test_exits_on_empty_queue_no_follow(self, tmp_path, monkeypatch):
        """pull_worker_loop exits immediately when queue is empty and follow=False."""
        model, _, pull_worker_loop = self._make_model(tmp_path, n_sims=0)

        monkeypatch.setattr(
            "stilt.workers._run_batch",
            lambda chunk, n_cores=1: [],
        )

        pull_worker_loop(model, n_cores=1, follow=False)

    def test_follow_polls_then_processes(self, tmp_path, monkeypatch):
        """pull_worker_loop with follow=True polls when empty, processes when work arrives."""
        model, _, pull_worker_loop = self._make_model(tmp_path, n_sims=0)

        call_count = [0]
        original_claims = model.repository.claim_pending_claims

        def counting_claims(n=1, worker_id="legacy", lease_ttl=1800.0):
            call_count[0] += 1
            result = original_claims(n, worker_id=worker_id, lease_ttl=lease_ttl)
            if call_count[0] == 3 and not result:
                r = Receptor(
                    time="202301011259",
                    longitude=-111.85,
                    latitude=40.77,
                    altitude=5.0,
                )
                model.repository.register_many(
                    [("hrrr_202301011259_-111.85_40.77_5", r)]
                )
                return original_claims(n, worker_id=worker_id, lease_ttl=lease_ttl)
            if call_count[0] > 5:
                raise KeyboardInterrupt
            return result

        model.repository.claim_pending_claims = counting_claims

        run_args_stub = MagicMock(spec=SimulationTask)
        model._build_run_args.return_value = run_args_stub

        monkeypatch.setattr(
            "stilt.workers._run_batch",
            lambda chunk, n_cores=1: [],
        )

        with pytest.raises(KeyboardInterrupt):
            pull_worker_loop(model, n_cores=1, follow=True, poll_interval=0.01)

        assert call_count[0] >= 4
        assert model._build_run_args.call_count >= 1

    def test_follow_uses_idle_backoff(self, tmp_path, monkeypatch):
        """Idle follow-mode polls back off up to a capped interval."""
        model, _, pull_worker_loop = self._make_model(tmp_path, n_sims=0)
        sleep_calls: list[float] = []
        claim_count = {"n": 0}

        def no_work(n=1, worker_id="legacy", lease_ttl=1800.0):
            del n, worker_id, lease_ttl
            claim_count["n"] += 1
            if claim_count["n"] > 3:
                raise KeyboardInterrupt
            return []

        model.repository.claim_pending_claims = no_work
        monkeypatch.setattr("stilt.workers.time.sleep", sleep_calls.append)

        with pytest.raises(KeyboardInterrupt):
            pull_worker_loop(model, n_cores=1, follow=True, poll_interval=0.5)

        assert sleep_calls == [0.5, 1.0, 2.0]

    def test_transactional_claim_loop_uses_uow_when_available(
        self, tmp_path, monkeypatch
    ):
        """Single-core workers prefer the transactional claim path when offered."""
        from stilt.workers import pull_worker_loop

        claim = SimulationClaim(
            sim_id="hrrr_202301011200_-111.85_40.77_5",
            claim_token="claim-1",
            worker_id="worker-a",
            claimed_at=dt.datetime(2026, 4, 13, 12, tzinfo=dt.timezone.utc),
            heartbeat_at=dt.datetime(2026, 4, 13, 12, tzinfo=dt.timezone.utc),
            expires_at=dt.datetime(2026, 4, 13, 12, 30, tzinfo=dt.timezone.utc),
        )
        transactional_repo = MagicMock()

        class _UoW:
            def __init__(self):
                self.claim = claim
                self.repository = transactional_repo
                self.released = False

            def release(self):
                self.released = True

        calls = {"n": 0}

        class _Ctx:
            def __enter__(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    return _UoW()
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        repo = MagicMock()
        repo.begin_claim_uow.side_effect = (
            lambda worker_id="legacy", lease_ttl=1800.0: _Ctx()
        )

        model = MagicMock()
        model.repository = repo
        run_args = MagicMock(spec=SimulationTask)
        run_args.model_copy.side_effect = lambda update: update
        model._build_run_args.return_value = run_args

        seen: list[dict] = []
        monkeypatch.setattr(
            "stilt.workers.run_worker",
            lambda args: seen.append(args) or MagicMock(),
        )

        pull_worker_loop(model, n_cores=1, follow=False)

        assert repo.begin_claim_uow.call_count == 2
        assert seen[0]["claim"] == claim
        assert seen[0]["repository"] is transactional_repo
        assert (
            not hasattr(repo, "claim_pending_claims")
            or repo.claim_pending_claims.call_count == 0
        )

    def test_transactional_claim_loop_releases_skipped_work(
        self, tmp_path, monkeypatch
    ):
        """Skipped transactional claims are rolled back via uow.release()."""
        from stilt.workers import pull_worker_loop

        claim = SimulationClaim(
            sim_id="hrrr_202301011200_-111.85_40.77_5",
            claim_token="claim-1",
            worker_id="worker-a",
            claimed_at=dt.datetime(2026, 4, 13, 12, tzinfo=dt.timezone.utc),
            heartbeat_at=dt.datetime(2026, 4, 13, 12, tzinfo=dt.timezone.utc),
            expires_at=dt.datetime(2026, 4, 13, 12, 30, tzinfo=dt.timezone.utc),
        )
        released: list[bool] = []

        class _UoW:
            def __init__(self):
                self.claim = claim
                self.repository = MagicMock()

            def release(self):
                released.append(True)

        calls = {"n": 0}

        class _Ctx:
            def __enter__(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    return _UoW()
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        repo = MagicMock()
        repo.begin_claim_uow.side_effect = (
            lambda worker_id="legacy", lease_ttl=1800.0: _Ctx()
        )

        model = MagicMock()
        model.repository = repo
        model._build_run_args.return_value = None

        run_worker_mock = MagicMock()
        monkeypatch.setattr("stilt.workers.run_worker", run_worker_mock)

        pull_worker_loop(model, n_cores=1, follow=False)

        assert released == [True]
        run_worker_mock.assert_not_called()
