"""Tests for the worker-side execution functions in ``stilt.execution.worker``."""

import contextlib
import datetime as dt
from types import SimpleNamespace

import pytest

from stilt.config import FootprintConfig, Grid, MetConfig, ModelConfig, STILTParams
from stilt.errors import ConfigValidationError, SimulationError
from stilt.execution import worker
from stilt.execution.worker import (
    SimulationResult,
    pull_simulations,
    run_simulation,
    run_simulations,
)
from stilt.meteorology import MetStream
from stilt.model import Model
from stilt.receptors import PointReceptor, Receptor
from stilt.simulation import SimID, Simulation
from stilt.store import LocalStore

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def receptor() -> Receptor:
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


@pytest.fixture
def other_receptor() -> Receptor:
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.86,
        latitude=40.78,
        altitude=10.0,
    )


@pytest.fixture
def met(tmp_path) -> MetStream:
    return MetStream(
        "hrrr",
        directory=tmp_path / "met",
        file_format="%Y%m%d_%H",
        file_tres="1h",
    )


@pytest.fixture
def params() -> STILTParams:
    return STILTParams(n_hours=-24, numpar=10)


@pytest.fixture
def store(tmp_path) -> LocalStore:
    """A store rooted away from the simulation directory, so publish() copies."""
    return LocalStore(tmp_path / "store")


@pytest.fixture
def sim(tmp_path, receptor, met, params, store) -> Simulation:
    sim_id = SimID.from_parts("hrrr", receptor)
    return Simulation(
        meteorology=met,
        receptor=receptor,
        params=params,
        directory=tmp_path / "compute" / sim_id,
        store=store,
    )


def _footprint_config(error: bool = False) -> FootprintConfig:
    return FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1),
        error=error,
    )


class _StubFootprint:
    """The only thing run_simulation reads from a footprint is ``is_empty``."""

    def __init__(self, is_empty: bool) -> None:
        self.is_empty = is_empty


def _write_stub_trajectory(sim: Simulation) -> None:
    sim.directory.mkdir(parents=True, exist_ok=True)
    sim.trajectories_path.write_bytes(b"traj")


def _model_config(tmp_path, **kwargs) -> ModelConfig:
    return ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        **kwargs,
    )


def _model(tmp_path, receptors, **config_kwargs) -> Model:
    return Model(
        project=tmp_path / "proj",
        config=_model_config(tmp_path, **config_kwargs),
        receptors=list(receptors),
    )


def _fake_run_simulation(calls: list[dict]):
    """Build a stand-in for run_simulation that records its arguments."""

    def fake(sim, footprints=None, *, skip_existing=True):
        calls.append(
            {
                "sim_id": str(sim.id),
                "footprints": footprints,
                "skip_existing": skip_existing,
            }
        )
        return SimulationResult(str(sim.id), "complete")

    return fake


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------


def test_simulation_result_is_frozen_with_optional_error():
    result = SimulationResult("hrrr_202301011200_-111.85_40.77_5", "complete")
    assert result.error is None
    with pytest.raises(AttributeError):
        result.status = "failed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# run_simulation: trajectory only
# ---------------------------------------------------------------------------


def test_run_simulation_trajectory_only_publishes_and_completes(
    sim, store, monkeypatch
):
    def fake_run_trajectories(*, write: bool = False, **kwargs):
        assert write is True
        _write_stub_trajectory(sim)

    monkeypatch.setattr(sim, "run_trajectories", fake_run_trajectories)
    monkeypatch.setattr(
        sim, "generate_footprint", lambda *a, **k: pytest.fail("no footprints")
    )

    result = run_simulation(sim)

    assert result == SimulationResult(str(sim.id), "complete")
    # publish() copied the trajectory into the store under its key.
    assert store.exists(sim.key(sim.trajectories_path))
    assert store.read_bytes(sim.key(sim.trajectories_path)) == b"traj"


def test_run_simulation_does_not_create_directory_before_running(sim, monkeypatch):
    """Construction is side-effect free; only running creates the sim directory."""
    assert not sim.directory.exists()
    monkeypatch.setattr(
        sim, "run_trajectories", lambda **kwargs: _write_stub_trajectory(sim)
    )

    run_simulation(sim)

    assert sim.directory.exists()


def test_run_simulation_simulation_error_is_failed_and_logged(sim, store, monkeypatch):
    def fake_run_trajectories(**kwargs):
        raise SimulationError("HYSPLIT failed")

    monkeypatch.setattr(sim, "run_trajectories", fake_run_trajectories)

    result = run_simulation(sim)

    assert result.status == "failed"
    assert result.error == "HYSPLIT failed"
    log_text = sim.log_path.read_text()
    assert "=== PYSTILT ERROR ===" in log_text
    assert "Phase: trajectory" in log_text
    assert "Type: SimulationError" in log_text
    assert "Message: HYSPLIT failed" in log_text
    # The failure log is published so remote readers can see why.
    assert store.exists(sim.key(sim.log_path))
    assert "HYSPLIT failed" in store.read_bytes(sim.key(sim.log_path)).decode()


def test_run_simulation_error_log_appends_to_existing_hysplit_log(sim, monkeypatch):
    sim.directory.mkdir(parents=True)
    sim.log_path.write_text("hysplit stdout\n")

    def fake_run_trajectories(**kwargs):
        raise SimulationError("boom")

    monkeypatch.setattr(sim, "run_trajectories", fake_run_trajectories)

    run_simulation(sim)

    log_text = sim.log_path.read_text()
    assert log_text.startswith("hysplit stdout\n")
    assert "=== PYSTILT ERROR ===" in log_text


def test_run_simulation_generic_exception_is_error(sim, monkeypatch):
    def fake_run_trajectories(**kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(sim, "run_trajectories", fake_run_trajectories)

    result = run_simulation(sim)

    assert result.status == "error"
    assert result.error == "unexpected"
    assert "Type: RuntimeError" in sim.log_path.read_text()


# ---------------------------------------------------------------------------
# run_simulation: footprints
# ---------------------------------------------------------------------------


def test_run_simulation_empty_footprint_writes_marker(sim, store, monkeypatch):
    monkeypatch.setattr(
        sim, "generate_footprint", lambda *a, **k: _StubFootprint(is_empty=True)
    )

    result = run_simulation(sim, {"slv": _footprint_config()})

    assert result.status == "complete-empty"
    assert sim.empty_footprint_path("slv").exists()
    assert not sim.footprint_path("slv").exists()
    # publish() covers the ``.empty`` marker too.
    assert store.exists(sim.key(sim.empty_footprint_path("slv")))


def test_run_simulation_nonempty_footprint_clears_stale_marker(sim, monkeypatch):
    sim.write_empty_footprint_marker("slv")
    monkeypatch.setattr(
        sim, "generate_footprint", lambda *a, **k: _StubFootprint(is_empty=False)
    )

    result = run_simulation(sim, {"slv": _footprint_config()}, skip_existing=False)

    assert result.status == "complete"
    assert not sim.empty_footprint_path("slv").exists()


def test_run_simulation_footprint_simulation_error_is_failed(sim, monkeypatch):
    def fake_generate(*args, **kwargs):
        raise SimulationError("Footprint failed")

    monkeypatch.setattr(sim, "generate_footprint", fake_generate)

    result = run_simulation(sim, {"slv": _footprint_config()})

    assert result.status == "failed"
    assert result.error == "Footprint failed"
    assert "Phase: footprint:slv" in sim.log_path.read_text()


def test_run_simulation_skips_existing_footprint(sim, monkeypatch):
    sim.directory.mkdir(parents=True)
    sim.footprint_path("slv").write_bytes(b"nc")
    monkeypatch.setattr(
        sim,
        "generate_footprint",
        lambda *a, **k: pytest.fail("existing footprint must not be regenerated"),
    )

    result = run_simulation(sim, {"slv": _footprint_config()}, skip_existing=True)

    assert result.status == "complete"


def test_run_simulation_skips_existing_empty_marker(sim, monkeypatch):
    sim.write_empty_footprint_marker("slv")
    monkeypatch.setattr(
        sim,
        "generate_footprint",
        lambda *a, **k: pytest.fail("empty footprint must not be regenerated"),
    )

    result = run_simulation(sim, {"slv": _footprint_config()}, skip_existing=True)

    assert result.status == "complete-empty"


def test_run_simulation_skip_existing_false_regenerates(sim, monkeypatch):
    sim.directory.mkdir(parents=True)
    sim.footprint_path("slv").write_bytes(b"nc")
    calls: list[str] = []

    def fake_generate(name, config, write, error):
        calls.append(name)
        return _StubFootprint(is_empty=False)

    monkeypatch.setattr(sim, "generate_footprint", fake_generate)

    run_simulation(sim, {"slv": _footprint_config()}, skip_existing=False)

    assert calls == ["slv"]


def test_run_simulation_error_footprint_uses_error_target(sim, monkeypatch):
    calls: list[tuple[str, bool]] = []

    def fake_generate(name, config, write, error):
        calls.append((name, error))
        return _StubFootprint(is_empty=False)

    monkeypatch.setattr(sim, "generate_footprint", fake_generate)

    result = run_simulation(sim, {"slv": _footprint_config(error=True)})

    assert result.status == "complete"
    assert calls == [("slv", False), ("slv", True)]


def test_run_simulation_error_footprint_skip_checks_error_name(sim, monkeypatch):
    """With skip_existing, the ``<name>_error`` target is checked independently."""
    sim.directory.mkdir(parents=True)
    sim.footprint_path("slv").write_bytes(b"nc")
    calls: list[tuple[str, bool]] = []

    def fake_generate(name, config, write, error):
        calls.append((name, error))
        return _StubFootprint(is_empty=True)

    monkeypatch.setattr(sim, "generate_footprint", fake_generate)

    result = run_simulation(sim, {"slv": _footprint_config(error=True)})

    # Main footprint already present -> only the error pass runs.
    assert calls == [("slv", True)]
    assert sim.empty_footprint_path("slv_error").exists()
    assert result.status == "complete"


def test_run_simulation_status_complete_if_any_footprint_nonempty(sim, monkeypatch):
    def fake_generate(name, config, write, error):
        return _StubFootprint(is_empty=(name == "coarse"))

    monkeypatch.setattr(sim, "generate_footprint", fake_generate)

    result = run_simulation(
        sim, {"fine": _footprint_config(), "coarse": _footprint_config()}
    )

    assert result.status == "complete"
    assert sim.empty_footprint_path("coarse").exists()
    assert not sim.empty_footprint_path("fine").exists()


def test_run_simulation_status_complete_empty_if_all_footprints_empty(sim, monkeypatch):
    monkeypatch.setattr(
        sim, "generate_footprint", lambda *a, **k: _StubFootprint(is_empty=True)
    )

    result = run_simulation(
        sim, {"fine": _footprint_config(), "coarse": _footprint_config()}
    )

    assert result.status == "complete-empty"


# ---------------------------------------------------------------------------
# run_simulations: inline
# ---------------------------------------------------------------------------


def test_run_simulations_inline_returns_results_in_order(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    sim_ids = [str(SimID.from_parts("hrrr", r)) for r in (receptor, other_receptor)]
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

    results = run_simulations(model, sim_ids, n_cores=1, skip_existing=True)

    assert [r.sim_id for r in results] == sim_ids
    assert [r.status for r in results] == ["complete", "complete"]
    assert [c["sim_id"] for c in calls] == sim_ids
    assert all(c["footprints"] is model.config.footprints for c in calls)


def test_run_simulations_empty_ids_returns_empty(tmp_path, receptor, monkeypatch):
    model = _model(tmp_path, [receptor])
    monkeypatch.setattr(
        worker, "run_simulation", lambda *a, **k: pytest.fail("must not run")
    )

    assert run_simulations(model, [], n_cores=1) == []


def test_run_simulations_inline_default_skip_reads_config(
    tmp_path, receptor, monkeypatch
):
    model = _model(tmp_path, [receptor], skip_existing=False)
    sim_id = str(SimID.from_parts("hrrr", receptor))
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

    run_simulations(model, [sim_id], n_cores=1)
    run_simulations(model, [sim_id], n_cores=1, skip_existing=True)

    assert [c["skip_existing"] for c in calls] == [False, True]


def test_run_simulations_inline_stops_after_interrupt(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    sim_ids = [str(SimID.from_parts("hrrr", r)) for r in (receptor, other_receptor)]
    seen: list[str] = []

    def fake(sim, footprints=None, *, skip_existing=True):
        seen.append(str(sim.id))
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "run_simulation", fake)

    results = run_simulations(model, sim_ids, n_cores=1)

    assert [(r.sim_id, r.status, r.error) for r in results] == [
        (sim_ids[0], "interrupted", "Worker preempted")
    ]
    assert seen == [sim_ids[0]]


def test_run_simulations_inline_continues_after_failed_result(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    sim_ids = [str(SimID.from_parts("hrrr", r)) for r in (receptor, other_receptor)]

    def fake(sim, footprints=None, *, skip_existing=True):
        if str(sim.id) == sim_ids[0]:
            return SimulationResult(str(sim.id), "failed", error="boom")
        return SimulationResult(str(sim.id), "complete")

    monkeypatch.setattr(worker, "run_simulation", fake)

    results = run_simulations(model, sim_ids, n_cores=1)

    assert [r.status for r in results] == ["failed", "complete"]
    assert results[0].error == "boom"


# ---------------------------------------------------------------------------
# run_simulations: process pool (plumbing only, with a synchronous fake Pool)
# ---------------------------------------------------------------------------


class _FakePool:
    """
    Synchronous stand-in for ``multiprocessing.Pool``.

    Runs the initializer in-process at construction and applies the mapped
    function immediately in ``imap_unordered``, yielding in *reverse* order so
    a test can prove results are re-ordered by index. Records ``terminate()``.
    """

    instances: list["_FakePool"] = []

    def __init__(self, n_cores, initializer=None, initargs=()):
        self.n_cores = n_cores
        self.terminated = False
        self.closed = False
        self.joined = False
        if initializer is not None:
            initializer(*initargs)
        _FakePool.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, func, iterable):
        for item in reversed(list(iterable)):
            yield func(item)

    def terminate(self):
        self.terminated = True

    def close(self):
        self.closed = True

    def join(self):
        self.joined = True


@pytest.fixture
def fake_pool(monkeypatch):
    """Patch the pool and keep the worker's module-level pool state isolated."""
    _FakePool.instances = []
    monkeypatch.setattr(worker.multiprocessing, "Pool", _FakePool)
    # The initializer installs a SIGTERM handler; keep it out of the test process.
    monkeypatch.setattr(worker.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_POOL_MODEL", None)
    monkeypatch.setattr(worker, "_POOL_SKIP", True)
    return _FakePool


def test_run_simulations_pool_rebuilds_model_and_orders_results(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    model = _model(tmp_path, [receptor, other_receptor], skip_existing=False)
    # Pool workers rebuild the Model from the project root: persist inputs.
    sim_ids = model.register()
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

    results = run_simulations(model, sim_ids, n_cores=2)

    [pool] = fake_pool.instances
    assert pool.n_cores == 2
    assert not pool.terminated
    assert pool.closed and pool.joined
    # The initializer rebuilt a fresh Model from the persisted project root.
    assert worker._POOL_MODEL is not None
    assert worker._POOL_MODEL is not model
    assert worker._POOL_MODEL.project.root == model.project.root
    assert worker._POOL_MODEL.compute_root == model.compute_root
    assert worker._POOL_SKIP is False
    # Results come back in input order even though the pool yielded reversed.
    assert [r.sim_id for r in results] == sim_ids
    assert [c["skip_existing"] for c in calls] == [False, False]


def test_run_simulations_pool_terminates_on_interrupted_result(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    model = _model(tmp_path, [receptor, other_receptor])
    sim_ids = model.register()

    def fake(sim, footprints=None, *, skip_existing=True):
        # The fake pool yields the *last* id first, so interrupt on the first id.
        if str(sim.id) == sim_ids[0]:
            raise KeyboardInterrupt
        return SimulationResult(str(sim.id), "complete")

    monkeypatch.setattr(worker, "run_simulation", fake)

    results = run_simulations(model, sim_ids, n_cores=2)

    [pool] = fake_pool.instances
    assert pool.terminated
    assert [(r.sim_id, r.status) for r in results] == [
        (sim_ids[0], "interrupted"),
        (sim_ids[1], "complete"),
    ]


def test_run_simulations_pool_keyboard_interrupt_terminates_and_returns(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    """A KeyboardInterrupt in the parent loop terminates the pool, keeping results."""
    model = _model(tmp_path, [receptor, other_receptor])
    sim_ids = model.register()

    def imap_then_interrupt(self, func, iterable):
        items = list(iterable)
        yield func(items[0])
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_pool, "imap_unordered", imap_then_interrupt)
    monkeypatch.setattr(
        worker,
        "run_simulation",
        lambda sim, footprints=None, *, skip_existing=True: SimulationResult(
            str(sim.id), "complete"
        ),
    )

    results = run_simulations(model, sim_ids, n_cores=2)

    [pool] = fake_pool.instances
    assert pool.terminated
    assert [r.sim_id for r in results] == [sim_ids[0]]


# ---------------------------------------------------------------------------
# pull_simulations
# ---------------------------------------------------------------------------


class _FakeClaim:
    def __init__(self, sim_id: str) -> None:
        self.sim_id = sim_id
        self.recorded: list[SimulationResult] = []

    def record(self, result: SimulationResult) -> None:
        self.recorded.append(result)


class _FakeQueue:
    """Yields each claim once, then ``None``."""

    def __init__(self, claims: list[_FakeClaim]) -> None:
        self._pending = list(claims)
        self.polls = 0

    @contextlib.contextmanager
    def claim_one(self):
        self.polls += 1
        yield self._pending.pop(0) if self._pending else None


def _pull_model(queue, *, skip_existing: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        queue=queue,
        config=SimpleNamespace(skip_existing=skip_existing, footprints={}),
        simulation=lambda sim_id: SimpleNamespace(id=sim_id),
    )


def test_pull_simulations_requires_a_queue():
    model = SimpleNamespace(queue=None)

    with pytest.raises(ConfigValidationError, match="Postgres work queue"):
        pull_simulations(model, follow=False)


def test_pull_simulations_records_result_on_claim(monkeypatch):
    claim = _FakeClaim("hrrr_202301011200_-111.85_40.77_5")
    queue = _FakeQueue([claim])
    calls: list[tuple[str, bool]] = []

    def fake(sim, footprints=None, *, skip_existing=True):
        calls.append((str(sim.id), skip_existing))
        return SimulationResult(str(sim.id), "complete")

    monkeypatch.setattr(worker, "run_simulation", fake)

    pull_simulations(_pull_model(queue, skip_existing=False), follow=False)

    assert calls == [(claim.sim_id, False)]
    assert claim.recorded == [SimulationResult(claim.sim_id, "complete")]
    # One claim, then one empty poll that ends the batch.
    assert queue.polls == 2


def test_pull_simulations_records_interrupted_result(monkeypatch):
    claim = _FakeClaim("hrrr_202301011200_-111.85_40.77_5")
    queue = _FakeQueue([claim])

    def fake(sim, footprints=None, *, skip_existing=True):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "run_simulation", fake)

    pull_simulations(_pull_model(queue), follow=False)

    assert [r.status for r in claim.recorded] == ["interrupted"]


def test_pull_simulations_follow_sleeps_on_empty_then_keeps_polling(monkeypatch):
    class _Stop(Exception):
        pass

    claim = _FakeClaim("hrrr_202301011200_-111.85_40.77_5")
    sleeps: list[float] = []

    class _Queue(_FakeQueue):
        @contextlib.contextmanager
        def claim_one(self):
            self.polls += 1
            if self.polls == 1:
                yield None  # empty -> sleep, keep going in follow mode
            elif self.polls == 2:
                yield self._pending.pop(0)
            else:
                raise _Stop  # end the otherwise-infinite follow loop

    queue = _Queue([claim])
    monkeypatch.setattr(worker.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        worker,
        "run_simulation",
        lambda sim, footprints=None, *, skip_existing=True: SimulationResult(
            str(sim.id), "complete"
        ),
    )

    with pytest.raises(_Stop):
        pull_simulations(_pull_model(queue), follow=True, poll_interval=0.5)

    assert sleeps == [0.5]
    assert len(claim.recorded) == 1
    assert queue.polls == 3
