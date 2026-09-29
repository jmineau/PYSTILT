"""Tests for the worker-side execution functions in ``stilt.execution.worker``."""

import contextlib
import datetime as dt
from types import SimpleNamespace

import pytest

from stilt.config import (
    FootprintConfig,
    Grid,
    MetConfig,
    ModelConfig,
    STILTParams,
    VariantConfig,
)
from stilt.errors import ConfigValidationError, SimulationError
from stilt.execution import worker
from stilt.execution.worker import (
    ReceptorResult,
    SimulationResult,
    pull_receptors,
    run_receptor,
    run_receptors,
    run_simulation,
)
from stilt.meteorology import MetStream
from stilt.model import Model
from stilt.receptors import PointReceptor, Receptor
from stilt.simulation import SimID, Simulation
from stilt.store import LocalStore

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

GRID = Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)


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


def _variant(params, *, name="hrrr", footprint=None, derived_from=None):
    data = {**params.model_dump(), **(footprint.model_dump() if footprint else {})}
    return VariantConfig(
        name=name, group=name, met="hrrr", derived_from=derived_from, **data
    )


def _make_sim(tmp_path, receptor, met, params, store, *, footprint=None):
    return Simulation(
        receptor,
        _variant(params, footprint=footprint),
        met=met,
        directory=tmp_path / "compute" / SimID(receptor.id, "hrrr"),
        store=store,
    )


@pytest.fixture
def sim(tmp_path, receptor, met, params, store) -> Simulation:
    """A trajectory-only simulation."""
    return _make_sim(tmp_path, receptor, met, params, store)


@pytest.fixture
def fsim(tmp_path, receptor, met, params, store) -> Simulation:
    """A simulation with a footprint whose trajectory already exists."""
    s = _make_sim(
        tmp_path, receptor, met, params, store, footprint=FootprintConfig(grid=GRID)
    )
    _write_stub_trajectory(s)
    return s


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

    def fake(sim, *, skip_existing=True):
        calls.append({"sim_id": str(sim.id), "skip_existing": skip_existing})
        return SimulationResult(str(sim.id), "complete")

    return fake


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_simulation_result_is_frozen_with_optional_error():
    result = SimulationResult("202301011200_-111.85_40.77_5/hrrr", "complete")
    assert result.error is None
    with pytest.raises(AttributeError):
        result.status = "failed"  # type: ignore[misc]


def test_receptor_result_reports_the_worst_simulation():
    results = [
        SimulationResult("r/a", "complete"),
        SimulationResult("r/b", "failed", error="boom"),
        SimulationResult("r/c", "complete"),
    ]
    summary = ReceptorResult.summarise("r", results)
    assert summary.status == "failed"
    assert summary.error == "boom"
    assert summary.simulations == tuple(results)
    assert ReceptorResult.summarise("r", []).status == "complete"


# ---------------------------------------------------------------------------
# run_simulation: trajectory
# ---------------------------------------------------------------------------


def test_run_simulation_trajectory_only_publishes_and_completes(
    sim, store, monkeypatch
):
    def fake_run_trajectories(*, write: bool = False, **kwargs):
        assert write is True
        _write_stub_trajectory(sim)

    monkeypatch.setattr(sim, "run_trajectories", fake_run_trajectories)
    monkeypatch.setattr(
        sim, "generate_footprint", lambda *a, **k: pytest.fail("no footprint")
    )

    result = run_simulation(sim)

    assert result == SimulationResult(str(sim.id), "complete", ran_hysplit=True)
    # publish() copied the trajectory into the store under its key.
    assert store.exists(sim.key(sim.trajectories_path))
    assert store.read_bytes(sim.key(sim.trajectories_path)) == b"traj"


def test_run_simulation_skips_an_existing_trajectory(sim, monkeypatch):
    _write_stub_trajectory(sim)
    monkeypatch.setattr(
        sim, "run_trajectories", lambda **k: pytest.fail("must not rerun HYSPLIT")
    )

    assert run_simulation(sim, skip_existing=True).status == "complete"


def test_run_simulation_reruns_the_trajectory_without_skip_existing(sim, monkeypatch):
    _write_stub_trajectory(sim)
    calls: list[bool] = []
    monkeypatch.setattr(sim, "run_trajectories", lambda **k: calls.append(True))

    run_simulation(sim, skip_existing=False)

    assert calls == [True]


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
# run_simulation: footprint
# ---------------------------------------------------------------------------


def test_run_simulation_publishes_empty_marker(fsim, store, monkeypatch):
    def fake_generate(**kwargs):
        fsim.write_empty_footprint_marker("outside_domain")
        return None

    monkeypatch.setattr(fsim, "generate_footprint", fake_generate)

    result = run_simulation(fsim)

    assert result.status == "complete"
    assert not fsim.footprint_path.exists()
    # publish() covers the ``.empty`` marker too.
    assert store.exists(fsim.key(fsim.empty_footprint_path))


def test_run_simulation_footprint_simulation_error_is_failed(fsim, monkeypatch):
    def fake_generate(**kwargs):
        raise SimulationError("Footprint failed")

    monkeypatch.setattr(fsim, "generate_footprint", fake_generate)

    result = run_simulation(fsim)

    assert result.status == "failed"
    assert result.error == "Footprint failed"
    assert "Phase: footprint" in fsim.log_path.read_text()


def test_run_simulation_skips_existing_footprint(fsim, monkeypatch):
    fsim.footprint_path.write_bytes(b"nc")
    monkeypatch.setattr(
        fsim,
        "generate_footprint",
        lambda **k: pytest.fail("existing footprint must not be regenerated"),
    )

    assert run_simulation(fsim, skip_existing=True).status == "complete"


def test_run_simulation_skips_existing_empty_marker(fsim, monkeypatch):
    fsim.write_empty_footprint_marker("outside_domain")
    monkeypatch.setattr(
        fsim,
        "generate_footprint",
        lambda **k: pytest.fail("empty footprint must not be regenerated"),
    )

    assert run_simulation(fsim, skip_existing=True).status == "complete"


def test_run_simulation_skip_existing_false_regenerates(fsim, monkeypatch):
    fsim.footprint_path.write_bytes(b"nc")
    calls: list[bool] = []

    def fake_generate(*, write):
        calls.append(write)
        return None

    monkeypatch.setattr(fsim, "run_trajectories", lambda **k: None)
    monkeypatch.setattr(fsim, "generate_footprint", fake_generate)

    run_simulation(fsim, skip_existing=False)

    assert calls == [True]


def test_run_simulation_backfills_a_missing_trajectory_and_remakes_the_footprint(
    tmp_path, receptor, met, params, store, monkeypatch
):
    """A lost trajectory is rerun, and the old footprint is remade from the new particles."""
    s = _make_sim(
        tmp_path, receptor, met, params, store, footprint=FootprintConfig(grid=GRID)
    )
    s.directory.mkdir(parents=True)
    s.footprint_path.write_bytes(b"nc")
    calls: list[str] = []
    monkeypatch.setattr(
        s,
        "run_trajectories",
        lambda **k: calls.append("hysplit") or _write_stub_trajectory(s),
    )
    monkeypatch.setattr(
        s,
        "generate_footprint",
        lambda **k: calls.append("footprint") or None,
    )

    result = run_simulation(s)

    assert result.status == "complete" and result.ran_hysplit
    assert calls == ["hysplit", "footprint"]


def test_run_receptor_remakes_derived_footprints_when_the_parent_reran(
    tmp_path, receptor, monkeypatch
):
    model = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={"hrrr": {}, "hrrr-s2": {"from": "hrrr", "smooth_factor": 2.0}},
    )
    model.register()
    parent = model.simulation((receptor.id, "hrrr"))
    derived = model.simulation((receptor.id, "hrrr-s2"))
    for sim in (parent, derived):
        sim.directory.mkdir(parents=True, exist_ok=True)
        sim.footprint_path.write_bytes(b"nc")
    calls: list[str] = []

    def fake_hysplit(**kwargs):
        calls.append("hysplit")
        _write_stub_trajectory(parent)
        parent._trajectories = "stub"  # the derived run reads this, not the stub file

    monkeypatch.setattr(parent, "run_trajectories", fake_hysplit)
    for sim in (parent, derived):

        def fake_generate(name=sim.variant, **kwargs):
            calls.append(name)
            return None

        monkeypatch.setattr(sim, "generate_footprint", fake_generate)

    result = run_receptor(model, str(receptor.id))

    assert result.status == "complete"
    assert calls == ["hysplit", "hrrr", "hrrr-s2"]


def test_run_simulation_derived_never_runs_hysplit(
    tmp_path, receptor, met, params, store, monkeypatch
):
    parent = _make_sim(tmp_path, receptor, met, params, store)
    derived = Simulation(
        receptor,
        _variant(
            params,
            name="hrrr-s2",
            footprint=FootprintConfig(grid=GRID, smooth_factor=2.0),
            derived_from="hrrr",
        ),
        parent=parent,
        directory=tmp_path / "compute" / SimID(receptor.id, "hrrr-s2"),
        store=store,
    )
    monkeypatch.setattr(derived, "run_trajectories", lambda **k: pytest.fail("no"))
    monkeypatch.setattr(parent, "run_trajectories", lambda **k: pytest.fail("no"))

    # Parent trajectory missing: a failure, not a HYSPLIT run.
    result = run_simulation(derived)
    assert result.status == "failed"
    assert "has no trajectory" in (result.error or "")

    monkeypatch.setattr(Simulation, "trajectories", property(lambda self: object()))
    monkeypatch.setattr(derived, "generate_footprint", lambda **k: None)
    assert run_simulation(derived).status == "complete"


# ---------------------------------------------------------------------------
# run_receptor
# ---------------------------------------------------------------------------


def test_run_receptor_runs_transport_variants_before_derived_ones(
    tmp_path, receptor, monkeypatch
):
    model = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={
            "s2": {"from": "hrrr", "smooth_factor": 2},
            "hrrr": {},
            "zi08": {"ziscale": 0.8},
        },
    )
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

    result = run_receptor(model, str(receptor.id), skip_existing=False)

    assert [c["sim_id"].split("/")[1] for c in calls] == ["hrrr", "zi08", "s2"]
    assert all(c["skip_existing"] is False for c in calls)
    assert result.status == "complete"
    assert result.receptor_id == str(receptor.id)
    assert len(result.simulations) == 3


def test_run_receptor_normalises_preemption(tmp_path, receptor, monkeypatch):
    model = _model(tmp_path, [receptor])

    def fake(sim, *, skip_existing=True):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "run_simulation", fake)

    result = run_receptor(model, str(receptor.id))

    assert result.status == "interrupted"
    assert result.error == "Worker preempted"


# ---------------------------------------------------------------------------
# run_receptors: inline
# ---------------------------------------------------------------------------


def test_run_receptors_inline_returns_results_in_order(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = [str(r.id) for r in (receptor, other_receptor)]
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

    results = run_receptors(model, ids, n_cores=1, skip_existing=True)

    assert [r.receptor_id for r in results] == ids
    assert [r.status for r in results] == ["complete", "complete"]
    assert [c["sim_id"] for c in calls] == [f"{rid}/hrrr" for rid in ids]


def test_run_receptors_empty_ids_returns_empty(tmp_path, receptor, monkeypatch):
    model = _model(tmp_path, [receptor])
    monkeypatch.setattr(
        worker, "run_simulation", lambda *a, **k: pytest.fail("must not run")
    )

    assert run_receptors(model, [], n_cores=1) == []


def test_run_receptors_inline_skips_existing_by_default(
    tmp_path, receptor, monkeypatch
):
    model = _model(tmp_path, [receptor])
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

    run_receptors(model, [str(receptor.id)], n_cores=1)
    run_receptors(model, [str(receptor.id)], n_cores=1, skip_existing=False)

    assert [c["skip_existing"] for c in calls] == [True, False]


def test_run_receptors_inline_stops_after_interrupt(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = [str(r.id) for r in (receptor, other_receptor)]
    seen: list[str] = []

    def fake(sim, *, skip_existing=True):
        seen.append(str(sim.id))
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "run_simulation", fake)

    results = run_receptors(model, ids, n_cores=1)

    assert [(r.receptor_id, r.status, r.error) for r in results] == [
        (ids[0], "interrupted", "Worker preempted")
    ]
    assert seen == [f"{ids[0]}/hrrr"]


def test_run_receptors_inline_continues_after_failed_result(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = [str(r.id) for r in (receptor, other_receptor)]

    def fake(sim, *, skip_existing=True):
        if sim.id.receptor == ids[0]:
            return SimulationResult(str(sim.id), "failed", error="boom")
        return SimulationResult(str(sim.id), "complete")

    monkeypatch.setattr(worker, "run_simulation", fake)

    results = run_receptors(model, ids, n_cores=1)

    assert [r.status for r in results] == ["failed", "complete"]
    assert results[0].error == "boom"


# ---------------------------------------------------------------------------
# run_receptors: process pool (plumbing only, with a synchronous fake Pool)
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


def test_run_receptors_pool_rebuilds_model_and_orders_results(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    model = _model(tmp_path, [receptor, other_receptor])
    # Pool workers rebuild the Model from the project root: persist inputs.
    ids = model.register()
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

    results = run_receptors(model, ids, n_cores=2, skip_existing=False)

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
    assert [r.receptor_id for r in results] == ids
    assert [c["skip_existing"] for c in calls] == [False, False]


def test_run_receptors_pool_terminates_on_interrupted_result(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = model.register()

    def fake(sim, *, skip_existing=True):
        # The fake pool yields the *last* id first, so interrupt on the first id.
        if sim.id.receptor == ids[0]:
            raise KeyboardInterrupt
        return SimulationResult(str(sim.id), "complete")

    monkeypatch.setattr(worker, "run_simulation", fake)

    results = run_receptors(model, ids, n_cores=2)

    [pool] = fake_pool.instances
    assert pool.terminated
    assert [(r.receptor_id, r.status) for r in results] == [
        (ids[0], "interrupted"),
        (ids[1], "complete"),
    ]


def test_run_receptors_pool_keyboard_interrupt_terminates_and_returns(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    """A KeyboardInterrupt in the parent loop terminates the pool, keeping results."""
    model = _model(tmp_path, [receptor, other_receptor])
    ids = model.register()

    def imap_then_interrupt(self, func, iterable):
        items = list(iterable)
        yield func(items[0])
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_pool, "imap_unordered", imap_then_interrupt)
    monkeypatch.setattr(
        worker,
        "run_simulation",
        lambda sim, *, skip_existing=True: SimulationResult(str(sim.id), "complete"),
    )

    results = run_receptors(model, ids, n_cores=2)

    [pool] = fake_pool.instances
    assert pool.terminated
    assert [r.receptor_id for r in results] == [ids[0]]


# ---------------------------------------------------------------------------
# pull_receptors
# ---------------------------------------------------------------------------

RID = "202301011200_-111.85_40.77_5"


class _FakeClaim:
    def __init__(self, receptor_id: str) -> None:
        self.receptor_id = receptor_id
        self.recorded: list[ReceptorResult] = []

    def record(self, result: ReceptorResult) -> None:
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


def _pull_model(queue) -> SimpleNamespace:
    return SimpleNamespace(queue=queue)


def test_pull_receptors_requires_a_queue():
    model = SimpleNamespace(queue=None)

    with pytest.raises(ConfigValidationError, match="Postgres work queue"):
        pull_receptors(model, follow=False)


def test_pull_receptors_records_result_on_claim(monkeypatch):
    claim = _FakeClaim(RID)
    queue = _FakeQueue([claim])
    calls: list[tuple[str, bool]] = []

    def fake(model, receptor_id, *, skip_existing=True):
        calls.append((receptor_id, skip_existing))
        return ReceptorResult(receptor_id, "complete")

    monkeypatch.setattr(worker, "run_receptor", fake)

    pull_receptors(_pull_model(queue), follow=False, skip_existing=False)

    assert calls == [(RID, False)]
    assert claim.recorded == [ReceptorResult(RID, "complete")]
    # One claim, then one empty poll that ends the batch.
    assert queue.polls == 2


def test_pull_receptors_records_interrupted_result(monkeypatch):
    claim = _FakeClaim(RID)
    queue = _FakeQueue([claim])
    monkeypatch.setattr(
        worker,
        "run_receptor",
        lambda model, rid, *, skip_existing=True: ReceptorResult(rid, "interrupted"),
    )

    pull_receptors(_pull_model(queue), follow=False)

    assert [r.status for r in claim.recorded] == ["interrupted"]


def test_pull_receptors_follow_sleeps_on_empty_then_keeps_polling(monkeypatch):
    class _Stop(Exception):
        pass

    claim = _FakeClaim(RID)
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
        "run_receptor",
        lambda model, rid, *, skip_existing=True: ReceptorResult(rid, "complete"),
    )

    with pytest.raises(_Stop):
        pull_receptors(_pull_model(queue), follow=True, poll_interval=0.5)

    assert sleeps == [0.5]
    assert len(claim.recorded) == 1
    assert queue.polls == 3
