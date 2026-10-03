"""Tests for the worker-side execution functions in ``stilt.execution.worker``."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from stilt.config import ProjectConfig
from stilt.exceptions import MeteorologyError, SimulationError
from stilt.execution import worker
from stilt.execution.worker import (
    SimulationResult,
    make_footprint,
    run_receptor,
    run_receptors,
)
from stilt.footprint.config import FootprintConfig
from stilt.meteorology import Met, MetConfig
from stilt.output import Output
from stilt.project import Project
from stilt.receptors import PointReceptor, Receptor
from stilt.simulation import Simulation
from stilt.spatial import Grid
from stilt.transforms import TransformContext
from stilt.transport import ModelInfo, ModelRun
from stilt.transport.hysplit import HysplitConfig
from stilt.variants import Variant

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
def met_config(tmp_path) -> MetConfig:
    return MetConfig(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h"
    )


@pytest.fixture
def met(met_config) -> Met:
    return Met("hrrr", met_config)


@pytest.fixture
def params() -> HysplitConfig:
    return HysplitConfig(n_hours=-24, numpar=10)


@pytest.fixture
def output(tmp_path) -> Output:
    return Output(tmp_path / "output")


def _variant(params, met_config, *, name="hrrr", footprint=None) -> Variant:
    return Variant(
        name=name,
        group=name,
        met="hrrr",
        met_config=met_config,
        transport=params,
        model=ModelInfo(version="v5.1.0"),
        footprint=footprint,
    )


def _make_sim(receptor, met_config, params, output, *, footprint=None, name="hrrr"):
    return Simulation(
        receptor, _variant(params, met_config, name=name, footprint=footprint), output
    )


@pytest.fixture
def compute_root(tmp_path):
    return tmp_path / "compute"


@pytest.fixture
def sim(receptor, met_config, params, output) -> Simulation:
    """A trajectory-only simulation."""
    return _make_sim(receptor, met_config, params, output)


@pytest.fixture
def fsim(receptor, met_config, params, output) -> Simulation:
    """A simulation with a footprint whose particles already exist."""
    s = _make_sim(
        receptor, met_config, params, output, footprint=FootprintConfig(grid=GRID)
    )
    _write_particles(s)
    return s


def _particles(receptor) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "time": [-60.0],
            "indx": [1.0],
            "long": [-111.9],
            "lati": [40.7],
            "zagl": [10.0],
            "foot": [1e-5],
        }
    )
    data["datetime"] = pd.Timestamp(receptor.time) + pd.to_timedelta(
        data["time"], unit="min"
    )
    return data


def _write_particles(sim: Simulation) -> pd.DataFrame:
    """Put a small particle file for *sim* in the output directory."""
    particles = _particles(sim.receptor)
    folder = sim.output.particles(sim.variant)
    folder.write(sim.receptor, particles, [])
    return particles


def _write_footprint(sim: Simulation, *, empty: bool = False) -> None:
    """Record a footprint (or an empty one) for *sim* in the output directory."""
    run = sim.output.particles(sim.variant)
    assert sim.variant.footprint is not None
    if empty:
        feet = run.footprints(sim.variant.footprint, name=sim.variant.name)
        feet.write_empty(sim.receptor, "outside_domain", name=sim.variant.name)
    else:
        make_footprint(
            sim,
            _particles(sim.receptor),
            context=TransformContext(receptor=sim.receptor, variant=sim.variant.name),
        )


def _model_config(tmp_path, **kwargs) -> ProjectConfig:
    return ProjectConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        **kwargs,
    )


def _model(tmp_path, receptors, **config_kwargs) -> Project:
    return Project.init(
        tmp_path / "proj",
        config=_model_config(tmp_path, **config_kwargs),
        receptors=list(receptors),
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_simulation_result_is_frozen_with_optional_error():
    result = SimulationResult("sim/hrrr", "complete")
    assert result.error is None
    with pytest.raises(AttributeError):
        result.status = "failed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# run_particles
# ---------------------------------------------------------------------------


def test_run_particles_starts_in_an_empty_directory(
    sim, met, compute_root, monkeypatch
):
    """A directory left by a stopped job is cleared before the model runs."""
    workdir = compute_root / sim.id
    workdir.mkdir(parents=True)
    (workdir / "PARTICLE_STILT.DAT").write_text("left over\n")
    seen: list[list[str]] = []

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, workdir, timeout=None):
            seen.append(sorted(p.name for p in workdir.iterdir()))
            raise RuntimeError("stop here")

    monkeypatch.setattr(worker, "get_model", lambda name: _Model())
    with pytest.raises(RuntimeError):
        worker.run_particles(sim, met=met, workdir=workdir)
    assert seen == [[]]


def test_run_particles_keeps_no_empty_scratch_copy(sim, met, compute_root, monkeypatch):
    """A run that fails before writing anything, such as on missing met, leaves no scratch copy."""

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, workdir, timeout=None):
            raise MeteorologyError("Insufficient number of meteorological files found.")

    monkeypatch.setattr(worker, "get_model", lambda name: _Model())
    with pytest.raises(MeteorologyError):
        worker.run_particles(sim, met=met, workdir=compute_root / sim.id)

    kept = sim.output.particles(sim.variant).scratch_path(sim.receptor.id)
    assert not kept.exists()
    assert not (compute_root / sim.id).exists()


def test_run_particles_without_particles_is_a_simulation_error(
    sim, met, compute_root, monkeypatch
):
    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, workdir, timeout=None):
            return ModelRun(particles=pd.DataFrame(), met_files=[])

    monkeypatch.setattr(worker, "get_model", lambda name: _Model())
    with pytest.raises(SimulationError) as caught:
        worker.run_particles(sim, met=met, workdir=compute_root / sim.id)
    assert caught.value.reason == "NO_PARTICLE_DATA"


# ---------------------------------------------------------------------------
# run_receptor: variants that share particles run as one group
# ---------------------------------------------------------------------------


def _fake_run_particles(monkeypatch, calls: list[str]):
    """Replace run_particles with one that writes particles and records the variant it ran for."""

    def fake(sim, *, met, workdir, keep_scratch=False, timeout=None):
        calls.append(sim.variant.name)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)


def _fake_make_footprint(monkeypatch, calls: list[tuple[str, int]]):
    """Replace make_footprint with one that records the variant and which particle table it got."""

    def fake(sim, particles, *, context):
        calls.append((sim.variant.name, id(particles)))
        _write_footprint(sim)

    monkeypatch.setattr(worker, "make_footprint", fake)


def _run_receptor(project, receptor, **kwargs):
    return run_receptor(
        project, str(receptor.id), compute_root=project.directory / "scratch", **kwargs
    )


SHARED = {"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}, "zi08": {"ziscale": 0.8}}


def test_a_particles_only_variant_runs_hysplit_and_completes(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor])
    hysplit: list[str] = []
    _fake_run_particles(monkeypatch, hysplit)
    monkeypatch.setattr(
        worker, "make_footprint", lambda *a, **k: pytest.fail("no footprint")
    )

    [result] = _run_receptor(project, receptor)

    assert result == SimulationResult(f"{receptor.id}/hrrr", "complete")
    assert hysplit == ["hrrr"]
    assert project.simulation(str(receptor.id), "hrrr").has_particles


def test_variants_that_share_particles_run_hysplit_once_and_reuse_the_table(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor], grid=GRID, variants=SHARED)
    hysplit: list[str] = []
    feet: list[tuple[str, int]] = []
    _fake_run_particles(monkeypatch, hysplit)
    _fake_make_footprint(monkeypatch, feet)

    results = _run_receptor(project, receptor)

    assert [r.status for r in results] == ["complete"] * 3
    assert [r.sim_id.split("/")[1] for r in results] == ["hrrr", "hrrr-s2", "zi08"]
    assert hysplit == ["hrrr", "zi08"]  # hrrr-s2 shares hrrr's particles
    assert [name for name, _ in feet] == ["hrrr", "hrrr-s2", "zi08"]
    # hrrr-s2's footprint is made from the table HYSPLIT just returned for hrrr.
    assert feet[0][1] == feet[1][1]


def test_existing_particles_are_read_once_for_every_footprint_of_the_group(
    tmp_path, receptor, monkeypatch
):
    project = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}},
    )
    _write_particles(project.simulation(str(receptor.id), "hrrr"))
    feet: list[tuple[str, int]] = []
    _fake_make_footprint(monkeypatch, feet)
    monkeypatch.setattr(
        worker, "run_particles", lambda *a, **k: pytest.fail("no HYSPLIT")
    )

    _run_receptor(project, receptor)

    assert [name for name, _ in feet] == ["hrrr", "hrrr-s2"]
    assert feet[0][1] == feet[1][1]


def test_existing_footprints_are_kept_and_empty_ones_count_as_done(
    tmp_path, receptor, monkeypatch
):
    project = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}},
    )
    hrrr = project.simulation(str(receptor.id), "hrrr")
    _write_particles(hrrr)
    _write_footprint(hrrr)
    _write_footprint(project.simulation(str(receptor.id), "hrrr-s2"), empty=True)
    monkeypatch.setattr(
        worker, "run_particles", lambda *a, **k: pytest.fail("no HYSPLIT")
    )
    monkeypatch.setattr(worker, "make_footprint", lambda *a, **k: pytest.fail("kept"))

    results = _run_receptor(project, receptor)

    assert [r.status for r in results] == ["complete", "complete"]


def test_without_skip_existing_hysplit_runs_once_per_group_and_every_footprint_is_remade(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor], grid=GRID, variants=SHARED)
    for name in SHARED:
        sim = project.simulation(str(receptor.id), name)
        _write_particles(sim)
        _write_footprint(sim)
    hysplit: list[str] = []
    feet: list[tuple[str, int]] = []
    _fake_run_particles(monkeypatch, hysplit)
    _fake_make_footprint(monkeypatch, feet)

    _run_receptor(project, receptor, skip_existing=False)

    assert hysplit == ["hrrr", "zi08"]
    assert [name for name, _ in feet] == ["hrrr", "hrrr-s2", "zi08"]


def test_rerun_particles_remake_a_footprint_that_already_existed(
    tmp_path, receptor, monkeypatch
):
    """Lost particles are rerun, and the old footprint is remade from the new ones."""
    project = _model(tmp_path, [receptor], grid=GRID)
    sim = project.simulation(str(receptor.id), "hrrr")
    _write_particles(sim)
    _write_footprint(sim)
    sim.particles_path.unlink()
    hysplit: list[str] = []
    feet: list[tuple[str, int]] = []
    _fake_run_particles(monkeypatch, hysplit)
    _fake_make_footprint(monkeypatch, feet)

    _run_receptor(project, receptor)

    assert hysplit == ["hrrr"] and [name for name, _ in feet] == ["hrrr"]


def test_run_receptor_takes_timeout_and_keep_scratch_from_execution(
    tmp_path, receptor, monkeypatch
):
    project = _model(
        tmp_path, [receptor], execution={"timeout": 120, "keep_scratch": True}
    )
    seen: list[dict] = []

    def fake(sim, **kwargs):
        seen.append(kwargs)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    _run_receptor(project, receptor)

    assert seen and all(k["timeout"] == 120 and k["keep_scratch"] for k in seen)


def test_run_receptor_normalises_preemption(tmp_path, receptor, monkeypatch):
    project = _model(tmp_path, [receptor], grid=GRID, variants=SHARED)
    hysplit: list[str] = []

    def fake(sim, **kwargs):
        if sim.variant.name == "zi08":
            raise KeyboardInterrupt
        hysplit.append(sim.variant.name)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    _fake_make_footprint(monkeypatch, [])

    results = _run_receptor(project, receptor)

    assert [(r.sim_id.split("/")[1], r.status) for r in results] == [
        ("hrrr", "complete"),
        ("hrrr-s2", "complete"),
        ("zi08", "interrupted"),
    ]
    assert results[-1].error == "Worker preempted"


# ---------------------------------------------------------------------------
# run_receptor: failures are recorded with the simulation
# ---------------------------------------------------------------------------


def test_a_failed_hysplit_run_fails_its_whole_group_once_and_is_recorded(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor], grid=GRID, variants=SHARED)
    hysplit: list[str] = []

    def fake(sim, **kwargs):
        hysplit.append(sim.variant.name)
        if sim.variant.name == "hrrr":
            raise SimulationError(
                "HYSPLIT failed (MET_COVERAGE).", reason="MET_COVERAGE"
            )
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    _fake_make_footprint(monkeypatch, [])

    results = _run_receptor(project, receptor)

    assert hysplit == ["hrrr", "zi08"]
    assert [(r.status, r.error) for r in results] == [
        ("failed", "HYSPLIT failed (MET_COVERAGE)."),
        ("failed", "HYSPLIT failed (MET_COVERAGE)."),
        ("complete", None),
    ]
    for name in ("hrrr", "hrrr-s2"):
        failure = project.simulation(str(receptor.id), name).failure
        assert failure is not None
        assert failure["step"] == "particles"
        assert failure["error"] == "SimulationError"
        assert failure["reason"] == "MET_COVERAGE"
        assert "traceback" not in failure
    assert project.simulation(str(receptor.id), "zi08").failure is None


def test_a_failed_footprint_is_the_variants_own_and_the_others_still_run(
    tmp_path, receptor, monkeypatch
):
    project = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}},
    )
    _fake_run_particles(monkeypatch, [])
    made: list[str] = []

    def fake(sim, particles, *, context):
        if sim.variant.name == "hrrr":
            raise ValueError("bad grid")
        made.append(sim.variant.name)
        _write_footprint(sim)

    monkeypatch.setattr(worker, "make_footprint", fake)

    results = _run_receptor(project, receptor)

    assert [(r.status, r.error) for r in results] == [
        ("error", "bad grid"),
        ("complete", None),
    ]
    assert made == ["hrrr-s2"]
    failure = project.simulation(str(receptor.id), "hrrr").failure
    assert failure is not None
    assert failure["step"] == "footprint" and failure["error"] == "ValueError"
    assert failure["reason"] is None
    assert "ValueError: bad grid" in failure["traceback"]
    assert project.simulation(str(receptor.id), "hrrr-s2").failure is None


def test_a_success_clears_the_failure_it_replaces(tmp_path, receptor, monkeypatch):
    project = _model(tmp_path, [receptor], grid=GRID)
    sim = project.simulation(str(receptor.id), "hrrr")

    def fail(sim, **kwargs):
        raise MeteorologyError("Insufficient number of meteorological files found.")

    monkeypatch.setattr(worker, "run_particles", fail)
    _run_receptor(project, receptor)
    assert sim.failure is not None and sim.failure["reason"] == "MISSING_MET_FILES"
    folder = project.output.find_particles(sim.variant)
    assert folder is not None and folder.failure_path(sim.receptor.id).exists()

    _fake_run_particles(monkeypatch, [])
    _fake_make_footprint(monkeypatch, [])
    _run_receptor(project, receptor)

    assert sim.is_complete() and sim.failure is None
    assert not folder.failure_path(sim.receptor.id).exists()


def test_the_failure_record_names_the_kept_log_and_scratch(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor])
    sim = project.simulation(str(receptor.id), "hrrr")

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, workdir, timeout=None):
            (workdir / "stilt.log").write_text("hycs_std said something\n")
            (workdir / "CONTROL").write_text("...")
            raise SimulationError(
                "HYSPLIT failed (FORTRAN_RUNTIME_ERROR).",
                reason="FORTRAN_RUNTIME_ERROR",
            )

    monkeypatch.setattr(worker, "get_model", lambda name: _Model())
    _run_receptor(project, receptor)

    failure = sim.failure
    assert failure is not None
    assert failure["log"] == str(sim.log_path) and "hycs_std said" in sim.log
    assert failure["scratch"] is not None
    assert (Path(failure["scratch"]) / "CONTROL").exists()


# ---------------------------------------------------------------------------
# run_receptors: inline
# ---------------------------------------------------------------------------


def _fake_run_receptor(calls: list[dict], status=None):
    """A stand-in for run_receptor: one complete hrrr result per receptor, unless *status* says otherwise."""

    def fake(project, receptor_id, *, compute_root, skip_existing=True):
        calls.append({"receptor": receptor_id, "skip_existing": skip_existing})
        state = (status or {}).get(receptor_id, ("complete", None))
        return [SimulationResult(f"{receptor_id}/hrrr", state[0], error=state[1])]

    return fake


def test_run_receptors_inline_returns_results_in_order(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = [str(r.id) for r in (receptor, other_receptor)]
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_receptor", _fake_run_receptor(calls))

    results = run_receptors(
        model, ids, compute_root=tmp_path / "scratch", n_cores=1, skip_existing=True
    )

    assert [r.sim_id.split("/")[0] for r in results] == ids
    assert [r.status for r in results] == ["complete", "complete"]
    assert [c["receptor"] for c in calls] == ids


def test_run_receptors_empty_ids_returns_empty(tmp_path, receptor, monkeypatch):
    model = _model(tmp_path, [receptor])
    monkeypatch.setattr(
        worker, "run_receptor", lambda *a, **k: pytest.fail("must not run")
    )

    assert run_receptors(model, [], compute_root=tmp_path / "scratch", n_cores=1) == []


def test_run_receptors_inline_skips_existing_by_default(
    tmp_path, receptor, monkeypatch
):
    model = _model(tmp_path, [receptor])
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_receptor", _fake_run_receptor(calls))

    run_receptors(
        model, [str(receptor.id)], compute_root=tmp_path / "scratch", n_cores=1
    )
    run_receptors(
        model,
        [str(receptor.id)],
        compute_root=tmp_path / "scratch",
        n_cores=1,
        skip_existing=False,
    )

    assert [c["skip_existing"] for c in calls] == [True, False]


def test_run_receptors_inline_stops_after_interrupt(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = [str(r.id) for r in (receptor, other_receptor)]
    calls: list[dict] = []
    monkeypatch.setattr(
        worker,
        "run_receptor",
        _fake_run_receptor(calls, {ids[0]: ("interrupted", "Worker preempted")}),
    )

    results = run_receptors(model, ids, compute_root=tmp_path / "scratch", n_cores=1)

    assert [(r.sim_id.split("/")[0], r.status, r.error) for r in results] == [
        (ids[0], "interrupted", "Worker preempted")
    ]
    assert [c["receptor"] for c in calls] == [ids[0]]


def test_run_receptors_inline_continues_after_failed_result(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = [str(r.id) for r in (receptor, other_receptor)]
    monkeypatch.setattr(
        worker, "run_receptor", _fake_run_receptor([], {ids[0]: ("failed", "boom")})
    )

    results = run_receptors(model, ids, compute_root=tmp_path / "scratch", n_cores=1)

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
    monkeypatch.setattr(worker, "_POOL_PROJECT", None)
    monkeypatch.setattr(worker, "_POOL_COMPUTE_ROOT", None)
    monkeypatch.setattr(worker, "_POOL_SKIP", True)
    return _FakePool


def test_run_receptors_pool_rebuilds_model_and_orders_results(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_receptor", _fake_run_receptor(calls))

    results = run_receptors(
        model, ids, compute_root=tmp_path / "scratch", n_cores=2, skip_existing=False
    )

    [pool] = fake_pool.instances
    assert pool.n_cores == 2
    assert not pool.terminated
    assert pool.closed and pool.joined
    # The initializer opened the project again from its directory.
    assert worker._POOL_PROJECT is not None
    assert worker._POOL_PROJECT is not model
    assert worker._POOL_PROJECT.directory == model.directory
    assert tmp_path / "scratch" == worker._POOL_COMPUTE_ROOT
    assert worker._POOL_SKIP is False
    # Results come back in input order even though the pool yielded reversed.
    assert [r.sim_id.split("/")[0] for r in results] == ids
    assert [c["skip_existing"] for c in calls] == [False, False]


def test_run_receptors_pool_terminates_on_interrupted_result(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])

    # The fake pool yields the *last* id first, so interrupt on the first id.
    monkeypatch.setattr(
        worker,
        "run_receptor",
        _fake_run_receptor([], {ids[0]: ("interrupted", "Worker preempted")}),
    )

    results = run_receptors(model, ids, compute_root=tmp_path / "scratch", n_cores=2)

    [pool] = fake_pool.instances
    assert pool.terminated
    assert [(r.sim_id.split("/")[0], r.status) for r in results] == [
        (ids[0], "interrupted"),
        (ids[1], "complete"),
    ]


def test_run_receptors_pool_keyboard_interrupt_terminates_and_returns(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    """A KeyboardInterrupt in the parent loop terminates the pool, keeping results."""
    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])

    def imap_then_interrupt(self, func, iterable):
        items = list(iterable)
        yield func(items[0])
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_pool, "imap_unordered", imap_then_interrupt)
    monkeypatch.setattr(worker, "run_receptor", _fake_run_receptor([]))

    results = run_receptors(model, ids, compute_root=tmp_path / "scratch", n_cores=2)

    [pool] = fake_pool.instances
    assert pool.terminated
    assert [r.sim_id.split("/")[0] for r in results] == [ids[0]]
