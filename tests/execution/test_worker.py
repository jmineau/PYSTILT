"""Tests for the worker-side execution functions in ``stilt.execution.worker``."""

import datetime as dt

import pandas as pd
import pytest

from stilt.config import (
    FootprintConfig,
    Grid,
    MetConfig,
    ProjectConfig,
)
from stilt.exceptions import (
    EmptyParticleOutputError,
    MeteorologyError,
    NoParticleOutputError,
    SimulationError,
)
from stilt.execution import worker
from stilt.execution.worker import (
    SimulationResult,
    make_footprint,
    run_receptor,
    run_receptors,
    run_simulation,
)
from stilt.meteorology import Met
from stilt.output import Output
from stilt.project import Project
from stilt.receptors import PointReceptor, Receptor
from stilt.simulation import Simulation
from stilt.transforms import TransformContext
from stilt.transport import ModelInfo
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


def _run(sim, met, compute_root, **kwargs) -> SimulationResult:
    return run_simulation(sim, met=met, compute_root=compute_root, **kwargs)


def _no_hysplit(monkeypatch):
    monkeypatch.setattr(
        worker, "run_particles", lambda *a, **k: pytest.fail("must not run HYSPLIT")
    )


def _fake_hysplit(monkeypatch, calls: list[str] | None = None):
    """Replace run_particles with one that writes particles and records the call."""

    def fake(sim, *, met, workdir, keep_scratch=False, **kwargs):
        if calls is not None:
            calls.append("hysplit")
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)


def _fake_footprint(monkeypatch, calls: list[str] | None = None, result=None):
    def fake(sim, trajectories, *, context, config=None, transforms=None):
        if calls is not None:
            calls.append("footprint")
        return result

    monkeypatch.setattr(worker, "make_footprint", fake)


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


def _fake_run_simulation(calls: list[dict]):
    """Build a stand-in for run_simulation that records its arguments."""

    def fake(sim, *, skip_existing=True, footprint_stale=False, **kwargs):
        calls.append(
            {
                "sim_id": str(sim.id),
                "skip_existing": skip_existing,
                "footprint_stale": footprint_stale,
            }
        )
        return SimulationResult(str(sim.id), "complete")

    return fake


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_simulation_result_is_frozen_with_optional_error():
    result = SimulationResult("sim/hrrr", "complete")
    assert result.error is None and not result.ran_hysplit
    with pytest.raises(AttributeError):
        result.status = "failed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# run_simulation
# ---------------------------------------------------------------------------


def test_run_simulation_trajectory_only_completes(sim, met, compute_root, monkeypatch):
    _fake_hysplit(monkeypatch)
    monkeypatch.setattr(
        worker, "make_footprint", lambda *a, **k: pytest.fail("no footprint")
    )

    result = _run(sim, met, compute_root)

    assert result == SimulationResult(str(sim.id), "complete", ran_hysplit=True)
    assert sim.has_particles


def test_run_simulation_skips_existing_particles(sim, met, compute_root, monkeypatch):
    _write_particles(sim)
    _no_hysplit(monkeypatch)

    result = _run(sim, met, compute_root)

    assert result == SimulationResult(str(sim.id), "complete", ran_hysplit=False)


def test_run_simulation_reruns_without_skip_existing(
    sim, met, compute_root, monkeypatch
):
    _write_particles(sim)
    calls: list[str] = []
    _fake_hysplit(monkeypatch, calls)

    result = _run(sim, met, compute_root, skip_existing=False)

    assert calls == ["hysplit"]
    assert result.ran_hysplit


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


def test_run_simulation_error_is_failed_and_logged(sim, met, compute_root, monkeypatch):
    def fail(*a, **k):
        raise SimulationError("HYSPLIT failed")

    monkeypatch.setattr(worker, "run_particles", fail)

    result = _run(sim, met, compute_root)

    assert result.status == "failed"
    assert result.error == "HYSPLIT failed"
    assert result.phase == "particles"
    log_text = sim.log
    assert "=== PYSTILT ERROR ===" in log_text
    assert "Phase: particles" in log_text
    assert "Type: SimulationError" in log_text
    assert "Message: HYSPLIT failed" in log_text
    assert sim.outcome == "failed:UNKNOWN"


@pytest.mark.parametrize("error", [NoParticleOutputError, EmptyParticleOutputError])
def test_run_simulation_without_particles_reads_back_as_no_particle_data(
    sim, met, compute_root, monkeypatch, error
):
    def fail(*a, **k):
        raise error("no particles")

    monkeypatch.setattr(worker, "run_particles", fail)

    _run(sim, met, compute_root)

    assert sim.outcome == "failed:NO_PARTICLE_DATA"


def test_run_simulation_error_log_appends_to_existing_hysplit_log(
    sim, met, compute_root, monkeypatch
):
    run = sim.output.particles(sim.variant)
    run.write_log(sim.receptor.id, "hysplit said hello\n")

    def fail(*a, **k):
        raise SimulationError("boom")

    monkeypatch.setattr(worker, "run_particles", fail)
    _run(sim, met, compute_root)

    text = sim.log
    assert text.startswith("hysplit said hello\n")
    assert "Message: boom" in text


def test_run_simulation_generic_exception_is_error(sim, met, compute_root, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(worker, "run_particles", fail)

    result = _run(sim, met, compute_root)

    assert result.status == "error"
    assert result.error == "unexpected"
    assert "Type: RuntimeError" in sim.log


def test_run_simulation_empty_footprint_is_complete(
    fsim, met, compute_root, monkeypatch
):
    def fake(sim, trajectories, *, context, config=None, transforms=None):
        _write_footprint(fsim, empty=True)
        return None

    monkeypatch.setattr(worker, "make_footprint", fake)

    result = _run(fsim, met, compute_root)

    assert result.status == "complete"
    assert fsim.is_complete() and fsim.footprint is None
    assert fsim.empty_reason == "outside_domain"


def test_run_simulation_footprint_error_is_failed(fsim, met, compute_root, monkeypatch):
    def fail(*a, **k):
        raise SimulationError("Footprint failed")

    monkeypatch.setattr(worker, "make_footprint", fail)

    result = _run(fsim, met, compute_root)

    assert result.status == "failed"
    assert "Phase: footprint" in fsim.log


def test_run_simulation_skips_existing_footprint(fsim, met, compute_root, monkeypatch):
    _write_footprint(fsim)
    monkeypatch.setattr(
        worker, "make_footprint", lambda *a, **k: pytest.fail("must not regenerate")
    )

    result = _run(fsim, met, compute_root)

    assert result.status == "complete" and not result.ran_hysplit


def test_run_simulation_skips_existing_empty_footprint(
    fsim, met, compute_root, monkeypatch
):
    _write_footprint(fsim, empty=True)
    monkeypatch.setattr(
        worker, "make_footprint", lambda *a, **k: pytest.fail("must not regenerate")
    )

    assert _run(fsim, met, compute_root).status == "complete"


def test_run_simulation_skip_existing_false_regenerates(
    fsim, met, compute_root, monkeypatch
):
    _write_footprint(fsim)
    calls: list[str] = []
    _fake_hysplit(monkeypatch, calls)
    _fake_footprint(monkeypatch, calls)

    _run(fsim, met, compute_root, skip_existing=False)

    assert calls == ["hysplit", "footprint"]


def test_run_simulation_footprint_stale_regenerates_only_the_footprint(
    fsim, met, compute_root, monkeypatch
):
    _write_footprint(fsim)
    calls: list[str] = []
    _fake_hysplit(monkeypatch, calls)
    _fake_footprint(monkeypatch, calls)

    result = _run(fsim, met, compute_root, footprint_stale=True)

    assert calls == ["footprint"]
    assert result.status == "complete" and not result.ran_hysplit


def test_run_simulation_backfills_missing_particles_and_remakes_the_footprint(
    receptor, met, met_config, params, output, compute_root, monkeypatch
):
    """Lost particles are rerun, and the old footprint is remade from the new ones."""
    s = _make_sim(
        receptor, met_config, params, output, footprint=FootprintConfig(grid=GRID)
    )
    _write_particles(s)
    _write_footprint(s)
    s.output.particles(s.variant).file(s.receptor.id).unlink()
    calls: list[str] = []
    _fake_hysplit(monkeypatch, calls)
    _fake_footprint(monkeypatch, calls)

    result = _run(s, met, compute_root)

    assert result.status == "complete" and result.ran_hysplit
    assert calls == ["hysplit", "footprint"]


def test_run_simulation_footprint_reads_the_stored_particles(
    fsim, met, compute_root, monkeypatch
):
    """With particles present and no HYSPLIT run, the footprint is made from the stored file."""
    seen: list = []

    def fake(sim, trajectories, *, context, config=None, transforms=None):
        seen.append(trajectories)
        return None

    monkeypatch.setattr(worker, "make_footprint", fake)
    _no_hysplit(monkeypatch)

    _run(fsim, met, compute_root)

    [traj] = seen
    assert isinstance(traj, pd.DataFrame) and len(traj) == 1


# ---------------------------------------------------------------------------
# run_receptor
# ---------------------------------------------------------------------------


def test_run_receptor_takes_timeout_and_keep_scratch_from_execution(
    tmp_path, receptor, monkeypatch
):
    model = _model(
        tmp_path, [receptor], execution={"timeout": 120, "keep_scratch": True}
    )
    seen: list[dict] = []

    def fake(sim, **kwargs):
        seen.append(kwargs)
        return SimulationResult(str(sim.id), "complete")

    monkeypatch.setattr(worker, "run_simulation", fake)
    run_receptor(model, str(receptor.id), compute_root=tmp_path / "scratch")

    assert seen and all(k["timeout"] == 120 and k["keep_scratch"] for k in seen)


def test_run_receptor_runs_every_variant_in_config_order(
    tmp_path, receptor, monkeypatch
):
    model = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={"s2": {"smooth_factor": 2}, "hrrr": {}, "zi08": {"ziscale": 0.8}},
    )
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

    result = run_receptor(
        model, str(receptor.id), compute_root=tmp_path / "scratch", skip_existing=False
    )

    assert [c["sim_id"].split("/")[1] for c in calls] == ["s2", "hrrr", "zi08"]
    assert [r.status for r in result] == ["complete"] * 3
    assert {r.sim_id.split("/")[0] for r in result} == {str(receptor.id)}


def test_run_receptor_remakes_sibling_footprints_when_the_particles_reran(
    tmp_path, receptor, monkeypatch
):
    """Variants with equal transport settings share one run: reuse, but remake footprints."""
    model = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={
            "hrrr": {},
            "hrrr-s2": {"smooth_factor": 2.0},
            "zi08": {"ziscale": 0.8},
        },
    )
    calls: list[dict] = []

    def fake(sim, *, skip_existing=True, footprint_stale=False, **kwargs):
        calls.append(
            {
                "variant": sim.variant.name,
                "skip": skip_existing,
                "stale": footprint_stale,
            }
        )
        # hrrr's particles were missing and HYSPLIT ran for it.
        return SimulationResult(
            str(sim.id), "complete", ran_hysplit=sim.variant.name == "hrrr"
        )

    monkeypatch.setattr(worker, "run_simulation", fake)

    run_receptor(
        model, str(receptor.id), compute_root=tmp_path / "scratch", skip_existing=True
    )

    assert calls == [
        {"variant": "hrrr", "skip": True, "stale": False},
        {"variant": "hrrr-s2", "skip": True, "stale": True},  # same run: reuse, remake
        {"variant": "zi08", "skip": True, "stale": False},  # its own run
    ]


def test_run_receptor_runs_failed_particles_once_per_transport_settings(
    tmp_path, receptor, monkeypatch
):
    """A failed HYSPLIT run fails every variant that shares it, without running again."""
    model = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={
            "hrrr": {},
            "hrrr-s2": {"smooth_factor": 2.0},
            "zi08": {"ziscale": 0.8},
        },
    )
    calls: list[str] = []

    def fake(sim, **kwargs):
        calls.append(sim.variant.name)
        if sim.variant.name == "hrrr":
            return SimulationResult(
                str(sim.id), "failed", error="met ends early", phase="particles"
            )
        return SimulationResult(str(sim.id), "complete", ran_hysplit=True)

    monkeypatch.setattr(worker, "run_simulation", fake)

    result = run_receptor(
        model, str(receptor.id), compute_root=tmp_path / "scratch", skip_existing=True
    )

    assert calls == ["hrrr", "zi08"]
    assert [(r.status, r.error) for r in result] == [
        ("failed", "met ends early"),
        ("failed", "met ends early"),
        ("complete", None),
    ]


def test_run_receptor_tries_again_after_a_footprint_failure(
    tmp_path, receptor, monkeypatch
):
    """A failed footprint does not stop a sibling variant with other footprint settings."""
    model = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}},
    )
    calls: list[str] = []

    def fake(sim, **kwargs):
        calls.append(sim.variant.name)
        if sim.variant.name == "hrrr":
            return SimulationResult(
                str(sim.id), "error", error="bad grid", phase="footprint"
            )
        return SimulationResult(str(sim.id), "complete")

    monkeypatch.setattr(worker, "run_simulation", fake)

    run_receptor(
        model, str(receptor.id), compute_root=tmp_path / "scratch", skip_existing=True
    )

    assert calls == ["hrrr", "hrrr-s2"]


def test_run_receptor_no_skip_reruns_each_run_once(tmp_path, receptor, monkeypatch):
    model = _model(
        tmp_path,
        [receptor],
        grid=GRID,
        variants={"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}},
    )
    calls: list[dict] = []

    def fake(sim, *, skip_existing=True, footprint_stale=False, **kwargs):
        calls.append(
            {
                "variant": sim.variant.name,
                "skip": skip_existing,
                "stale": footprint_stale,
            }
        )
        return SimulationResult(str(sim.id), "complete", ran_hysplit=not skip_existing)

    monkeypatch.setattr(worker, "run_simulation", fake)

    run_receptor(
        model, str(receptor.id), compute_root=tmp_path / "scratch", skip_existing=False
    )

    # The second variant must not rerun the shared particles again.
    assert calls == [
        {"variant": "hrrr", "skip": False, "stale": False},
        {"variant": "hrrr-s2", "skip": True, "stale": True},
    ]


def test_run_receptor_normalises_preemption(tmp_path, receptor, monkeypatch):
    model = _model(tmp_path, [receptor])

    def fake(sim, *, skip_existing=True, footprint_stale=False, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "run_simulation", fake)

    result = run_receptor(model, str(receptor.id), compute_root=tmp_path / "scratch")

    assert [(r.status, r.error) for r in result] == [
        ("interrupted", "Worker preempted")
    ]


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

    results = run_receptors(
        model, ids, compute_root=tmp_path / "scratch", n_cores=1, skip_existing=True
    )

    assert [r.sim_id.split("/")[0] for r in results] == ids
    assert [r.status for r in results] == ["complete", "complete"]
    assert [c["sim_id"] for c in calls] == [f"{rid}/hrrr" for rid in ids]


def test_run_receptors_empty_ids_returns_empty(tmp_path, receptor, monkeypatch):
    model = _model(tmp_path, [receptor])
    monkeypatch.setattr(
        worker, "run_simulation", lambda *a, **k: pytest.fail("must not run")
    )

    assert run_receptors(model, [], compute_root=tmp_path / "scratch", n_cores=1) == []


def test_run_receptors_inline_skips_existing_by_default(
    tmp_path, receptor, monkeypatch
):
    model = _model(tmp_path, [receptor])
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

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
    seen: list[str] = []

    def fake(sim, *, skip_existing=True, footprint_stale=False, **kwargs):
        seen.append(str(sim.id))
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "run_simulation", fake)

    results = run_receptors(model, ids, compute_root=tmp_path / "scratch", n_cores=1)

    assert [(r.sim_id.split("/")[0], r.status, r.error) for r in results] == [
        (ids[0], "interrupted", "Worker preempted")
    ]
    assert seen == [f"{ids[0]}/hrrr"]


def test_run_receptors_inline_continues_after_failed_result(
    tmp_path, receptor, other_receptor, monkeypatch
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = [str(r.id) for r in (receptor, other_receptor)]

    def fake(sim, *, skip_existing=True, footprint_stale=False, **kwargs):
        if sim.id.receptor == ids[0]:
            return SimulationResult(str(sim.id), "failed", error="boom")
        return SimulationResult(str(sim.id), "complete")

    monkeypatch.setattr(worker, "run_simulation", fake)

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
    monkeypatch.setattr(worker, "run_simulation", _fake_run_simulation(calls))

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

    def fake(sim, *, skip_existing=True, footprint_stale=False, **kwargs):
        # The fake pool yields the *last* id first, so interrupt on the first id.
        if sim.id.receptor == ids[0]:
            raise KeyboardInterrupt
        return SimulationResult(str(sim.id), "complete")

    monkeypatch.setattr(worker, "run_simulation", fake)

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
    monkeypatch.setattr(
        worker,
        "run_simulation",
        lambda sim, *, skip_existing=True, footprint_stale=False, **kw: (
            SimulationResult(str(sim.id), "complete")
        ),
    )

    results = run_receptors(model, ids, compute_root=tmp_path / "scratch", n_cores=2)

    [pool] = fake_pool.instances
    assert pool.terminated
    assert [r.sim_id.split("/")[0] for r in results] == [ids[0]]
