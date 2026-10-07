"""Tests for the worker-side execution functions in ``stilt.execution.worker``."""

import datetime as dt
from typing import ClassVar

import pandas as pd
import pytest

from stilt.config import ProjectConfig, Variant
from stilt.exceptions import MeteorologyError, SimulationError
from stilt.execution import worker
from stilt.execution.config import ExecutionConfig
from stilt.execution.worker import make_footprint, run_receptor, run_receptors
from stilt.footprint.config import FootprintConfig
from stilt.meteorology import Met, MetConfig
from stilt.output import Output
from stilt.project import Project
from stilt.receptors import PointReceptor, Receptor
from stilt.simulation import Simulation
from stilt.spatial import Grid
from stilt.transport import ModelRun, TransportConfig
from stilt.transport.hysplit import FailureReason, HysplitConfig

from ..fixtures.factories import make_met_config, make_project_config, make_variant

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
    return make_met_config(tmp_path / "met")


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
    return make_variant(
        name, transport=params, met_config=met_config, footprint=footprint
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
            "particle": [1.0],
            "lon": [-111.9],
            "lat": [40.7],
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
    sim.output.write_particles(sim.variant, sim.receptor, particles, [])
    return particles


def _write_footprint(sim: Simulation, *, empty: bool = False) -> None:
    """Record a footprint (or an empty one) for *sim* in the output directory."""
    assert sim.variant.footprint is not None
    if empty:
        sim.output.write_empty_footprint(sim.variant, sim.receptor)
    else:
        make_footprint(sim, _particles(sim.receptor))


def _model_config(tmp_path, **kwargs) -> ProjectConfig:
    return make_project_config(tmp_path, **kwargs)


def _model(tmp_path, receptors, **config_kwargs) -> Project:
    return Project.init(
        tmp_path / "proj",
        config=_model_config(tmp_path, **config_kwargs),
        receptors=list(receptors),
    )


# ---------------------------------------------------------------------------
# run_particles
# ---------------------------------------------------------------------------


def test_run_particles_starts_in_an_empty_directory(
    sim, met, compute_root, monkeypatch
):
    """A directory left by a stopped job is cleared before the model runs."""
    workdir = compute_root / sim.receptor.id / sim.variant.name
    workdir.mkdir(parents=True)
    (workdir / "PARTICLE_STILT.DAT").write_text("left over\n")
    seen: list[list[str]] = []

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            seen.append(sorted(p.name for p in workdir.iterdir()))
            raise RuntimeError("stop here")

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())
    with pytest.raises(RuntimeError):
        worker.run_particles(sim, met=met, workdir=workdir)
    assert seen == [[]]


def test_a_run_stopped_partway_leaves_its_start_in_the_log(
    sim, met, compute_root, monkeypatch
):
    """The log is written when the run starts, so a cut-off run reads as interrupted."""

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            raise KeyboardInterrupt  # SIGTERM from Slurm, mid-run

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())
    with pytest.raises(KeyboardInterrupt):
        worker.run_particles(sim, met=met, workdir=compute_root / "w")

    assert "PYSTILT started this run" in sim.log
    assert not sim.has_particles and sim.failure is None


def test_a_model_that_needs_no_workdir_gets_none(sim, met, compute_root, monkeypatch):
    seen: list = []

    class _Model:
        name = "hysplit"
        needs_workdir = False

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            seen.append(workdir)
            raise SimulationError("stop here", log="model log\n")

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())
    monkeypatch.setattr(worker, "get_model", lambda name: _Model())
    workdir = compute_root / sim.receptor.id / sim.variant.name
    with pytest.raises(SimulationError):
        worker.run_particles(sim, met=met, workdir=workdir)

    assert seen == [None]
    assert not workdir.exists()
    assert sim.log == "model log\n"
    assert sim.kept_workdir is None or not sim.kept_workdir.exists()


def test_run_particles_keeps_no_empty_scratch_copy(sim, met, compute_root, monkeypatch):
    """A run that fails before writing anything, such as on missing met, leaves no scratch copy."""

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            raise MeteorologyError("Insufficient number of meteorological files found.")

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())
    with pytest.raises(MeteorologyError):
        worker.run_particles(
            sim, met=met, workdir=compute_root / sim.receptor.id / sim.variant.name
        )

    kept = sim.output.kept_workdir(sim.variant, sim.receptor.id)
    assert kept is None or not kept.exists()
    assert not (compute_root / sim.receptor.id / sim.variant.name).exists()


def test_run_particles_without_particles_is_a_simulation_error(
    sim, met, compute_root, monkeypatch
):
    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            return ModelRun(particles=pd.DataFrame(), met_files=[])

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())
    with pytest.raises(SimulationError) as caught:
        worker.run_particles(
            sim, met=met, workdir=compute_root / sim.receptor.id / sim.variant.name
        )
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

    def fake(sim, particles):
        calls.append((sim.variant.name, id(particles)))
        _write_footprint(sim)

    monkeypatch.setattr(worker, "make_footprint", fake)


def _run_receptor(project, receptor, **kwargs):
    return run_receptor(
        project, str(receptor.id), compute_root=project.directory / "scratch", **kwargs
    )


SHARED = {"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}, "zi08": {"ziscale": 0.8}}


def test_a_receptor_leaves_no_folders_under_the_compute_root(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor], variants={"hrrr": {"realizations": 2}})
    compute_root = project.directory / "scratch"

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            (workdir / "CONTROL").write_text("")
            raise RuntimeError("stop here")

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())

    assert len(_run_receptor(project, receptor)) == 2  # both realizations failed
    assert list(compute_root.iterdir()) == []


def test_the_execution_settings_given_reach_the_hysplit_run(
    tmp_path, receptor, monkeypatch
):
    """An override passed in, not config.yaml's, sets the timeout and keep_scratch."""
    project = _model(tmp_path, [receptor])
    seen: dict = {}

    def fake(sim, **kwargs):
        seen.update(kwargs)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    override = ExecutionConfig(timeout=42, keep_scratch=True)
    _run_receptor(project, receptor, execution=override)

    assert (seen["timeout"], seen["keep_scratch"]) == (42, True)


def test_a_particles_only_variant_runs_hysplit_and_completes(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor])
    hysplit: list[str] = []
    _fake_run_particles(monkeypatch, hysplit)
    monkeypatch.setattr(
        worker, "make_footprint", lambda *a, **k: pytest.fail("no footprint")
    )

    assert _run_receptor(project, receptor) == []
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

    assert _run_receptor(project, receptor) == []
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

    assert _run_receptor(project, receptor) == []
    assert project.incomplete().empty


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


def test_run_receptor_stops_on_preemption_with_what_finished_written(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor], grid=GRID, variants=SHARED)
    hysplit: list[str] = []

    def fake(sim, **kwargs):
        if sim.variant.name == "zi08":
            raise KeyboardInterrupt
        hysplit.append(sim.variant.name)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    _fake_make_footprint(monkeypatch, [])

    with pytest.raises(KeyboardInterrupt):
        _run_receptor(project, receptor)

    st = project.status()
    assert dict(zip(st.variant, st.state, strict=True)) == {
        "hrrr": "complete",
        "hrrr-s2": "complete",
        "zi08": "pending",
    }


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
            # The driver passes the enum; the record must still write.
            raise SimulationError(
                "HYSPLIT failed (MET_COVERAGE).", reason=FailureReason.MET_COVERAGE
            )
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    _fake_make_footprint(monkeypatch, [])

    problems = _run_receptor(project, receptor)

    assert hysplit == ["hrrr", "zi08"]
    # One line for the group whose run failed.
    assert problems == [
        "hrrr failed during particles (MET_COVERAGE): HYSPLIT failed (MET_COVERAGE)."
    ]
    for name in ("hrrr", "hrrr-s2"):
        failure = project.simulation(str(receptor.id), name).failure
        assert failure is not None
        assert failure["step"] == "particles"
        assert failure["reason"] == "MET_COVERAGE"
        assert failure["message"] == "HYSPLIT failed (MET_COVERAGE)."
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

    def fake(sim, particles):
        if sim.variant.name == "hrrr":
            raise ValueError("bad grid")
        made.append(sim.variant.name)
        _write_footprint(sim)

    monkeypatch.setattr(worker, "make_footprint", fake)

    problems = _run_receptor(project, receptor)

    assert problems == ["hrrr failed during footprint (ValueError): bad grid"]
    assert made == ["hrrr-s2"]
    failure = project.simulation(str(receptor.id), "hrrr").failure
    assert failure is not None
    assert failure["step"] == "footprint" and failure["reason"] == "ValueError"
    assert failure["message"] == "bad grid"
    assert "ValueError: bad grid" in failure["traceback"]
    assert project.simulation(str(receptor.id), "hrrr-s2").failure is None
    # The record belongs to the footprint folder, found by its settings hash.
    sim = project.simulation(str(receptor.id), "hrrr")
    assert project.output.failure("footprints", sim.variant, sim.receptor.id) == failure


def test_a_success_clears_the_failure_it_replaces(tmp_path, receptor, monkeypatch):
    project = _model(tmp_path, [receptor], grid=GRID)
    sim = project.simulation(str(receptor.id), "hrrr")

    def fail(sim, **kwargs):
        raise MeteorologyError("Insufficient number of meteorological files found.")

    monkeypatch.setattr(worker, "run_particles", fail)
    _run_receptor(project, receptor)
    assert sim.failure is not None and sim.failure["reason"] == "MISSING_MET_FILES"
    assert project.output.failure("particles", sim.variant, sim.receptor.id)

    _fake_run_particles(monkeypatch, [])
    _fake_make_footprint(monkeypatch, [])
    _run_receptor(project, receptor)

    assert sim.is_complete and sim.failure is None
    assert project.output.failure("particles", sim.variant, sim.receptor.id) is None


def test_a_failed_run_keeps_its_log_and_working_directory(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor])
    sim = project.simulation(str(receptor.id), "hrrr")

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            (workdir / "CONTROL").write_text("...")
            raise SimulationError(
                "HYSPLIT failed (FORTRAN_RUNTIME_ERROR).",
                reason="FORTRAN_RUNTIME_ERROR",
                log="hycs_std said something\n",
            )

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())
    _run_receptor(project, receptor)

    failure = sim.failure
    assert failure is not None and failure["reason"] == "FORTRAN_RUNTIME_ERROR"
    assert set(failure) == {"step", "reason", "message", "time"}
    assert "hycs_std said" in sim.log
    assert sim.kept_workdir is not None
    assert (sim.kept_workdir / "CONTROL").exists()


# ---------------------------------------------------------------------------
# run_receptors: inline
# ---------------------------------------------------------------------------


def _fake_run_receptor(calls: list[dict], status=None):
    """
    A stand-in for run_receptor that records its calls.

    *status* maps a receptor to ``"interrupted"`` (it raises
    KeyboardInterrupt, as a preempted run does) or ``"failed"`` (it reports
    one problem); the others complete.
    """

    def fake(
        project,
        receptor_id,
        *,
        compute_root,
        execution=None,
        skip_existing=True,
        batched=True,
    ):
        calls.append(
            {
                "receptor": receptor_id,
                "execution": execution,
                "skip_existing": skip_existing,
            }
        )
        state = (status or {}).get(receptor_id)
        if state == "interrupted":
            raise KeyboardInterrupt
        return ["hrrr failed during particles (X): boom"] if state == "failed" else []

    return fake


def test_run_receptors_inline_runs_in_order_and_logs_progress(
    tmp_path, receptor, other_receptor, monkeypatch, caplog
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = [str(r.id) for r in (receptor, other_receptor)]
    calls: list[dict] = []
    monkeypatch.setattr(
        worker, "run_receptor", _fake_run_receptor(calls, {ids[0]: "failed"})
    )

    with caplog.at_level("INFO", logger="stilt.execution.worker"):
        run_receptors(
            model,
            ids,
            compute_root=tmp_path / "scratch",
            execution=ExecutionConfig(cpus=1),
            skip_existing=True,
        )

    assert [c["receptor"] for c in calls] == ids
    assert caplog.messages == [
        f"[1/2] {ids[0]}: hrrr failed during particles (X): boom",
        f"[2/2] {ids[1]} complete",
    ]


def test_run_receptors_empty_ids_runs_nothing(tmp_path, receptor, monkeypatch):
    model = _model(tmp_path, [receptor])
    monkeypatch.setattr(
        worker, "run_receptor", lambda *a, **k: pytest.fail("must not run")
    )
    run_receptors(
        model, [], compute_root=tmp_path / "scratch", execution=ExecutionConfig(cpus=1)
    )


def test_run_receptors_inline_skips_existing_by_default(
    tmp_path, receptor, monkeypatch
):
    model = _model(tmp_path, [receptor])
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_receptor", _fake_run_receptor(calls))

    run_receptors(
        model,
        [str(receptor.id)],
        compute_root=tmp_path / "scratch",
        execution=ExecutionConfig(cpus=1),
    )
    run_receptors(
        model,
        [str(receptor.id)],
        compute_root=tmp_path / "scratch",
        execution=ExecutionConfig(cpus=1),
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
        worker, "run_receptor", _fake_run_receptor(calls, {ids[0]: "interrupted"})
    )

    run_receptors(
        model, ids, compute_root=tmp_path / "scratch", execution=ExecutionConfig(cpus=1)
    )

    assert [c["receptor"] for c in calls] == [ids[0]]


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


def test_run_receptors_pool_rebuilds_the_project_in_each_worker(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])
    calls: list[dict] = []
    monkeypatch.setattr(worker, "run_receptor", _fake_run_receptor(calls))

    run_receptors(
        model,
        ids,
        compute_root=tmp_path / "scratch",
        execution=ExecutionConfig(cpus=2),
        skip_existing=False,
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
    assert ExecutionConfig(cpus=2) == worker._POOL_EXECUTION
    assert sorted(c["receptor"] for c in calls) == sorted(ids)
    assert [c["skip_existing"] for c in calls] == [False, False]


def test_run_receptors_pool_terminates_when_a_worker_is_stopped(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])
    calls: list[dict] = []
    # The fake pool yields the *last* id first, so it is stopped on the second.
    monkeypatch.setattr(
        worker, "run_receptor", _fake_run_receptor(calls, {ids[0]: "interrupted"})
    )

    run_receptors(
        model, ids, compute_root=tmp_path / "scratch", execution=ExecutionConfig(cpus=2)
    )

    [pool] = fake_pool.instances
    assert pool.terminated and not pool.closed
    assert [c["receptor"] for c in calls] == [ids[1], ids[0]]


def test_run_receptors_pool_keyboard_interrupt_terminates(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    """A KeyboardInterrupt in the parent loop terminates the pool."""
    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])
    calls: list[dict] = []

    def imap_then_interrupt(self, func, iterable):
        items = list(iterable)
        yield func(items[0])
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_pool, "imap_unordered", imap_then_interrupt)
    monkeypatch.setattr(worker, "run_receptor", _fake_run_receptor(calls))

    run_receptors(
        model, ids, compute_root=tmp_path / "scratch", execution=ExecutionConfig(cpus=2)
    )

    [pool] = fake_pool.instances
    assert pool.terminated
    assert [c["receptor"] for c in calls] == [ids[0]]


# ---------------------------------------------------------------------------
# A batched transport model
# ---------------------------------------------------------------------------


class BatchConfig(TransportConfig):
    """The batched toy model's config: the shared parameters only."""


class BatchedToy:
    """A toy model that runs many receptors per call and needs no workdir, as an emulator would."""

    name = "toy"
    config_class = BatchConfig
    batched = True
    needs_workdir = False
    calls: ClassVar[list[list[str]]] = []
    skip: ClassVar[set[str]] = set()  # receptors it returns no rows for

    def version(self, config):
        return "1.0"

    def data_files(self, config):
        return None

    def run(self, receptor, config, met, window, workdir=None, timeout=None):
        raise AssertionError("a batched model is run with run_many")

    def run_many(self, receptors, config, met, windows, workdir=None, timeout=None):
        assert workdir is None
        BatchedToy.calls.append([str(r.id) for r in receptors])
        tables = [
            pd.DataFrame(
                {
                    "receptor": str(r.id),
                    "particle": [1, 2, 1, 2],
                    "time": [0, 0, -60, -60],
                    "lon": r.longitude,
                    "lat": r.latitude,
                    "zagl": r.altitude,
                    "foot": [0.0, 0.0, 0.01, 0.02],
                }
            )
            for r in receptors
            if str(r.id) not in self.skip
        ]
        return ModelRun(particles=pd.concat(tables), log="toy ran\n")


BATCHED_TOY = f"{__name__}.BatchedToy"


@pytest.fixture
def batched_toy():
    BatchedToy.calls = []
    BatchedToy.skip = set()
    return BatchedToy


def test_a_batched_model_runs_every_receptor_of_a_variant_in_one_call(
    tmp_path, receptor, other_receptor, batched_toy
):
    third = PointReceptor(
        time=dt.datetime(2023, 1, 1, 14),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    project = _model(
        tmp_path,
        [receptor, other_receptor, third],
        grid=GRID,
        hnf_plume=False,
        variants={
            "toy": {"model": BATCHED_TOY},
            "toy-s2": {"model": BATCHED_TOY, "smooth_factor": 2.0},
        },
    )
    ids = [str(r.id) for r in (receptor, other_receptor, third)]
    batched_toy.skip = {ids[2]}

    run_receptors(project, ids, compute_root=tmp_path / "scratch")

    # One call for the variants that share particles, all three receptors in it.
    assert batched_toy.calls == [ids]
    for rid in ids[:2]:
        for variant in ("toy", "toy-s2"):
            sim = project.simulation(rid, variant)
            assert sim.is_complete, (rid, variant)
        assert project.simulation(rid, "toy").log == "toy ran\n"
    # The receptor the model gave no rows fails alone.
    failed = project.simulation(ids[2], "toy")
    assert not failed.has_particles
    assert failed.failure["reason"] == "NO_PARTICLE_DATA"
    # No workdir was made for a model that does not ask for one.
    assert not (tmp_path / "scratch").exists()

    # Run again: nothing is missing, so the model is not called.
    batched_toy.skip = set()
    run_receptors(project, ids[:2], compute_root=tmp_path / "scratch")
    assert batched_toy.calls == [ids]
