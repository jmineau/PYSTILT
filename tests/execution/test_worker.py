"""Tests for the worker-side execution functions in ``stilt.execution.worker``."""

import datetime as dt
import errno
from concurrent.futures import Future
from typing import ClassVar

import pandas as pd
import pytest
from typer.testing import CliRunner

from stilt.cli import _exit_code, app
from stilt.config import ProjectConfig, Variant
from stilt.exceptions import MeteorologyError, SimulationError, WorkerDied
from stilt.execution import worker
from stilt.execution.config import ExecutionConfig
from stilt.execution.worker import make_footprint, run_receptor, run_receptors
from stilt.footprint.config import FootprintConfig
from stilt.output import Output
from stilt.project import Project
from stilt.receptors import PointReceptor, Receptor
from stilt.simulation import Simulation
from stilt.spatial import Grid
from stilt.transport import ModelRun, TransportConfig
from stilt.transport.hysplit import FailureReason, HysplitConfig, Met, MetConfig

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
            "age": [-60.0],
            "particle": [1.0],
            "lon": [-111.9],
            "lat": [40.7],
            "zagl": [10.0],
            "foot": [1e-5],
        }
    )
    data["time"] = pd.Timestamp(receptor.time) + pd.to_timedelta(
        data["age"], unit="min"
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
        worker.run_particles(sim, workdir=workdir)
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
        worker.run_particles(sim, workdir=compute_root / "w")

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
        worker.run_particles(sim, workdir=workdir)

    assert seen == [None]
    assert not workdir.exists()
    assert sim.log == "model log\n"
    assert sim.kept_workdir is None or not sim.kept_workdir.exists()


def test_run_particles_keeps_no_empty_scratch_copy(sim, met, compute_root, monkeypatch):
    """A run that fails before writing anything, such as on missing met, leaves no scratch copy."""

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            raise MeteorologyError("No met file for 2023-01-01 11:00.")

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())
    with pytest.raises(MeteorologyError):
        worker.run_particles(
            sim, workdir=compute_root / sim.receptor.id / sim.variant.name
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
            sim, workdir=compute_root / sim.receptor.id / sim.variant.name
        )
    assert caught.value.reason == "NO_PARTICLE_DATA"


# ---------------------------------------------------------------------------
# run_receptor: variants that share particles run as one group
# ---------------------------------------------------------------------------


def _fake_run_particles(monkeypatch, calls: list[str]):
    """Replace run_particles with one that writes particles and records the variant it ran for."""

    def fake(sim, *, workdir, keep_workdir=False, timeout=None):
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


@pytest.mark.parametrize("numpar", [200, 100], ids=["other-settings", "same-settings"])
def test_runs_that_share_a_compute_root_never_share_a_workdir(
    tmp_path, receptor, monkeypatch, numpar
):
    """Two projects run the same receptor and variant name at once in one compute root."""
    first = _model(tmp_path / "a", [receptor], numpar=100)
    second = _model(tmp_path / "b", [receptor], numpar=numpar)
    compute_root = tmp_path / "compute"
    ended: dict[str, str | None] = {}

    class _Model:
        name = "hysplit"
        started = 0

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            mine = f"run {_Model.started}"
            _Model.started += 1
            (workdir / "CONTROL").write_text(mine)
            if mine == "run 0":  # the second project starts while the first runs
                run_receptor(second, str(receptor.id), compute_root=compute_root)
            control = workdir / "CONTROL"
            ended[mine] = control.read_text() if control.exists() else None
            raise RuntimeError("stop here")

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())
    run_receptor(first, str(receptor.id), compute_root=compute_root)

    # Each run finds its own input in its workdir when it ends.
    assert ended == {"run 0": "run 0", "run 1": "run 1"}
    assert list(compute_root.iterdir()) == []


def test_each_run_gets_a_new_workdir_named_after_its_simulation(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor], variants={"hrrr": {"realizations": 2}})
    compute_root = tmp_path / "compute"
    workdirs: list = []

    def fake(sim, *, workdir, keep_workdir=False, timeout=None):
        workdirs.append(workdir)
        raise RuntimeError("stop here")

    monkeypatch.setattr(worker, "run_particles", fake)
    for _ in range(2):  # the same simulations again
        run_receptor(project, str(receptor.id), compute_root=compute_root)

    assert len(set(workdirs)) == 4
    assert {w.parent for w in workdirs} == {compute_root}
    names = sorted(w.name.rsplit("_", 1)[0] for w in workdirs)
    assert names == [f"{receptor.id}_hrrr_{k}" for k in (0, 0, 1, 1)]


def test_the_execution_settings_given_reach_the_hysplit_run(
    tmp_path, receptor, monkeypatch
):
    """An override passed in, not config.yaml's, sets the timeout and keep_workdir."""
    project = _model(tmp_path, [receptor])
    seen: dict = {}

    def fake(sim, **kwargs):
        seen.update(kwargs)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    override = ExecutionConfig(timeout=42, keep_workdir=True)
    _run_receptor(project, receptor, execution=override)

    assert (seen["timeout"], seen["keep_workdir"]) == (42, True)


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


def test_run_receptor_takes_timeout_and_keep_workdir_from_execution(
    tmp_path, receptor, monkeypatch
):
    project = _model(
        tmp_path, [receptor], execution={"timeout": 120, "keep_workdir": True}
    )
    seen: list[dict] = []

    def fake(sim, **kwargs):
        seen.append(kwargs)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    _run_receptor(project, receptor)

    assert seen and all(k["timeout"] == 120 and k["keep_workdir"] for k in seen)


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
        raise MeteorologyError("No met file for 2023-01-01 11:00.")

    monkeypatch.setattr(worker, "run_particles", fail)
    _run_receptor(project, receptor)
    assert sim.failure is not None and sim.failure["reason"] == "MET_COVERAGE"
    assert project.output.failure("particles", sim.variant, sim.receptor.id)

    _fake_run_particles(monkeypatch, [])
    _fake_make_footprint(monkeypatch, [])
    _run_receptor(project, receptor)

    assert sim.is_complete and sim.failure is None
    assert project.output.failure("particles", sim.variant, sim.receptor.id) is None


# A connection or I/O error that a retry may get past is not a failure of the
# simulation: no record is written, so it stays unfinished and `stilt run`
# exits with 3 (stilt._transient.is_transient sorts the errors).


def _client_error(status, code):
    """Return what botocore raises for a response with this status and error code."""
    error = type("ClientError", (Exception,), {})("the service answered")
    error.response = {
        "ResponseMetadata": {"HTTPStatusCode": status},
        "Error": {"Code": code},
    }
    return error


def _raised_from(outer, inner):
    """Return *outer* with *inner* as its ``__cause__``, as ``raise outer from inner`` does."""
    outer.__cause__ = inner
    return outer


def _s3fs_error(error_errno, status, code):
    """Return what s3fs raises for a ClientError: an OSError that keeps it as its cause."""
    return _raised_from(
        OSError(error_errno, "the store answered"), _client_error(status, code)
    )


def _states(project):
    return dict(zip(project.status().receptor, project.status().state, strict=True))


def _fails_with(error):
    def fail(*args, **kwargs):
        raise error

    return fail


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("the bucket stopped answering"),
        TimeoutError("read timed out"),
        OSError(errno.EIO, "Input/output error"),
        # s3fs turns a busy answer into an OSError and keeps the ClientError.
        _s3fs_error(errno.EBUSY, 503, "SlowDown"),
    ],
)
def test_a_connection_error_in_a_simulation_leaves_no_failure_record(
    tmp_path, receptor, monkeypatch, error
):
    project = _model(tmp_path, [receptor])
    sim = project.simulation(str(receptor.id), "hrrr")
    monkeypatch.setattr(worker, "run_particles", _fails_with(error))

    problems = _run_receptor(project, receptor)

    assert len(problems) == 1
    assert problems[0].startswith("hrrr did not finish during particles")
    assert "a retry may finish it" in problems[0]
    assert sim.failure is None
    assert project.output.failure("particles", sim.variant, sim.receptor.id) is None
    assert _states(project) == {sim.receptor.id: "pending"}


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("denied"),  # AccessDenied
        FileNotFoundError("no such key"),  # NoSuchKey, NoSuchBucket
        OSError(errno.ENOSPC, "No space left on device"),
        OSError(errno.EINVAL, "s3fs's PermanentRedirect"),
        RuntimeError("a bug"),
        ValueError("bad grid"),
        # An answer the service gave, through a library that wraps it in a path error.
        _raised_from(FileNotFoundError("gone"), _client_error(404, "NoSuchKey")),
        _s3fs_error(errno.EINVAL, 400, "InvalidBucketName"),
    ],
)
def test_an_error_that_a_retry_would_repeat_is_still_a_recorded_failure(
    tmp_path, receptor, monkeypatch, error
):
    project = _model(tmp_path, [receptor])
    sim = project.simulation(str(receptor.id), "hrrr")
    monkeypatch.setattr(worker, "run_particles", _fails_with(error))

    problems = _run_receptor(project, receptor)

    assert problems[0].startswith("hrrr failed during particles")
    assert sim.failure is not None and sim.failure["step"] == "particles"
    assert _states(project) == {sim.receptor.id: "failed"}


def test_a_simulation_error_is_a_failure_whatever_caused_it(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor])
    sim = project.simulation(str(receptor.id), "hrrr")

    def fail(sim, **kwargs):
        raise SimulationError("HYSPLIT failed.", reason="FORTRAN_RUNTIME_ERROR") from (
            ConnectionResetError("reset")
        )

    monkeypatch.setattr(worker, "run_particles", fail)
    _run_receptor(project, receptor)

    assert sim.failure is not None and sim.failure["reason"] == "FORTRAN_RUNTIME_ERROR"
    assert _states(project) == {sim.receptor.id: "failed"}


def test_a_connection_error_while_the_particles_are_written_is_interrupted(
    tmp_path, receptor, monkeypatch
):
    """The run had started and its log says so; there is no particle file and no record."""
    project = _model(tmp_path, [receptor])
    sim = project.simulation(str(receptor.id), "hrrr")

    class _Model:
        name = "hysplit"

        def run(self, receptor, params, met, window, workdir=None, timeout=None):
            return ModelRun(particles=_particles(receptor), met_files=[])

    monkeypatch.setattr("stilt.transport.get_model", lambda name: _Model())
    monkeypatch.setattr(
        Output, "write_particles", _fails_with(ConnectionError("bucket down"))
    )
    problems = _run_receptor(project, receptor)

    assert "did not finish during particles" in problems[0]
    assert not sim.has_particles and sim.failure is None
    assert _states(project) == {sim.receptor.id: "interrupted"}


def test_a_connection_error_in_a_footprint_leaves_the_particles_and_no_record(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor], grid=GRID)
    sim = project.simulation(str(receptor.id), "hrrr")
    _fake_run_particles(monkeypatch, [])
    monkeypatch.setattr(
        worker, "make_footprint", _fails_with(OSError(errno.ESTALE, "stale"))
    )

    problems = _run_receptor(project, receptor)

    assert problems[0].startswith("hrrr did not finish during footprint")
    assert sim.has_particles and not sim.has_footprint and sim.failure is None
    assert _states(project) == {sim.receptor.id: "pending"}

    # The next run keeps the particles and makes the footprint.
    ran: list[str] = []
    _fake_run_particles(monkeypatch, ran)
    _fake_make_footprint(monkeypatch, [])
    assert _run_receptor(project, receptor) == []
    assert ran == [] and sim.is_complete


def test_a_rerun_after_a_connection_error_finishes_the_simulation(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor])
    sim = project.simulation(str(receptor.id), "hrrr")
    monkeypatch.setattr(worker, "run_particles", _fails_with(ConnectionError("down")))
    _run_receptor(project, receptor)
    assert not sim.is_complete

    _fake_run_particles(monkeypatch, [])
    assert _run_receptor(project, receptor) == []

    assert sim.is_complete and sim.failure is None
    assert _states(project) == {sim.receptor.id: "complete"}


def test_a_connection_error_removes_the_failure_record_of_an_earlier_attempt(
    tmp_path, receptor, monkeypatch
):
    """The earlier attempt failed for a reason that is mended; this one did not finish, so it is not "failed"."""
    project = _model(tmp_path, [receptor])
    sim = project.simulation(str(receptor.id), "hrrr")
    monkeypatch.setattr(
        worker, "run_particles", _fails_with(MeteorologyError("No met file."))
    )
    _run_receptor(project, receptor)
    assert sim.failure is not None

    monkeypatch.setattr(worker, "run_particles", _fails_with(ConnectionError("down")))
    _run_receptor(project, receptor)

    assert sim.failure is None
    assert _states(project) == {sim.receptor.id: "pending"}


def test_an_earlier_record_that_cannot_be_removed_does_not_stop_the_run(
    tmp_path, receptor, other_receptor, monkeypatch
):
    """With the output unreachable the old record stays, and the run goes on to the next receptor."""
    project = _model(tmp_path, [receptor, other_receptor])
    monkeypatch.setattr(worker, "run_particles", _fails_with(ConnectionError("down")))
    monkeypatch.setattr(
        Output, "clear_failure", _fails_with(ConnectionError("down too"))
    )

    for each in (receptor, other_receptor):
        problems = _run_receptor(project, each)
        assert "did not finish" in problems[0]


def test_a_mixed_run_is_3_until_the_connection_error_is_mended_and_then_1(
    tmp_path, receptor, other_receptor, monkeypatch
):
    """The retry runs only what did not finish; the real failure stays, and ends the run with 1."""
    project = _model(tmp_path, [receptor, other_receptor])
    transient, real = str(receptor.id), str(other_receptor.id)

    def first(sim, **kwargs):
        if sim.receptor.id == transient:
            raise ConnectionError("down")
        raise MeteorologyError("No met file.")

    monkeypatch.setattr(worker, "run_particles", first)
    run_receptors(
        project,
        [transient, real],
        compute_root=tmp_path / "compute",
        execution=ExecutionConfig(cpus=1),
    )
    assert _states(project) == {transient: "pending", real: "failed"}
    assert _exit_code(project.status()) == 3

    def second(sim, **kwargs):
        if sim.receptor.id == real:
            raise MeteorologyError("No met file.")
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", second)
    run_receptors(
        project,
        [transient, real],
        compute_root=tmp_path / "compute",
        execution=ExecutionConfig(cpus=1),
    )
    assert _states(project) == {transient: "complete", real: "failed"}
    assert _exit_code(project.status()) == 1


def test_stilt_run_exits_3_after_a_connection_error_in_a_simulation_and_0_on_the_rerun(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor])
    (tmp_path / "met").mkdir(exist_ok=True)
    runner = CliRunner()
    monkeypatch.setattr(worker, "run_particles", _fails_with(ConnectionError("down")))

    down = runner.invoke(app, ["run", str(project.directory)])
    assert down.exit_code == 3, down.output
    assert "pending=1" in down.output and "failed=0" in down.output

    _fake_run_particles(monkeypatch, [])
    up = runner.invoke(app, ["run", str(project.directory)])
    assert up.exit_code == 0, up.output


def test_stilt_run_still_exits_1_for_a_simulation_that_failed(
    tmp_path, receptor, monkeypatch
):
    project = _model(tmp_path, [receptor])
    (tmp_path / "met").mkdir(exist_ok=True)
    monkeypatch.setattr(
        worker, "run_particles", _fails_with(MeteorologyError("No met file."))
    )
    result = CliRunner().invoke(app, ["run", str(project.directory)])
    assert result.exit_code == 1, result.output


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
# run_receptors: process pool (plumbing only, with a synchronous fake executor)
# ---------------------------------------------------------------------------


class _FakeExecutor:
    """
    Synchronous stand-in for ``ProcessPoolExecutor``.

    Runs the initializer in-process at construction and the function at
    ``submit``, returning a finished future that holds its result or its
    error. Records how it was stopped.
    """

    instances: list["_FakeExecutor"] = []

    def __init__(self, max_workers, initializer=None, initargs=()):
        self.max_workers = max_workers
        self.submitted = 0
        self.shutdowns: list[dict] = []
        self._processes: dict = {}
        if initializer is not None:
            initializer(*initargs)
        _FakeExecutor.instances.append(self)

    def submit(self, func, item):
        self.submitted += 1
        future: Future = Future()
        try:
            future.set_result(func(item))
        except Exception as error:  # a KeyboardInterrupt goes up, as a signal does
            future.set_exception(error)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdowns.append({"wait": wait, "cancel_futures": cancel_futures})


@pytest.fixture
def fake_pool(monkeypatch):
    """Patch the executor and keep the worker's module-level pool state isolated."""
    _FakeExecutor.instances = []
    monkeypatch.setattr(worker, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(worker, "_check_workers_can_start", lambda cpus: None)
    # The initializer installs a SIGTERM handler; keep it out of the test process.
    monkeypatch.setattr(worker.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_POOL_PROJECT", None)
    monkeypatch.setattr(worker, "_POOL_COMPUTE_ROOT", None)
    monkeypatch.setattr(worker, "_POOL_SKIP", True)
    return _FakeExecutor


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
    assert pool.max_workers == 2
    # A normal finish waits for the workers to exit on their own.
    assert pool.shutdowns == [{"wait": True, "cancel_futures": False}]
    # The initializer opened the project again from its directory.
    assert worker._POOL_PROJECT is not None
    assert worker._POOL_PROJECT is not model
    assert worker._POOL_PROJECT.directory == model.directory
    assert tmp_path / "scratch" == worker._POOL_COMPUTE_ROOT
    assert worker._POOL_SKIP is False
    assert ExecutionConfig(cpus=2) == worker._POOL_EXECUTION
    assert sorted(c["receptor"] for c in calls) == sorted(ids)
    assert [c["skip_existing"] for c in calls] == [False, False]


def test_run_receptors_pool_logs_every_receptor_once(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool, caplog
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])
    monkeypatch.setattr(
        worker, "run_receptor", _fake_run_receptor([], {ids[0]: "failed"})
    )

    with caplog.at_level("INFO", logger="stilt.execution.worker"):
        run_receptors(
            model,
            ids,
            compute_root=tmp_path / "scratch",
            execution=ExecutionConfig(cpus=2),
        )

    # Receptors are logged as they finish, in either order, counted 1 then 2.
    lines = {m.split("] ", 1)[1] for m in caplog.messages}
    assert lines == {
        f"{ids[0]}: hrrr failed during particles (X): boom",
        f"{ids[1]} complete",
    }
    assert sorted(m.split("]")[0] for m in caplog.messages) == ["[1/2", "[2/2"]


def test_run_receptors_pool_stops_its_workers_and_says_how_far_it_got(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool, caplog
):
    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])
    calls: list[dict] = []
    # The fake runs each receptor as it is submitted, so the second is stopped
    # before the first one's result is read.
    monkeypatch.setattr(
        worker, "run_receptor", _fake_run_receptor(calls, {ids[1]: "interrupted"})
    )

    with caplog.at_level("INFO", logger="stilt.execution.worker"):
        run_receptors(
            model,
            ids,
            compute_root=tmp_path / "scratch",
            execution=ExecutionConfig(cpus=2),
        )

    [pool] = fake_pool.instances
    # The workers are stopped, and what has not started is cancelled.
    assert pool.shutdowns == [{"wait": False, "cancel_futures": True}]
    assert [c["receptor"] for c in calls] == ids
    assert "stopped after 0 of 2 receptors; the rest did not run" in caplog.text


def test_a_pool_worker_exits_on_sigterm_and_ignores_ctrl_c(
    tmp_path, receptor, monkeypatch, fake_pool
):
    """A worker that survived SIGTERM would keep the stop waiting for ever (#309)."""
    import signal

    model = _model(tmp_path, [receptor])
    handlers: dict[int, object] = {}
    monkeypatch.setattr(
        worker.signal,
        "signal",
        lambda signum, handler: handlers.update({signum: handler}),
    )

    worker._init_pool_worker(
        str(model.directory),
        str(model.output.directory),
        str(tmp_path / "scratch"),
        ExecutionConfig(cpus=2),
        True,
    )

    assert handlers[signal.SIGINT] is signal.SIG_IGN
    handler = handlers[signal.SIGTERM]
    assert callable(handler)
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)


def test_run_receptors_pool_worker_error_stops_the_workers_and_is_not_masked(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    """
    A worker's own error must reach the caller, with the workers stopped (#320).

    An error that escapes a receptor's own handling (anything outside
    ``run_receptor``'s particles and footprint ``try`` blocks) is re-raised
    by the future's ``result()``.
    """
    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])

    def fail(*args, **kwargs):
        raise ConnectionError("output bucket unreachable")

    monkeypatch.setattr(worker, "run_receptor", fail)

    with pytest.raises(ConnectionError, match="output bucket unreachable"):
        run_receptors(
            model,
            ids,
            compute_root=tmp_path / "scratch",
            execution=ExecutionConfig(cpus=2),
        )

    [pool] = fake_pool.instances
    assert pool.shutdowns == [{"wait": False, "cancel_futures": True}]


def test_run_receptors_pool_dead_worker_stops_the_run_and_names_what_was_running(
    tmp_path, receptor, other_receptor, monkeypatch, fake_pool
):
    """A worker killed outright breaks the pool: stop, and name the receptors in progress."""
    from concurrent.futures.process import BrokenProcessPool

    model = _model(tmp_path, [receptor, other_receptor])
    ids = list(model.receptors["receptor"])

    def kill(*args, **kwargs):
        raise BrokenProcessPool("a process was terminated abruptly")

    monkeypatch.setattr(worker, "run_receptor", kill)

    with pytest.raises(WorkerDied) as stopped:
        run_receptors(
            model,
            ids,
            compute_root=tmp_path / "scratch",
            execution=ExecutionConfig(cpus=2),
        )

    message = str(stopped.value)
    assert all(rid in message for rid in ids)
    assert "memory" in message
    assert "run it again" in message
    [pool] = fake_pool.instances
    assert pool.shutdowns == [{"wait": False, "cancel_futures": True}]


#: A transport model in its own package, which pool workers import by path.
_TOY_PACKAGE = """
import pandas as pd
from pydantic import BaseModel
from stilt.transport import ModelRun, TransportConfig

class ToyConfig(TransportConfig):
    pass

class ToyMet(BaseModel):
    weather: str

    def settings(self):
        return {"weather": self.weather}

class ToyModel:
    name = "toy"
    config_class = ToyConfig
    met_config_class = ToyMet

    def version(self, config):
        return "1.0"

    def data_files(self, config):
        return None

    def run(self, receptor, config, met, window, workdir=None, timeout=None):
        particles = pd.DataFrame({
            "particle": [1, 2, 1, 2],
            "age": [0, 0, -60, -60],
            "lon": receptor.longitude,
            "lat": receptor.latitude,
            "zagl": receptor.altitude,
            "foot": [0.0, 0.0, 0.01, 0.01],
        })
        return ModelRun(particles=particles)
"""


def test_a_run_on_two_cpus_writes_to_the_output_it_was_given(tmp_path):
    """Pool workers open the project again; they must write where the parent looks (#188)."""
    import os
    import subprocess
    import sys

    (tmp_path / "toypool").mkdir()
    (tmp_path / "toypool" / "__init__.py").write_text(_TOY_PACKAGE)
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yaml").write_text(
        "model: toypool.ToyModel\n"
        "n_hours: -1\n"
        "hnf_plume: false\n"
        "mets:\n  toy:\n    weather: hrrr\n"
        "variants:\n  toy: {}\n"
    )
    (project / "receptors.csv").write_text(
        "time,longitude,latitude,altitude\n"
        + "".join(f"2023-01-01 {h:02d}:00:00,-111.85,40.77,5.0\n" for h in range(4))
    )
    override = tmp_path / "elsewhere"

    done = subprocess.run(
        [sys.executable, "-m", "stilt", "run", str(project), "--cpus", "2"]
        + ["--output", str(override), "--compute-root", str(tmp_path / "work")],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert len(list(override.glob("particles/*/date=*/*.parquet"))) == 4
    assert not (project / "output").exists()


#: A toy model whose run at altitude 99 is killed outright, as the out-of-memory
#: killer would, when ``TOY_KILL`` is set; the others wait, so they are still
#: in progress when the pool finds its worker dead.
_KILLING_TOY_PACKAGE = """
import os
import signal
import time

from toypool import ToyModel

class KillingToy(ToyModel):
    def run(self, receptor, config, met, window, workdir=None, timeout=None):
        if os.environ.get("TOY_KILL"):
            if receptor.altitude == 99.0:
                os.kill(os.getpid(), signal.SIGKILL)  # the worker's own pid
            time.sleep(30)  # until the pool stops it
        return super().run(receptor, config, met, window, workdir, timeout)
"""


def test_a_worker_killed_outright_stops_the_run_and_a_rerun_finishes_it(tmp_path):
    """A pool worker that dies without Python's say used to hang the run for ever (#382)."""
    import os
    import subprocess
    import sys

    for name, source in (("toypool", _TOY_PACKAGE), ("toykill", _KILLING_TOY_PACKAGE)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(source)
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yaml").write_text(
        "model: toykill.KillingToy\n"
        "n_hours: -1\n"
        "hnf_plume: false\n"
        "mets:\n  toy:\n    weather: hrrr\n"
        "variants:\n  toy: {}\n"
    )
    # The first two start together on two cpus; the killed one is the first.
    altitudes = [99.0, 5.0, 5.0, 5.0]
    (project / "receptors.csv").write_text(
        "time,longitude,latitude,altitude\n"
        + "".join(
            f"2023-01-01 {h:02d}:00:00,-111.85,40.77,{altitude}\n"
            for h, altitude in enumerate(altitudes)
        )
    )
    command = [sys.executable, "-m", "stilt", "run", str(project), "--cpus", "2"]
    command += ["--compute-root", str(tmp_path / "work")]
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}

    # On main this hangs; the timeout turns that into a failure.
    killed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        env={**env, "TOY_KILL": "1"},
    )

    ids = list(Project(project).receptors["receptor"])
    assert killed.returncode == 3, killed.stdout + killed.stderr
    # It names the two receptors in progress, not the two that had not started.
    assert all(rid in killed.stderr for rid in ids[:2]), killed.stderr
    assert not any(rid in killed.stderr for rid in ids[2:]), killed.stderr
    assert "memory" in killed.stderr
    assert "run it again" in killed.stderr

    rerun = subprocess.run(
        command, capture_output=True, text=True, timeout=120, env=env
    )

    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert len(list((project / "output").glob("particles/*/date=*/*.parquet"))) == 4


@pytest.mark.parametrize(
    ("alive", "died"),
    [([True, True], False), ([True, False], True), ([False, True], True)],
)
def test_a_dead_pool_worker_is_found_by_looking_at_the_workers(alive, died):
    """The loop looks at the workers itself, since the pool can miss a kill (#400)."""
    from types import SimpleNamespace

    processes = {
        i: SimpleNamespace(is_alive=lambda state=state: state)
        for i, state in enumerate(alive)
    }
    executor = SimpleNamespace(_processes=processes)

    assert worker._a_worker_died(executor) is died


# ---------------------------------------------------------------------------
# A batched transport model
# ---------------------------------------------------------------------------


class BatchConfig(TransportConfig):
    """The batched toy model's config: the shared parameters only."""


class BatchedToy:
    """A toy model that runs many receptors per call and needs no workdir, as an emulator would."""

    name = "toy"
    config_class = BatchConfig
    met_config_class = MetConfig
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
                    "age": [0, 0, config.n_hours * 60, config.n_hours * 60],
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
