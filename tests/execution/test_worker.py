"""Tests for the worker-side execution functions in ``runtraj.execution.worker``."""

import datetime as dt
import errno
from concurrent.futures import Future
from typing import ClassVar

import pandas as pd
import pytest
from runtraj.cli import _exit_code, app
from runtraj.config import ProjectConfig, Variant
from runtraj.exceptions import MeteorologyError, SimulationError
from runtraj.execution import worker
from runtraj.execution.config import ExecutionConfig
from runtraj.execution.worker import run_release, run_releases
from runtraj.output import Output
from runtraj.releases import PointRelease, Release
from runtraj.simulation import Simulation
from runtraj.spatial import Grid
from runtraj.transport import ModelRun, TransportConfig
from runtraj.transport.hysplit import FailureReason, HysplitConfig, Met, MetConfig
from typer.testing import CliRunner

from stilt import Project
from stilt.footprint.stage import store_empty_footprint

from ..fixtures.factories import make_met_config, make_project_config, make_variant
from ..fixtures.footprints import run_footprint

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

GRID = Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)


@pytest.fixture
def release() -> Release:
    return PointRelease(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


@pytest.fixture
def other_release() -> Release:
    return PointRelease(
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


def _make_sim(release, met_config, params, output, *, footprint=None, name="hrrr"):
    return Simulation(
        release, _variant(params, met_config, name=name, footprint=footprint), output
    )


@pytest.fixture
def compute_root(tmp_path):
    return tmp_path / "compute"


@pytest.fixture
def sim(release, met_config, params, output) -> Simulation:
    """A trajectory-only simulation."""
    return _make_sim(release, met_config, params, output)


def _particles(release) -> pd.DataFrame:
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
    data["time"] = pd.Timestamp(release.time) + pd.to_timedelta(data["age"], unit="min")
    return data


def _write_particles(sim: Simulation) -> pd.DataFrame:
    """Put a small particle file for *sim* in the output directory."""
    particles = _particles(sim.release)
    sim.output.write_particles(sim.variant, sim.release, particles, [])
    return particles


def _write_footprint(sim: Simulation, *, empty: bool = False) -> None:
    """Record a footprint (or an empty one) for *sim* in the output directory."""
    assert sim.variant.stages.get("footprint") is not None
    if empty:
        store_empty_footprint(sim.output, sim.variant, sim.release)
    else:
        # The particles the worker gave the simulation, or a small table.
        run_footprint(sim, sim.__dict__.get("particles", _particles(sim.release)))


def _model_config(tmp_path, **kwargs) -> ProjectConfig:
    return make_project_config(tmp_path, **kwargs)


def _model(tmp_path, releases, **config_kwargs) -> Project:
    return Project.init(
        tmp_path / "proj",
        config=_model_config(tmp_path, **config_kwargs),
        releases=list(releases),
    )


# ---------------------------------------------------------------------------
# run_particles
# ---------------------------------------------------------------------------


def test_run_particles_starts_in_an_empty_directory(
    sim, met, compute_root, monkeypatch
):
    """A directory left by a stopped job is cleared before the model runs."""
    workdir = compute_root / sim.release.id / sim.variant.name
    workdir.mkdir(parents=True)
    (workdir / "PARTICLE_STILT.DAT").write_text("left over\n")
    seen: list[list[str]] = []

    class _Model:
        name = "hysplit"

        def run(self, release, params, met, window, workdir=None, timeout=None):
            seen.append(sorted(p.name for p in workdir.iterdir()))
            raise RuntimeError("stop here")

    monkeypatch.setattr("runtraj.transport.get_model", lambda name: _Model())
    with pytest.raises(RuntimeError):
        worker.run_particles(sim, workdir=workdir)
    assert seen == [[]]


def test_a_run_stopped_partway_leaves_its_start_in_the_log(
    sim, met, compute_root, monkeypatch
):
    """The log is written when the run starts, so a cut-off run reads as interrupted."""

    class _Model:
        name = "hysplit"

        def run(self, release, params, met, window, workdir=None, timeout=None):
            raise KeyboardInterrupt  # SIGTERM from Slurm, mid-run

    monkeypatch.setattr("runtraj.transport.get_model", lambda name: _Model())
    with pytest.raises(KeyboardInterrupt):
        worker.run_particles(sim, workdir=compute_root / "w")

    assert "runtraj started this run" in sim.log
    assert not sim.has_particles and sim.failure is None


def test_a_model_that_needs_no_workdir_gets_none(sim, met, compute_root, monkeypatch):
    seen: list = []

    class _Model:
        name = "hysplit"
        needs_workdir = False

        def run(self, release, params, met, window, workdir=None, timeout=None):
            seen.append(workdir)
            raise SimulationError("stop here", log="model log\n")

    monkeypatch.setattr("runtraj.transport.get_model", lambda name: _Model())
    monkeypatch.setattr(worker, "get_model", lambda name: _Model())
    workdir = compute_root / sim.release.id / sim.variant.name
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

        def run(self, release, params, met, window, workdir=None, timeout=None):
            raise MeteorologyError("No met file for 2023-01-01 11:00.")

    monkeypatch.setattr("runtraj.transport.get_model", lambda name: _Model())
    with pytest.raises(MeteorologyError):
        worker.run_particles(
            sim, workdir=compute_root / sim.release.id / sim.variant.name
        )

    kept = sim.output.kept_workdir(sim.variant, sim.release.id)
    assert kept is None or not kept.exists()
    assert not (compute_root / sim.release.id / sim.variant.name).exists()


def test_run_particles_without_particles_is_a_simulation_error(
    sim, met, compute_root, monkeypatch
):
    class _Model:
        name = "hysplit"

        def run(self, release, params, met, window, workdir=None, timeout=None):
            return ModelRun(particles=pd.DataFrame(), met_files=[])

    monkeypatch.setattr("runtraj.transport.get_model", lambda name: _Model())
    with pytest.raises(SimulationError) as caught:
        worker.run_particles(
            sim, workdir=compute_root / sim.release.id / sim.variant.name
        )
    assert caught.value.reason == "NO_PARTICLE_DATA"


# ---------------------------------------------------------------------------
# run_release: variants that share particles run as one group
# ---------------------------------------------------------------------------


def _fake_run_particles(monkeypatch, calls: list[str]):
    """Replace run_particles with one that writes particles and records the variant it ran for."""

    def fake(sim, *, workdir, keep_workdir=False, timeout=None):
        calls.append(sim.variant.name)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)


def _fake_make_footprint(monkeypatch, calls: list[tuple[str, int]]):
    """Replace the worker's run of a stage with one that records the variant and which particle table it read."""

    def fake(sim, name):
        assert name == "footprint"
        calls.append((sim.variant.name, id(sim.particles)))
        _write_footprint(sim)

    monkeypatch.setattr(worker, "run_stage", fake)


def _run_release(project, release, **kwargs):
    return run_release(
        project, str(release.id), compute_root=project.directory / "scratch", **kwargs
    )


SHARED = {"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}, "zi08": {"ziscale": 0.8}}


def test_each_run_gets_a_new_workdir_named_after_its_simulation(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release], variants={"hrrr": {"realizations": 2}})
    compute_root = tmp_path / "compute"
    workdirs: list = []

    def fake(sim, *, workdir, keep_workdir=False, timeout=None):
        workdirs.append(workdir)
        raise RuntimeError("stop here")

    monkeypatch.setattr(worker, "run_particles", fake)
    for _ in range(2):  # the same simulations again
        run_release(project, str(release.id), compute_root=compute_root)

    assert len(set(workdirs)) == 4
    assert {w.parent for w in workdirs} == {compute_root}
    names = sorted(w.name.rsplit("_", 1)[0] for w in workdirs)
    assert names == [f"{release.id}_hrrr_{k}" for k in (0, 0, 1, 1)]


def test_the_execution_settings_given_reach_the_hysplit_run(
    tmp_path, release, monkeypatch
):
    """An override passed in, not config.yaml's, sets the timeout and keep_workdir."""
    project = _model(tmp_path, [release])
    seen: dict = {}

    def fake(sim, **kwargs):
        seen.update(kwargs)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    override = ExecutionConfig(timeout=42, keep_workdir=True)
    _run_release(project, release, execution=override)

    assert (seen["timeout"], seen["keep_workdir"]) == (42, True)


def test_a_particles_only_variant_runs_hysplit_and_completes(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release])
    hysplit: list[str] = []
    _fake_run_particles(monkeypatch, hysplit)
    monkeypatch.setattr(
        worker, "run_stage", lambda *a, **k: pytest.fail("no footprint")
    )

    assert _run_release(project, release) == []
    assert hysplit == ["hrrr"]
    assert project.simulation(str(release.id), "hrrr").has_particles


def test_variants_that_share_particles_run_hysplit_once_and_reuse_the_table(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release], grid=GRID, variants=SHARED)
    hysplit: list[str] = []
    feet: list[tuple[str, int]] = []
    _fake_run_particles(monkeypatch, hysplit)
    _fake_make_footprint(monkeypatch, feet)

    assert _run_release(project, release) == []
    assert hysplit == ["hrrr", "zi08"]  # hrrr-s2 shares hrrr's particles
    assert [name for name, _ in feet] == ["hrrr", "hrrr-s2", "zi08"]
    # hrrr-s2's footprint is made from the table HYSPLIT just returned for hrrr.
    assert feet[0][1] == feet[1][1]


def test_existing_particles_are_read_once_for_every_footprint_of_the_group(
    tmp_path, release, monkeypatch
):
    project = _model(
        tmp_path,
        [release],
        grid=GRID,
        variants={"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}},
    )
    _write_particles(project.simulation(str(release.id), "hrrr"))
    feet: list[tuple[str, int]] = []
    _fake_make_footprint(monkeypatch, feet)
    monkeypatch.setattr(
        worker, "run_particles", lambda *a, **k: pytest.fail("no HYSPLIT")
    )

    _run_release(project, release)

    assert [name for name, _ in feet] == ["hrrr", "hrrr-s2"]
    assert feet[0][1] == feet[1][1]


def test_existing_footprints_are_kept_and_empty_ones_count_as_done(
    tmp_path, release, monkeypatch
):
    project = _model(
        tmp_path,
        [release],
        grid=GRID,
        variants={"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}},
    )
    hrrr = project.simulation(str(release.id), "hrrr")
    _write_particles(hrrr)
    _write_footprint(hrrr)
    _write_footprint(project.simulation(str(release.id), "hrrr-s2"), empty=True)
    monkeypatch.setattr(
        worker, "run_particles", lambda *a, **k: pytest.fail("no HYSPLIT")
    )
    monkeypatch.setattr(worker, "run_stage", lambda *a, **k: pytest.fail("kept"))

    assert _run_release(project, release) == []
    assert project.incomplete().empty


def test_without_skip_existing_hysplit_runs_once_per_group_and_every_footprint_is_remade(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release], grid=GRID, variants=SHARED)
    for name in SHARED:
        sim = project.simulation(str(release.id), name)
        _write_particles(sim)
        _write_footprint(sim)
    hysplit: list[str] = []
    feet: list[tuple[str, int]] = []
    _fake_run_particles(monkeypatch, hysplit)
    _fake_make_footprint(monkeypatch, feet)

    _run_release(project, release, skip_existing=False)

    assert hysplit == ["hrrr", "zi08"]
    assert [name for name, _ in feet] == ["hrrr", "hrrr-s2", "zi08"]


def test_rerun_particles_remake_a_footprint_that_already_existed(
    tmp_path, release, monkeypatch
):
    """Lost particles are rerun, and the old footprint is remade from the new ones."""
    project = _model(tmp_path, [release], grid=GRID)
    sim = project.simulation(str(release.id), "hrrr")
    _write_particles(sim)
    _write_footprint(sim)
    sim.particles_path.unlink()
    hysplit: list[str] = []
    feet: list[tuple[str, int]] = []
    _fake_run_particles(monkeypatch, hysplit)
    _fake_make_footprint(monkeypatch, feet)

    _run_release(project, release)

    assert hysplit == ["hrrr"] and [name for name, _ in feet] == ["hrrr"]


def test_run_release_takes_timeout_and_keep_workdir_from_execution(
    tmp_path, release, monkeypatch
):
    project = _model(
        tmp_path, [release], execution={"timeout": 120, "keep_workdir": True}
    )
    seen: list[dict] = []

    def fake(sim, **kwargs):
        seen.append(kwargs)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    _run_release(project, release)

    assert seen and all(k["timeout"] == 120 and k["keep_workdir"] for k in seen)


def test_run_release_stops_on_preemption_with_what_finished_written(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release], grid=GRID, variants=SHARED)
    hysplit: list[str] = []

    def fake(sim, **kwargs):
        if sim.variant.name == "zi08":
            raise KeyboardInterrupt
        hysplit.append(sim.variant.name)
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", fake)
    _fake_make_footprint(monkeypatch, [])

    with pytest.raises(KeyboardInterrupt):
        _run_release(project, release)

    st = project.status()
    assert dict(zip(st.variant, st.state, strict=True)) == {
        "hrrr": "complete",
        "hrrr-s2": "complete",
        "zi08": "pending",
    }


# ---------------------------------------------------------------------------
# run_release: failures are recorded with the simulation
# ---------------------------------------------------------------------------


def test_a_failed_hysplit_run_fails_its_whole_group_once_and_is_recorded(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release], grid=GRID, variants=SHARED)
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

    problems = _run_release(project, release)

    assert hysplit == ["hrrr", "zi08"]
    # One line for the group whose run failed.
    assert problems == [
        "hrrr failed during particles (MET_COVERAGE): HYSPLIT failed (MET_COVERAGE)."
    ]
    for name in ("hrrr", "hrrr-s2"):
        failure = project.simulation(str(release.id), name).failure
        assert failure is not None
        assert failure["stage"] == "particles"
        assert failure["reason"] == "MET_COVERAGE"
        assert failure["message"] == "HYSPLIT failed (MET_COVERAGE)."
        assert "traceback" not in failure
    assert project.simulation(str(release.id), "zi08").failure is None


def test_a_failed_footprint_is_the_variants_own_and_the_others_still_run(
    tmp_path, release, monkeypatch
):
    project = _model(
        tmp_path,
        [release],
        grid=GRID,
        variants={"hrrr": {}, "hrrr-s2": {"smooth_factor": 2.0}},
    )
    _fake_run_particles(monkeypatch, [])
    made: list[str] = []

    def fake(sim, name):
        if sim.variant.name == "hrrr":
            raise ValueError("bad grid")
        made.append(sim.variant.name)
        _write_footprint(sim)

    monkeypatch.setattr(worker, "run_stage", fake)

    problems = _run_release(project, release)

    assert problems == ["hrrr failed during footprint (ValueError): bad grid"]
    assert made == ["hrrr-s2"]
    failure = project.simulation(str(release.id), "hrrr").failure
    assert failure is not None
    assert failure["stage"] == "footprint" and failure["reason"] == "ValueError"
    assert failure["message"] == "bad grid"
    assert "ValueError: bad grid" in failure["traceback"]
    assert project.simulation(str(release.id), "hrrr-s2").failure is None
    # The record belongs to the footprint folder, found by its settings hash.
    sim = project.simulation(str(release.id), "hrrr")
    assert project.output.failure("footprint", sim.variant, sim.release.id) == failure


def test_a_success_clears_the_failure_it_replaces(tmp_path, release, monkeypatch):
    project = _model(tmp_path, [release], grid=GRID)
    sim = project.simulation(str(release.id), "hrrr")

    def fail(sim, **kwargs):
        raise MeteorologyError("No met file for 2023-01-01 11:00.")

    monkeypatch.setattr(worker, "run_particles", fail)
    _run_release(project, release)
    assert sim.failure is not None and sim.failure["reason"] == "MET_COVERAGE"
    assert project.output.failure("particles", sim.variant, sim.release.id)

    _fake_run_particles(monkeypatch, [])
    _fake_make_footprint(monkeypatch, [])
    _run_release(project, release)

    assert sim.is_complete and sim.failure is None
    assert project.output.failure("particles", sim.variant, sim.release.id) is None


# A connection or I/O error that a retry may get past is not a failure of the
# simulation: no record is written, so it stays unfinished and `stilt run`
# exits with 3 (runtraj._transient.is_transient sorts the errors).


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
    return dict(zip(project.status().release, project.status().state, strict=True))


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
    tmp_path, release, monkeypatch, error
):
    project = _model(tmp_path, [release])
    sim = project.simulation(str(release.id), "hrrr")
    monkeypatch.setattr(worker, "run_particles", _fails_with(error))

    problems = _run_release(project, release)

    assert len(problems) == 1
    assert problems[0].startswith("hrrr did not finish during particles")
    assert "a retry may finish it" in problems[0]
    assert sim.failure is None
    assert project.output.failure("particles", sim.variant, sim.release.id) is None
    assert _states(project) == {sim.release.id: "pending"}


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
    tmp_path, release, monkeypatch, error
):
    project = _model(tmp_path, [release])
    sim = project.simulation(str(release.id), "hrrr")
    monkeypatch.setattr(worker, "run_particles", _fails_with(error))

    problems = _run_release(project, release)

    assert problems[0].startswith("hrrr failed during particles")
    assert sim.failure is not None and sim.failure["stage"] == "particles"
    assert _states(project) == {sim.release.id: "failed"}


def test_a_simulation_error_is_a_failure_whatever_caused_it(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release])
    sim = project.simulation(str(release.id), "hrrr")

    def fail(sim, **kwargs):
        raise SimulationError("HYSPLIT failed.", reason="FORTRAN_RUNTIME_ERROR") from (
            ConnectionResetError("reset")
        )

    monkeypatch.setattr(worker, "run_particles", fail)
    _run_release(project, release)

    assert sim.failure is not None and sim.failure["reason"] == "FORTRAN_RUNTIME_ERROR"
    assert _states(project) == {sim.release.id: "failed"}


def test_a_connection_error_while_the_particles_are_written_is_interrupted(
    tmp_path, release, monkeypatch
):
    """The run had started and its log says so; there is no particle file and no record."""
    project = _model(tmp_path, [release])
    sim = project.simulation(str(release.id), "hrrr")

    class _Model:
        name = "hysplit"

        def run(self, release, params, met, window, workdir=None, timeout=None):
            return ModelRun(particles=_particles(release), met_files=[])

    monkeypatch.setattr("runtraj.transport.get_model", lambda name: _Model())
    monkeypatch.setattr(
        Output, "write_particles", _fails_with(ConnectionError("bucket down"))
    )
    problems = _run_release(project, release)

    assert "did not finish during particles" in problems[0]
    assert not sim.has_particles and sim.failure is None
    assert _states(project) == {sim.release.id: "interrupted"}


def test_a_connection_error_in_a_footprint_leaves_the_particles_and_no_record(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release], grid=GRID)
    sim = project.simulation(str(release.id), "hrrr")
    _fake_run_particles(monkeypatch, [])
    monkeypatch.setattr(
        worker, "run_stage", _fails_with(OSError(errno.ESTALE, "stale"))
    )

    problems = _run_release(project, release)

    assert problems[0].startswith("hrrr did not finish during footprint")
    assert sim.has_particles and not sim.has("footprint") and sim.failure is None
    assert _states(project) == {sim.release.id: "pending"}

    # The next run keeps the particles and makes the footprint.
    ran: list[str] = []
    _fake_run_particles(monkeypatch, ran)
    _fake_make_footprint(monkeypatch, [])
    assert _run_release(project, release) == []
    assert ran == [] and sim.is_complete


def test_a_rerun_after_a_connection_error_finishes_the_simulation(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release])
    sim = project.simulation(str(release.id), "hrrr")
    monkeypatch.setattr(worker, "run_particles", _fails_with(ConnectionError("down")))
    _run_release(project, release)
    assert not sim.is_complete

    _fake_run_particles(monkeypatch, [])
    assert _run_release(project, release) == []

    assert sim.is_complete and sim.failure is None
    assert _states(project) == {sim.release.id: "complete"}


def test_a_connection_error_removes_the_failure_record_of_an_earlier_attempt(
    tmp_path, release, monkeypatch
):
    """The earlier attempt failed for a reason that is mended; this one did not finish, so it is not "failed"."""
    project = _model(tmp_path, [release])
    sim = project.simulation(str(release.id), "hrrr")
    monkeypatch.setattr(
        worker, "run_particles", _fails_with(MeteorologyError("No met file."))
    )
    _run_release(project, release)
    assert sim.failure is not None

    monkeypatch.setattr(worker, "run_particles", _fails_with(ConnectionError("down")))
    _run_release(project, release)

    assert sim.failure is None
    assert _states(project) == {sim.release.id: "pending"}


def test_a_mixed_run_is_3_until_the_connection_error_is_mended_and_then_1(
    tmp_path, release, other_release, monkeypatch
):
    """The retry runs only what did not finish; the real failure stays, and ends the run with 1."""
    project = _model(tmp_path, [release, other_release])
    transient, real = str(release.id), str(other_release.id)

    def first(sim, **kwargs):
        if sim.release.id == transient:
            raise ConnectionError("down")
        raise MeteorologyError("No met file.")

    monkeypatch.setattr(worker, "run_particles", first)
    run_releases(
        project,
        [transient, real],
        compute_root=tmp_path / "compute",
        execution=ExecutionConfig(local={"cpus": 1}),
    )
    assert _states(project) == {transient: "pending", real: "failed"}
    assert _exit_code(project.status()) == 3

    def second(sim, **kwargs):
        if sim.release.id == real:
            raise MeteorologyError("No met file.")
        return _write_particles(sim)

    monkeypatch.setattr(worker, "run_particles", second)
    run_releases(
        project,
        [transient, real],
        compute_root=tmp_path / "compute",
        execution=ExecutionConfig(local={"cpus": 1}),
    )
    assert _states(project) == {transient: "complete", real: "failed"}
    assert _exit_code(project.status()) == 1


def test_stilt_run_exits_3_after_a_connection_error_in_a_simulation_and_0_on_the_rerun(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release])
    (tmp_path / "met").mkdir(exist_ok=True)
    runner = CliRunner()
    monkeypatch.setattr(worker, "run_particles", _fails_with(ConnectionError("down")))

    down = runner.invoke(app, ["run", str(project.directory)])
    assert down.exit_code == 3, down.output
    assert "pending=1" in down.output and "failed=0" in down.output

    _fake_run_particles(monkeypatch, [])
    up = runner.invoke(app, ["run", str(project.directory)])
    assert up.exit_code == 0, up.output


def test_a_failed_run_keeps_its_log_and_working_directory(
    tmp_path, release, monkeypatch
):
    project = _model(tmp_path, [release])
    sim = project.simulation(str(release.id), "hrrr")

    class _Model:
        name = "hysplit"

        def run(self, release, params, met, window, workdir=None, timeout=None):
            (workdir / "CONTROL").write_text("...")
            raise SimulationError(
                "HYSPLIT failed (FORTRAN_RUNTIME_ERROR).",
                reason="FORTRAN_RUNTIME_ERROR",
                log="hycs_std said something\n",
            )

    monkeypatch.setattr("runtraj.transport.get_model", lambda name: _Model())
    _run_release(project, release)

    failure = sim.failure
    assert failure is not None and failure["reason"] == "FORTRAN_RUNTIME_ERROR"
    assert set(failure) == {"stage", "reason", "message", "time"}
    assert "hycs_std said" in sim.log
    assert sim.kept_workdir is not None
    assert (sim.kept_workdir / "CONTROL").exists()


# ---------------------------------------------------------------------------
# run_releases: inline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# run_releases: process pool (plumbing only, with a synchronous fake executor)
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


def test_run_releases_pool_worker_error_stops_the_workers_and_is_not_masked(
    tmp_path, release, other_release, monkeypatch, fake_pool
):
    """
    A worker's own error must reach the caller, with the workers stopped (#320).

    An error that escapes a release's own handling (anything outside
    ``run_release``'s particles and footprint ``try`` blocks) is re-raised
    by the future's ``result()``.
    """
    model = _model(tmp_path, [release, other_release])
    ids = list(model.releases["release"])

    def fail(*args, **kwargs):
        raise ConnectionError("output bucket unreachable")

    monkeypatch.setattr(worker, "run_release", fail)

    with pytest.raises(ConnectionError, match="output bucket unreachable"):
        run_releases(
            model,
            ids,
            compute_root=tmp_path / "scratch",
            execution=ExecutionConfig(local={"cpus": 2}),
        )

    [pool] = fake_pool.instances
    assert pool.shutdowns == [{"wait": False, "cancel_futures": True}]


# ---------------------------------------------------------------------------
# A batched transport model
# ---------------------------------------------------------------------------


class BatchConfig(TransportConfig):
    """The batched toy model's config: the shared parameters only."""


class BatchedToy:
    """A toy model that runs many releases per call and needs no workdir, as an emulator would."""

    name = "toy"
    config_class = BatchConfig
    met_config_class = MetConfig
    batched = True
    needs_workdir = False
    calls: ClassVar[list[list[str]]] = []
    skip: ClassVar[set[str]] = set()  # releases it returns no rows for

    def sha256(self, config):
        return None

    def version(self, config):
        return "1.0"

    def data_files(self, config):
        return None

    def run(self, release, config, met, window, workdir=None, timeout=None):
        raise AssertionError("a batched model is run with run_many")

    def run_many(self, releases, config, met, windows, workdir=None, timeout=None):
        assert workdir is None
        BatchedToy.calls.append([str(r.id) for r in releases])
        tables = [
            pd.DataFrame(
                {
                    "release": str(r.id),
                    "particle": [1, 2, 1, 2],
                    "age": [0, 0, config.n_hours * 60, config.n_hours * 60],
                    "lon": r.longitude,
                    "lat": r.latitude,
                    "zagl": r.altitude,
                    "foot": [0.0, 0.0, 0.01, 0.02],
                }
            )
            for r in releases
            if str(r.id) not in self.skip
        ]
        return ModelRun(particles=pd.concat(tables), log="toy ran\n")


BATCHED_TOY = f"{__name__}.BatchedToy"


@pytest.fixture
def batched_toy():
    BatchedToy.calls = []
    BatchedToy.skip = set()
    return BatchedToy


def test_a_batched_model_runs_every_release_of_a_variant_in_one_call(
    tmp_path, release, other_release, batched_toy
):
    third = PointRelease(
        time=dt.datetime(2023, 1, 1, 14),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    project = _model(
        tmp_path,
        [release, other_release, third],
        grid=GRID,
        hnf_plume=False,
        models={"toy": {"class": BATCHED_TOY, "n_hours": -24}},
        variants={"toy": {}, "toy-s2": {"smooth_factor": 2.0}},
    )
    ids = [str(r.id) for r in (release, other_release, third)]
    batched_toy.skip = {ids[2]}

    run_releases(project, ids, compute_root=tmp_path / "scratch")

    # One call for the variants that share particles, all three releases in it.
    assert batched_toy.calls == [ids]
    for rid in ids[:2]:
        for variant in ("toy", "toy-s2"):
            sim = project.simulation(rid, variant)
            assert sim.is_complete, (rid, variant)
        assert project.simulation(rid, "toy").log == "toy ran\n"
    # The release the model gave no rows fails alone.
    failed = project.simulation(ids[2], "toy")
    assert not failed.has_particles
    assert failed.failure["reason"] == "NO_PARTICLE_DATA"
    # No workdir was made for a model that does not ask for one.
    assert not (tmp_path / "scratch").exists()

    # Run again: nothing is missing, so the model is not called.
    batched_toy.skip = set()
    run_releases(project, ids[:2], compute_root=tmp_path / "scratch")
    assert batched_toy.calls == [ids]
