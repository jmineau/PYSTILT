"""Tests for stilt.cli - Typer command-line interface."""

import os
import subprocess
import sys
from dataclasses import replace

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import stilt.__main__
from stilt.cli import _error_exit_code, _resolve_project, app
from stilt.config import ProjectConfig
from stilt.execution.config import ExecutionConfig
from stilt.project import Project
from stilt.spatial import Grid

from .fixtures.factories import make_met_config, make_project_config

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path):
    """Write a minimal config.yaml + receptors.csv so _resolve_project succeeds."""
    cfg = make_project_config(
        tmp_path,
        # Local, with the time limit a Slurm run of it needs.
        execution={"time": "01:00:00"},
    )
    cfg.to_yaml(tmp_path / "config.yaml")
    (tmp_path / "receptors.csv").write_text(
        "time,longitude,latitude,altitude\n2023-01-01 12:00:00,-111.85,40.77,5.0\n"
    )


class _Calls(list):
    """What `stilt run` asked of the project; ``states`` is the status the fake run returns."""

    states: list[str] = []


@pytest.fixture
def calls(monkeypatch):
    """Record what `stilt run` asks of the project, and run nothing."""
    recorded = _Calls()

    def run(self, **kwargs):
        recorded.append(("run", kwargs))
        n = len(recorded.states)
        return pd.DataFrame(
            {
                "receptor": [f"r{i}" for i in range(n)],
                "variant": ["hrrr"] * n,
                "state": recorded.states,
                "reason": [
                    "met_missing" if s == "failed" else None for s in recorded.states
                ],
            }
        )

    def submit(self, **kwargs):
        recorded.append(("submit", kwargs))
        return "12345"

    monkeypatch.setattr("stilt.cli.Project.run", run)
    monkeypatch.setattr("stilt.cli.Project.submit", submit)
    return recorded


def test_python_m_entrypoint_invokes_cli(monkeypatch):
    called = False

    def fake_app() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(stilt.__main__, "app", fake_app)

    stilt.__main__.main()

    assert called is True


# ---------------------------------------------------------------------------
# _resolve_project helper
# ---------------------------------------------------------------------------


def test_resolve_project_exits_when_no_config_yaml(tmp_path):
    """Exits with code 2 when no config.yaml is found: the command line named no project."""
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 2


def test_resolve_project_returns_path_when_config_exists(tmp_path):
    """Returns the resolved path when config.yaml is present."""
    (tmp_path / "config.yaml").write_text("n_hours: -24\n")
    resolved = _resolve_project(tmp_path)
    assert resolved == str(tmp_path.resolve())


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def test_status_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 2


def test_status_prints_project_info(tmp_path):
    """``status`` prints one summary line with the project root and counts."""
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 0
    assert (
        f"Project: {tmp_path.resolve()}  total=1  complete=0  failed=0  "
        "interrupted=0  pending=1" in result.output
    )


def test_status_counts_full_simulation_completion(tmp_path):
    """A complete trajectory without required footprints is not done yet."""
    from stilt.project import Project
    from stilt.receptors import PointReceptor

    cfg = make_project_config(
        tmp_path,
        grid=Grid(
            xmin=-114.0,
            xmax=-113.0,
            ymin=39.0,
            ymax=40.0,
            xres=0.1,
            yres=0.1,
        ),
        variants={"hrrr": {}},
    )

    receptor = PointReceptor(
        time="2023-01-01 12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    project = Project.init(tmp_path, config=cfg, receptors=[receptor])

    # Particles exist but the required footprint does not: not complete.
    sim = project.simulation(receptor.id, "hrrr")
    particles = pd.DataFrame(
        {
            "age": [-60.0],
            "particle": [1.0],
            "lon": [-113.5],
            "lat": [39.5],
            "zagl": [10.0],
            "foot": [1e-5],
        }
    )
    sim.output.write_particles(sim.variant, receptor, particles, [])

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=1  complete=0  failed=0  interrupted=0  pending=1" in result.output

    # Once the footprint is present too, the simulation counts as complete.
    from stilt.execution.worker import make_footprint

    sim = project.simulation(receptor.id, "hrrr")  # a fresh value
    make_footprint(sim, sim.particles)

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=1  complete=1  failed=0  interrupted=0  pending=0" in result.output


def test_status_counts_each_state_per_variant(tmp_path, monkeypatch):
    """Failed simulations are counted as failed, not pending (#192)."""
    make_project_config(
        tmp_path, variants={"hrrr": {}, "fine": {"numpar": 50}}
    ).to_yaml(tmp_path / "config.yaml")
    (tmp_path / "receptors.csv").write_text("time,longitude,latitude,altitude\n")
    table = pd.DataFrame(
        {
            "variant": ["hrrr", "hrrr", "hrrr", "fine", "fine", "fine"],
            "state": ["complete", "failed", "pending"] + ["interrupted"] * 3,
            "reason": [None, "MET_COVERAGE", None, None, None, None],
        }
    )
    monkeypatch.setattr(Project, "status", lambda self, sel=None: table)
    monkeypatch.setattr("stilt.cli._print_folders", lambda project: None)

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "total=6  complete=1  failed=1  interrupted=3  pending=1" in result.output
    assert (
        "  hrrr: total=3  complete=1  failed=1  interrupted=0  pending=1"
        in result.output
    )
    assert (
        "  fine: total=3  complete=0  failed=0  interrupted=3  pending=0"
        in result.output
    )


def test_cli_help_lists_current_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    expected = {
        "check",
        "init",
        "run",
        "status",
        "submit",
    }
    for command in expected:
        assert command in result.output

    registered = {
        cmd.name or cmd.callback.__name__.replace("_", "-")  # type: ignore[union-attr]
        for cmd in app.registered_commands
    }
    assert registered == expected


# ---------------------------------------------------------------------------
# check command
# ---------------------------------------------------------------------------


def _write_checkable_config(tmp_path, n_receptors=1, files=True):
    """Write a project of hourly receptors from noon, and met files for them unless ``files`` is false."""
    import datetime as dt

    from .fixtures.factories import make_met_files

    _write_minimal_config(tmp_path)
    times = [
        dt.datetime(2023, 1, 1, 12) + dt.timedelta(hours=h) for h in range(n_receptors)
    ]
    (tmp_path / "receptors.csv").write_text(
        "time,longitude,latitude,altitude\n"
        + "".join(f"{t:%Y-%m-%d %H:%M:%S},-111.85,40.77,5.0\n" for t in times)
    )
    if files:
        for time in times:
            make_met_files(tmp_path / "met", time, -24)


def test_check_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 2


def test_check_finding_nothing_exits_0(tmp_path):
    _write_checkable_config(tmp_path)

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f"Project: {tmp_path.resolve()}  receptors=1" in result.output
    assert "Variants: hrrr (met hrrr, particles only)" in result.output
    assert "No problems found." in result.output


def test_check_lists_what_would_stop_a_run_and_exits_1(tmp_path, calls):
    _write_checkable_config(tmp_path, files=False)

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 1, result.output
    lines = result.output.splitlines()
    header = next(line for line in lines if line.startswith("check"))
    assert header.split() == ["check", "variant", "receptor", "message"]
    row = next(line for line in lines if line.startswith("met_files"))
    assert "202301011200_-111.85_40.77_5" in row
    assert "no met file for 2022-12-31 12:00" in row
    assert "Found 1 problem: met_files 1" in result.output
    assert calls == []  # nothing ran


def test_check_shows_the_first_rows_of_each_check_and_counts_the_rest(tmp_path):
    _write_checkable_config(tmp_path, n_receptors=12, files=False)

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 1, result.output
    shown = [line for line in result.output.splitlines() if "no met file" in line]
    assert len(shown) == 10
    assert "... 2 more (project.check() lists them all)" in result.output
    assert "Found 12 problems: met_files 12" in result.output


def test_check_exits_2_when_the_project_files_are_invalid(tmp_path):
    _write_checkable_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(config.read_text() + "nonsense: 1\n")

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 2
    assert "nonsense" in result.output


def _point_met_at(tmp_path, directory):
    """Point the project's met at *directory*, as an unedited `stilt init` project's placeholder does."""
    config = tmp_path / "config.yaml"
    text = config.read_text()
    assert str(tmp_path / "met") in text
    config.write_text(text.replace(str(tmp_path / "met"), str(directory)))


@pytest.mark.parametrize(
    "command", [["run"], ["submit"], ["status"], ["status", "--json"]]
)
def test_a_met_directory_that_does_not_exist_exits_2_in_one_line(
    tmp_path, calls, command
):
    """The line `stilt check` lists, not a traceback, and 2: nothing ran (#280)."""
    _write_minimal_config(tmp_path)
    missing = tmp_path / "no-such-met"
    _point_met_at(tmp_path, missing)

    result = runner.invoke(app, [*command, str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert result.output == f"Error: Met 'hrrr': {missing} does not exist.\n"
    assert calls == []
    checked = runner.invoke(app, ["check", str(tmp_path)])
    assert checked.exit_code == 1
    assert f"{missing} does not exist." in checked.output


def test_a_met_directory_with_no_arl_file_exits_2_in_one_line(tmp_path, calls):
    _write_minimal_config(tmp_path)
    empty = tmp_path / "empty-met"
    empty.mkdir()
    (empty / "README").write_text("not a met file")
    _point_met_at(tmp_path, empty)

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert result.output.startswith(f"Error: Met 'hrrr': No ARL file under {empty}.")
    assert len(result.output.splitlines()) == 1
    assert calls == []


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


def test_run_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 2


def test_run_runs_the_project_with_its_execution_settings(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    [(verb, kwargs)] = calls
    assert verb == "run"
    assert kwargs["execution"] == ExecutionConfig(time="01:00:00")  # config.yaml's
    assert kwargs["skip_existing"] is True
    # Resolved once, so the run uses the directory the banner shows.
    from stilt.execution.runner import resolve_compute_root
    from stilt.project import Project

    scratch = resolve_compute_root(Project(tmp_path))
    assert kwargs["compute_root"] == str(scratch)
    assert f"Compute root: {scratch}" in result.output


def test_run_prints_a_startup_summary_and_the_status(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0
    assert (
        f"Starting run: project={tmp_path.resolve()}  backend=local  "
        "tasks=1  cpus=1  skip=existing"
    ) in result.output
    assert "Compute root:" in result.output
    assert f"Output: {tmp_path.resolve() / 'output'}" in result.output
    assert "Receptors loaded: 1" in result.output
    assert "Running here, one line per receptor" in result.output
    assert f"Project: {tmp_path.resolve()}  total=" in result.output


def test_run_forwards_compute_root_and_shows_it(tmp_path, calls):
    _write_minimal_config(tmp_path)
    compute_root = tmp_path / "scratch"

    result = runner.invoke(
        app, ["run", str(tmp_path), "--compute-root", str(compute_root)]
    )

    assert result.exit_code == 0
    assert f"Compute root: {compute_root.resolve()}" in result.output
    assert calls[0][1]["compute_root"] == str(compute_root)


def test_run_no_skip_passes_false(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path), "--no-skip"])

    assert result.exit_code == 0
    assert calls[0][1]["skip_existing"] is False
    assert "skip=no-skip" in result.output


def test_run_options_override_the_execution_settings(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(
        app, ["run", str(tmp_path), "--backend", "local", "--cpus", "4"]
    )

    assert result.exit_code == 0
    execution = calls[0][1]["execution"]
    assert (execution.backend, execution.cpus) == ("local", 4)
    assert "tasks=1  cpus=4" in result.output


def test_run_on_slurm_waits_for_the_job(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(
        app, ["run", str(tmp_path), "--backend", "slurm", "--n-workers", "2"]
    )

    assert result.exit_code == 0, result.output
    assert [verb for verb, _ in calls] == ["run"]
    assert calls[0][1]["execution"].n_workers == 2
    assert "Submitting to Slurm and waiting for the job to finish" in result.output


def test_run_writes_to_the_output_given(tmp_path, calls, monkeypatch):
    _write_minimal_config(tmp_path)
    opened = []
    real = Project.__init__

    def record(self, path, *, output=None):
        opened.append(output)
        real(self, path, output=output)

    monkeypatch.setattr(Project, "__init__", record)

    elsewhere = str(tmp_path / "elsewhere")
    result = runner.invoke(app, ["run", str(tmp_path), "--output", elsewhere])

    assert result.exit_code == 0, result.output
    assert elsewhere in opened
    assert f"Output: {elsewhere}" in result.output


def test_run_reads_execution_settings_from_a_file(tmp_path, calls):
    _write_minimal_config(tmp_path)
    settings = tmp_path / "execution.yaml"
    settings.write_text("backend: slurm\ncpus: 3\ntimeout: 900\nkeep_workdir: true\n")

    result = runner.invoke(
        app,
        [
            "run",
            str(tmp_path),
            "--execution",
            str(settings),
            "--task",
            "0/2",
            "--cpus",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    execution = calls[0][1]["execution"]
    # The file's settings, then the options, and a task runs here.
    assert (execution.backend, execution.cpus, execution.timeout) == ("local", 2, 900)
    assert execution.keep_workdir is True


@pytest.mark.parametrize(
    ("states", "code"),
    [
        ([], 0),
        (["complete", "complete"], 0),
        (["complete", "failed"], 1),
        (["failed", "pending"], 3),
        (["complete", "interrupted"], 3),
    ],
)
def test_run_exit_code_says_how_the_run_ended(tmp_path, calls, states, code):
    _write_minimal_config(tmp_path)
    calls.states = states

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == code, result.output


# ---------------------------------------------------------------------------
# An error the command did not expect: 3 to retry, 4 not to (#310)
# ---------------------------------------------------------------------------


def _error_named(name, *bases):
    """Return an exception of a class called *name*, as a cloud library would raise it."""
    return type(name, bases or (Exception,), {})("it failed")


def _client_error(status, code="Error"):
    """Return what botocore raises for a response with this status and error code."""
    error = _error_named("ClientError")
    error.response = {
        "ResponseMetadata": {"HTTPStatusCode": status},
        "Error": {"Code": code},
    }
    return error


def _with_status(attribute, status):
    """Return an error that carries an HTTP status as aiohttp, requests or gcsfs set it."""
    error = RuntimeError("the service answered")
    setattr(error, attribute, status)
    return error


@pytest.mark.parametrize(
    ("error", "code"),
    [
        # Connection and I/O errors: run it again.
        (ConnectionRefusedError("refused"), 3),
        (TimeoutError("slow"), 3),
        (OSError(5, "Input/output error"), 3),
        (OSError(116, "Stale file handle"), 3),
        # Cloud libraries' errors that are not OSError, matched by class name.
        (_error_named("EndpointConnectionError"), 3),
        (_error_named("ConnectTimeoutError"), 3),
        (_error_named("ReadTimeoutError"), 3),
        # aiohttp's ServerDisconnectedError is a ClientConnectionError.
        (
            _error_named(
                "ServerDisconnectedError",
                type("ClientConnectionError", (Exception,), {}),
            ),
            3,
        ),
        (_error_named("ClientConnectionError"), 3),
        (_client_error(503, "SlowDown"), 3),
        (_client_error(500, "InternalError"), 3),
        (_client_error(400, "ThrottlingException"), 3),
        (_client_error(429), 3),
        # The status of a busy service, wherever the library keeps it.
        (_with_status("status", 503), 3),
        (_with_status("status_code", 429), 3),
        (_with_status("code", 502), 3),
        (_with_status("status", 404), 4),
        (_with_status("code", "not a status"), 4),
        # Not transient: the path or the credentials are wrong, or it is a bug.
        (_client_error(403, "AccessDenied"), 4),
        (_client_error(404, "NoSuchKey"), 4),
        (_client_error(None, "NoSuchKey"), 4),
        (FileNotFoundError("no such file"), 4),
        (PermissionError("denied"), 4),
        (IsADirectoryError("a directory"), 4),
        (RuntimeError("a bug"), 4),
        (KeyError("k"), 4),
        (ValueError("bad"), 4),
        (_error_named("SSLError"), 4),
    ],
)
def test_error_exit_code_sorts_retry_from_bug(error, code):
    assert _error_exit_code(error) == code


def test_a_cloud_error_is_sorted_with_the_libraries_not_imported():
    """A botocore error is 3 and a denied one 4, as for the stand-ins above."""
    botocore = pytest.importorskip("botocore.exceptions")
    refused = botocore.EndpointConnectionError(endpoint_url="http://127.0.0.1:9")
    denied = botocore.ClientError(
        {
            "Error": {"Code": "AccessDenied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "ListObjectsV2",
    )
    assert _error_exit_code(refused) == 3
    assert _error_exit_code(botocore.ReadTimeoutError(endpoint_url="u")) == 3
    assert _error_exit_code(denied) == 4


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (_error_named("EndpointConnectionError"), 3),
        (ConnectionResetError("reset"), 3),
        (OSError(5, "I/O error"), 3),
        (RuntimeError("a bug"), 4),
        (KeyError("k"), 4),
        (FileNotFoundError("no such file"), 4),
    ],
)
def test_run_exits_3_for_a_connection_error_and_4_for_any_other(
    tmp_path, monkeypatch, error, code
):
    """An error that escapes the run is never 1, the code of "some simulations failed"."""
    _write_minimal_config(tmp_path)

    def run(self, **kwargs):
        raise error

    monkeypatch.setattr("stilt.cli.Project.run", run)
    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == code, result.output
    assert str(error) in result.output
    assert f"stilt exits with {code}" in result.output
    assert "Traceback (most recent call last)" in result.output


def test_run_says_where_an_unexpected_error_happened(tmp_path, monkeypatch):
    """The traceback is kept, so a bug can be found."""
    _write_minimal_config(tmp_path)

    def run(self, **kwargs):
        raise RuntimeError("a bug somewhere")

    monkeypatch.setattr("stilt.cli.Project.run", run)
    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 4
    assert "in run" in result.output
    assert "RuntimeError: a bug somewhere" in result.output


def test_run_exits_3_when_ctrl_c_arrives_outside_the_workers(tmp_path, monkeypatch):
    """Ctrl-C while waiting for a Slurm job was "Aborted!" with exit 1."""
    _write_minimal_config(tmp_path)

    def run(self, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("stilt.cli.Project.run", run)
    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 3, result.output


def test_a_failed_simulation_is_still_1_and_a_wrong_command_line_2(tmp_path, calls):
    _write_minimal_config(tmp_path)
    calls.states = ["complete", "failed"]
    assert runner.invoke(app, ["run", str(tmp_path)]).exit_code == 1
    assert runner.invoke(app, ["run", str(tmp_path), "--task", "3/3"]).exit_code == 2
    assert runner.invoke(app, ["run", str(tmp_path / "no_project")]).exit_code == 2
    receptors = tmp_path / "missing.txt"
    assert (
        runner.invoke(app, ["run", str(tmp_path), "--receptors", str(receptors)])
    ).exit_code == 2


def test_a_value_error_of_the_run_is_still_2(tmp_path, monkeypatch):
    """The run's own refusals (a receptor that is not in the project) are the command line's."""
    _write_minimal_config(tmp_path)

    def run(self, **kwargs):
        raise ValueError("no such receptor")

    monkeypatch.setattr("stilt.cli.Project.run", run)
    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 2
    assert "Error: no such receptor" in result.output


@pytest.mark.parametrize(
    ("command", "method"),
    [("submit", "submit"), ("status", "status"), ("check", "check")],
)
def test_the_other_commands_exit_4_on_an_unexpected_error(
    tmp_path, monkeypatch, command, method
):
    """Python's own 1 for `status` and `submit` was not a code anyone defined; `check`'s 1 is "found problems"."""
    _write_minimal_config(tmp_path)

    def broken(self, *args, **kwargs):
        raise RuntimeError("a bug somewhere")

    monkeypatch.setattr(f"stilt.cli.Project.{method}", broken)
    result = runner.invoke(app, [command, str(tmp_path)])

    assert result.exit_code == 4, result.output


def test_status_exits_3_on_a_connection_error(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)

    def broken(self, *args, **kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr("stilt.cli.Project.status", broken)

    assert runner.invoke(app, ["status", str(tmp_path)]).exit_code == 3


def test_an_unreachable_object_store_exits_3_not_1(tmp_path):
    """The reproduction of #310: `stilt run` with an output bucket nothing answers."""
    pytest.importorskip("s3fs")
    _write_minimal_config(tmp_path)
    env = {
        **os.environ,
        "AWS_ENDPOINT_URL": "http://127.0.0.1:9",
        "AWS_ACCESS_KEY_ID": "x",
        "AWS_SECRET_ACCESS_KEY": "x",
        "AWS_DEFAULT_REGION": "us-east-1",
    }
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "stilt",
            "run",
            str(tmp_path),
            "--output",
            "s3://stilt/output",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    assert done.returncode == 3, done.stderr
    assert "stilt exits with 3" in done.stderr


@pytest.mark.parametrize("args", [["run", "--bogus-flag"], ["nosuchcmd"]])
def test_a_wrong_command_line_exits_2_not_interrupted(tmp_path, calls, args):
    """Click exits 2 on a typo, so 2 means a wrong command line and never "retry"."""
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, [*args, str(tmp_path)])

    assert result.exit_code == 2
    assert calls == []


def test_a_task_takes_its_receptors_from_the_submissions_rows(tmp_path, monkeypatch):
    """A task reads receptors.parquet, not all of receptors.csv (#191)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _write_minimal_config(tmp_path)
    project = Project(tmp_path)
    rows = project._rows
    pq.write_table(
        pa.Table.from_pandas(rows, preserve_index=False), tmp_path / "r.parquet"
    )
    (tmp_path / "receptors.csv").write_text("not a receptor table\n")
    seen = []

    def run(self, **kwargs):
        seen.append((list(self.receptors["receptor"]), kwargs["receptors"]))
        return pd.DataFrame({"receptor": [], "variant": [], "state": [], "reason": []})

    monkeypatch.setattr("stilt.cli.Project.run", run)
    result = runner.invoke(
        app,
        [
            "run",
            str(tmp_path),
            "--receptors",
            str(tmp_path / "r.parquet"),
            "--task",
            "0/1",
        ],
    )

    assert result.exit_code == 0, result.output
    ids = list(rows["receptor"])
    assert seen == [(ids, ids)]


def test_run_task_runs_its_share_here(tmp_path, calls):
    _write_minimal_config(tmp_path)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    config["execution"] = {"backend": "slurm", "n_workers": 8, "time": "01:00:00"}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    calls.states = ["complete", "failed"]

    result = runner.invoke(app, ["run", str(tmp_path), "--task", "1/3"])

    assert result.exit_code == 1, result.output
    [(verb, kwargs)] = calls
    assert kwargs["task"] == (1, 3)
    assert kwargs["receptors"] is None
    assert kwargs["execution"].backend == "local"
    # The banner names the task, not the local backend the task runs on.
    assert "task 1 of 3  cpus=1" in result.output
    assert "backend=" not in result.output
    assert "Receptors of this task: 1, 4, 7, ..." in result.output
    # A task sums up what it ran, not the whole project.
    assert (
        "This run:  total=2  complete=1  failed=1  interrupted=0  pending=0"
        in result.output
    )
    assert "failed: met_missing 1" in result.output
    assert "Project:" not in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--task", "3/3"], "0 <= I < N"),
        (["--task", "two/3"], "takes I/N"),
        (["--task", "0/2", "--backend", "slurm"], "drop --backend slurm"),
        (["--receptors", "missing.txt"], "cannot read"),
    ],
)
def test_run_task_and_receptors_refuse_bad_values(tmp_path, calls, args, message):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path), *args])

    assert result.exit_code == 2
    assert message in result.output
    assert calls == []


def test_run_receptors_reads_ids_from_a_file(tmp_path, calls):
    _write_minimal_config(tmp_path)
    listed = tmp_path / "ids.txt"
    listed.write_text("# failed last time\nr1\n\n  r2  \n")

    result = runner.invoke(app, ["run", str(tmp_path), "--receptors", str(listed)])

    assert result.exit_code == 0, result.output
    assert calls[0][1]["receptors"] == ["r1", "r2"]
    assert "Receptors listed: 2" in result.output


def test_run_reports_receptors_not_in_the_project(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    listed = tmp_path / "ids.txt"
    listed.write_text("nope\n")

    result = runner.invoke(app, ["run", str(tmp_path), "--receptors", str(listed)])

    assert result.exit_code == 2
    assert "not in this project" in result.output


def test_submit_submits_to_slurm_and_returns(tmp_path, calls):
    _write_minimal_config(tmp_path)  # its backend is local

    result = runner.invoke(app, ["submit", str(tmp_path), "--n-workers", "2"])

    assert result.exit_code == 0, result.output
    assert [verb for verb, _ in calls] == ["submit"]
    execution = calls[0][1]["execution"]
    assert (execution.backend, execution.n_workers) == ("slurm", 2)
    assert "Submitting to Slurm and returning" in result.output
    assert "Submitted job: 12345" in result.output


def test_submit_without_a_time_limit_says_so(tmp_path, calls):
    _write_minimal_config(tmp_path)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    del config["execution"]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    result = runner.invoke(app, ["submit", str(tmp_path)])

    assert result.exit_code == 2
    assert "execution.time is required with backend: slurm" in result.output
    assert "Traceback" not in result.output
    assert calls == []


def test_submit_receptors_reads_ids_from_a_file(tmp_path, calls):
    _write_minimal_config(tmp_path)
    listed = tmp_path / "ids.txt"
    listed.write_text("r1\nr2\n")

    result = runner.invoke(app, ["submit", str(tmp_path), "--receptors", str(listed)])

    assert result.exit_code == 0, result.output
    assert calls[0][1]["receptors"] == ["r1", "r2"]
    assert "Receptors listed: 2" in result.output


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


def test_init_creates_config_yaml(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0
    assert (project / "config.yaml").exists()


def test_init_creates_receptors_csv(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0
    assert (project / "receptors.csv").exists()


def test_init_receptors_csv_reads_after_appending_a_row(tmp_path):
    """The starter receptors.csv has no line that parses as a bad receptor (#49)."""
    from stilt.receptors import read_receptors

    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0

    csv = project / "receptors.csv"
    assert read_receptors(csv) == []
    with csv.open("a") as f:
        f.write("2023-01-01 12:00:00,-111.85,40.77,5\n")
    receptors = read_receptors(csv)
    assert len(receptors) == 1
    assert receptors[0].altitude == 5


def test_init_prints_confirmation(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0
    assert "Initialized STILT project" in result.output


def test_init_writes_science_first_commented_config(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0

    text = (project / "config.yaml").read_text()
    parsed = yaml.safe_load(text)

    assert "#" in text
    assert ProjectConfig.from_yaml(project / "config.yaml")
    assert list(parsed) == [
        "mets",
        "grid",
        "variants",
        "n_hours",
        "numpar",
        "varsiwant",
        "hnf_plume",
        "output",
    ]
    assert parsed["output"] == "./output"
    assert parsed["grid"]["xmin"] == -113.0
    loaded = ProjectConfig.from_yaml(project / "config.yaml")
    assert loaded.footprint.grid is not None and loaded.footprint.grid.xmin == -113.0
    assert list(loaded.resolve()) == ["hrrr"]
    assert parsed["variants"] == {"hrrr": {}}
    assert text.index("mets:") < text.index("grid:")
    assert text.index("grid:") < text.index("n_hours:")
    assert text.index("numpar:") < text.index("# execution:")


def test_init_config_omits_advanced_and_internal_knobs(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0

    text = (project / "config.yaml").read_text()

    assert "ichem" not in text
    assert "idsp" not in text
    assert "kagl" not in text


def test_init_aborts_when_config_exists(tmp_path):
    project = tmp_path / "existing"
    project.mkdir()
    (project / "config.yaml").write_text("n_hours: -24\n")
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 2


def test_status_lists_output_folders_no_variant_uses(tmp_path):
    from stilt.project import Project

    _write_minimal_config(tmp_path)
    project = Project(tmp_path)
    # A run made under settings the config no longer has.
    variant = project.variants["hrrr"]
    stale = replace(
        variant,
        name="old",
        transport=variant.transport.model_copy(update={"numpar": 7}),
    )
    project.output.write_log(stale, str(project.receptors["receptor"].iloc[0]), "")

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "(no variant)" in result.output
    assert "settings=old-" in result.output
    assert "never deletes" in result.output


def test_status_counts_failures_by_reason(tmp_path):
    from stilt.project import Project
    from stilt.receptors import PointReceptor

    receptors = [
        PointReceptor(
            time=f"2023-01-01 {h}:00", longitude=-111.85, latitude=40.77, altitude=5
        )
        for h in (12, 13, 14)
    ]
    project = Project.init(
        tmp_path,
        mets={"hrrr": make_met_config(tmp_path / "met")},
        receptors=receptors,
        variants={"hrrr": {}},
    )
    reasons = ["MET_COVERAGE", "MET_COVERAGE", None]
    for receptor, reason in zip(receptors, reasons, strict=True):
        sim = project.simulation(str(receptor.id), "hrrr")
        sim.output.record_failure(
            "particles",
            sim.variant,
            sim.receptor.id,
            {"step": "particles", "reason": reason or "ValueError", "message": "m"},
        )

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "failed: MET_COVERAGE 2, ValueError 1" in result.output


def test_status_json_is_the_same_as_data(tmp_path):
    import json

    from stilt.project import Project
    from stilt.receptors import PointReceptor

    _write_minimal_config(tmp_path)
    project = Project(tmp_path)
    receptor = PointReceptor(
        time="2023-01-01 12:00:00", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    variant = project.variants["hrrr"]
    project.output.record_failure(
        "particles", variant, receptor.id, {"reason": "MET_COVERAGE"}
    )

    result = runner.invoke(app, ["status", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["project"] == str(tmp_path.resolve())
    assert data["simulations"] == {
        "total": 1,
        "complete": 0,
        "failed": 1,
        "interrupted": 0,
        "pending": 0,
    }
    assert data["variants"]["hrrr"]["failed"] == 1
    assert data["failed"] == {"MET_COVERAGE": 1}
    [folder] = data["folders"]
    assert (folder["kind"], folder["variants"], folder["files"]) == (
        "particles",
        ["hrrr"],
        0,
    )


def test_status_lists_folders_and_what_differs(tmp_path):
    import datetime as dt

    from stilt.config import ProjectConfig
    from stilt.project import Project
    from stilt.receptors import PointReceptor

    from .fixtures.particles import finished

    receptor = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    met = make_met_config(tmp_path / "met")
    old = Project.init(
        tmp_path / "old",
        config=ProjectConfig(
            mets={"hrrr": met},
            variants={"hrrr": {}},
            numpar=500,
            hnf_plume=False,
            output=str(tmp_path / "out"),
        ),
        receptors=[receptor],
    )
    sim = old.simulation(receptor.id, "hrrr")
    particles = pd.DataFrame(
        {
            "age": [-60.0],
            "particle": [1.0],
            "lon": [-111.9],
            "lat": [40.7],
            "zagl": [10.0],
            "foot": [1e-5],
        }
    )
    sim.output.write_particles(
        sim.variant, receptor, finished(particles, receptor, sim.transport), []
    )
    new = Project.init(
        tmp_path / "new",
        config=ProjectConfig(
            mets={"hrrr": met},
            variants={"hrrr": {}},
            numpar=1000,
            hnf_plume=False,
            output=str(tmp_path / "out"),
        ),
        receptors=[receptor],
    )

    result = runner.invoke(app, ["status", str(new.directory)])

    assert result.exit_code == 0, result.output
    assert f"Output: {tmp_path / 'out'}" in result.output
    assert "1 file  (no variant)" in result.output
    assert "numpar: 500 (config: 1000)" in result.output
    assert "PYSTILT never deletes one" in result.output
    table = old.folders()
    assert table[["kind", "name", "files", "variant", "differs"]].values.tolist() == [
        ["particles", "hrrr", 1, "hrrr", ""]
    ]
