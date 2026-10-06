"""Tests for stilt.cli - Typer command-line interface."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import stilt.__main__
from stilt.cli import _resolve_project, app
from stilt.config import ProjectConfig
from stilt.execution.config import ExecutionConfig
from stilt.spatial import Grid
from stilt.variants import resolve

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path):
    """Write a minimal config.yaml + receptors.csv so _resolve_project succeeds."""
    cfg = ProjectConfig(
        mets={
            "hrrr": {
                "directory": tmp_path / "met",
                "file_format": "%Y%m%d_%H",
                "file_tres": "1h",
            }
        },
    )
    cfg.to_yaml(tmp_path / "config.yaml")
    (tmp_path / "receptors.csv").write_text(
        "time,longitude,latitude,altitude\n2023-01-01 12:00:00,-111.85,40.77,5.0\n"
    )


@pytest.fixture
def calls(monkeypatch):
    """Record what `stilt run` asks of the project, and run nothing."""
    recorded: list[tuple[str, dict]] = []

    def run(self, **kwargs):
        recorded.append(("run", kwargs))
        return []

    def submit(self, **kwargs):
        recorded.append(("submit", kwargs))
        return [SimpleNamespace(job_id="12345_0"), SimpleNamespace(job_id="12345_1")]

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
    """Exits with code 1 when no config.yaml is found."""
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1


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
    assert result.exit_code == 1


def test_status_prints_project_info(tmp_path):
    """status prints one summary line with the project root and counts."""
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 0
    assert (
        f"Project: {tmp_path.resolve()}  total=1  completed=0  pending=1"
        in result.output
    )


def test_status_counts_full_simulation_completion(tmp_path):
    """A complete trajectory without required footprints is not done yet."""
    from stilt.project import Project
    from stilt.receptors import PointReceptor

    cfg = ProjectConfig(
        mets={
            "hrrr": {
                "directory": tmp_path / "met",
                "file_format": "%Y%m%d_%H",
                "file_tres": "1h",
            }
        },
        grid=Grid(
            xmin=-114.0,
            xmax=-113.0,
            ymin=39.0,
            ymax=40.0,
            xres=0.1,
            yres=0.1,
        ),
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
            "time": [-60.0],
            "indx": [1.0],
            "long": [-113.5],
            "lati": [39.5],
            "zagl": [10.0],
            "foot": [1e-5],
        }
    )
    folder = sim.output.particles(sim.variant)
    folder.write(receptor, particles, [])

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=1  completed=0  pending=1" in result.output

    # Once the footprint is present too, the simulation counts as complete.
    from stilt.execution import make_footprint

    sim = project.simulation(receptor.id, "hrrr")  # a fresh value
    make_footprint(sim, sim.particles)

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=1  completed=1  pending=0" in result.output


def test_cli_help_lists_current_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    expected = {
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
# run command
# ---------------------------------------------------------------------------


def test_run_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1


def test_run_runs_the_project_with_its_execution_settings(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    [(verb, kwargs)] = calls
    assert verb == "run"
    assert kwargs["execution"] == ExecutionConfig.model_validate({})
    assert kwargs["skip_existing"] is True
    # Resolved once, so the run uses the directory the banner shows.
    from stilt.execution import resolve_workdir
    from stilt.project import Project

    scratch = resolve_workdir(Project(tmp_path))
    assert kwargs["workdir"] == str(scratch)
    assert f"Workdir: {scratch}" in result.output


def test_run_prints_a_startup_summary_and_the_status(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0
    assert (
        f"Starting run: project={tmp_path.resolve()}  backend=local  "
        "tasks=1  cpus=1  skip=existing"
    ) in result.output
    assert "Workdir:" in result.output
    assert f"Output: {tmp_path.resolve() / 'output'}" in result.output
    assert "Receptors loaded: 1" in result.output
    assert "Execution mode: local, one line per receptor" in result.output
    assert f"Project: {tmp_path.resolve()}  total=" in result.output


def test_run_forwards_workdir_and_shows_it(tmp_path, calls):
    _write_minimal_config(tmp_path)
    workdir = tmp_path / "scratch"

    result = runner.invoke(app, ["run", str(tmp_path), "--workdir", str(workdir)])

    assert result.exit_code == 0
    assert f"Workdir: {workdir.resolve()}" in result.output
    assert calls[0][1]["workdir"] == str(workdir)


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
    assert "Execution mode: submit-and-wait" in result.output
    assert "waiting for the job" in result.output


def test_submit_submits_to_slurm_and_returns(tmp_path, calls):
    _write_minimal_config(tmp_path)  # its backend is local

    result = runner.invoke(app, ["submit", str(tmp_path), "--n-workers", "2"])

    assert result.exit_code == 0, result.output
    assert [verb for verb, _ in calls] == ["submit"]
    execution = calls[0][1]["execution"]
    assert (execution.backend, execution.n_workers) == ("slurm", 2)
    assert "Execution mode: submit-and-return" in result.output
    assert "Submitted job: 12345" in result.output


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
    assert list(resolve(loaded)) == ["hrrr"]
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
    assert result.exit_code == 1


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
    project.output.particles(stale)

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "particles folders" in result.output
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
        mets={
            "hrrr": {
                "directory": tmp_path / "met",
                "file_format": "%Y%m%d_%H",
                "file_tres": "1h",
            }
        },
        receptors=receptors,
    )
    reasons = ["MISSING_MET_FILES", "MISSING_MET_FILES", None]
    for receptor, reason in zip(receptors, reasons, strict=True):
        sim = project.simulation(str(receptor.id), "hrrr")
        sim.output.particles(sim.variant).record_failure(
            sim.receptor.id,
            {"step": "particles", "reason": reason or "ValueError", "message": "m"},
        )

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "failed: MISSING_MET_FILES 2, ValueError 1" in result.output
