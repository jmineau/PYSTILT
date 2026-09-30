"""Tests for stilt.cli - Typer command-line interface."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import yaml
from typer.testing import CliRunner

import stilt.__main__
from stilt.cli import _resolve_project, app
from stilt.config import ExecutionConfig, Grid, ModelConfig
from stilt.execution import register

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path):
    """Write a minimal config.yaml + receptors.csv so _resolve_project succeeds."""
    cfg = ModelConfig(
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


class _FakeHandle:
    detached = False

    def wait(self):
        return None


def _fake_model_factory(captured: list[dict]):
    """
    Build a Model stand-in that records its constructor kwargs.

    The stand-in exposes just enough surface for the CLI's startup summary,
    progress line, and status print.
    """

    class _FakeModel:
        def __init__(self, project):
            self.record = {"project": project}
            captured.append(self.record)
            self.project = SimpleNamespace(
                root=project, directory=Path(project), name=Path(project).name
            )
            self.output = SimpleNamespace(path=Path(project) / "output")
            self.receptors = []
            self.variants = {"hrrr": None}
            self.config = SimpleNamespace(execution=ExecutionConfig.model_validate({}))

        def status(self):
            return pd.DataFrame(
                columns=["receptor", "variant", "trajectory", "footprint", "complete"]
            )

        def unreferenced(self):
            return {"particles": [], "footprints": []}

        def run(self, skip_existing=True, wait=True, compute_root=None, execution=None):
            self.record["compute_root"] = compute_root
            return _FakeHandle()

    return _FakeModel


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
    from stilt.model import Model
    from stilt.receptors import PointReceptor

    cfg = ModelConfig(
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
    model = Model(project=tmp_path, config=cfg, receptors=[receptor])
    assert register(model) == [str(receptor.id)]

    # Particles exist but the required footprint does not: not complete.
    sim = model.simulation((receptor.id, "hrrr"))
    from stilt.trajectory import Trajectories

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
    run = sim.output.run(sim.variant.name, sim.variant.transport)
    run.write_particles(
        Trajectories(receptor=receptor, params=sim.params, met_files=[], data=particles)
    )

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=1  completed=0  pending=1" in result.output

    # Once the footprint is present too, the simulation counts as complete.
    from stilt.execution import write_footprint

    sim = model.simulation(
        (receptor.id, "hrrr")
    )  # a fresh value; the old one cached no particles
    write_footprint(sim, sim.trajectories, context=model.transform_context(sim))

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


def test_run_invokes_model_run(tmp_path, monkeypatch):
    """run passes the config's execution settings to model.run."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    fake_handle.detached = False  # local execution has finished
    calls: list = []

    def fake_run(
        self, skip_existing=None, wait=True, compute_root=None, execution=None
    ):
        calls.append(
            {"execution": execution, "skip_existing": skip_existing, "wait": wait}
        )
        return fake_handle

    monkeypatch.setattr("stilt.cli.Model.run", fake_run)

    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 0
    [call] = calls
    assert call["execution"] == ExecutionConfig.model_validate({})
    assert call["skip_existing"] is True
    assert call["wait"] is False
    # A local run is over when model.run returns, so there is nothing to wait for.
    fake_handle.wait.assert_not_called()


def test_run_prints_startup_and_wait_messages(tmp_path, monkeypatch):
    """run prints a startup summary before blocking locally."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    fake_handle.detached = False  # local execution has finished

    monkeypatch.setattr(
        "stilt.cli.Model.run",
        lambda self, skip_existing=None, wait=True, compute_root=None, execution=None: (
            fake_handle
        ),
    )

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0
    assert (
        f"Starting run: project={tmp_path.resolve()}  backend=local  "
        "tasks=1  cpus=1  skip=existing"
    ) in result.output
    assert "Compute root:" in result.output
    assert f"Output: {tmp_path.resolve() / 'output'}" in result.output
    assert "Receptors loaded: 1" in result.output
    assert "Execution mode: local, one line per receptor" in result.output
    assert f"Project: {tmp_path.resolve()}  total=" in result.output


def test_run_startup_summary_uses_project_root_and_compute_root(tmp_path, monkeypatch):
    """run startup output shows the project root and a non-default compute root."""
    _write_minimal_config(tmp_path)
    captured: list[dict] = []
    monkeypatch.setattr("stilt.cli.Model", _fake_model_factory(captured))

    compute_root = tmp_path / "scratch"
    result = runner.invoke(
        app, ["run", str(tmp_path), "--compute-root", str(compute_root)]
    )

    assert result.exit_code == 0
    assert f"project={tmp_path.resolve()}" in result.output
    assert f"Compute root: {compute_root.resolve()}" in result.output


def test_run_no_skip_passes_false(tmp_path, monkeypatch):
    """--no-skip passes skip_existing=False to model.run() and shows in the summary."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    fake_handle.detached = False  # local execution -> inline wait
    calls: list = []

    def fake_run(
        self, skip_existing=None, wait=True, compute_root=None, execution=None
    ):
        calls.append(skip_existing)
        return fake_handle

    monkeypatch.setattr("stilt.cli.Model.run", fake_run)

    result = runner.invoke(app, ["run", str(tmp_path), "--no-skip"])
    assert result.exit_code == 0
    assert calls == [False]
    assert "skip=no-skip" in result.output


def test_run_forwards_compute_root(tmp_path, monkeypatch):
    """run forwards --compute-root to model.run()."""
    _write_minimal_config(tmp_path)
    captured: list[dict] = []
    monkeypatch.setattr("stilt.cli.Model", _fake_model_factory(captured))

    result = runner.invoke(
        app,
        ["run", str(tmp_path), "--compute-root", str(tmp_path / "scratch")],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "project": str(tmp_path.resolve()),
            "compute_root": str(tmp_path / "scratch"),
        }
    ]


def test_run_backend_and_workers_override_the_config(tmp_path, monkeypatch):
    """--backend and --n-workers replace those execution settings for this run."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    fake_handle.detached = False
    captured_executor = []

    def fake_run(
        self, skip_existing=None, wait=True, compute_root=None, execution=None
    ):
        captured_executor.append(execution)
        return fake_handle

    monkeypatch.setattr("stilt.cli.Model.run", fake_run)

    result = runner.invoke(
        app, ["run", str(tmp_path), "--backend", "local", "--cpus", "4"]
    )
    assert result.exit_code == 0
    assert len(captured_executor) == 1
    assert captured_executor[0].backend == "local"
    assert captured_executor[0].cpus == 4
    assert "tasks=1  cpus=4" in result.output


def test_run_slurm_fire_and_forget(tmp_path, monkeypatch):
    """For a SlurmHandle, prints job_id and exits without blocking by default."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock(detached=True, job_id="12345")

    monkeypatch.setattr(
        "stilt.cli.Model.run",
        lambda self, skip_existing=None, wait=True, compute_root=None, execution=None: (
            fake_handle
        ),
    )

    result = runner.invoke(
        app, ["run", str(tmp_path), "--backend", "slurm", "--n-workers", "2"]
    )
    assert result.exit_code == 0
    assert "Execution mode: submit-and-return" in result.output
    assert "Submitted job: 12345" in result.output
    # --wait not passed → fire-and-forget, handle.wait() must NOT be called.
    fake_handle.wait.assert_not_called()


def test_run_slurm_with_wait_flag_blocks(tmp_path, monkeypatch):
    """--wait causes the CLI to call handle.wait() for a SlurmHandle."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock(detached=True, job_id="99")

    monkeypatch.setattr(
        "stilt.cli.Model.run",
        lambda self, skip_existing=None, wait=True, compute_root=None, execution=None: (
            fake_handle
        ),
    )

    result = runner.invoke(
        app,
        ["run", str(tmp_path), "--backend", "slurm", "--n-workers", "2", "--wait"],
    )
    assert result.exit_code == 0
    assert "Execution mode: submit-and-wait" in result.output
    assert "Waiting for job completion" in result.output
    fake_handle.wait.assert_called_once()


def test_run_detached_handle_prints_job_id(tmp_path, monkeypatch):
    """Any detached handle (e.g. Slurm) prints the submitted job ID."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock(detached=True, job_id="42")
    monkeypatch.setattr(
        "stilt.cli.Model.run",
        lambda self, skip_existing=None, wait=True, compute_root=None, execution=None: (
            fake_handle
        ),
    )

    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 0
    assert "Submitted job: 42" in result.output


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
    assert ModelConfig.from_yaml(project / "config.yaml")
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
    loaded = ModelConfig.from_yaml(project / "config.yaml")
    assert loaded.grid is not None and loaded.grid.xmin == -113.0
    assert list(loaded.resolve_variants()) == ["hrrr"]
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
    from stilt.model import Model

    _write_minimal_config(tmp_path)
    model = Model(project=tmp_path)
    register(model)
    # A run made under settings the config no longer has.
    stale = model.variants["hrrr"].transport.model_copy(update={"numpar": 7})
    model.output.run("old", stale)

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "particles folders" in result.output
    assert "settings=old-" in result.output
    assert "never deletes" in result.output
