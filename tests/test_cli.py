"""Tests for stilt.cli - Typer command-line interface."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import yaml
from typer.testing import CliRunner

import stilt.__main__
from stilt.cli import _resolve_project, app
from stilt.config import FootprintConfig, Grid, ModelConfig
from stilt.model import StatusCounts
from stilt.project import SIMULATIONS_PREFIX

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
        def __init__(self, project, compute_root=None):
            captured.append({"project": project, "compute_root": compute_root})
            is_cloud = "://" in str(project)
            self.project = SimpleNamespace(
                root=project,
                is_cloud=is_cloud,
                simulations_dir=(
                    None if is_cloud else Path(project) / SIMULATIONS_PREFIX
                ),
            )
            self.compute_root = (
                Path(compute_root)
                if compute_root is not None
                else self.project.simulations_dir
            )
            self.receptors = []
            self.config = SimpleNamespace(execution={})

        def status(self):
            return StatusCounts()

        def run(self, executor=None, skip_existing=None, wait=True):
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


def test_resolve_project_returns_cloud_uri_unchanged():
    assert _resolve_project("s3://bucket/project") == "s3://bucket/project"


def test_resolve_project_accepts_output_root_without_config(tmp_path):
    (tmp_path / SIMULATIONS_PREFIX).mkdir(parents=True)
    resolved = _resolve_project(tmp_path, require_inputs=False)
    assert resolved == str(tmp_path.resolve())


def test_resolve_project_rejects_empty_dir_without_inputs_required(tmp_path):
    from typer import Exit

    try:
        _resolve_project(tmp_path, require_inputs=False)
    except Exit as exc:
        assert exc.exit_code == 1
    else:
        raise AssertionError("expected typer.Exit")


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
    from stilt.simulation import SimID

    cfg = ModelConfig(
        mets={
            "hrrr": {
                "directory": tmp_path / "met",
                "file_format": "%Y%m%d_%H",
                "file_tres": "1h",
            }
        },
        footprints={
            "slv": FootprintConfig(
                grid=Grid(
                    xmin=-114.0,
                    xmax=-113.0,
                    ymin=39.0,
                    ymax=40.0,
                    xres=0.1,
                    yres=0.1,
                )
            )
        },
    )

    receptor = PointReceptor(
        time="2023-01-01 12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    model = Model(project=tmp_path, config=cfg, receptors=[receptor])
    sid = str(SimID.from_parts("hrrr", receptor))
    assert model.register() == [sid]

    # Trajectory exists but the required footprint does not → not complete.
    sim = model.simulation(sid)
    sim.directory.mkdir(parents=True, exist_ok=True)
    sim.trajectories_path.write_bytes(b"x")

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=1  completed=0  pending=1" in result.output

    # Once the footprint is present too, the simulation counts as complete.
    sim.footprint_path("slv").write_bytes(b"x")

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=1  completed=1  pending=0" in result.output


def test_cli_help_lists_current_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    expected = {
        "init",
        "run",
        "register",
        "pull-worker",
        "push-worker",
        "serve",
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
    """run always calls model.run(wait=False) and waits inline for local handles."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    fake_handle.detached = False  # local execution -> inline wait
    calls: list = []

    def fake_run(self, executor=None, skip_existing=None, wait=True):
        calls.append(
            {"executor": executor, "skip_existing": skip_existing, "wait": wait}
        )
        return fake_handle

    monkeypatch.setattr("stilt.cli.Model.run", fake_run)

    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 0
    assert calls == [{"executor": None, "skip_existing": None, "wait": False}]
    # Local handle — wait() must always be called so no orphan workers.
    fake_handle.wait.assert_called_once()


def test_run_prints_startup_and_wait_messages(tmp_path, monkeypatch):
    """run prints a startup summary before blocking locally."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    fake_handle.detached = False  # local execution -> inline wait

    monkeypatch.setattr(
        "stilt.cli.Model.run",
        lambda self, executor=None, skip_existing=None, wait=True: fake_handle,
    )

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0
    assert (
        f"Starting run: project={tmp_path.resolve()}  backend=local  "
        "dispatch=push  workers=1  skip=config"
    ) in result.output
    assert "Compute root:" not in result.output  # default compute root
    assert "Receptors loaded: 1" in result.output
    assert "Execution mode: local-blocking" in result.output
    assert "Workers launched. Waiting for completion..." in result.output
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
    assert f"Compute root: {compute_root}" in result.output


def test_run_no_skip_passes_false(tmp_path, monkeypatch):
    """--no-skip passes skip_existing=False to model.run() and shows in the summary."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    fake_handle.detached = False  # local execution -> inline wait
    calls: list = []

    def fake_run(self, executor=None, skip_existing=None, wait=True):
        calls.append(skip_existing)
        return fake_handle

    monkeypatch.setattr("stilt.cli.Model.run", fake_run)

    result = runner.invoke(app, ["run", str(tmp_path), "--no-skip"])
    assert result.exit_code == 0
    assert calls == [False]
    assert "skip=no-skip" in result.output


def test_run_accepts_cloud_project_uri(monkeypatch):
    """Cloud project URIs are forwarded unchanged into Model construction."""
    captured: list[dict] = []
    monkeypatch.setattr("stilt.cli.Model", _fake_model_factory(captured))

    result = runner.invoke(app, ["run", "s3://bucket/project"])

    assert result.exit_code == 0
    assert captured == [{"project": "s3://bucket/project", "compute_root": None}]
    assert "project=s3://bucket/project" in result.output


def test_run_forwards_compute_root(tmp_path, monkeypatch):
    """run forwards --compute-root into Model construction."""
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


def test_run_config_option_resolves_project_from_its_parent(tmp_path, monkeypatch):
    """--config PATH uses the config file's parent as the project root."""
    _write_minimal_config(tmp_path)
    captured: list[dict] = []
    monkeypatch.setattr("stilt.cli.Model", _fake_model_factory(captured))

    result = runner.invoke(app, ["run", "--config", str(tmp_path / "config.yaml")])

    assert result.exit_code == 0
    assert captured == [{"project": str(tmp_path.resolve()), "compute_root": None}]


def test_run_config_option_exits_when_file_missing(tmp_path):
    result = runner.invoke(app, ["run", "--config", str(tmp_path / "nope.yaml")])

    assert result.exit_code == 1
    assert "config file not found" in result.output


def test_run_backend_override_builds_executor(tmp_path, monkeypatch):
    """--backend and --n-workers build an executor override passed to model.run()."""
    from stilt.execution import LocalExecutor

    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    fake_handle.detached = False
    captured_executor = []

    def fake_run(self, executor=None, skip_existing=None, wait=True):
        captured_executor.append(executor)
        return fake_handle

    monkeypatch.setattr("stilt.cli.Model.run", fake_run)

    result = runner.invoke(
        app, ["run", str(tmp_path), "--backend", "local", "--n-workers", "4"]
    )
    assert result.exit_code == 0
    assert len(captured_executor) == 1
    assert isinstance(captured_executor[0], LocalExecutor)
    assert captured_executor[0].n_workers == 4
    assert "workers=4" in result.output


def test_run_slurm_fire_and_forget(tmp_path, monkeypatch):
    """For a SlurmHandle, prints job_id and exits without blocking by default."""
    from stilt.execution import SlurmHandle

    _write_minimal_config(tmp_path)

    fake_handle = SlurmHandle("12345")
    fake_handle.wait = MagicMock()

    monkeypatch.setattr(
        "stilt.cli.Model.run",
        lambda self, executor=None, skip_existing=None, wait=True: fake_handle,
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
    from stilt.execution import SlurmHandle

    _write_minimal_config(tmp_path)

    fake_handle = SlurmHandle("99")
    fake_handle.wait = MagicMock()

    monkeypatch.setattr(
        "stilt.cli.Model.run",
        lambda self, executor=None, skip_existing=None, wait=True: fake_handle,
    )

    result = runner.invoke(
        app,
        ["run", str(tmp_path), "--backend", "slurm", "--n-workers", "2", "--wait"],
    )
    assert result.exit_code == 0
    assert "Execution mode: submit-and-wait" in result.output
    assert "Waiting for job completion..." in result.output
    fake_handle.wait.assert_called_once()


def test_run_detached_handle_prints_job_id(tmp_path, monkeypatch):
    """Any detached handle (e.g. Slurm) prints the submitted job ID."""
    from stilt.execution import SlurmHandle

    _write_minimal_config(tmp_path)

    fake_handle = SlurmHandle("42")
    fake_handle.wait = MagicMock()
    monkeypatch.setattr(
        "stilt.cli.Model.run",
        lambda self, executor=None, skip_existing=None, wait=True: fake_handle,
    )

    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 0
    assert "Submitted job: 42" in result.output


# ---------------------------------------------------------------------------
# pull-worker command
# ---------------------------------------------------------------------------


def test_pull_worker_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["pull-worker", str(tmp_path)])
    assert result.exit_code == 1


def test_pull_worker_calls_pull_simulations(tmp_path, monkeypatch):
    """pull-worker calls pull_simulations on the model."""
    _write_minimal_config(tmp_path)

    loop_calls: list[dict] = []

    def fake_loop(model, follow=False, poll_interval=10.0, *, skip_existing=None):
        loop_calls.append({"follow": follow})

    monkeypatch.setattr("stilt.cli.pull_simulations", fake_loop)

    result = runner.invoke(app, ["pull-worker", str(tmp_path)])
    assert result.exit_code == 0
    assert loop_calls == [{"follow": False}]


def test_pull_worker_follow_flag_forwarded(tmp_path, monkeypatch):
    """--follow is forwarded to pull_simulations."""
    _write_minimal_config(tmp_path)

    loop_calls: list[dict] = []

    def fake_loop(model, follow=False, poll_interval=10.0, *, skip_existing=None):
        loop_calls.append({"follow": follow})

    monkeypatch.setattr("stilt.cli.pull_simulations", fake_loop)

    result = runner.invoke(app, ["pull-worker", str(tmp_path), "--follow"])
    assert result.exit_code == 0
    assert loop_calls == [{"follow": True}]


def test_pull_worker_accepts_cloud_project_uri(monkeypatch):
    """pull-worker can bootstrap from a cloud project ref."""
    captured: list[dict] = []
    monkeypatch.setattr("stilt.cli.Model", _fake_model_factory(captured))
    monkeypatch.setattr(
        "stilt.cli.pull_simulations",
        lambda model, follow=False, poll_interval=10.0, skip_existing=None: None,
    )

    result = runner.invoke(app, ["pull-worker", "gs://bucket/project"])

    assert result.exit_code == 0
    assert captured == [{"project": "gs://bucket/project", "compute_root": None}]


def test_pull_worker_forwards_compute_root(tmp_path, monkeypatch):
    """pull-worker forwards --compute-root into Model construction."""
    _write_minimal_config(tmp_path)
    captured: list[dict] = []
    monkeypatch.setattr("stilt.cli.Model", _fake_model_factory(captured))
    monkeypatch.setattr(
        "stilt.cli.pull_simulations",
        lambda model, follow=False, poll_interval=10.0, skip_existing=None: None,
    )

    result = runner.invoke(
        app,
        ["pull-worker", str(tmp_path), "--compute-root", str(tmp_path / "scratch")],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "project": str(tmp_path.resolve()),
            "compute_root": str(tmp_path / "scratch"),
        }
    ]


# ---------------------------------------------------------------------------
# push-worker command
# ---------------------------------------------------------------------------


def test_push_worker_calls_run_simulations(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    chunk = tmp_path / "task_0.txt"
    chunk.write_text("hrrr_202301011200_abc\n\nhrrr_202301011200_def\n")

    sim_list_calls: list[dict] = []

    def fake_run(model, sim_ids, *, n_cores=1, skip_existing=None):
        sim_list_calls.append(
            {"sim_ids": sim_ids, "n_cores": n_cores, "skip_existing": skip_existing}
        )
        return []

    monkeypatch.setattr("stilt.cli.run_simulations", fake_run)

    result = runner.invoke(
        app,
        ["push-worker", str(tmp_path), "--chunk", str(chunk), "--cpus", "4"],
    )

    assert result.exit_code == 0
    assert sim_list_calls == [
        {
            "sim_ids": ["hrrr_202301011200_abc", "hrrr_202301011200_def"],
            "n_cores": 4,
            "skip_existing": None,
        }
    ]


def test_push_worker_forwards_skip_existing_flags(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    chunk = tmp_path / "task_0.txt"
    chunk.write_text("hrrr_202301011200_abc\n")

    seen: list[bool | None] = []

    def fake_run(model, sim_ids, *, n_cores=1, skip_existing=None):
        seen.append(skip_existing)
        return []

    monkeypatch.setattr("stilt.cli.run_simulations", fake_run)

    base = ["push-worker", str(tmp_path), "--chunk", str(chunk)]
    assert runner.invoke(app, [*base, "--skip-existing"]).exit_code == 0
    assert runner.invoke(app, [*base, "--no-skip-existing"]).exit_code == 0
    assert seen == [True, False]


def test_push_worker_forwards_compute_root(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    chunk = tmp_path / "task_0.txt"
    chunk.write_text("hrrr_202301011200_abc\n")
    captured: list[dict] = []

    monkeypatch.setattr("stilt.cli.Model", _fake_model_factory(captured))
    monkeypatch.setattr(
        "stilt.cli.run_simulations",
        lambda model, sim_ids, n_cores=1, skip_existing=None: [],
    )

    result = runner.invoke(
        app,
        [
            "push-worker",
            str(tmp_path),
            "--chunk",
            str(chunk),
            "--compute-root",
            str(tmp_path / "scratch"),
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "project": str(tmp_path.resolve()),
            "compute_root": str(tmp_path / "scratch"),
        }
    ]


def test_push_worker_requires_chunk_option(tmp_path):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["push-worker", str(tmp_path)])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


def test_serve_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["serve", str(tmp_path)])
    assert result.exit_code == 1


def test_serve_calls_pull_simulations_in_follow_mode(tmp_path, monkeypatch):
    """serve is the user-facing long-lived queue consumer command."""
    _write_minimal_config(tmp_path)

    loop_calls: list[dict] = []

    def fake_loop(model, follow=False, poll_interval=10.0, *, skip_existing=None):
        loop_calls.append({"follow": follow})

    monkeypatch.setattr("stilt.cli.pull_simulations", fake_loop)

    result = runner.invoke(app, ["serve", str(tmp_path)])
    assert result.exit_code == 0
    assert loop_calls == [{"follow": True}]


def test_serve_accepts_cloud_project_uri(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr("stilt.cli.Model", _fake_model_factory(captured))
    monkeypatch.setattr(
        "stilt.cli.pull_simulations",
        lambda model, follow=False, poll_interval=10.0, skip_existing=None: None,
    )

    result = runner.invoke(app, ["serve", "gs://bucket/project"])

    assert result.exit_code == 0
    assert captured == [{"project": "gs://bucket/project", "compute_root": None}]


def test_serve_forwards_compute_root(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    captured: list[dict] = []
    loop_calls: list[dict] = []

    def fake_loop(model, follow=False, poll_interval=10.0, *, skip_existing=None):
        loop_calls.append({"follow": follow})

    monkeypatch.setattr("stilt.cli.Model", _fake_model_factory(captured))
    monkeypatch.setattr("stilt.cli.pull_simulations", fake_loop)

    result = runner.invoke(
        app,
        ["serve", str(tmp_path), "--compute-root", str(tmp_path / "scratch")],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "project": str(tmp_path.resolve()),
            "compute_root": str(tmp_path / "scratch"),
        }
    ]
    assert loop_calls == [{"follow": True}]


# ---------------------------------------------------------------------------
# register command
# ---------------------------------------------------------------------------


def test_register_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["register", str(tmp_path)])
    assert result.exit_code == 1


def test_register_registers_project_receptors(tmp_path, monkeypatch):
    """register calls Model.register() with no receptors and prints the count."""
    _write_minimal_config(tmp_path)

    register_calls: list = []

    def fake_register(model, receptors=None):
        del model
        register_calls.append(receptors)
        return ["sim_id_1", "sim_id_2"]

    monkeypatch.setattr("stilt.cli.Model.register", fake_register)

    result = runner.invoke(app, ["register", str(tmp_path)])
    assert result.exit_code == 0
    assert "Registered 2 simulation(s)." in result.output
    assert register_calls == [None]


def test_register_with_receptors_file(tmp_path, monkeypatch):
    """--receptors PATH loads receptors from file and passes them to register()."""
    _write_minimal_config(tmp_path)

    receptors_csv = tmp_path / "my_receptors.csv"
    receptors_csv.write_text(
        "time,longitude,latitude,altitude\n2023-01-01 12:00:00,-111.85,40.77,5.0\n"
    )

    register_calls: list = []

    def fake_register(model, receptors=None):
        del model
        register_calls.append(receptors)
        return ["sim_id_1"]

    monkeypatch.setattr("stilt.cli.Model.register", fake_register)

    result = runner.invoke(
        app, ["register", str(tmp_path), "--receptors", str(receptors_csv)]
    )
    assert result.exit_code == 0
    assert "Registered 1 simulation(s)." in result.output
    assert len(register_calls) == 1
    assert register_calls[0] is not None
    assert len(register_calls[0]) == 1


def test_register_writes_project_inputs(tmp_path):
    """Without mocks, register persists config.yaml + receptors.csv and reports ids."""
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["register", str(tmp_path)])

    assert result.exit_code == 0
    assert "Registered 1 simulation(s)." in result.output
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "receptors.csv").exists()


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
        "footprints",
        "n_hours",
        "numpar",
        "varsiwant",
        "hnf_plume",
        "skip_existing",
    ]
    assert "grids" not in parsed
    assert "grid" not in parsed["footprints"]["default"]
    assert parsed["footprints"]["default"]["xmin"] == -113.0
    assert (
        ModelConfig.from_yaml(project / "config.yaml").footprints["default"].grid.xmin
        == -113.0
    )
    assert text.index("mets:") < text.index("footprints:")
    assert text.index("footprints:") < text.index("n_hours:")
    assert text.index("numpar:") < text.index("skip_existing:")
    assert text.index("skip_existing:") < text.index("# execution:")


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
