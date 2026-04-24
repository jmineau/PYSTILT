"""Tests for stilt.cli - Typer command-line interface."""

from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

import stilt.__main__
from stilt.cli import _resolve_project_dir, app
from stilt.config import FootprintConfig, Grid, ModelConfig
from stilt.index import IndexCounts

runner = CliRunner()


def _fake_state_for_cli_summary():
    class _FakeState:
        def counts(self):
            return IndexCounts()

        def rebuild(self):
            return None

    return _FakeState()


def test_python_m_entrypoint_invokes_cli(monkeypatch):
    called = False

    def fake_app() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(stilt.__main__, "app", fake_app)

    stilt.__main__.main()

    assert called is True


# ---------------------------------------------------------------------------
# _resolve_project_dir helper
# ---------------------------------------------------------------------------


def test_resolve_project_dir_exits_when_no_config_yaml(tmp_path):
    """Exits with code 1 when no config.yaml is found."""
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1


def test_resolve_project_dir_returns_path_when_config_exists(tmp_path):
    """Returns the resolved path when config.yaml is present."""
    (tmp_path / "config.yaml").write_text("n_hours: -24\n")
    resolved = _resolve_project_dir(tmp_path)
    assert resolved == str(tmp_path.resolve())


def test_resolve_project_dir_returns_cloud_uri_unchanged():
    assert _resolve_project_dir("s3://bucket/project") == "s3://bucket/project"


def test_resolve_project_dir_accepts_durable_output_root_without_config(tmp_path):
    (tmp_path / "simulations").mkdir()
    resolved = _resolve_project_dir(tmp_path, require_inputs=False)
    assert resolved == str(tmp_path.resolve())


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def test_status_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 1


def test_status_prints_project_info(tmp_path):
    """status command prints a summary line."""
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 0
    assert "total=" in result.output
    assert "completed=" in result.output


def test_status_accepts_output_dir_without_project_arg(tmp_path):
    (tmp_path / "simulations").mkdir()

    result = runner.invoke(app, ["status", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=" in result.output


def test_status_counts_full_simulation_completion(tmp_path):
    """A complete trajectory without required footprints is not done yet."""
    from stilt.execution import SimulationResult
    from stilt.model import Model
    from stilt.receptor import Receptor
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
    cfg.to_yaml(tmp_path / "config.yaml")

    receptor = Receptor(
        time="2023-01-01 12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    model = Model(project=tmp_path, config=cfg, receptors=[receptor])
    sid = str(SimID.from_parts("hrrr", receptor))
    model.register_pending()
    model.index.record(
        SimulationResult(
            sim_id=SimID(sid),
            status="complete",
            traj_present=True,
        )
    )

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "completed=0" in result.output


def test_cli_help_lists_current_queue_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "register" in result.output
    assert "pull-worker" in result.output
    assert "serve" in result.output
    assert "enqueue" not in result.output


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


def test_run_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1


def test_run_invokes_model_run(tmp_path, monkeypatch):
    """run command always calls model.run(wait=False) and completes inline for local."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    calls: list = []

    def fake_run(self, executor=None, skip_existing=None, rebuild=None, wait=True):
        calls.append({"skip_existing": skip_existing, "rebuild": rebuild, "wait": wait})
        return fake_handle

    monkeypatch.setattr("stilt.model.Model.run", fake_run)

    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["skip_existing"] is None  # no --no-skip flag, so None (use config)
    assert calls[0]["rebuild"] is None
    assert calls[0]["wait"] is False  # CLI always passes wait=False to model.run
    # Local handle — wait() must always be called so no orphan workers.
    fake_handle.wait.assert_called_once()


def test_run_prints_startup_and_wait_messages(tmp_path, monkeypatch):
    """run prints a startup summary before blocking locally."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()

    monkeypatch.setattr(
        "stilt.model.Model.run",
        lambda self, executor=None, skip_existing=None, rebuild=None, wait=True,: (
            fake_handle
        ),
    )

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0
    assert "Starting run:" in result.output
    assert "Workers launched. Waiting for completion..." in result.output


def test_run_startup_summary_uses_explicit_roots(tmp_path, monkeypatch):
    """run startup output uses project_root/output_root rather than project slug."""
    _write_minimal_config(tmp_path)

    class _FakeHandle:
        def wait(self):
            return None

    class _FakeModel:
        def __init__(self, project, output_dir=None, compute_root=None):
            self.project = "slug-only"
            from types import SimpleNamespace

            self.layout = SimpleNamespace(
                project_root=project,
                output_root=output_dir,
                output_dir=str(Path(project) / ".cache" / "outputs"),
                is_cloud_output=True,
            )
            self.compute_root = compute_root
            self.receptors = []
            self.config = type("Cfg", (), {"execution": {}})()
            self.index = _fake_state_for_cli_summary()

        def status(self, scene_id=None):
            return self.index.counts()

        def run(self, executor=None, skip_existing=None, rebuild=None, wait=True):
            return _FakeHandle()

    monkeypatch.setattr("stilt.cli.Model", _FakeModel)

    compute_root = tmp_path / "scratch"
    result = runner.invoke(
        app,
        [
            "run",
            str(tmp_path),
            "--output-dir",
            "s3://bucket/project",
            "--compute-root",
            str(compute_root),
        ],
    )

    assert result.exit_code == 0
    assert f"project={tmp_path.resolve()}" in result.output
    assert "project=slug-only" not in result.output
    assert "Output root: s3://bucket/project" in result.output
    assert f"Compute root: {compute_root}" in result.output


def test_run_no_skip_passes_false(tmp_path, monkeypatch):
    """--no-skip passes skip_existing=False to model.run()."""
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    calls: list = []

    def fake_run(self, executor=None, skip_existing=None, rebuild=None, wait=True):
        calls.append(skip_existing)
        return fake_handle

    monkeypatch.setattr("stilt.model.Model.run", fake_run)

    result = runner.invoke(app, ["run", str(tmp_path), "--no-skip"])
    assert result.exit_code == 0
    assert calls[0] is False


def test_run_rebuild_flag_passes_true(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    rebuild_calls: list[bool | None] = []

    def fake_run(self, executor=None, skip_existing=None, rebuild=None, wait=True):
        rebuild_calls.append(rebuild)
        return fake_handle

    monkeypatch.setattr("stilt.model.Model.run", fake_run)

    result = runner.invoke(app, ["run", str(tmp_path), "--rebuild"])

    assert result.exit_code == 0
    assert rebuild_calls == [True]


def test_run_no_rebuild_flag_passes_false(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)

    fake_handle = MagicMock()
    rebuild_calls: list[bool | None] = []

    def fake_run(self, executor=None, skip_existing=None, rebuild=None, wait=True):
        rebuild_calls.append(rebuild)
        return fake_handle

    monkeypatch.setattr("stilt.model.Model.run", fake_run)

    result = runner.invoke(app, ["run", str(tmp_path), "--no-rebuild"])

    assert result.exit_code == 0
    assert rebuild_calls == [False]


def test_run_accepts_cloud_project_uri(monkeypatch):
    """Cloud project URIs are forwarded unchanged into Model construction."""
    captured: list[dict] = []

    class _FakeHandle:
        def wait(self):
            return None

    class _FakeModel:
        def __init__(self, project, output_dir=None, compute_root=None):
            captured.append(
                {
                    "project": project,
                    "output_dir": output_dir,
                    "compute_root": compute_root,
                }
            )
            self.project = "project"
            from types import SimpleNamespace

            self.layout = SimpleNamespace(
                project_root=project,
                output_root=output_dir or project,
                output_dir=None,
                is_cloud_output=True,
            )
            self.compute_root = compute_root
            self.receptors = []
            self.config = type("Cfg", (), {"execution": {}})()
            self.index = _fake_state_for_cli_summary()

        def status(self, scene_id=None):
            return self.index.counts()

        def run(self, executor=None, skip_existing=None, rebuild=None, wait=True):
            return _FakeHandle()

    monkeypatch.setattr("stilt.cli.Model", _FakeModel)

    result = runner.invoke(app, ["run", "s3://bucket/project"])

    assert result.exit_code == 0
    assert captured == [
        {
            "project": "s3://bucket/project",
            "output_dir": None,
            "compute_root": None,
        }
    ]


def test_run_forwards_output_dir_and_compute_root(tmp_path, monkeypatch):
    """run forwards the new durable-output and compute-root options."""
    _write_minimal_config(tmp_path)
    captured: list[dict] = []

    class _FakeHandle:
        def wait(self):
            return None

    class _FakeModel:
        def __init__(self, project, output_dir=None, compute_root=None):
            captured.append(
                {
                    "project": project,
                    "output_dir": output_dir,
                    "compute_root": compute_root,
                }
            )
            self.project = "project"
            from types import SimpleNamespace

            self.layout = SimpleNamespace(
                project_root=project,
                output_root=output_dir or project,
                output_dir=output_dir,
                is_cloud_output=output_dir is not None,
            )
            self.compute_root = compute_root
            self.receptors = []
            self.config = type("Cfg", (), {"execution": {}})()
            self.index = _fake_state_for_cli_summary()

        def status(self, scene_id=None):
            return self.index.counts()

        def run(self, executor=None, skip_existing=None, rebuild=None, wait=True):
            return _FakeHandle()

    monkeypatch.setattr("stilt.cli.Model", _FakeModel)

    result = runner.invoke(
        app,
        [
            "run",
            str(tmp_path),
            "--output-dir",
            "s3://bucket/project",
            "--compute-root",
            str(tmp_path / "scratch"),
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "project": str(tmp_path.resolve()),
            "output_dir": "s3://bucket/project",
            "compute_root": str(tmp_path / "scratch"),
        }
    ]


def test_run_backend_override_builds_executor(tmp_path, monkeypatch):
    """--backend and --n-workers build an executor override passed to model.run()."""
    from stilt.execution import LocalExecutor

    _write_minimal_config(tmp_path)

    captured_executor = []

    def fake_run(self, executor=None, skip_existing=None, rebuild=None, wait=True):
        captured_executor.append(executor)
        return MagicMock()

    monkeypatch.setattr("stilt.model.Model.run", fake_run)

    result = runner.invoke(
        app, ["run", str(tmp_path), "--backend", "local", "--n-workers", "4"]
    )
    assert result.exit_code == 0
    assert len(captured_executor) == 1
    assert isinstance(captured_executor[0], LocalExecutor)
    assert captured_executor[0].n_workers == 4


def test_run_slurm_fire_and_forget(tmp_path, monkeypatch):
    """For a SlurmHandle, prints job_id and exits without blocking by default."""
    from stilt.execution import SlurmHandle

    _write_minimal_config(tmp_path)

    fake_handle = SlurmHandle("12345")
    fake_handle.wait = MagicMock()

    monkeypatch.setattr(
        "stilt.model.Model.run",
        lambda self, executor=None, skip_existing=None, rebuild=None, wait=True,: (
            fake_handle
        ),
    )

    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 0
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
        "stilt.model.Model.run",
        lambda self, executor=None, skip_existing=None, rebuild=None, wait=True,: (
            fake_handle
        ),
    )

    result = runner.invoke(app, ["run", str(tmp_path), "--wait"])
    assert result.exit_code == 0
    assert "Waiting for Slurm job completion..." in result.output
    fake_handle.wait.assert_called_once()


# ---------------------------------------------------------------------------
# run --no-wait prints job_id (legacy test name kept for reference)
# ---------------------------------------------------------------------------


def test_run_no_wait_prints_job_id(tmp_path, monkeypatch):
    """Slurm fire-and-forget path prints the submitted job ID."""
    from stilt.execution import SlurmHandle

    _write_minimal_config(tmp_path)

    fake_handle = SlurmHandle("42")
    fake_handle.wait = MagicMock()
    monkeypatch.setattr(
        "stilt.model.Model.run",
        lambda self, executor=None, skip_existing=None, rebuild=None, wait=True,: (
            fake_handle
        ),
    )

    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 0
    assert "Submitted job: 42" in result.output


# ---------------------------------------------------------------------------
# worker command
# ---------------------------------------------------------------------------


def test_pull_worker_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["pull-worker", str(tmp_path)])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# worker command
# ---------------------------------------------------------------------------


def test_pull_worker_calls_pull_worker_loop(tmp_path, monkeypatch):
    """pull-worker calls pull_simulations on the model."""
    _write_minimal_config(tmp_path)

    loop_calls: list[dict] = []

    def fake_loop(model, follow=False, poll_interval=10.0):
        loop_calls.append({"follow": follow})

    monkeypatch.setattr("stilt.cli.pull_simulations", fake_loop)

    result = runner.invoke(app, ["pull-worker", str(tmp_path)])
    assert result.exit_code == 0
    assert len(loop_calls) == 1
    assert loop_calls[0]["follow"] is False


def test_pull_worker_follow_flag_forwarded(tmp_path, monkeypatch):
    """--follow is forwarded to pull_simulations."""
    _write_minimal_config(tmp_path)

    loop_calls: list[dict] = []

    def fake_loop(model, follow=False, poll_interval=10.0):
        loop_calls.append({"follow": follow})

    monkeypatch.setattr("stilt.cli.pull_simulations", fake_loop)

    result = runner.invoke(app, ["pull-worker", str(tmp_path), "--follow"])
    assert result.exit_code == 0
    assert loop_calls[0]["follow"] is True


def test_pull_worker_accepts_cloud_project_uri(monkeypatch):
    """pull-worker can bootstrap from a cloud project ref."""
    captured: list[dict] = []

    class _FakeModel:
        def __init__(self, project, output_dir=None, compute_root=None):
            captured.append(
                {
                    "project": project,
                    "output_dir": output_dir,
                    "compute_root": compute_root,
                }
            )

    monkeypatch.setattr("stilt.cli.Model", _FakeModel)
    monkeypatch.setattr(
        "stilt.cli.pull_simulations",
        lambda model, follow=False, poll_interval=10.0: None,
    )

    result = runner.invoke(app, ["pull-worker", "gs://bucket/project"])

    assert result.exit_code == 0
    assert captured == [
        {
            "project": "gs://bucket/project",
            "output_dir": None,
            "compute_root": None,
        }
    ]


def test_pull_worker_forwards_output_dir_and_compute_root(tmp_path, monkeypatch):
    """pull-worker forwards durable-output and compute-root bootstrap options."""
    _write_minimal_config(tmp_path)
    captured: list[dict] = []

    class _FakeModel:
        def __init__(self, project, output_dir=None, compute_root=None):
            captured.append(
                {
                    "project": project,
                    "output_dir": output_dir,
                    "compute_root": compute_root,
                }
            )

    monkeypatch.setattr("stilt.cli.Model", _FakeModel)
    monkeypatch.setattr(
        "stilt.cli.pull_simulations",
        lambda model, follow=False, poll_interval=10.0: None,
    )

    result = runner.invoke(
        app,
        [
            "pull-worker",
            str(tmp_path),
            "--output-dir",
            "gs://bucket/project",
            "--compute-root",
            str(tmp_path / "scratch"),
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "project": str(tmp_path.resolve()),
            "output_dir": "gs://bucket/project",
            "compute_root": str(tmp_path / "scratch"),
        }
    ]


def test_push_worker_calls_push_simulations(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    chunk = tmp_path / "task_0.txt"
    chunk.write_text("hrrr_202301011200_abc\nhrrr_202301011200_def\n")

    sim_list_calls: list[dict] = []

    def fake_run(model, sim_ids, n_cores=1, skip_existing=None):
        sim_list_calls.append(
            {"sim_ids": sim_ids, "n_cores": n_cores, "skip_existing": skip_existing}
        )

    monkeypatch.setattr("stilt.cli.push_simulations", fake_run)

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


def test_serve_calls_worker_loop_in_follow_mode(tmp_path, monkeypatch):
    """serve is the user-facing long-lived queue consumer command."""
    _write_minimal_config(tmp_path)

    loop_calls: list[dict] = []

    def fake_loop(model, follow=False, poll_interval=10.0):
        loop_calls.append({"follow": follow})

    monkeypatch.setattr("stilt.cli.pull_simulations", fake_loop)

    result = runner.invoke(app, ["serve", str(tmp_path)])
    assert result.exit_code == 0
    assert loop_calls == [{"follow": True}]


def test_serve_accepts_cloud_project_uri(monkeypatch):
    captured: list[dict] = []

    class _FakeModel:
        def __init__(self, project, output_dir=None, compute_root=None):
            captured.append(
                {
                    "project": project,
                    "output_dir": output_dir,
                    "compute_root": compute_root,
                }
            )

    monkeypatch.setattr("stilt.cli.Model", _FakeModel)
    monkeypatch.setattr(
        "stilt.cli.pull_simulations",
        lambda model, follow=False, poll_interval=10.0: None,
    )

    result = runner.invoke(app, ["serve", "gs://bucket/project"])

    assert result.exit_code == 0
    assert captured == [
        {
            "project": "gs://bucket/project",
            "output_dir": None,
            "compute_root": None,
        }
    ]


def test_serve_forwards_output_dir_and_compute_root(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    captured: list[dict] = []
    loop_calls: list[dict] = []

    class _FakeModel:
        def __init__(self, project, output_dir=None, compute_root=None):
            captured.append(
                {
                    "project": project,
                    "output_dir": output_dir,
                    "compute_root": compute_root,
                }
            )

    def fake_loop(model, follow=False, poll_interval=10.0):
        loop_calls.append({"follow": follow})

    monkeypatch.setattr("stilt.cli.Model", _FakeModel)
    monkeypatch.setattr("stilt.cli.pull_simulations", fake_loop)

    result = runner.invoke(
        app,
        [
            "serve",
            str(tmp_path),
            "--output-dir",
            "gs://bucket/project",
            "--compute-root",
            str(tmp_path / "scratch"),
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "project": str(tmp_path.resolve()),
            "output_dir": "gs://bucket/project",
            "compute_root": str(tmp_path / "scratch"),
        }
    ]
    assert loop_calls == [{"follow": True}]


# ---------------------------------------------------------------------------
# rebuild command
# ---------------------------------------------------------------------------


def test_rebuild_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["rebuild", str(tmp_path)])
    assert result.exit_code == 1


def test_rebuild_calls_repository_rebuild(tmp_path, monkeypatch):
    """rebuild command calls state.rebuild() and prints status."""
    _write_minimal_config(tmp_path)

    rebuild_calls: list = []
    monkeypatch.setattr(
        "stilt.index.sqlite.SqliteIndex.rebuild",
        lambda self: rebuild_calls.append(True),
    )

    result = runner.invoke(app, ["rebuild", str(tmp_path)])
    assert result.exit_code == 0
    assert rebuild_calls  # was called at least once
    assert "total=" in result.output


def test_rebuild_accepts_output_dir_without_project_arg(tmp_path, monkeypatch):
    (tmp_path / "simulations").mkdir()

    rebuild_calls: list = []
    monkeypatch.setattr(
        "stilt.index.sqlite.SqliteIndex.rebuild",
        lambda self: rebuild_calls.append(True),
    )

    result = runner.invoke(app, ["rebuild", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert rebuild_calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path):
    """Write a minimal config.yaml so _resolve_project_dir succeeds."""
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


# ---------------------------------------------------------------------------
# register command
# ---------------------------------------------------------------------------


def test_register_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["register", str(tmp_path)])
    assert result.exit_code == 1


def test_register_registers_receptors(tmp_path, monkeypatch):
    """register command calls the model registration boundary and prints the count."""
    _write_minimal_config(tmp_path)

    register_calls: list = []

    def fake_register_pending(model, receptors=None, scene_id=None):
        del model
        register_calls.append({"receptors": receptors, "scene_id": scene_id})
        return ["sim_id_1", "sim_id_2"]

    monkeypatch.setattr("stilt.cli.Model.register_pending", fake_register_pending)

    result = runner.invoke(app, ["register", str(tmp_path)])
    assert result.exit_code == 0
    assert "2" in result.output
    assert len(register_calls) == 1


def test_register_with_receptors_file(tmp_path, monkeypatch):
    """--receptors PATH loads receptors from file."""
    _write_minimal_config(tmp_path)

    receptors_csv = tmp_path / "my_receptors.csv"
    receptors_csv.write_text(
        "time,longitude,latitude,altitude\n2023-01-01 12:00:00,-111.85,40.77,5.0\n"
    )

    register_calls: list = []

    def fake_register_pending(model, receptors=None, scene_id=None):
        del model
        register_calls.append({"receptors": receptors, "scene_id": scene_id})
        return ["sim_id_1"]

    monkeypatch.setattr("stilt.cli.Model.register_pending", fake_register_pending)

    result = runner.invoke(
        app, ["register", str(tmp_path), "--receptors", str(receptors_csv)]
    )
    assert result.exit_code == 0
    assert len(register_calls) == 1
    assert register_calls[0]["receptors"] is not None
    assert len(register_calls[0]["receptors"]) == 1


def test_register_forwards_scene_id(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)

    register_calls: list = []

    def fake_register_pending(model, receptors=None, scene_id=None):
        del model
        register_calls.append({"receptors": receptors, "scene_id": scene_id})
        return ["sim_id_1"]

    monkeypatch.setattr("stilt.cli.Model.register_pending", fake_register_pending)

    result = runner.invoke(app, ["register", str(tmp_path), "--scene-id", "scene-a"])

    assert result.exit_code == 0
    assert len(register_calls) == 1
    assert register_calls[0]["scene_id"] == "scene-a"
    assert register_calls[0]["receptors"] is not None


def test_status_filters_one_scene(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)

    class _FakeModel:
        def __init__(self, project, output_dir=None):
            self.project = "my_project"

        def status(self, scene_id=None):
            assert scene_id == "scene-a"
            return IndexCounts(total=2, completed=1, running=0, pending=1, failed=0)

    monkeypatch.setattr("stilt.cli.Model", _FakeModel)

    result = runner.invoke(app, ["status", str(tmp_path), "--scene-id", "scene-a"])

    assert result.exit_code == 0
    assert "Scene: scene-a" in result.output
    assert "total=2" in result.output


def test_status_groups_counts_by_scene(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)

    class _FakeModel:
        def __init__(self, project, output_dir=None):
            self.project = "my_project"

        def scene_counts(self):
            return {
                "scene-a": IndexCounts(
                    total=2, completed=1, running=0, pending=1, failed=0
                ),
                "scene-b": IndexCounts(
                    total=1, completed=0, running=0, pending=1, failed=0
                ),
            }

    monkeypatch.setattr("stilt.cli.Model", _FakeModel)

    result = runner.invoke(app, ["status", str(tmp_path), "--by-scene"])

    assert result.exit_code == 0
    assert "Scene: scene-a" in result.output
    assert "Scene: scene-b" in result.output


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


def test_init_aborts_when_config_exists(tmp_path):
    project = tmp_path / "existing"
    project.mkdir()
    (project / "config.yaml").write_text("n_hours: -24\n")
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 1
