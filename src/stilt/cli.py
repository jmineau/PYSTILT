"""STILT command-line interface.

Thin Typer wrapper over the model and runtime APIs. Each command loads a
project or durable output root, delegates to ``Model`` or execution helpers,
and prints a brief status summary. All heavy lifting lives in ``model.py``,
``service/``, and ``execution/``; this module has no orchestration logic of
its own.

Usage examples::

    stilt init                        # scaffold a new project in cwd
    stilt init ./my_project           # scaffold a new project in ./my_project
    stilt run                         # run locally, block until done
    stilt run ./my_project --no-skip  # re-run all simulations
    stilt run --wait                  # submit to Slurm and block until done
    stilt pull-worker ./my_project                  # drain pending simulations
    stilt push-worker ./my_project --chunk chunks/run_01/task_0.txt
    stilt serve ./my_project                         # long-lived streaming mode
    stilt register ./my_project --scene-id overpass_001       # register a scene group
    stilt rebuild                     # rebuild durable index from disk
    stilt status                      # show status from cwd
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import BaseModel

from stilt.config import STILTParams
from stilt.execution import (
    SlurmHandle,
    get_executor,
    pull_simulations,
    push_simulations,
    resolve_backend,
)
from stilt.model import Model
from stilt.receptor import read_receptors
from stilt.storage import (
    ProjectFiles,
    is_cloud_project,
)

app = typer.Typer(
    name="stilt",
    help="STILT model command-line interface.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


def _to_commented_yaml(model_cls: type[BaseModel], data: dict[str, Any]) -> str:
    """Render *data* as YAML with a ``#`` comment above each key that has a description."""
    fields = model_cls.model_fields
    lines: list[str] = []
    for key, value in data.items():
        field = fields.get(key)
        if field is not None and field.description:
            desc = " ".join(field.description.split())
            lines.append(f"# {desc}")
        lines.append(yaml.dump({key: value}, default_flow_style=False).rstrip())
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Shared options / arguments
# ---------------------------------------------------------------------------

_PROJECT_DIR_ARG = typer.Argument(
    None,
    help="Path or URI of the STILT project. Defaults to the current directory.",
)
_NEW_PROJECT_DIR_ARG = typer.Argument(
    None,
    help="Path to the new STILT project directory. Defaults to the current directory.",
)
_REQUIRED_PROJECT_DIR_ARG = typer.Argument(
    ..., help="Path or URI of the STILT project."
)
_NO_SKIP = typer.Option(
    False, "--no-skip", help="Re-run simulations that already have output."
)
_OUTPUT_DIR = typer.Option(
    None,
    "--output-dir",
    help="Durable output root. May be a local path or object-storage URI.",
)
_COMPUTE_ROOT = typer.Option(
    None,
    "--compute-root",
    help="Parent directory under which worker simulation dirs are created.",
)


def _resolve_project_dir(
    path: str | Path | None, *, require_inputs: bool = True
) -> str:
    """Resolve a local root path or pass through a cloud project/output URI."""
    raw = str(path or Path.cwd())
    if is_cloud_project(raw):
        return raw

    resolved = Path(raw).resolve()
    has_inputs = (resolved / "config.yaml").exists()
    files = ProjectFiles(resolved)
    has_index = files.simulations_dir.exists() or files.index_db_path.exists()
    if require_inputs and not has_inputs:
        typer.echo(
            f"Error: '{resolved}' does not look like a STILT project directory "
            "(no config.yaml found).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not require_inputs and not (has_inputs or has_index):
        typer.echo(
            f"Error: '{resolved}' does not look like a STILT project or durable output root.",
            err=True,
        )
        raise typer.Exit(code=1)
    return str(resolved)


def _resolve_model_root(
    project_dir: str | Path | None,
    output_dir: str | None,
    *,
    require_inputs: bool,
) -> tuple[str, str | None]:
    """Return the root passed to Model plus any separate durable output override."""
    root = project_dir if project_dir is not None else output_dir
    resolved = _resolve_project_dir(root, require_inputs=require_inputs)
    if project_dir is not None:
        return resolved, output_dir
    return resolved, None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    project_dir: Path = _NEW_PROJECT_DIR_ARG,
) -> None:
    """Scaffold a new STILT project directory with a default config.yaml.

    Creates a starter config.yaml and receptors.csv. Edit both files
    before running ``stilt run``.

    Examples
    --------
    ::

        stilt init
        stilt init /path/to/project
    """
    project_dir = (project_dir or Path.cwd()).resolve()
    files = ProjectFiles(project_dir)
    config_path = files.config_path
    receptors_path = files.receptors_path

    if config_path.exists():
        typer.echo(
            f"Error: '{project_dir}' already contains a config.yaml. Aborting.",
            err=True,
        )
        raise typer.Exit(code=1)

    project_dir.mkdir(parents=True, exist_ok=True)

    stilt_yaml = _to_commented_yaml(STILTParams, {"n_hours": -24, "numpar": 1000})
    config_path.write_text(
        "# STILT project configuration\n"
        "# Edit mets.hrrr.directory to point to your ARL meteorology files.\n"
        "\n" + stilt_yaml + "\nmets:\n"
        "  hrrr:\n"
        "    directory: /path/to/arl/meteorology\n"
        '    file_format: "%Y%m%d_%H"\n'
        "    file_tres: 1h\n"
        "\n"
        "# execution:\n"
        "#   backend: local  # increase n_workers for parallel; use 'slurm' for HPC\n"
        "#   n_workers: 1\n"
    )

    receptors_path.write_text(
        "time,longitude,latitude,altitude\n"
        "# Example: 2023-01-01 12:00:00,-111.85,40.77,5\n"
    )

    typer.echo(f"Initialized STILT project at '{project_dir}'")
    typer.echo("  config.yaml   — edit met directory and footprint settings")
    typer.echo("  receptors.csv — add receptor times/locations here")


@app.command()
def run(
    project_dir: str | None = _PROJECT_DIR_ARG,
    config_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        help="Path to config.yaml. Project dir defaults to its parent.",
    ),
    no_skip: bool = _NO_SKIP,
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Override execution backend: local | slurm.",
    ),
    n_workers: int | None = typer.Option(
        None,
        "--n-workers",
        help="Override number of workers (overrides config.yaml execution.n_workers).",
    ),
    wait: bool = typer.Option(
        False,
        "--wait/--no-wait",
        help=(
            "Block until submitted Slurm jobs finish before returning. "
            "By default, ``stilt run`` returns immediately after ``sbatch`` when "
            "using backend: slurm (fire-and-forget). Local backends always "
            "complete inline regardless of this flag."
        ),
    ),
    rebuild: bool | None = typer.Option(  # noqa: B008
        None,
        "--rebuild/--no-rebuild",
        help=(
            "Rebuild the durable index from outputs before planning. "
            "Defaults to auto: enabled when skip-existing is in effect."
        ),
    ),
    output_dir: str | None = _OUTPUT_DIR,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Run trajectories (and footprints if configured).

    Reads ``config.yaml`` in the project directory.  If footprint configs are
    defined there, footprints are generated after trajectories.

    For ``backend: local`` (default), the command always blocks until all
    simulations complete.  For ``backend: slurm``, the command submits the
    job array and returns immediately — use ``--wait`` to poll until done.
    Pass ``--no-skip`` to re-run simulations that already have output.
    """
    # Resolve project dir: --config parent takes precedence over positional arg.
    if config_path is not None:
        resolved_dir = str(config_path.resolve().parent)
        resolved_output = output_dir
        if (
            not (Path(resolved_dir) / "config.yaml").exists()
            and not config_path.exists()
        ):
            typer.echo(f"Error: config file not found: {config_path}", err=True)
            raise typer.Exit(code=1)
    else:
        resolved_dir, resolved_output = _resolve_model_root(
            project_dir, output_dir, require_inputs=True
        )

    model = Model(
        project=resolved_dir,
        output_dir=resolved_output,
        compute_root=compute_root,
    )

    # Build executor override if --backend or --n-workers are given.
    executor = None
    execution = dict(model.config.execution or {})
    if backend is not None or n_workers is not None:
        if backend is not None:
            execution["backend"] = backend
        if n_workers is not None:
            execution["n_workers"] = n_workers
        executor = get_executor(execution)

    skip_existing = None if not no_skip else False
    _print_run_start(
        model,
        execution=execution,
        skip_existing=skip_existing,
        wait=wait,
    )
    handle = model.run(
        executor=executor,
        skip_existing=skip_existing,
        rebuild=rebuild,
        wait=False,
    )
    if isinstance(handle, SlurmHandle):
        typer.echo(f"Submitted job: {handle.job_id}")
        if wait:
            typer.echo("Waiting for Slurm job completion...")
            _wait_with_progress(model, handle)
            _print_status(model)
    else:
        # Local backend — always complete inline so no orphan workers.
        typer.echo("Workers launched. Waiting for completion...")
        _wait_with_progress(model, handle)
        _print_status(model)


@app.command("register")
def register(
    project_dir: str = _REQUIRED_PROJECT_DIR_ARG,
    receptors_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--receptors",
        help=(
            "Path to a receptors CSV to register. "
            "Defaults to the project's receptors.csv."
        ),
    ),
    output_dir: str | None = _OUTPUT_DIR,
    scene_id: str | None = typer.Option(
        None,
        "--scene-id",
        help="Optional grouping identifier for this scene submission.",
    ),
) -> None:
    """Register pending work with an optional scene grouping."""
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=True
    )
    model = Model(project=resolved_dir, output_dir=resolved_output)

    if receptors_path is not None:
        receptors = read_receptors(receptors_path)
    else:
        receptors = model.receptors

    registration = model.register_pending(receptors=receptors, scene_id=scene_id)
    typer.echo(f"Registered {len(registration)} simulation(s).")


@app.command("pull-worker")
def pull_worker(
    project_dir: str = _REQUIRED_PROJECT_DIR_ARG,
    follow: bool = typer.Option(
        False,
        "--follow/--no-follow",
        help=(
            "Keep polling when the queue is empty. "
            "Use for long-lived streaming deployments (Kubernetes follow mode)."
        ),
    ),
    output_dir: str | None = _OUTPUT_DIR,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Drain pending simulations from the durable project index.

    Atomically pulls and processes simulations until the queue is empty
    (batch mode) or indefinitely (``--follow``).

    Examples
    --------
    ::

        stilt pull-worker ./hrrr_24h
        stilt pull-worker ./hrrr_24h --follow
    """
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=True
    )
    model = Model(
        project=resolved_dir,
        output_dir=resolved_output,
        compute_root=compute_root,
    )
    pull_simulations(model, follow=follow)


@app.command("push-worker")
def push_worker(
    project_dir: str = _REQUIRED_PROJECT_DIR_ARG,
    chunk: str = typer.Option(
        ...,
        "--chunk",
        help="Path to one immutable chunk file.",
    ),
    cpus: int = typer.Option(
        1, "--cpus", help="Number of CPU cores to use for within-task parallelism."
    ),
    skip_existing: bool | None = typer.Option(  # noqa: B008
        None,
        "--skip-existing/--no-skip-existing",
        help=(
            "Respect durable outputs that already exist. "
            "Defaults to config.yaml when omitted."
        ),
    ),
    output_dir: str | None = _OUTPUT_DIR,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Run one immutable chunk shard without queue polling or heartbeats."""
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=True
    )
    model = Model(
        project=resolved_dir, output_dir=resolved_output, compute_root=compute_root
    )
    sim_ids = [
        s for line in Path(chunk).read_text().splitlines() if (s := line.strip())
    ]
    push_simulations(model, sim_ids, n_cores=cpus, skip_existing=skip_existing)


@app.command()
def serve(
    project_dir: str = _REQUIRED_PROJECT_DIR_ARG,
    output_dir: str | None = _OUTPUT_DIR,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Run long-lived queue workers that keep polling for new simulations.

    This is the user-facing streaming consumer command. It is equivalent to
    ``stilt pull-worker --follow`` but uses language that better matches the
    deployment model for always-on queue consumers.

    Examples
    --------
    ::

        stilt serve ./hrrr_24h
    """
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=True
    )
    model = Model(
        project=resolved_dir,
        output_dir=resolved_output,
        compute_root=compute_root,
    )
    pull_simulations(model, follow=True)


@app.command()
def rebuild(
    project_dir: str | None = _PROJECT_DIR_ARG,
    output_dir: str | None = _OUTPUT_DIR,
) -> None:
    """Rebuild durable index rows by scanning simulation output on disk.

    Useful after manual file operations or interrupted runs that left the
    SQLite database out of sync with the filesystem.
    """
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=False
    )
    model = Model(project=resolved_dir, output_dir=resolved_output)
    model.index.rebuild()
    _print_status(model)


@app.command()
def status(
    project_dir: str | None = _PROJECT_DIR_ARG,
    output_dir: str | None = _OUTPUT_DIR,
    scene_id: str | None = typer.Option(
        None,
        "--scene-id",
        help="Show counts for one registered scene only.",
    ),
    by_scene: bool = typer.Option(
        False,
        "--by-scene",
        help="Print grouped counts for every registered scene.",
    ),
) -> None:
    """Show simulation counts for a project."""
    resolved_dir, resolved_output = _resolve_model_root(
        project_dir, output_dir, require_inputs=False
    )
    model = Model(project=resolved_dir, output_dir=resolved_output)
    _print_status(model, scene_id=scene_id, by_scene=by_scene)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_status(
    model: Model,
    *,
    scene_id: str | None = None,
    by_scene: bool = False,
) -> None:
    """Print a project status summary."""
    if by_scene:
        grouped = model.scene_counts()
        if not grouped:
            typer.echo("Scenes: none")
            return
        for name, status in grouped.items():
            typer.echo(
                f"Scene: {name}  total={status.total}  completed={status.completed}  "
                f"running={status.running}  pending={status.pending}  failed={status.failed}"
            )
        return
    status = model.status(scene_id=scene_id)
    label = (
        f"Scene: {scene_id}" if scene_id is not None else f"Project: {model.project}"
    )
    typer.echo(
        f"{label}  total={status.total}  completed={status.completed}  "
        f"running={status.running}  pending={status.pending}  failed={status.failed}"
    )


def _print_run_start(
    model: Model,
    *,
    execution: dict[str, Any],
    skip_existing: bool | None,
    wait: bool,
) -> None:
    """Print a concise startup summary for ``stilt run``."""
    backend = resolve_backend(execution)
    executor = get_executor(execution)
    dispatch = executor.dispatch
    project_root = model.layout.project_root
    output_root = model.layout.output_root
    local_output_dir = model.layout.output_dir
    is_cloud_output = model.layout.is_cloud_output
    compute_root = model.compute_root
    receptors = model.receptors
    worker_count = executor.n_workers
    mode = "config" if skip_existing is None else "no-skip"
    typer.echo(
        "Starting run: "
        f"project={project_root}  backend={backend}  dispatch={dispatch}  "
        f"workers={worker_count}  skip={mode}"
    )
    if output_root is not None and output_root != project_root:
        typer.echo(f"Output root: {output_root}")
    if compute_root is not None:
        if is_cloud_output or local_output_dir is None:
            typer.echo(f"Compute root: {compute_root}")
        else:
            default_compute_root = Path(str(local_output_dir)) / "simulations" / "by-id"
            if str(compute_root) != str(default_compute_root):
                typer.echo(f"Compute root: {compute_root}")
    typer.echo(f"Receptors loaded: {len(receptors)}")
    typer.echo(
        "Execution mode: " + ("submit-and-wait" if wait else "submit-and-return")
        if backend == "slurm"
        else "Execution mode: local-blocking"
    )


def _format_progress_line(model: Model) -> str | None:
    """Return a one-line progress summary, or ``None`` if unavailable."""
    try:
        status = model.status()
    except Exception:
        return None
    return (
        "Progress: "
        f"total={status.total}  completed={status.completed}  "
        f"running={status.running}  pending={status.pending}  failed={status.failed}"
    )


def _wait_with_progress(
    model: Model,
    handle: object,
    *,
    poll_interval: float = 5.0,
) -> None:
    """Wait on a job handle while periodically printing queue progress."""
    done = threading.Event()
    errors: list[BaseException] = []

    def _wait() -> None:
        try:
            handle.wait()  # type: ignore[attr-defined]
        except BaseException as exc:  # pragma: no cover - re-raised in caller thread
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_wait, daemon=True)
    thread.start()

    last_line: str | None = None
    initial_line = _format_progress_line(model)
    if initial_line is not None:
        typer.echo(initial_line)
        last_line = initial_line

    while not done.wait(timeout=poll_interval):
        line = _format_progress_line(model)
        if line is None:
            typer.echo("Progress: still running...")
            continue
        if line != last_line:
            typer.echo(line)
            last_line = line
        else:
            typer.echo(f"{line}  (no change)")

    thread.join()
    if errors:
        raise errors[0]
