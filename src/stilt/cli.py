"""
STILT command-line interface.

Thin Typer wrapper over :class:`~stilt.model.Model` and the worker functions.
Each command loads a project root, delegates, and prints a brief summary.

Usage examples::

    stilt init                        # scaffold a new project in cwd
    stilt init ./my_project           # scaffold a new project in ./my_project
    stilt run                         # run locally, block until done
    stilt run ./my_project --no-skip  # re-run all simulations
    stilt run --wait                  # submit to Slurm and block until done
    stilt register ./my_project       # persist inputs / seed the work queue
    stilt push-worker ./my_project --chunk chunks/run_01/task_0.txt
    stilt pull-worker ./my_project    # drain the Postgres work queue
    stilt serve ./my_project          # long-lived queue worker
    stilt status                      # show completion counts from cwd
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import typer

from stilt.execution import (
    get_executor,
    pull_simulations,
    resolve_backend,
    run_simulations,
)
from stilt.model import Model
from stilt.project import CONFIG_KEY, RECEPTORS_KEY, SIMULATIONS_PREFIX
from stilt.receptors import read_receptors
from stilt.store import is_uri

app = typer.Typer(
    name="stilt",
    help="STILT model command-line interface.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


def _starter_config_yaml() -> str:
    """Return the science-first commented starter config written by ``stilt init``."""
    return """# PYSTILT project configuration
# See docs for details: https://jmineau.github.io/PYSTILT


# Meteorology sources. Edit directory to point to your ARL files.
mets:
  hrrr:  # Unique name for this met source.
    directory: /path/to/arl/meteorology
    file_format: "%Y%m%d_%H"
    file_tres: 1h  # Temporal resolution of met files.


# Named footprint products to compute for each receptor.
# For multiple footprints, define reusable top-level grids and reference them by name.
footprints:
  default:  # Name of this footprint product.
    xmin: -113.0
    xmax: -110.5
    ymin: 40.0
    ymax: 42.0
    xres: 0.01
    yres: 0.01


# Common run controls. Negative n_hours means backward in time.
n_hours: -24
numpar: 1000
varsiwant: [time, indx, long, lati, zagl, foot, mlht, pres, dens, samt, sigw, tlgr]
hnf_plume: true  # rescale footprints via a gaussian plume model in the hyper-near field
skip_existing: true  # skip simulation outputs that already exist


# Execution is optional. Local execution is the default.
# execution:
#   backend: local  # or "slurm"
#   n_workers: 1
"""


# ---------------------------------------------------------------------------
# Shared options / arguments
# ---------------------------------------------------------------------------

_PROJECT_ARG = typer.Argument(
    None,
    help="Path or URI of the STILT project. Defaults to the current directory.",
)
_NEW_PROJECT_ARG = typer.Argument(
    None,
    help="Path to the new STILT project directory. Defaults to the current directory.",
)
_REQUIRED_PROJECT_ARG = typer.Argument(..., help="Path or URI of the STILT project.")
_NO_SKIP = typer.Option(
    False, "--no-skip", help="Re-run simulations that already have output."
)
_COMPUTE_ROOT = typer.Option(
    None,
    "--compute-root",
    help="Parent directory under which worker simulation dirs are created.",
)


def _resolve_project(path: str | Path | None, *, require_inputs: bool = True) -> str:
    """Resolve a local project root, or pass a cloud URI through unchanged."""
    raw = str(path or Path.cwd())
    if is_uri(raw):
        return raw

    resolved = Path(raw).resolve()
    has_inputs = (resolved / CONFIG_KEY).exists()
    has_outputs = (resolved / SIMULATIONS_PREFIX).exists()
    if require_inputs and not has_inputs:
        typer.echo(
            f"Error: '{resolved}' does not look like a STILT project directory "
            "(no config.yaml found).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not require_inputs and not (has_inputs or has_outputs):
        typer.echo(
            f"Error: '{resolved}' does not look like a STILT project.",
            err=True,
        )
        raise typer.Exit(code=1)
    return str(resolved)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(project: Path = _NEW_PROJECT_ARG) -> None:
    """
    Scaffold a new STILT project directory with a default config.yaml.

    Creates a starter config.yaml and receptors.csv. Edit both files
    before running ``stilt run``.
    """
    project = (project or Path.cwd()).resolve()
    config_path = project / CONFIG_KEY
    receptors_path = project / RECEPTORS_KEY

    if config_path.exists():
        typer.echo(
            f"Error: '{project}' already contains a config.yaml. Aborting.",
            err=True,
        )
        raise typer.Exit(code=1)

    project.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_starter_config_yaml())
    receptors_path.write_text(
        "time,longitude,latitude,altitude\n"
        "# Example: 2023-01-01 12:00:00,-111.85,40.77,5\n"
    )

    typer.echo(f"Initialized STILT project at '{project}'")
    typer.echo("  config.yaml   — edit met directory and footprint settings")
    typer.echo("  receptors.csv — add receptor times/locations here")


@app.command()
def run(
    project: str | None = _PROJECT_ARG,
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
            "By default ``stilt run`` returns right after ``sbatch`` for "
            "backend: slurm. Local runs always complete inline."
        ),
    ),
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """
    Run trajectories (and footprints if configured).

    Reads ``config.yaml`` in the project directory. For ``backend: local``
    (default) the command blocks until all simulations complete. For
    ``backend: slurm`` it submits the job array and returns — use ``--wait``
    to poll until done. Pass ``--no-skip`` to re-run existing simulations.
    """
    if config_path is not None:
        resolved = str(config_path.resolve().parent)
        if not (Path(resolved) / CONFIG_KEY).exists() and not config_path.exists():
            typer.echo(f"Error: config file not found: {config_path}", err=True)
            raise typer.Exit(code=1)
    else:
        resolved = _resolve_project(project, require_inputs=True)

    model = Model(project=resolved, compute_root=compute_root)

    executor = None
    execution = dict(model.config.execution or {})
    if backend is not None or n_workers is not None:
        if backend is not None:
            execution["backend"] = backend
        if n_workers is not None:
            execution["n_workers"] = n_workers
        executor = get_executor(execution)

    skip_existing = None if not no_skip else False
    _print_run_start(model, execution=execution, skip_existing=skip_existing, wait=wait)
    handle = model.run(executor=executor, skip_existing=skip_existing, wait=False)

    if handle.detached:
        typer.echo(f"Submitted job: {handle.job_id}")
        if wait:
            typer.echo("Waiting for job completion...")
            _wait_with_progress(model, handle)
            _print_status(model)
    else:
        typer.echo("Workers launched. Waiting for completion...")
        _wait_with_progress(model, handle)
        _print_status(model)


@app.command("register")
def register(
    project: str = _REQUIRED_PROJECT_ARG,
    receptors_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--receptors",
        help="Receptors CSV to add to the project. Defaults to the project's own.",
    ),
) -> None:
    """Persist project inputs and, when a queue is configured, enqueue simulations."""
    model = Model(project=_resolve_project(project, require_inputs=True))
    receptors = read_receptors(receptors_path) if receptors_path is not None else None
    sim_ids = model.register(receptors=receptors)
    typer.echo(f"Registered {len(sim_ids)} simulation(s).")


@app.command("pull-worker")
def pull_worker(
    project: str = _REQUIRED_PROJECT_ARG,
    follow: bool = typer.Option(
        False,
        "--follow/--no-follow",
        help="Keep polling when the queue is empty (long-lived deployments).",
    ),
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """
    Drain pending simulations from the Postgres work queue.

    Atomically claims and runs simulations until the queue is empty
    (batch mode) or indefinitely (``--follow``).
    """
    model = Model(
        project=_resolve_project(project, require_inputs=True),
        compute_root=compute_root,
    )
    pull_simulations(model, follow=follow)


@app.command("push-worker")
def push_worker(
    project: str = _REQUIRED_PROJECT_ARG,
    chunk: str = typer.Option(..., "--chunk", help="Path to one chunk file."),
    cpus: int = typer.Option(
        1, "--cpus", help="Number of CPU cores to use within this task."
    ),
    skip_existing: bool | None = typer.Option(  # noqa: B008
        None,
        "--skip-existing/--no-skip-existing",
        help="Respect outputs that already exist. Defaults to config.yaml.",
    ),
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Run the simulation ids listed in one chunk file (one per line)."""
    model = Model(
        project=_resolve_project(project, require_inputs=True),
        compute_root=compute_root,
    )
    sim_ids = [
        s for line in Path(chunk).read_text().splitlines() if (s := line.strip())
    ]
    run_simulations(model, sim_ids, n_cores=cpus, skip_existing=skip_existing)


@app.command()
def serve(
    project: str = _REQUIRED_PROJECT_ARG,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Run a long-lived queue worker (equivalent to ``pull-worker --follow``)."""
    model = Model(
        project=_resolve_project(project, require_inputs=True),
        compute_root=compute_root,
    )
    pull_simulations(model, follow=True)


@app.command()
def status(project: str | None = _PROJECT_ARG) -> None:
    """Show simulation completion counts for a project."""
    model = Model(project=_resolve_project(project, require_inputs=True))
    _print_status(model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_counts(model: Model) -> str:
    """Format a model's simulation counts for one status line."""
    counts = model.status()
    return (
        f"total={counts.total}  completed={counts.completed}  pending={counts.pending}"
    )


def _print_status(model: Model) -> None:
    """Print a project status summary."""
    typer.echo(f"Project: {model.project.root}  {_format_counts(model)}")


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
    mode = "config" if skip_existing is None else "no-skip"
    typer.echo(
        "Starting run: "
        f"project={model.project.root}  backend={backend}  "
        f"dispatch={executor.dispatch}  workers={executor.n_workers}  skip={mode}"
    )
    default_compute_root = (
        None if model.project.is_cloud else model.project.simulations_dir
    )
    if model.compute_root != default_compute_root:
        typer.echo(f"Compute root: {model.compute_root}")
    typer.echo(f"Receptors loaded: {len(model.receptors)}")
    typer.echo(
        "Execution mode: " + ("submit-and-wait" if wait else "submit-and-return")
        if backend == "slurm"
        else "Execution mode: local-blocking"
    )


def _format_progress_line(model: Model) -> str | None:
    """Return a one-line progress summary, or ``None`` if unavailable."""
    try:
        return f"Progress: {_format_counts(model)}"
    except Exception:
        return None


def _wait_with_progress(
    model: Model,
    handle: object,
    *,
    poll_interval: float = 5.0,
) -> None:
    """Wait on a job handle while periodically printing progress."""
    done = threading.Event()
    errors: list[BaseException] = []

    def _wait() -> None:
        """Wait on the handle in a worker thread, storing any exception."""
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
