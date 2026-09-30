"""
The ``stilt`` command-line interface.

Each command opens a project with :class:`~stilt.Model`, calls it, and
prints a short summary. Examples::

    stilt init                        # start a project in the current directory
    stilt init ./my_project           # start a project in ./my_project
    stilt run                         # run locally and wait until done
    stilt run ./my_project --no-skip  # run every simulation again
    stilt run --wait                  # with Slurm, wait for the jobs to finish
    stilt register ./my_project       # save inputs and fill the work queue
    stilt push-worker ./my_project --chunk chunks/run_01/task_0.txt
    stilt pull-worker ./my_project    # run receptors from the Postgres queue
    stilt serve ./my_project          # keep taking work from the queue
    stilt status                      # count finished simulations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

from stilt.execution import (
    Executor,
    get_executor,
    pull_receptors,
    resolve_compute_root,
    run_receptors,
)
from stilt.execution import register as register_inputs
from stilt.model import Model
from stilt.project import CONFIG_KEY, RECEPTORS_KEY
from stilt.receptors import read_receptors

app = typer.Typer(
    name="stilt",
    help="Run STILT simulations and check on a project.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


def _starter_config_yaml() -> str:
    """Return the commented starter ``config.yaml`` written by ``stilt init``."""
    return """# PYSTILT project configuration
# See docs for details: https://jmineau.github.io/PYSTILT


# Meteorology, by name. Edit directory to point to your ARL files, or replace
# file_format and file_tres with "download: hrrr" to download them from NOAA.
mets:
  hrrr:  # Unique name for this met.
    directory: /path/to/arl/meteorology
    file_format: "%Y%m%d_%H"
    file_tres: 6h  # Hours each met file covers; the docs' HRRR files hold six.


# Footprint grid. Remove it (or set grid: null) for trajectory-only runs.
grid:
  xmin: -113.0
  xmax: -110.5
  ymin: 40.0
  ymax: 42.0
  xres: 0.01
  yres: 0.01


# Variants. Every receptor runs once per variant, with the settings in this
# file as the defaults. An entry with no overrides runs the defaults as they
# are; add others to run the same receptors under other settings, e.g. a
# mixed-layer bracket or a wind-error ensemble (see the docs). Only the
# variants listed here run.
variants:
  hrrr: {}
#  hrrr-zi08: {ziscale: 0.8}


# Common run controls. Negative n_hours means backward in time.
n_hours: -24
numpar: 1000
varsiwant: [time, indx, long, lati, zagl, foot, mlht, pres, dens, samt, sigw, tlgr]
hnf_plume: true  # rescale footprints via a gaussian plume model in the hyper-near field


# Results go to this directory, relative to the project unless absolute.
# Several projects can name the same directory and share runs.
output: ./output


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
    help="Directory for the new project. Defaults to the current directory.",
)
_REQUIRED_PROJECT_ARG = typer.Argument(..., help="Path or URI of the STILT project.")
_NO_SKIP = typer.Option(
    False, "--no-skip", help="Run simulations again even if their outputs exist."
)
_COMPUTE_ROOT = typer.Option(
    None,
    "--compute-root",
    help="Scratch directory HYSPLIT runs under. Defaults to PYSTILT_COMPUTE_ROOT, then $TMPDIR/pystilt/<project>.",
)


def _resolve_project(path: str | Path | None) -> str:
    """Return the absolute project path, exiting if it has no config.yaml."""
    raw = str(path or Path.cwd())
    resolved = Path(raw).resolve()
    if not (resolved / CONFIG_KEY).exists():
        typer.echo(
            f"Error: '{resolved}' does not look like a STILT project directory "
            "(no config.yaml found).",
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
    Create a new project with a starter config.yaml and receptors.csv.

    Edit both files before running stilt run. Stops if the directory
    already has a config.yaml.
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
    # Header only: every other line of a receptors file is read as a receptor.
    receptors_path.write_text("time,longitude,latitude,altitude\n")

    typer.echo(f"Initialized STILT project at '{project}'")
    typer.echo("  config.yaml   — edit met directory and footprint settings")
    typer.echo("  receptors.csv — add receptor times/locations here, one per line:")
    typer.echo("                  2023-01-01 12:00:00,-111.85,40.77,5")


@app.command()
def run(
    project: str | None = _PROJECT_ARG,
    no_skip: bool = _NO_SKIP,
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Where to run: local or slurm. Overrides execution.backend in config.yaml.",
    ),
    n_workers: int | None = typer.Option(
        None,
        "--n-workers",
        help="Number of workers. Overrides execution.n_workers in config.yaml.",
    ),
    wait: bool = typer.Option(
        False,
        "--wait/--no-wait",
        help=(
            "Wait for submitted Slurm jobs to finish. Without it, a Slurm run "
            "returns once the jobs are submitted. Local runs always wait."
        ),
    ),
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """
    Run every unfinished simulation in a project.

    Runs HYSPLIT for each receptor and variant, then the footprint when the
    variant has a grid. Simulations whose outputs exist are skipped unless
    --no-skip is given. A local run returns when all simulations are done.
    A Slurm run submits a job array and returns. Add --wait to wait for it.
    """
    resolved = _resolve_project(project)
    model = Model(project=resolved)
    # Progress is the worker's one line per finished receptor.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("stilt.execution.worker").setLevel(logging.INFO)

    execution = dict(model.config.execution or {})
    if backend is not None:
        execution["backend"] = backend
    if n_workers is not None:
        execution["n_workers"] = n_workers
    executor = get_executor(execution)

    _print_run_start(
        model,
        executor,
        backend=execution.get("backend", "local"),
        compute_root=compute_root,
        skip_existing=not no_skip,
        wait=wait,
    )
    # A local run has finished when this returns; a Slurm job has been submitted.
    handle = model.run(
        executor=executor,
        skip_existing=not no_skip,
        wait=False,
        compute_root=compute_root,
    )
    if handle.detached:
        typer.echo(f"Submitted job: {handle.job_id}")
        if not wait:
            return
        typer.echo("Waiting for job completion (squeue shows its tasks)...")
        handle.wait()
    _print_status(model)


@app.command("register")
def register(
    project: str = _REQUIRED_PROJECT_ARG,
    receptors_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--receptors",
        help="Receptors CSV to add to the project. Defaults to the project's receptors.csv.",
    ),
) -> None:
    """
    Save a project's settings and receptors, and queue its receptors.

    Receptors go to the Postgres work queue only when PYSTILT_DB_URL is set.
    """
    model = Model(project=_resolve_project(project))
    receptors = read_receptors(receptors_path) if receptors_path is not None else None
    receptor_ids = register_inputs(model, receptors=receptors)
    typer.echo(
        f"Registered {len(receptor_ids)} receptor(s) x {len(model.variants)} variant(s)."
    )


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
    Run receptors from the Postgres work queue.

    Each receptor is claimed by one worker only. The worker stops when the
    queue is empty, or keeps waiting for more work with --follow. Needs
    PYSTILT_DB_URL.
    """
    model = Model(project=_resolve_project(project))
    pull_receptors(model, follow=follow, compute_root=compute_root)


@app.command("push-worker")
def push_worker(
    project: str = _REQUIRED_PROJECT_ARG,
    chunk: str = typer.Option(
        ..., "--chunk", help="File listing the receptor ids to run, one per line."
    ),
    cpus: int = typer.Option(
        1, "--cpus", help="Number of receptors to run at once in this task."
    ),
    no_skip: bool = _NO_SKIP,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """
    Run the receptors listed in one chunk file.

    Slurm array tasks call this, one chunk file per task.
    """
    model = Model(project=_resolve_project(project))
    receptor_ids = [
        s for line in Path(chunk).read_text().splitlines() if (s := line.strip())
    ]
    run_receptors(
        model,
        receptor_ids,
        compute_root=compute_root,
        n_cores=cpus,
        skip_existing=not no_skip,
    )


@app.command()
def serve(
    project: str = _REQUIRED_PROJECT_ARG,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """Keep running receptors from the work queue. Same as pull-worker --follow."""
    model = Model(project=_resolve_project(project))
    pull_receptors(model, follow=True, compute_root=compute_root)


@app.command()
def status(project: str | None = _PROJECT_ARG) -> None:
    """Count finished and pending simulations, per variant when there are several."""
    model = Model(project=_resolve_project(project))
    _print_status(model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _counts(table: Any) -> str:
    """Return a ``total / completed / pending`` line for a status table."""
    done = int(table["complete"].sum())
    return f"total={len(table)}  completed={done}  pending={len(table) - done}"


def _print_status(model: Model) -> None:
    """Print a project status summary, per variant when there are several."""
    table = model.status()
    typer.echo(f"Project: {model.project.root}  {_counts(table)}")
    if len(model.variants) > 1:
        for variant, rows in table.groupby("variant", sort=False):
            typer.echo(f"  {variant}: {_counts(rows)}")
    unreferenced = model.unreferenced()
    for kind, keys in unreferenced.items():
        if keys:
            typer.echo(
                f"{kind} folders in {model.output.path} that no variant here uses: "
                f"{', '.join('settings=' + k for k in keys)}  (from changed settings, "
                "dropped variants, or another project; PYSTILT never deletes them)"
            )


def _print_run_start(
    model: Model,
    executor: Executor,
    *,
    backend: str,
    compute_root: str | None,
    skip_existing: bool,
    wait: bool,
) -> None:
    """Print the settings ``stilt run`` is about to use."""
    mode = "existing" if skip_existing else "no-skip"
    typer.echo(
        "Starting run: "
        f"project={model.project.root}  backend={backend}  "
        f"dispatch={executor.dispatch}  workers={executor.n_workers}  skip={mode}"
    )
    typer.echo(f"Output: {model.output.path}")
    typer.echo(f"Compute root: {resolve_compute_root(model.project, compute_root)}")
    typer.echo(f"Receptors loaded: {len(model.receptors)}")
    typer.echo(f"Variants: {', '.join(model.variants)}")
    typer.echo(
        "Execution mode: " + ("submit-and-wait" if wait else "submit-and-return")
        if backend != "local"
        else "Execution mode: local, one line per receptor as it finishes"
    )
