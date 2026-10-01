"""
The ``stilt`` command-line interface.

Each command opens a :class:`~stilt.Project`, calls it, and
prints a short summary. Examples::

    stilt init                        # start a project in the current directory
    stilt init ./my_project           # start a project in ./my_project
    stilt run                         # run locally and wait until done
    stilt run ./my_project --no-skip  # run every simulation again
    stilt run --backend slurm         # submit a Slurm job array and return
    stilt run --wait                  # with Slurm, wait for the jobs to finish
    stilt status                      # count finished simulations
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from stilt.config import ExecutionConfig
from stilt.execution import resolve_compute_root
from stilt.project import CONFIG_KEY, RECEPTORS_KEY, Project

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
#   cpus: 1         # receptors at once (per array task on Slurm)
#   n_workers: 1    # Slurm array tasks
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
        help="Number of Slurm array tasks. Overrides execution.n_workers in config.yaml.",
    ),
    cpus: int | None = typer.Option(
        None,
        "--cpus",
        help=(
            "Receptors each task runs at once (local processes, or CPUs per "
            "Slurm task). Overrides execution.cpus in config.yaml."
        ),
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
    opened = Project(_resolve_project(project))
    # Progress is the worker's one line per finished receptor.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("stilt.execution.worker").setLevel(logging.INFO)

    overrides: dict[str, Any] = {}
    if backend is not None:
        overrides["backend"] = backend
    if n_workers is not None:
        overrides["n_workers"] = n_workers
    if cpus is not None:
        overrides["cpus"] = cpus
    execution = ExecutionConfig.model_validate(
        {**opened.config.execution.model_dump(exclude_unset=True), **overrides}
    )

    _print_run_start(
        opened,
        execution,
        compute_root=compute_root,
        skip_existing=not no_skip,
        wait=wait,
    )
    options: dict[str, Any] = {
        "skip_existing": not no_skip,
        "compute_root": compute_root,
        "execution": execution,
    }
    if execution.backend == "slurm" and not wait:
        jobs = opened.submit(**options)
        if jobs:
            typer.echo(f"Submitted job: {str(jobs[0].job_id).split('_')[0]}")
            return
    else:
        if execution.backend == "slurm":
            typer.echo(
                "Submitted; waiting for the job to finish (squeue shows its tasks)..."
            )
        opened.run(**options)
    _print_status(opened)


@app.command()
def status(project: str | None = _PROJECT_ARG) -> None:
    """Count finished and pending simulations, per variant when there are several."""
    _print_status(Project(_resolve_project(project)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _counts(total: int, pending: int) -> str:
    """Return a ``total / completed / pending`` line."""
    return f"total={total}  completed={total - pending}  pending={pending}"


def _print_status(project: Project) -> None:
    """Print a project status summary, per variant when there are several."""
    sims = project.simulations
    pending = project.incomplete(sims)
    typer.echo(f"Project: {project.directory}  {_counts(len(sims), len(pending))}")
    if len(project.variants) > 1:
        total = Counter(sims["variant"])
        waiting = Counter(pending["variant"])
        for variant in project.variants:
            typer.echo(f"  {variant}: {_counts(total[variant], waiting[variant])}")
    unreferenced = project.unreferenced()
    for kind, keys in unreferenced.items():
        if keys:
            typer.echo(
                f"{kind} folders in {project.output.path} that no variant here uses: "
                f"{', '.join('settings=' + k for k in keys)}  (from changed settings, "
                "dropped variants, or another project; PYSTILT never deletes them)"
            )


def _print_run_start(
    project: Project,
    execution: ExecutionConfig,
    *,
    compute_root: str | None,
    skip_existing: bool,
    wait: bool,
) -> None:
    """Print the settings ``stilt run`` is about to use."""
    backend = execution.backend
    mode = "existing" if skip_existing else "no-skip"
    typer.echo(
        "Starting run: "
        f"project={project.directory}  backend={backend}  "
        f"tasks={1 if backend == 'local' else execution.n_workers}  "
        f"cpus={execution.cpus}  skip={mode}"
    )
    typer.echo(f"Output: {project.output.path}")
    if backend == "local" or compute_root is not None:
        scratch = resolve_compute_root(project, compute_root)
        typer.echo(f"Compute root: {scratch}")
    else:
        typer.echo("Compute root: each task's own $TMPDIR (or PYSTILT_COMPUTE_ROOT)")
    typer.echo(f"Receptors loaded: {len(project.receptors)}")
    typer.echo(f"Variants: {', '.join(project.variants)}")
    typer.echo(
        "Execution mode: " + ("submit-and-wait" if wait else "submit-and-return")
        if backend != "local"
        else "Execution mode: local, one line per receptor as it finishes"
    )
