"""
The ``stilt`` command-line interface.

Each command opens a :class:`~stilt.Project`, calls it, and
prints a short summary. Examples::

    stilt init                        # start a project in the current directory
    stilt init ./my_project           # start a project in ./my_project
    stilt run                         # run, and wait until done
    stilt run ./my_project --no-skip  # run every simulation again
    stilt run --backend slurm         # run as a Slurm job array, and wait
    stilt submit                      # submit a Slurm job array and return
    stilt status                      # count finished simulations
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from stilt.execution import resolve_compute_root
from stilt.execution.config import ExecutionConfig
from stilt.project import Project

app = typer.Typer(
    name="stilt",
    help="Run STILT simulations and check on a project.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared options / arguments
# ---------------------------------------------------------------------------

_PROJECT_ARG = typer.Argument(
    None,
    help="Path of the STILT project. Defaults to the current directory.",
)
_NEW_PROJECT_ARG = typer.Argument(
    None,
    help="Directory for the new project. Defaults to the current directory.",
)
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
    if not Project(resolved).config_path.exists():
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
    try:
        Project.init(project, starter=True)
    except FileExistsError:
        typer.echo(
            f"Error: '{project}' already contains a config.yaml. Aborting.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    typer.echo(f"Initialized STILT project at '{project}'")
    typer.echo("  config.yaml   — edit met directory and footprint settings")
    typer.echo("  receptors.csv — add receptor times/locations here, one per line:")
    typer.echo("                  2023-01-01 12:00:00,-111.85,40.77,5")


_BACKEND = typer.Option(
    None,
    "--backend",
    help="Where to run: local or slurm. Overrides execution.backend in config.yaml.",
)
_N_WORKERS = typer.Option(
    None,
    "--n-workers",
    help="Number of Slurm array tasks. Overrides execution.n_workers in config.yaml.",
)
_CPUS = typer.Option(
    None,
    "--cpus",
    help=(
        "Receptors each task runs at once (local processes, or CPUs per "
        "Slurm task). Overrides execution.cpus in config.yaml."
    ),
)


def _start(
    project: str | None,
    *,
    backend: str | None,
    n_workers: int | None,
    cpus: int | None,
    no_skip: bool,
    compute_root: str | None,
    waits: bool,
) -> tuple[Project, dict[str, Any]]:
    """Open the project, print what is about to run, and return the run's options."""
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
    if execution.backend == "local":
        # Resolved once, here, so the banner shows the directory the run uses.
        # A Slurm task resolves its own, on its node.
        compute_root = str(resolve_compute_root(opened, compute_root))
    _print_run_start(
        opened,
        execution,
        compute_root=compute_root,
        skip_existing=not no_skip,
        waits=waits,
    )
    options = {
        "skip_existing": not no_skip,
        "compute_root": compute_root,
        "execution": execution,
    }
    return opened, options


@app.command()
def run(
    project: str | None = _PROJECT_ARG,
    no_skip: bool = _NO_SKIP,
    backend: str | None = _BACKEND,
    n_workers: int | None = _N_WORKERS,
    cpus: int | None = _CPUS,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """
    Run every unfinished simulation in a project, and wait until they are done.

    Runs HYSPLIT for each receptor and variant, then the footprint when the
    variant has a grid. Simulations whose outputs exist are skipped unless
    --no-skip is given. With the Slurm backend this submits a job array and
    waits for it; use stilt submit to return as soon as it is submitted.
    """
    opened, options = _start(
        project,
        backend=backend,
        n_workers=n_workers,
        cpus=cpus,
        no_skip=no_skip,
        compute_root=compute_root,
        waits=True,
    )
    if options["execution"].backend == "slurm":
        typer.echo(
            "Submitted; waiting for the job to finish (squeue shows its tasks)..."
        )
    opened.run(**options)
    _print_status(opened)


@app.command()
def submit(
    project: str | None = _PROJECT_ARG,
    no_skip: bool = _NO_SKIP,
    n_workers: int | None = _N_WORKERS,
    cpus: int | None = _CPUS,
    compute_root: str | None = _COMPUTE_ROOT,
) -> None:
    """
    Submit every unfinished simulation in a project to Slurm, and return.

    The receptors are split among a Slurm job array's tasks, with the
    resources under execution in config.yaml. Check on them with stilt
    status.
    """
    opened, options = _start(
        project,
        backend="slurm",
        n_workers=n_workers,
        cpus=cpus,
        no_skip=no_skip,
        compute_root=compute_root,
        waits=False,
    )
    jobs = opened.submit(**options)
    if jobs:
        typer.echo(f"Submitted job: {str(jobs[0].job_id).split('_')[0]}")
    else:
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
    pending = sims.incomplete()
    typer.echo(f"Project: {project.directory}  {_counts(len(sims), len(pending))}")
    if len(project.variants) > 1:
        total = Counter(sims["variant"])
        waiting = Counter(pending["variant"])
        for variant in project.variants:
            typer.echo(f"  {variant}: {_counts(total[variant], waiting[variant])}")
    failed = pending.failures()
    if len(failed):
        causes = Counter(
            reason if isinstance(reason, str) else error
            for reason, error in zip(failed["reason"], failed["error"], strict=True)
        )
        listed = ", ".join(f"{cause} {n}" for cause, n in causes.most_common())
        typer.echo(f"failed: {listed}  (sim.failure says why)")
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
    waits: bool,
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
    if compute_root is not None:
        typer.echo(f"Compute root: {compute_root}")
    else:
        typer.echo("Compute root: each task's own $TMPDIR (or PYSTILT_COMPUTE_ROOT)")
    typer.echo(f"Receptors loaded: {len(project.receptors)}")
    typer.echo(f"Variants: {', '.join(project.variants)}")
    typer.echo(
        "Execution mode: " + ("submit-and-wait" if waits else "submit-and-return")
        if backend != "local"
        else "Execution mode: local, one line per receptor as it finishes"
    )
