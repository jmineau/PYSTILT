"""
The ``stilt`` command-line interface.

Each command opens a :class:`~stilt.Project`, calls it, and
prints a short summary. Examples::

    stilt init                        # start a project in the current directory
    stilt init ./my_project           # start a project in ./my_project
    stilt run                         # run, and wait until done
    stilt run ./my_project --no-skip  # run every simulation again
    stilt run --backend slurm         # run as a Slurm job array, and wait
    stilt run --task 3/10             # run task 3 of 10 here (a job array's task)
    stilt run --receptors ids.txt     # run only the receptors listed in ids.txt
    stilt submit                      # submit a Slurm job array and return
    stilt status                      # count finished simulations; list the settings folders

``stilt run`` exits with 0 when every simulation it ran is complete, 1 when
some failed, and 3 when some did not finish because the run was
interrupted (Ctrl-C, or SIGTERM from a scheduler). Every command exits with
2 when its command line is wrong: an unknown option, a bad value, or a
directory that is not a project.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd
import typer
import yaml
from pydantic import ValidationError

from stilt.execution.config import ExecutionConfig
from stilt.execution.runner import resolve_compute_root
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
    help="Directory the transport model runs under, one workdir per simulation. Defaults to PYSTILT_COMPUTE_ROOT, then $TMPDIR/pystilt/<project>.",
)


#: Exit codes. 2 is also Click's own for an unknown option or command, so a
#: driver never takes a typo for a run to retry.
EXIT_COMPLETE, EXIT_FAILED, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 3


def _fail(message: str) -> NoReturn:
    """Print an error about the command line or the project, and exit with 2."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=EXIT_USAGE)


def _settings_problems(error: ValidationError) -> str:
    """Return what was wrong with some execution settings, one problem per clause."""
    problems = []
    for e in error.errors():
        message = e["msg"].removeprefix("Value error, ")
        where = ".".join(str(part) for part in e["loc"])
        problems.append(f"execution.{where}: {message}" if where else message)
    return "; ".join(problems)


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
        raise typer.Exit(code=EXIT_USAGE)
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
        raise typer.Exit(code=EXIT_USAGE) from None

    typer.echo(f"Initialized STILT project at '{project}'")
    typer.echo("  config.yaml    edit the met directory and footprint settings")
    typer.echo("  receptors.csv  add receptor times and locations, one per line:")
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
_RECEPTORS = typer.Option(
    None,
    "--receptors",
    help="File of receptor ids, one per line. Only those receptors run.",
)
_EXECUTION = typer.Option(
    None,
    "--execution",
    help=(
        "YAML file of execution settings, used in place of the execution "
        "section of config.yaml. Each Slurm task reads its submission's."
    ),
)
_TASK = typer.Option(
    None,
    "--task",
    help=(
        "Run task I of N here, as one task of a job array: every Nth "
        "receptor of the project, starting at I (0 to N-1). Runs in this "
        "process whatever the backend."
    ),
)


def _parse_task(task: str) -> tuple[int, int]:
    """Return ``(i, n)`` from ``--task I/N``, or exit."""
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", task)
    if match is None:
        _fail(f"--task takes I/N, such as 3/10; got {task!r}.")
    i, n = int(match.group(1)), int(match.group(2))
    if not 0 <= i < n:
        _fail(f"--task I/N needs 0 <= I < N; got {task!r}.")
    return i, n


def _read_ids(path: Path) -> list[str]:
    """Return the receptor ids in a file, one per line; blank lines and # comments are skipped."""
    try:
        text = path.read_text()
    except OSError as error:
        _fail(f"cannot read {path}: {error}")
    lines = (line.strip() for line in text.splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _exit_code(table: pd.DataFrame) -> int:
    """Return the exit code for the status table of the simulations a run ran."""
    if bool(table["state"].isin(["pending", "interrupted"]).any()):
        return EXIT_INTERRUPTED
    if (table["state"] == "failed").any():
        return EXIT_FAILED
    return EXIT_COMPLETE


def _start(
    project: str | None,
    *,
    backend: str | None,
    n_workers: int | None,
    cpus: int | None,
    no_skip: bool,
    compute_root: str | None,
    waits: bool,
    receptor_ids: list[str] | None = None,
    task: tuple[int, int] | None = None,
    execution_file: Path | None = None,
) -> tuple[Project, dict[str, Any]]:
    """Open the project, print what is about to run, and return the run's options."""
    opened = Project(_resolve_project(project))
    if execution_file is None:
        settings = opened.config.execution.model_dump(exclude_unset=True)
    else:
        try:
            settings = yaml.safe_load(execution_file.read_text()) or {}
        except (OSError, yaml.YAMLError) as error:
            _fail(f"cannot read {execution_file}: {error}")
    if task is not None:
        if backend == "slurm":
            _fail("--task runs here, as one task of a job array; drop --backend slurm.")
        backend = "local"
    # Progress is the worker's one line per finished receptor.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("stilt.execution").setLevel(logging.INFO)

    overrides: dict[str, Any] = {}
    if backend is not None:
        overrides["backend"] = backend
    if n_workers is not None:
        overrides["n_workers"] = n_workers
    if cpus is not None:
        overrides["cpus"] = cpus
    try:
        execution = ExecutionConfig.model_validate({**settings, **overrides})
    except ValidationError as error:
        _fail(_settings_problems(error))
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
        receptor_ids=receptor_ids,
        task=task,
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
    receptors: Path | None = _RECEPTORS,
    task: str | None = _TASK,
    execution: Path | None = _EXECUTION,
) -> None:
    """
    Run every unfinished simulation in a project, and wait until they are done.

    Runs the transport model for each receptor and variant, then the footprint when the
    variant has a grid. Simulations whose outputs exist are skipped unless
    --no-skip is given. With the Slurm backend this submits a job array and
    waits for it; use stilt submit to return as soon as it is submitted.

    --receptors limits the run to the receptors listed in a file. --task
    I/N runs one share of them here, for each task of a job array or a
    Kubernetes indexed Job. Exits with 0 when every simulation that ran is
    complete, 1 when some failed, 3 when some did not finish, and 2 when
    the command line is wrong.
    """
    receptor_ids = None if receptors is None else _read_ids(receptors)
    share = None if task is None else _parse_task(task)
    opened, options = _start(
        project,
        backend=backend,
        n_workers=n_workers,
        cpus=cpus,
        no_skip=no_skip,
        compute_root=compute_root,
        waits=True,
        receptor_ids=receptor_ids,
        task=share,
        execution_file=execution,
    )
    try:
        table = opened.run(receptors=receptor_ids, task=share, **options)
    except ValueError as error:
        _fail(str(error))
    if receptor_ids is None and share is None:
        _print_status(opened)
    else:
        _print_status(opened, table)
    raise typer.Exit(code=_exit_code(table))


@app.command()
def submit(
    project: str | None = _PROJECT_ARG,
    no_skip: bool = _NO_SKIP,
    n_workers: int | None = _N_WORKERS,
    cpus: int | None = _CPUS,
    compute_root: str | None = _COMPUTE_ROOT,
    receptors: Path | None = _RECEPTORS,
) -> None:
    """
    Submit every unfinished simulation in a project to Slurm, and return.

    The receptors are split among a Slurm job array's tasks, with the
    resources under execution in config.yaml. --receptors limits it to the
    receptors listed in a file. Check on them with stilt status.
    """
    receptor_ids = None if receptors is None else _read_ids(receptors)
    opened, options = _start(
        project,
        backend="slurm",
        n_workers=n_workers,
        cpus=cpus,
        no_skip=no_skip,
        compute_root=compute_root,
        waits=False,
        receptor_ids=receptor_ids,
    )
    try:
        job_id = opened.submit(receptors=receptor_ids, **options)
    except ValueError as error:
        _fail(str(error))
    if job_id is not None:
        typer.echo(f"Submitted job: {job_id}")
    else:
        _print_status(opened)


@app.command()
def status(project: str | None = _PROJECT_ARG) -> None:
    """
    Count finished and unfinished simulations, and list the output's settings folders.

    The counts are per variant when there are several. Each settings folder
    holds the results of one set of settings; the list says how many result
    files it holds, which variants use it, and, for a folder no variant
    uses, how its settings differ from the variant of its name.
    """
    opened = Project(_resolve_project(project))
    _print_status(opened)
    _print_folders(opened)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _counts(total: int, pending: int) -> str:
    """Return a ``total / completed / pending`` line."""
    return f"total={total}  completed={total - pending}  pending={pending}"


def _print_status(project: Project, ran: pd.DataFrame | None = None) -> None:
    """
    Print a project status summary, per variant when there are several.

    With *ran*, the status table of a run, it sums up those simulations
    only.
    """
    table = project.status() if ran is None else ran
    pending = table[table.state != "complete"]
    label = f"Project: {project.directory}" if ran is None else "This run:"
    typer.echo(f"{label}  {_counts(len(table), len(pending))}")
    if len(project.variants) > 1:
        total = Counter(table["variant"])
        waiting = Counter(pending["variant"])
        for variant in project.variants:
            typer.echo(f"  {variant}: {_counts(total[variant], waiting[variant])}")
    causes = Counter(table.loc[table.state == "failed", "reason"])
    if causes:
        listed = ", ".join(f"{cause} {n}" for cause, n in causes.most_common())
        typer.echo(
            f"failed: {listed}  (why: the .failure.yaml beside each log, "
            f"under {project.output.directory / 'logs'})"
        )


def _print_folders(project: Project) -> None:
    """Print the output directory's settings folders, which variants use each, and how the others differ."""
    table = project.folders()
    typer.echo(f"Output: {project.output.directory}")
    if table.empty:
        typer.echo("  no settings folders yet")
        return
    kind_width = max(len(k) for k in table["kind"])
    name_width = max(len(f) for f in table["folder"]) + len("settings=")
    counts = [f"{n:,} file" + ("" if n == 1 else "s") for n in table["files"]]
    count_width = max(len(c) for c in counts)
    for row, count in zip(table.to_dict("records"), counts, strict=True):
        used = row["variant"] or "(no variant)"
        line = (
            f"  {row['kind']:<{kind_width}}  {'settings=' + row['folder']:<{name_width}}"
            f"  {count:>{count_width}}  {used}"
        )
        typer.echo(f"{line}  {row['differs']}" if row["differs"] else line)
    if (table["variant"] == "").any():
        typer.echo(
            "  A folder no variant uses is from changed settings, a dropped "
            "variant, or another project. PYSTILT never deletes one."
        )


def _print_run_start(
    project: Project,
    execution: ExecutionConfig,
    *,
    compute_root: str | None,
    skip_existing: bool,
    waits: bool,
    receptor_ids: list[str] | None = None,
    task: tuple[int, int] | None = None,
) -> None:
    """Print the settings ``stilt run`` is about to use."""
    backend = execution.backend
    mode = "existing" if skip_existing else "no-skip"
    if task is not None:
        # A task of a job array runs here; "backend=local" would read as if
        # the array had not been used.
        where = f"task {task[0]} of {task[1]}"
    else:
        tasks = 1 if backend == "local" else execution.n_workers
        where = f"backend={backend}  tasks={tasks}"
    typer.echo(
        f"Starting run: project={project.directory}  {where}  "
        f"cpus={execution.cpus}  skip={mode}"
    )
    typer.echo(f"Output: {project.output.directory}")
    if compute_root is not None:
        typer.echo(f"Compute root: {compute_root}")
    else:
        typer.echo("Compute root: each task's own $TMPDIR (or PYSTILT_COMPUTE_ROOT)")
    typer.echo(f"Receptors loaded: {len(project.receptors)}")
    if receptor_ids is not None:
        typer.echo(f"Receptors listed: {len(receptor_ids)}")
    if task is not None:
        i, n = task
        typer.echo(f"Receptors of this task: {i}, {i + n}, {i + 2 * n}, ...")
    typer.echo(f"Variants: {', '.join(project.variants)}")
    if task is not None or backend == "local":
        typer.echo("Running here, one line per receptor as it finishes")
    elif waits:
        typer.echo(
            "Submitting to Slurm and waiting for the job to finish (squeue shows its tasks)"
        )
    else:
        typer.echo("Submitting to Slurm and returning")
