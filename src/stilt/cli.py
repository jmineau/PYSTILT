"""
The particle commands of the ``stilt`` command-line interface.

Each command opens a :class:`stilt.project.Project`, the particle part, calls
it, and prints a short summary. The ``stilt`` command is these and
``stilt init``, which starts a STILT project (:mod:`stilt.stilt_cli`).
Examples::

    stilt check                       # list what would stop a run, before any run
    stilt fetch                       # get the met files the runs need, once
    stilt run                         # run, and wait until done
    stilt run ./my_project --no-skip  # run every simulation again
    stilt run --backend slurm         # run as a Slurm job array, and wait
    stilt run --task 3/10             # run task 3 of 10 here (a job array's task)
    stilt run --releases ids.txt     # run only the releases listed in ids.txt
    stilt submit                      # submit a Slurm job array and return
    stilt status                      # count finished simulations; list the settings folders

``stilt run`` exits with 0 when every simulation it ran is complete, 1 when
some failed, and 3 when some did not finish because the run was
interrupted (Ctrl-C, or SIGTERM from a scheduler), a worker process was killed
(often for memory), or a connection or I/O error
stopped them (an output bucket that stopped answering while a simulation wrote). ``stilt check`` exits
with 0 when it finds nothing and 1 when it lists anything. ``stilt fetch``
exits with 0 when every met file the runs need is there, 1 when the archive
lacks some, and 3 when it was stopped (Ctrl-C or SIGTERM) or the archive was
busy or unreachable. Every command
exits with 2 when its command line is wrong (an unknown option, a bad value,
or a directory that is not a project) or the project's files are invalid
(``config.yaml`` that is not valid YAML or has a misspelled setting, a bad
row in ``releases.csv``). ``stilt run`` and ``stilt submit`` also
exit with 2 when a met directory does not exist or holds no ARL file, which
``stilt check`` lists. ``stilt status`` does too, unless the output holds no
result the project could own: then it counts every simulation as pending.

An error the command did not expect ends it too, after its traceback: with
3 when the error is a connection or an I/O error (a bucket that cannot be
reached or is busy, a stale disk), which a retry may get past, and with 4 for
any other (a bug in PYSTILT or the project, a bucket in the wrong region, a
403 or a 404, which a retry will repeat).
``is_transient`` sorts them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import signal
import threading
import traceback
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd
import typer
import yaml
from pydantic import ValidationError
from typer.core import TyperGroup

from stilt._transient import is_transient
from stilt.exceptions import MissingClass, WorkerDied
from stilt.execution.config import ExecutionConfig
from stilt.execution.runner import pending, resolve_compute_root
from stilt.project import STATES, Project

#: Exit codes. 2 is also Click's own for an unknown option or command, so a
#: driver never takes a typo for a run to retry. 3 says "run it again": the
#: run was stopped, or hit a connection or I/O error; what finished is on
#: disk. 4 says "do not": an unexpected error that a retry would repeat. 1 is
#: the code of a run whose simulations failed, and no other.
EXIT_COMPLETE, EXIT_FAILED, EXIT_USAGE, EXIT_INTERRUPTED, EXIT_BUG = 0, 1, 2, 3, 4


class _Group(TyperGroup):
    """
    The ``stilt`` command group, which ends a command that raised in an exit code.

    Python's own exit code for an uncaught error is 1, which is also the
    code of a run whose simulations failed. A driver that decides on a
    retry from the code cannot tell the two apart. So an error the command
    did not handle prints its traceback and ends with
    3 if a retry may get past it (``is_transient``)
    and 4 if not, and a ``KeyboardInterrupt`` that no
    worker caught (Ctrl-C while waiting for a Slurm job, say) with 3, as one
    a worker caught does. Typer's and Click's own exceptions (an exit,
    a wrong command line) pass, matched by module so that it holds whether
    Typer vendors Click or not. So does a ``BrokenPipeError``, which is
    stdout closed by ``head``: Click ends quietly, with 1, as it did before.
    """

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Run the command, and turn an error it did not handle into an exit code."""
        try:
            return super().invoke(*args, **kwargs)
        except KeyboardInterrupt:
            typer.echo("Interrupted.", err=True)
            raise typer.Exit(code=EXIT_INTERRUPTED) from None
        except BrokenPipeError:
            raise  # stdout was closed (`stilt status | head`): Click ends quietly, with 1
        except Exception as error:
            if type(error).__module__.partition(".")[0] in ("typer", "click"):
                raise  # an exit, or a wrong command line: Click's own, and 2
            code = EXIT_INTERRUPTED if is_transient(error) else EXIT_BUG
            traceback.print_exception(error)
            what = "a connection or I/O error, which a retry may get past"
            if code == EXIT_BUG:
                what = "an unexpected error, which a retry would repeat"
            typer.echo(f"stilt exits with {code}: {what}.", err=True)
            raise typer.Exit(code=code) from None


app = typer.Typer(
    name="stilt",
    help="Run STILT simulations and check on a project.",
    no_args_is_help=True,
    cls=_Group,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared options / arguments
# ---------------------------------------------------------------------------

_PROJECT_ARG = typer.Argument(
    None,
    help="Path of the STILT project. Defaults to the current directory.",
)
_NO_SKIP = typer.Option(
    False, "--no-skip", help="Run simulations again even if their outputs exist."
)
_COMPUTE_ROOT = typer.Option(
    None,
    "--compute-root",
    help="Directory the transport model runs under, a new workdir for each run. Defaults to PYSTILT_COMPUTE_ROOT, then $TMPDIR/pystilt/<project>.",
)


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
    if not (resolved / "config.yaml").exists():
        typer.echo(
            f"Error: '{resolved}' does not look like a STILT project directory "
            "(no config.yaml found).",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)
    return str(resolved)


def _open(
    project: str | Path | None,
    output: str | None = None,
    rows: pd.DataFrame | None = None,
) -> Project:
    """
    Open the project, exiting with 2 when its files are invalid or the met of a variant cannot be read.

    With *rows* (from ``--releases releases.parquet``) the project takes
    its releases from them, so ``releases.csv`` is not read.

    A variant's settings folder is named by its met's weather product, read
    from the header of the met's first file. Without the file every command
    that counts or runs simulations would stop with a traceback;
    ``stilt check`` lists the same problem as a finding.
    """
    opened = Project(_resolve_project(project), output=output)
    if rows is not None:
        opened._take_rows(rows)
    _require_inputs(opened)
    _require_met(opened)
    return opened


def _require_inputs(project: Project) -> None:
    """
    Exit with 2 and the problem when ``config.yaml`` or ``releases.csv`` is invalid.

    That includes a class the config names that is not installed here, and
    a stage whose settings do not fit its variant's transport model.
    """
    try:
        _ = project.config
        _ = project.releases
        _ = project.variants
    except (MissingClass, ValueError) as error:
        _fail(str(error))


def _require_build(project: Project) -> None:
    """Exit with 2 and the line ``stilt check`` lists when a variant's transport model cannot be found, such as no ``hycs_std``."""
    try:
        for variant in project.variants.values():
            _ = variant.build
    except FileNotFoundError as error:
        _fail(str(error))


def _require_met(project: Project) -> None:
    """Exit with 2 and the line ``stilt check`` lists when a variant's met cannot be read."""
    try:
        for variant in project.variants.values():
            _ = variant.met_settings
    except (FileNotFoundError, ValueError) as error:
        _fail(str(error))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


#: Rows of each check ``stilt check`` prints; ``project.check()`` has them all.
_CHECK_ROWS = 10


@app.command()
def check(project: str | None = _PROJECT_ARG) -> None:
    """
    List what would stop a project's runs, before any run.

    For each variant, checks that the met's directory exists and holds ARL
    files, that every met file each release's run needs is there (naming
    the hours), that the first of them holds more than one time period,
    which HYSPLIT needs, that each release is inside the met's crop box and the
    footprint grid, that an averaging-kernel table matches the
    releases, and that ``varsiwant`` holds ``zsfc`` when releases above sea
    level need it. Nothing is run or downloaded. It looks for files, not inside
    them: a run that passes can still fail as MET_COVERAGE, when a met file
    is damaged or cut short, or the release is outside the files' domain.
    Exits with 0 when it finds nothing, 1 when it lists anything, and 2 when
    the command line is wrong or the project's files are invalid.
    """
    opened = Project(_resolve_project(project))
    try:
        problems = opened.check()
    except (MissingClass, ValueError) as error:
        _fail(str(error))
    _print_check(opened, problems)
    raise typer.Exit(code=EXIT_FAILED if len(problems) else EXIT_COMPLETE)


_BACKEND = typer.Option(
    None,
    "--backend",
    help="Where to run: local or slurm. Overrides execution.backend in config.yaml.",
)
_N_WORKERS = typer.Option(
    None,
    "--n-workers",
    help="Number of Slurm array tasks. Overrides execution.slurm.n_workers in config.yaml.",
)
_CPUS = typer.Option(
    None,
    "--cpus",
    help=(
        "Releases each task runs at once (local processes, or CPUs per "
        "Slurm task). Overrides cpus in the backend's block of execution "
        "in config.yaml."
    ),
)
_RELEASES = typer.Option(
    None,
    "--releases",
    help=(
        "File of release ids, one per line, or the releases.parquet a Slurm "
        "submission writes. Only those releases run."
    ),
)
_OUTPUT = typer.Option(
    None,
    "--output",
    help="Output directory to use instead of config.yaml's: a path, or a URL such as s3://bucket/output.",
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
        "release of the project, starting at I (0 to N-1). Runs in this "
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


def _read_releases(path: Path) -> tuple[list[str], pd.DataFrame | None]:
    """
    Return the release ids in a file, and their checked rows when the file holds them.

    A text file lists ids, one per line; blank lines and # comments are
    skipped. A Parquet file is the ``releases.parquet`` a Slurm submission
    writes: the checked rows of its releases, in the order they run.
    """
    try:
        if path.suffix == ".parquet":
            rows = pd.read_parquet(path)
            return list(dict.fromkeys(rows["release"].astype(str))), rows
        text = path.read_text()
    except (OSError, ValueError, KeyError) as error:
        _fail(f"cannot read {path}: {error}")
    lines = (line.strip() for line in text.splitlines())
    return [line for line in lines if line and not line.startswith("#")], None


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
    release_ids: list[str] | None = None,
    release_rows: pd.DataFrame | None = None,
    task: tuple[int, int] | None = None,
    execution_file: Path | None = None,
    output: str | None = None,
) -> tuple[Project, dict[str, Any]]:
    """
    Open the project, print what is about to run, and return the run's options.

    With *release_rows* (from ``--releases releases.parquet``) the
    project takes its releases from them, not from ``releases.csv``.
    """
    opened = _open(project, output=output, rows=release_rows)
    _require_build(opened)
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
    # Progress is the worker's one line per finished release.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("stilt.execution").setLevel(logging.INFO)

    # Each option replaces its setting in the block of the backend it is for.
    if backend is not None:
        settings["backend"] = backend
    if n_workers is not None:
        settings["slurm"] = {**(settings.get("slurm") or {}), "n_workers": n_workers}
    if cpus is not None:
        block = settings.get("backend", "local")
        settings[block] = {**(settings.get(block) or {}), "cpus": cpus}
    try:
        execution = ExecutionConfig.model_validate(settings)
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
        release_ids=release_ids,
        task=task,
    )
    options = {
        "skip_existing": not no_skip,
        "compute_root": compute_root,
        "execution": execution,
    }
    return opened, options


@contextlib.contextmanager
def _stop_once() -> Iterator[None]:
    """
    Turn the first SIGINT or SIGTERM into ``KeyboardInterrupt`` inside the block, and ignore any after it.

    A pod or a Slurm job is stopped with SIGTERM, which ends Python at once
    and skips the ``finally`` blocks that remove a download's partial file
    (as for ``stilt run``, :func:`stilt.execution.worker._sigterm_as_interrupt`
    makes it raise instead). A second signal while that cleanup runs would
    raise again and could skip it, so after the first the rest stay ignored,
    through the exit. Handlers can be set only from the main thread;
    elsewhere this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    stopped = False

    def stop(signum: int, frame: object) -> None:
        nonlocal stopped
        stopped = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        if not stopped:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


@app.command()
def fetch(
    project: str | None = _PROJECT_ARG,
    releases: Path | None = _RELEASES,
    no_skip: bool = typer.Option(
        False,
        "--no-skip",
        help="Get the met of every simulation, also those whose particles exist.",
    ),
    output: str | None = _OUTPUT,
) -> None:
    """
    Get the met files a project's unfinished simulations need, then exit.

    For each met, gets the files that cover the time of every simulation
    that has yet to run, as a run would ask for them. A met with download
    set downloads what is missing, cropped to its subgrid, so the runs that
    follow find every file and download nothing: run this once where the
    files are shared, before many workers start. A met without download is
    only checked for missing files, as stilt check does. --releases limits
    it to the releases listed in a file, as for stilt run; the fetch job
    stilt submit sends before a job array reads its releases.parquet.
    --no-skip gets the met of every simulation, for a stilt run --no-skip.
    Exits with 0 when
    every file is there, 1 when the archive lacks some, 2 when the command
    line is wrong, the project's files are invalid or a met cannot be read, and 3 when stopped by Ctrl-C or
    SIGTERM, after removing the file it was downloading, or when the archive
    was busy or the connection failed (4 for any other unexpected error).
    """
    release_ids, rows = (None, None) if releases is None else _read_releases(releases)
    opened = _open(project, output=output, rows=rows)
    sel = None
    if release_ids is not None:
        try:
            # Checks the ids as a run does; lists no output without skip_existing.
            release_ids = pending(opened, releases=release_ids, skip_existing=False)
        except ValueError as error:
            _fail(f"{releases}: {error}")
        sel = opened.simulations["release"].isin(release_ids)
    # arlmet logs each file it downloads, which is the progress to show.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("arlmet.archives").setLevel(logging.INFO)
    count = len(opened.releases) if release_ids is None else len(release_ids)
    typer.echo(f"Project: {opened.directory}  releases={count}")
    try:
        with _stop_once():
            files = opened.fetch(sel, skip_existing=not no_skip)
    except (ImportError, ValueError) as error:
        _fail(str(error))
    except KeyboardInterrupt:
        typer.echo(
            "Interrupted. The partial download was removed; fetch again to go on."
        )
        raise typer.Exit(code=EXIT_INTERRUPTED) from None
    problems = _print_fetch(files)
    raise typer.Exit(code=EXIT_FAILED if problems else EXIT_COMPLETE)


@app.command()
def run(
    project: str | None = _PROJECT_ARG,
    no_skip: bool = _NO_SKIP,
    backend: str | None = _BACKEND,
    n_workers: int | None = _N_WORKERS,
    cpus: int | None = _CPUS,
    compute_root: str | None = _COMPUTE_ROOT,
    releases: Path | None = _RELEASES,
    task: str | None = _TASK,
    execution: Path | None = _EXECUTION,
    output: str | None = _OUTPUT,
) -> None:
    """
    Run every unfinished simulation in a project, and wait until they are done.

    Runs the transport model for each release and variant, then the footprint when the
    variant has one. Simulations whose outputs exist are skipped unless
    --no-skip is given. With the Slurm backend this submits a job array and
    waits for it; use stilt submit to return as soon as it is submitted.

    --releases limits the run to the releases listed in a file. --task
    I/N runs one share of them here, for each task of a job array or a
    Kubernetes indexed Job.

    Exits with 0 when every simulation that ran is complete, 1 when some
    failed, 3 when some did not finish (the run was stopped, a worker process
    was killed, often for memory, or a connection
    or I/O error stopped them while they wrote), and 2 when
    the command line is wrong, the project's files are invalid or a met
    cannot be read (nothing ran). An
    error the run did not expect prints its traceback and exits with 3 when
    it is a connection or I/O error, such as an output bucket that cannot
    be reached, which a retry may get past, and with 4 when it is any
    other, which a retry would repeat.
    """
    release_ids, rows = (None, None) if releases is None else _read_releases(releases)
    share = None if task is None else _parse_task(task)
    opened, options = _start(
        project,
        backend=backend,
        n_workers=n_workers,
        cpus=cpus,
        no_skip=no_skip,
        compute_root=compute_root,
        waits=True,
        release_ids=release_ids,
        release_rows=rows,
        task=share,
        execution_file=execution,
        output=output,
    )
    try:
        table = opened.run(releases=release_ids, task=share, **options)
    except ValueError as error:
        _fail(str(error))
    except WorkerDied as error:
        # A worker was killed, most likely for memory; the message says what to do.
        typer.echo(f"stilt run stopped: {error}", err=True)
        raise typer.Exit(code=EXIT_INTERRUPTED) from None
    if release_ids is None and share is None:
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
    releases: Path | None = _RELEASES,
    output: str | None = _OUTPUT,
) -> None:
    """
    Submit every unfinished simulation in a project to Slurm, and return.

    The releases are split among a Slurm job array's tasks, with the
    resources under execution in config.yaml. --releases limits it to the
    releases listed in a file. Check on them with stilt status.
    """
    release_ids, rows = (None, None) if releases is None else _read_releases(releases)
    opened, options = _start(
        project,
        backend="slurm",
        n_workers=n_workers,
        cpus=cpus,
        no_skip=no_skip,
        compute_root=compute_root,
        waits=False,
        release_ids=release_ids,
        release_rows=rows,
        output=output,
    )
    try:
        job_id = opened.submit(releases=release_ids, **options)
    except ValueError as error:
        _fail(str(error))
    if job_id is not None:
        typer.echo(f"Submitted job: {job_id}")
    else:
        _print_status(opened)


@app.command()
def status(
    project: str | None = _PROJECT_ARG,
    as_json: bool = typer.Option(
        False, "--json", help="Print the same as JSON, for a program to read."
    ),
    output: str | None = _OUTPUT,
) -> None:
    """
    Count finished and unfinished simulations, and list the output's settings folders.

    The counts are per variant when there are several. Each settings folder
    holds the results of one set of settings; the list says how many result
    files it holds, which variants use it, and, for a folder this project
    does not use, how its settings differ from the variant of its name.

    The settings folders are named by the hash of the settings, which includes
    the met's weather product, read from the met's files. When a met cannot
    be read, status still counts every simulation as pending if no folder in
    the output has the variant's other settings, since then no result can
    exist. Otherwise it needs the met, and exits with 2, as it does when the
    command line is wrong or the project's files are invalid.

    The hash also includes the checksum of hycs_std. On a machine without
    hycs_std, a variant's folder is the one it made (by its name) with its
    other settings; status exits with 2 when the variant made two, with two
    builds of hycs_std.
    """
    opened = Project(_resolve_project(project), output=output)
    _require_inputs(opened)
    try:
        table = opened.status()
        folders = opened.folders()
    except FileNotFoundError:
        _require_met(opened)
        _require_build(opened)
        raise
    if as_json:
        typer.echo(json.dumps(_status_data(opened, table, folders), indent=2))
        return
    _print_status(opened, status=table)
    _print_folders(opened, folders)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _counts(states: Iterable[str]) -> str:
    """Return a ``total=  complete=  failed=  interrupted=  pending=`` line."""
    return "  ".join(f"{name}={n}" for name, n in _state_counts(states).items())


def _print_status(
    project: Project,
    ran: pd.DataFrame | None = None,
    status: pd.DataFrame | None = None,
) -> None:
    """
    Print a project status summary, per variant when there are several.

    With *ran*, the status table of a run, it sums up those simulations
    only. *status* is the project's status table, when the caller has it.
    """
    shown = status if ran is None else ran
    table = project.status() if shown is None else shown
    label = f"Project: {project.directory}" if ran is None else "This run:"
    typer.echo(f"{label}  {_counts(table['state'])}")
    if len(project.variants) > 1:
        for variant in project.variants:
            states = table.loc[table["variant"] == variant, "state"]
            typer.echo(f"  {variant}: {_counts(states)}")
    causes = Counter(table.loc[table.state == "failed", "reason"])
    if causes:
        listed = ", ".join(f"{cause} {n}" for cause, n in causes.most_common())
        typer.echo(
            f"failed: {listed}  (why: each one's .failure.yaml, "
            f"under {project.output.directory / 'logs'})"
        )


def _state_counts(states: Iterable[str]) -> dict[str, int]:
    """Return how many simulations are in each state, and the total."""
    counts = Counter(states)
    return {"total": counts.total(), **{s: counts[s] for s in STATES}}


def _status_data(
    project: Project, table: pd.DataFrame, folders: pd.DataFrame
) -> dict[str, Any]:
    """
    Return what ``stilt status`` prints, as plain data.

    *table* is the project's status table and *folders* its settings folders.

    ``simulations`` and ``variants`` count the simulations in each state,
    ``failed`` counts the failed ones by reason, and ``folders`` lists the
    settings folders as :meth:`stilt.Project.folders` does.
    """
    failed = table.loc[table.state == "failed", "reason"]
    return {
        "project": str(project.directory),
        "output": str(project.output.directory),
        "simulations": _state_counts(table["state"]),
        "variants": {
            name: _state_counts(table.loc[table.variant == name, "state"])
            for name in project.variants
        },
        "failed": dict(Counter(failed.fillna("unknown")).most_common()),
        "folders": [
            {
                "kind": row["kind"],
                "folder": row["folder"],
                "name": row["name"],
                "files": int(row["files"]),
                "variants": [v for v in row["variant"].split(", ") if v],
                "differs": row["differs"],
            }
            for row in folders.to_dict("records")
        ],
    }


def _print_check(project: Project, problems: pd.DataFrame) -> None:
    """
    Print the releases and variants checked, then the problems found as a table.

    The first :data:`_CHECK_ROWS` rows of each check are shown, with a count
    of the rest.
    """
    typer.echo(f"Project: {project.directory}  releases={len(project.releases)}")
    variants = ", ".join(
        f"{name} (met {v.met}, {', '.join(v.stages) or 'particles only'})"
        for name, v in project.variants.items()
    )
    typer.echo(f"Variants: {variants}")
    if problems.empty:
        typer.echo("No problems found.")
        return
    lines: list[tuple[str, str, str, str]] = []
    for name, rows in problems.groupby("check", sort=False):
        for row in rows.head(_CHECK_ROWS).itertuples():
            release = "" if pd.isna(row.release) else str(row.release)
            lines.append((str(name), str(row.variant), release, str(row.message)))
        if len(rows) > _CHECK_ROWS:
            more = (
                f"... {len(rows) - _CHECK_ROWS} more (project.check() lists them all)"
            )
            lines.append((str(name), "", "", more))
    header = ("check", "variant", "release", "message")
    widths = [max(len(line[i]) for line in [header, *lines]) for i in range(3)]
    typer.echo()
    for line in [header, *lines]:
        cells = [f"{t:<{w}}" for t, w in zip(line[:3], widths, strict=True)]
        typer.echo("  ".join([*cells, line[3]]).rstrip())
    counts = problems.groupby("check", sort=False).size()
    listed = ", ".join(f"{name} {n}" for name, n in counts.items())
    noun = "problem" if len(problems) == 1 else "problems"
    typer.echo(f"\nFound {len(problems)} {noun}: {listed}")


def _print_fetch(files: pd.DataFrame) -> int:
    """
    Print how many met files each met has, then the times whose files could not be had.

    Returns the number of those times. The first :data:`_CHECK_ROWS` of them
    are shown, with a count of the rest.
    """
    if files.empty:
        typer.echo("No unfinished simulation needs met files.")
        return 0
    have = files[files["file"].notna()]
    failed = files[files["message"].notna()]
    for name, rows in have.groupby("met", sort=False):
        noun = "file" if len(rows) == 1 else "files"
        typer.echo(f"Met {name}: {len(rows)} {noun} ready")
    for name, rows in failed.groupby("met", sort=False):
        typer.echo(f"\nMet {name}: could not get the files for these times:")
        for message in rows["message"].head(_CHECK_ROWS):
            typer.echo(f"  {message}")
        if len(rows) > _CHECK_ROWS:
            typer.echo(f"  ... {len(rows) - _CHECK_ROWS} more")
    return len(failed)


def _print_folders(project: Project, table: pd.DataFrame) -> None:
    """
    Print the output directory's settings folders, this project's first.

    The folders no variant of this project uses (changed settings, a
    dropped variant, or another project sharing the output directory) follow
    under their own heading, with how their settings differ. *table* is
    :meth:`stilt.Project.folders`.
    """
    typer.echo(f"Output: {project.output.directory}")
    if table.empty:
        typer.echo("  no settings folders yet")
        return
    kind_width = max(len(k) for k in table["kind"])
    name_width = max(len(f) for f in table["folder"]) + len("settings=")
    counts = [f"{n:,} file" + ("" if n == 1 else "s") for n in table["files"]]
    count_width = max(len(c) for c in counts)

    def lines(rows: pd.DataFrame, indent: str) -> None:
        for row in rows.to_dict("records"):
            count = counts[row["_i"]]
            line = (
                f"{indent}{row['kind']:<{kind_width}}  "
                f"{'settings=' + row['folder']:<{name_width}}  {count:>{count_width}}"
            )
            line = f"{line}  {row['variant']}" if row["variant"] else line
            typer.echo(f"{line}  {row['differs']}" if row["differs"] else line)

    table = table.assign(_i=range(len(table)))
    used = table[table["variant"] != ""]
    others = table[table["variant"] == ""]
    if others.empty:
        lines(used, "  ")
        return
    typer.echo(
        "  Used by this project:"
        if not used.empty
        else "  Used by this project: none yet"
    )
    lines(used, "    ")
    typer.echo(
        "  Not used by this project (changed settings, a dropped variant, or "
        "another project sharing this output; PYSTILT never deletes one):"
    )
    lines(others, "    ")


def _print_run_start(
    project: Project,
    execution: ExecutionConfig,
    *,
    compute_root: str | None,
    skip_existing: bool,
    waits: bool,
    release_ids: list[str] | None = None,
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
        tasks = 1 if backend == "local" else execution.slurm.n_workers
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
    typer.echo(f"Releases loaded: {len(project.releases)}")
    if release_ids is not None:
        typer.echo(f"Releases listed: {len(release_ids)}")
    if task is not None:
        i, n = task
        typer.echo(f"Releases of this task: {i}, {i + n}, {i + 2 * n}, ...")
    typer.echo(f"Variants: {', '.join(project.variants)}")
    if task is not None or backend == "local":
        typer.echo("Running here, one line per release as it finishes")
    elif waits:
        typer.echo(
            "Submitting to Slurm and waiting for the job to finish (squeue shows its tasks)"
        )
    else:
        typer.echo("Submitting to Slurm and returning")
