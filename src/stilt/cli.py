"""
The ``stilt`` command-line interface.

Each command opens a :class:`~stilt.Project`, calls it, and
prints a short summary. Examples::

    stilt init                        # start a project in the current directory
    stilt init ./my_project           # start a project in ./my_project
    stilt check                       # list what would stop a run, before any run
    stilt run                         # run, and wait until done
    stilt run ./my_project --no-skip  # run every simulation again
    stilt run --backend slurm         # run as a Slurm job array, and wait
    stilt run --task 3/10             # run task 3 of 10 here (a job array's task)
    stilt run --receptors ids.txt     # run only the receptors listed in ids.txt
    stilt submit                      # submit a Slurm job array and return
    stilt status                      # count finished simulations; list the settings folders

``stilt run`` exits with 0 when every simulation it ran is complete, 1 when
some failed, and 3 when some did not finish because the run was
interrupted (Ctrl-C, or SIGTERM from a scheduler). ``stilt check`` exits
with 0 when it finds nothing and 1 when it lists anything. Every command
exits with 2 when its command line is wrong: an unknown option, a bad value,
or a directory that is not a project. ``stilt run``, ``stilt submit``, and
``stilt status`` also exit with 2 when a met directory does not exist or
holds no ARL file, which ``stilt check`` lists.

An error the command did not expect ends it too, after its traceback: with
3 when the error is a connection or an I/O error (a bucket that cannot be
reached or is busy, a stale disk), which a retry may get past, and with 4 for
any other (a bug in PYSTILT or the project, a bucket in the wrong region, a
403 or a 404, which a retry will repeat).
``_error_exit_code`` sorts them.
"""

from __future__ import annotations

import errno
import json
import logging
import re
import socket
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

from stilt.execution.config import ExecutionConfig
from stilt.execution.runner import resolve_compute_root
from stilt.project import STATES, Project

#: Exit codes. 2 is also Click's own for an unknown option or command, so a
#: driver never takes a typo for a run to retry. 3 says "run it again": the
#: run was stopped, or hit a connection or I/O error; what finished is on
#: disk. 4 says "do not": an unexpected error that a retry would repeat. 1 is
#: the code of a run whose simulations failed, and no other.
EXIT_COMPLETE, EXIT_FAILED, EXIT_USAGE, EXIT_INTERRUPTED, EXIT_BUG = 0, 1, 2, 3, 4

#: Errors from the cloud libraries that do not inherit from ``OSError`` and
#: mean the network failed or the service is briefly unavailable. Matched by
#: class name along an exception's ``__mro__``, because botocore, aiohttp and
#: urllib3 are optional and not imported here.
_NETWORK_ERRORS = frozenset(
    {
        # botocore (also the errors s3fs and aiobotocore raise)
        "EndpointConnectionError",
        "ProxyConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "ConnectionClosedError",
        "IncompleteReadError",
        # aiohttp (s3fs, gcsfs, and fsspec's HTTP filesystem)
        "ClientConnectionError",
        "ServerConnectionError",
        "ClientPayloadError",
        # urllib3
        "ProtocolError",
        # ftplib: a 4xx reply ("421 too many users"), transient by the FTP
        # standard; error_perm (5xx), error_reply and error_proto are not
        "error_temp",
    }
)

#: ``OSError`` subclasses that name a path, not the network: the file is not
#: there, or may not be used, and a retry meets the same answer.
_PATH_ERRORS = (
    FileNotFoundError,
    FileExistsError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
)

#: HTTP statuses of a service that is busy or briefly down. Any other 3xx, 4xx
#: or 5xx is an answer that a retry repeats: a bucket in another region, a bad
#: request, a 403 or a 404.
_BUSY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

#: Codes of botocore's ``ClientError`` for a service that asks the caller to slow
#: down or try again, whatever the status that carries them.
_BUSY_CODES = frozenset(
    {
        "Throttling",
        "ThrottlingException",
        "RequestLimitExceeded",
        "RequestThrottled",
        "SlowDown",
        "RequestTimeout",
        "TooManyRequestsException",
        "ProvisionedThroughputExceededException",
        "ServiceUnavailable",
        "InternalError",
        "OperationAborted",
    }
)

#: Errnos of an ``OSError`` that no type, status or cause explains, which are
#: input/output trouble that goes away: an I/O error, a stale NFS handle, a busy
#: or interrupted call, a connection that failed. Any other errno (EINVAL,
#: ENOSPC, EROFS, none at all) is a refusal or a full disk that a retry meets again.
_TRANSIENT_ERRNOS = frozenset(
    getattr(errno, name)
    for name in (
        "EIO",
        "EREMOTEIO",
        "ESTALE",
        "EBUSY",
        "EAGAIN",
        "EINTR",
        "ETIMEDOUT",
        "ECONNRESET",
        "ECONNREFUSED",
        "ECONNABORTED",
        "ENETDOWN",
        "ENETUNREACH",
        "EHOSTDOWN",
        "EHOSTUNREACH",
    )
    if hasattr(errno, name)
)


def _http_status(error: BaseException) -> int | None:
    """
    Return the HTTP status an error from a cloud or HTTP library carries, or None.

    Read from botocore's ``ClientError.response`` (a dict), from the
    ``status_code`` or ``status`` of the response a ``requests`` error holds,
    and from the error's own ``status``, ``status_code`` or ``code`` (aiohttp,
    gcsfs, ``urllib.error.HTTPError``).
    """
    response = getattr(error, "response", None)
    if isinstance(response, dict):  # botocore's ClientError
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status if isinstance(status, int) else None
    holders = (response, error) if response is not None else (error,)
    for holder in holders:
        for attribute in ("status", "status_code", "code"):
            status = getattr(holder, attribute, None)
            if isinstance(status, int) and 100 <= status <= 599:
                return status
    return None


def _causes(error: BaseException) -> Iterator[BaseException]:
    """Yield an error and what it was raised from or while handling, outermost first."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        else:
            current = None if current.__suppress_context__ else current.__context__


def _says(error: BaseException) -> int | None:
    """Return 3 or 4 when this one error says so by its status, code or type; else None."""
    response = getattr(error, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    status = _http_status(error)
    if code in _BUSY_CODES or status in _BUSY_STATUSES:
        return EXIT_INTERRUPTED
    if status is not None and status >= 300:
        return EXIT_BUG
    if isinstance(error, ConnectionError | TimeoutError | socket.gaierror):
        return EXIT_INTERRUPTED
    if not _NETWORK_ERRORS.isdisjoint(c.__name__ for c in type(error).__mro__):
        return EXIT_INTERRUPTED
    return None


def _error_exit_code(error: Exception) -> int:
    """
    Return the exit code for an error the command did not expect.

    3 (:data:`EXIT_INTERRUPTED`, "run it again") when the error is a
    connection or I/O error that a retry may get past, 4
    (:data:`EXIT_BUG`, "do not") when it is any other.

    The error and its ``__cause__`` or ``__context__`` chain are read,
    outermost first, and the first that says decides. An error says 3 by a
    busy service's HTTP status (408, 429, 500, 502, 503, 504) or botocore
    code (:data:`_BUSY_CODES`), by being a ``ConnectionError``,
    ``TimeoutError`` or name-resolution error, or by a class named in
    :data:`_NETWORK_ERRORS` (which includes ftplib's ``error_temp``, a 4xx
    reply): the cloud libraries are optional, so the ``__mro__``'s class
    names are read, not the classes imported. It says 4 by any other 3xx, 4xx
    or 5xx status (a bucket in another region, a bad request, a 403, a 404).
    s3fs turns a ``ClientError`` into a plain ``OSError`` and keeps it in
    ``__cause__``, and fsspec's HTTP filesystem raises ``FileNotFoundError``
    for every failure, so the chain is read before a path error (a file that
    is not there or may not be used) is taken at its word. If nothing in the
    chain says, an ``OSError`` that is not about a path is 3 when its errno
    is input/output trouble that goes away (:data:`_TRANSIENT_ERRNOS`, such
    as s3fs's ``EIO`` for a code it does not know), and anything else,
    ftplib's ``error_perm``, ``error_reply`` and ``error_proto`` included,
    is 4.

    Parameters
    ----------
    error : Exception
        The error that ended the command.

    Returns
    -------
    int
        3 or 4.
    """
    chain = list(_causes(error))
    for each in chain:
        verdict = _says(each)
        if verdict is not None:
            return verdict
    transient = any(
        isinstance(each, OSError)
        and not isinstance(each, _PATH_ERRORS)
        and each.errno in _TRANSIENT_ERRNOS
        for each in chain
    )
    return EXIT_INTERRUPTED if transient else EXIT_BUG


class _Group(TyperGroup):
    """
    The ``stilt`` command group, which ends a command that raised in an exit code.

    Python's own exit code for an uncaught error is 1, which is also the
    code of a run whose simulations failed. A driver that decides on a
    retry from the code cannot tell the two apart. So an error the command
    did not handle prints its traceback and ends with
    ``_error_exit_code``'s 3 or 4, and a ``KeyboardInterrupt`` that no
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
            code = _error_exit_code(error)
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
    if not Project(resolved).config_path.exists():
        typer.echo(
            f"Error: '{resolved}' does not look like a STILT project directory "
            "(no config.yaml found).",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)
    return str(resolved)


def _open(project: str | Path | None, output: str | None = None) -> Project:
    """
    Open the project, exiting with 2 when the met of a variant cannot be read.

    A variant's settings folder is named by its met's weather product, read
    from the header of the met's first file. Without the file every command
    that counts or runs simulations would stop with a traceback;
    ``stilt check`` lists the same problem as a finding.
    """
    opened = Project(_resolve_project(project), output=output)
    try:
        for variant in opened.variants.values():
            _ = variant.met_settings
    except FileNotFoundError as error:
        _fail(str(error))
    return opened


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


#: Rows of each check ``stilt check`` prints; ``project.check()`` has them all.
_CHECK_ROWS = 10


@app.command()
def check(project: str | None = _PROJECT_ARG) -> None:
    """
    List what would stop a project's runs, before any run.

    For each variant, checks that the met's directory exists and holds ARL
    files, that every met file each receptor's run needs is there (naming
    the hours), that the first of them holds more than one time period,
    which HYSPLIT needs, that each receptor is inside the met's crop box and the
    footprint grid, and that an averaging-kernel table matches the
    receptors. Nothing is run or downloaded. It looks for files, not inside
    them: a run that passes can still fail as MET_COVERAGE, when a met file
    is damaged or cut short, or the receptor is outside the files' domain.
    Exits with 0 when it finds nothing, 1 when it lists anything, and 2 when
    the command line is wrong or the project's files are invalid.
    """
    opened = Project(_resolve_project(project))
    try:
        problems = opened.check()
    except ValueError as error:
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
    help=(
        "File of receptor ids, one per line, or the receptors.parquet a Slurm "
        "submission writes. Only those receptors run."
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


def _read_receptors(path: Path) -> tuple[list[str], pd.DataFrame | None]:
    """
    Return the receptor ids in a file, and their checked rows when the file holds them.

    A text file lists ids, one per line; blank lines and # comments are
    skipped. A Parquet file is the ``receptors.parquet`` a Slurm submission
    writes: the checked rows of its receptors, in the order they run.
    """
    try:
        if path.suffix == ".parquet":
            rows = pd.read_parquet(path)
            return list(dict.fromkeys(rows["receptor"].astype(str))), rows
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
    receptor_ids: list[str] | None = None,
    receptor_rows: pd.DataFrame | None = None,
    task: tuple[int, int] | None = None,
    execution_file: Path | None = None,
    output: str | None = None,
) -> tuple[Project, dict[str, Any]]:
    """
    Open the project, print what is about to run, and return the run's options.

    With *receptor_rows* (from ``--receptors receptors.parquet``) the
    project takes its receptors from them, not from ``receptors.csv``.
    """
    opened = _open(project, output=output)
    if receptor_rows is not None:
        opened._take_rows(receptor_rows)
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
    output: str | None = _OUTPUT,
) -> None:
    """
    Run every unfinished simulation in a project, and wait until they are done.

    Runs the transport model for each receptor and variant, then the footprint when the
    variant has a grid. Simulations whose outputs exist are skipped unless
    --no-skip is given. With the Slurm backend this submits a job array and
    waits for it; use stilt submit to return as soon as it is submitted.

    --receptors limits the run to the receptors listed in a file. --task
    I/N runs one share of them here, for each task of a job array or a
    Kubernetes indexed Job.

    Exits with 0 when every simulation that ran is complete, 1 when some
    failed, 3 when some did not finish (the run was stopped), and 2 when
    the command line is wrong or a met cannot be read (nothing ran). An
    error the run did not expect prints its traceback and exits with 3 when
    it is a connection or I/O error, such as an output bucket that cannot
    be reached, which a retry may get past, and with 4 when it is any
    other, which a retry would repeat.
    """
    receptor_ids, rows = (
        (None, None) if receptors is None else _read_receptors(receptors)
    )
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
        receptor_rows=rows,
        task=share,
        execution_file=execution,
        output=output,
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
    output: str | None = _OUTPUT,
) -> None:
    """
    Submit every unfinished simulation in a project to Slurm, and return.

    The receptors are split among a Slurm job array's tasks, with the
    resources under execution in config.yaml. --receptors limits it to the
    receptors listed in a file. Check on them with stilt status.
    """
    receptor_ids, rows = (
        (None, None) if receptors is None else _read_receptors(receptors)
    )
    opened, options = _start(
        project,
        backend="slurm",
        n_workers=n_workers,
        cpus=cpus,
        no_skip=no_skip,
        compute_root=compute_root,
        waits=False,
        receptor_ids=receptor_ids,
        receptor_rows=rows,
        output=output,
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
    files it holds, which variants use it, and, for a folder no variant
    uses, how its settings differ from the variant of its name. Exits with
    2 when the command line is wrong or a met cannot be read.
    """
    opened = _open(project, output=output)
    if as_json:
        typer.echo(json.dumps(_status_data(opened), indent=2))
        return
    _print_status(opened)
    _print_folders(opened)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _counts(states: Iterable[str]) -> str:
    """Return a ``total=  complete=  failed=  interrupted=  pending=`` line."""
    return "  ".join(f"{name}={n}" for name, n in _state_counts(states).items())


def _print_status(project: Project, ran: pd.DataFrame | None = None) -> None:
    """
    Print a project status summary, per variant when there are several.

    With *ran*, the status table of a run, it sums up those simulations
    only.
    """
    table = project.status() if ran is None else ran
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


def _status_data(project: Project) -> dict[str, Any]:
    """
    Return what ``stilt status`` prints, as plain data.

    ``simulations`` and ``variants`` count the simulations in each state,
    ``failed`` counts the failed ones by reason, and ``folders`` lists the
    settings folders as :meth:`stilt.Project.folders` does.
    """
    table = project.status()
    failed = table.loc[table.state == "failed", "reason"]
    folders = project.folders()
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
    Print the receptors and variants checked, then the problems found as a table.

    The first :data:`_CHECK_ROWS` rows of each check are shown, with a count
    of the rest.
    """
    typer.echo(f"Project: {project.directory}  receptors={len(project.receptors)}")
    variants = ", ".join(
        f"{name} (met {v.met}, {'footprint' if v.footprint is not None else 'particles only'})"
        for name, v in project.variants.items()
    )
    typer.echo(f"Variants: {variants}")
    if problems.empty:
        typer.echo("No problems found.")
        return
    lines: list[tuple[str, str, str, str]] = []
    for name, rows in problems.groupby("check", sort=False):
        for row in rows.head(_CHECK_ROWS).itertuples():
            receptor = "" if pd.isna(row.receptor) else str(row.receptor)
            lines.append((str(name), str(row.variant), receptor, str(row.message)))
        if len(rows) > _CHECK_ROWS:
            more = (
                f"... {len(rows) - _CHECK_ROWS} more (project.check() lists them all)"
            )
            lines.append((str(name), "", "", more))
    header = ("check", "variant", "receptor", "message")
    widths = [max(len(line[i]) for line in [header, *lines]) for i in range(3)]
    typer.echo()
    for line in [header, *lines]:
        cells = [f"{t:<{w}}" for t, w in zip(line[:3], widths, strict=True)]
        typer.echo("  ".join([*cells, line[3]]).rstrip())
    counts = problems.groupby("check", sort=False).size()
    listed = ", ".join(f"{name} {n}" for name, n in counts.items())
    noun = "problem" if len(problems) == 1 else "problems"
    typer.echo(f"\nFound {len(problems)} {noun}: {listed}")


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
