"""
Running a project: finding the receptors with missing results and handing them to workers.

With ``backend: slurm``, :func:`submit` writes a job array script into
``_slurm/<stamp>/`` in the project and submits it with ``sbatch``. Each task
of the array runs ``stilt run --task``, so a task is a command line that can
be read, and run again, by hand.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import yaml

from stilt._paths import absolute
from stilt.execution.config import ExecutionConfig

if TYPE_CHECKING:
    import pandas as pd

    from stilt.project import Project

logger = logging.getLogger(__name__)


#: Seconds before a Slurm task's time limit that Slurm sends it SIGUSR1,
#: which stops the task and requeues it (``--signal=B:USR1@120``).
NOTICE_SECONDS = 120

#: Times a Slurm task requeues itself before it stops for good.
MAX_REQUEUES = 10

#: Slurm job states of a task that has not ended.
_ACTIVE = frozenset(
    {
        "PENDING",
        "RUNNING",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESIZING",
        "SUSPENDED",
        "CONFIGURING",
        "COMPLETING",
        "SIGNALING",
        "STAGE_OUT",
    }
)


# ---------------------------------------------------------------------------
# Shares of the receptors
# ---------------------------------------------------------------------------


def task_share(receptor_ids: list[str], task: int, n_tasks: int) -> list[str]:
    """
    Return task *task*'s share of *receptor_ids*, split *n_tasks* ways.

    The share is every *n_tasks*-th receptor, starting at *task*, which runs
    from 0 to ``n_tasks - 1`` as the index of a job array does.

    Raises
    ------
    ValueError
        If *task* is not from 0 to ``n_tasks - 1``.
    """
    if n_tasks < 1 or not 0 <= task < n_tasks:
        raise ValueError(
            f"A task runs from 0 to n_tasks - 1; got task {task} of {n_tasks}."
        )
    return receptor_ids[task::n_tasks]


# ---------------------------------------------------------------------------
# Running a project
# ---------------------------------------------------------------------------


def _project_slug(directory: str | Path) -> str:
    """Return a lowercase, hyphenated name for a project directory, for the Slurm job name."""
    name = Path(str(directory).rstrip("/")).name or "project"
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower().replace("_", "-"))
    return re.sub(r"-{2,}", "-", slug).strip("-") or "project"


def resolve_compute_root(
    project: Project, compute_root: str | Path | None = None
) -> Path:
    """
    Return the compute root of *project*: the directory its workdirs are made in.

    That is *compute_root* when given, else the ``PYSTILT_COMPUTE_ROOT``
    environment variable when it is set and not empty, else
    ``$TMPDIR/pystilt/<project name>``. The path is absolute and resolved, so
    a worker handed it gets the same directory.
    """
    if compute_root is None:
        compute_root = os.environ.get("PYSTILT_COMPUTE_ROOT") or None
    if compute_root is not None:
        return absolute(compute_root)
    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return absolute(Path(tmp_root) / "pystilt" / project.name)


def _pending(
    project: Project,
    skip_existing: bool,
    receptors: Iterable[str] | None = None,
    task: tuple[int, int] | None = None,
) -> list[str]:
    """
    Return the ids of the receptors to run, each once.

    *receptors* limits them to those, in that order. *task* ``(i, n)``
    takes share ``i`` of ``n`` of them (of all the project's receptors
    without *receptors*) before the complete ones are dropped, so the tasks
    of a job array split the receptors the same way whenever each starts.
    """
    sims = project.simulations
    every = list(dict.fromkeys(sims["receptor"]))
    if receptors is None:
        chosen = every
    else:
        chosen = list(dict.fromkeys(str(r) for r in receptors))
        known = set(every)
        unknown = [r for r in chosen if r not in known]
        if unknown:
            raise ValueError(
                f"{len(unknown)} receptor ids are not in this project, such as "
                f"{unknown[:3]}."
            )
    if task is not None:
        chosen = task_share(chosen, *task)
    if skip_existing and chosen:
        left = set(project.incomplete(sims[sims["receptor"].isin(chosen)])["receptor"])
        chosen = [r for r in chosen if r in left]
    return chosen


def _status(project: Project, receptor_ids: list[str]) -> pd.DataFrame:
    """Return the status table of the simulations of *receptor_ids*."""
    sims = project.simulations
    return project.status(sims[sims["receptor"].isin(receptor_ids)])


def run(
    project: Project,
    *,
    receptors: Iterable[str] | None = None,
    task: tuple[int, int] | None = None,
    execution: ExecutionConfig | None = None,
    skip_existing: bool = True,
    compute_root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run every simulation of a project that has not finished, and wait for it.

    Each receptor with missing results runs once: the transport model once for each
    distinct set of transport settings whose particles are missing, then the
    footprint of every variant that has a grid. With ``backend: local`` the
    receptors run in this process (``cpus`` at a time). With
    ``backend: slurm`` they are submitted as one job array (:func:`submit`)
    and this waits for it.

    Parameters
    ----------
    project : Project
        Project to run.
    receptors : iterable of str, optional
        Run only these receptors (their ids), in this order.
    task : tuple of (int, int), optional
        ``(i, n)``: run share ``i`` of ``n`` of the receptors here, in this
        process, whatever the backend. The share is every ``n``-th
        receptor in project order (or in *receptors* order), starting at
        ``i`` (:func:`task_share`), whether it is complete or not, and the
        complete ones are then skipped. Each task of a job array or a
        Kubernetes indexed Job runs one share, and every task splits the
        receptors the same way whenever it starts.
    execution : ExecutionConfig, optional
        Where to run and with what resources. Defaults to the ``execution``
        settings of the project's config.
    skip_existing : bool, default True
        Skip simulations whose results all exist. ``False`` runs every
        simulation again.
    compute_root : str or Path, optional
        Directory the workdirs are made in, one per simulation
        (:func:`resolve_compute_root`). On Slurm it is resolved on the
        compute node unless given here.

    Returns
    -------
    pandas.DataFrame
        The status table (:meth:`stilt.Project.status`) of the
        simulations that ran: ``state`` says which are complete, which
        failed and why, and which did not finish.

    Raises
    ------
    ValueError
        If a receptor id is not in the project, or *task* is not from 0 to
        ``n - 1``.

    Notes
    -----
    In a Slurm job, a task (``task`` given) that stops with work left
    requeues itself with ``scontrol requeue`` when it was preempted (Slurm
    set the job's ``PreemptTime``) or got SIGUSR1, which the job array
    script asks Slurm to send :data:`NOTICE_SECONDS` before the time
    limit. It starts again later and skips what it finished. A task
    stopped by ``scancel`` is not requeued.
    """
    execution = execution if execution is not None else project.config.execution
    pending = _pending(project, skip_existing, receptors, task)
    if not pending:
        logger.info("run: every simulation is complete; nothing to do")
    elif execution.backend == "slurm" and task is None:
        _wait(_submit(project, pending, execution, skip_existing, compute_root))
    else:
        logger.info("run(%s): %d receptors", ", ".join(project.variants), len(pending))
        # In this process, so Ctrl-C and SIGTERM stop the workers cleanly
        # and progress prints as it happens.
        from .worker import run_receptors

        def work() -> None:
            run_receptors(
                project,
                pending,
                compute_root=resolve_compute_root(project, compute_root),
                execution=execution,
                skip_existing=skip_existing,
            )

        if task is None:
            work()
        else:
            return _run_task(project, pending, work)
    return _status(project, pending)


def submit(
    project: Project,
    *,
    receptors: Iterable[str] | None = None,
    execution: ExecutionConfig | None = None,
    skip_existing: bool = True,
    compute_root: str | Path | None = None,
) -> str | None:
    """
    Submit every simulation of a project that has not finished to Slurm.

    The receptors with missing results are split among up to ``n_workers``
    tasks of one job array, and this returns once it is submitted. The
    submission is a folder ``_slurm/<date_time>_<id>/`` in the project:
    ``receptors.txt`` lists the receptors to run, ``execution.yaml`` holds
    the execution settings, ``job.sh`` is the script given to ``sbatch``,
    and ``<task>.log`` is each task's log. Task ``i`` runs
    ``stilt run <project> --receptors receptors.txt --task i/N``. A task
    that is preempted or nears its time limit requeues itself and skips
    what it finished (see :func:`run`).

    Parameters
    ----------
    project, receptors, execution, skip_existing, compute_root
        As for :func:`run`.

    Returns
    -------
    str or None
        The Slurm job id of the array, or ``None`` when nothing needs to
        run. ``squeue -j <id>`` follows it and ``scancel <id>`` stops it.

    Raises
    ------
    ValueError
        If the execution backend is not Slurm.
    """
    execution = execution if execution is not None else project.config.execution
    if execution.backend != "slurm":
        raise ValueError(
            f"submit sends work to Slurm, and this run's backend is "
            f"{execution.backend!r}. Use run(), or set execution.backend to slurm."
        )
    pending = _pending(project, skip_existing, receptors)
    if not pending:
        logger.info("submit: every simulation is complete; nothing to do")
        return None
    return _submit(project, pending, execution, skip_existing, compute_root)


def _sbatch_lines(options: dict[str, Any]) -> list[str]:
    """Return ``#SBATCH`` lines; ``True`` is a bare flag, and ``None`` and ``False`` are left out."""
    lines = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = "--" + str(key).replace("_", "-")
        lines.append(f"#SBATCH {flag}" if value is True else f"#SBATCH {flag}={value}")
    return lines


def job_script(
    project: Project,
    execution: ExecutionConfig,
    folder: Path,
    n_tasks: int,
    *,
    skip_existing: bool = True,
    compute_root: str | Path | None = None,
) -> str:
    """
    Return the ``sbatch`` script of a job array whose tasks each run a share of the receptors.

    Task ``i`` runs ``stilt run <project> --receptors <folder>/receptors.txt
    --task i/n_tasks --execution <folder>/execution.yaml`` with the Python
    that called this. The ``#SBATCH`` lines come from *execution*: its
    resources, ``--requeue``, ``--signal=B:USR1@120`` (see :func:`run`),
    and its ``slurm`` options, which can replace any of them. Its ``setup``
    commands run before the task.
    """
    options: dict[str, Any] = {
        "job-name": f"pystilt-{_project_slug(project.directory)}",
        "array": f"0-{n_tasks - 1}"
        + (f"%{execution.array_parallelism}" if execution.array_parallelism else ""),
        "cpus-per-task": execution.cpus,
        "time": execution.time_minutes,
        "mem": execution.mem,
        "partition": execution.partition,
        "account": execution.account,
        "qos": execution.qos,
        "output": folder / "%a.log",
        # A requeued task writes on at the end of its log.
        "open-mode": "append",
        "requeue": True,
        "signal": f"B:USR1@{NOTICE_SECONDS}",
    }
    for key, value in execution.slurm.items():
        options[str(key).replace("_", "-")] = value

    command = [
        sys.executable,
        "-m",
        "stilt",
        "run",
        str(project.directory),
        "--receptors",
        str(folder / "receptors.txt"),
        "--task",
        f"$SLURM_ARRAY_TASK_ID/{n_tasks}",
        "--execution",
        str(folder / "execution.yaml"),
    ]
    if compute_root is not None:
        command += ["--compute-root", str(resolve_compute_root(project, compute_root))]
    # $SLURM_ARRAY_TASK_ID must expand, so that word is quoted with "".
    words = [f'"{w}"' if w.startswith("$") else shlex.quote(w) for w in command]
    # A task's log starts with where and when it ran, again after a requeue,
    # so a task on a bad node can be found from its log.
    header = (
        'echo "$(date -u +%FT%TZ) $SLURMD_NODENAME '
        "job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} "
        'restart ${SLURM_RESTART_COUNT:-0}"'
    )
    lines = ["#!/bin/bash", *_sbatch_lines(options), "", header, *execution.setup]
    if not skip_existing:
        # A requeued task keeps what it finished before.
        lines += [
            'skip="--no-skip"',
            '[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && skip=""',
        ]
        words.append("$skip")
    # exec, so this process is the one Slurm signals.
    lines.append("exec " + " ".join(words))
    return "\n".join(lines) + "\n"


def _submit(
    project: Project,
    pending: list[str],
    execution: ExecutionConfig,
    skip_existing: bool,
    compute_root: str | Path | None,
) -> str:
    """Write a submission folder for *pending* receptors, submit its job array, and return the job id."""
    # One folder per submission, so a later array never overwrites these.
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    folder = project.directory / "_slurm" / stamp
    folder.mkdir(parents=True)
    (folder / "receptors.txt").write_text("\n".join(pending) + "\n")
    (folder / "execution.yaml").write_text(
        yaml.safe_dump(execution.model_dump(exclude_unset=True), sort_keys=False)
    )
    n_tasks = max(1, min(execution.n_workers, len(pending)))
    script = folder / "job.sh"
    script.write_text(
        job_script(
            project,
            execution,
            folder,
            n_tasks,
            skip_existing=skip_existing,
            compute_root=compute_root,
        )
    )
    result = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch refused {script}: {result.stderr.strip() or result.stdout.strip()}"
        )
    job_id = result.stdout.strip().split(";")[0]
    logger.info("Submitted job: %s (%d tasks); logs in %s", job_id, n_tasks, folder)
    return job_id


def _task_states(job_id: str) -> dict[str, tuple[str, str]] | None:
    """
    Return ``{task: (state, exit code)}`` for the tasks of a job array, from ``sacct``.

    ``None`` when ``sacct`` cannot answer, as when the accounting database
    is slow to hear of a new job.
    """
    result = subprocess.run(
        ["sacct", "-j", job_id, "-X", "-n", "-P", "--format=JobID,State,ExitCode"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    states = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) == 3:
            task, state, code = parts
            states[task] = (state.split()[0] if state else "", code)
    return states or None


def _wait(job_id: str, poll: float = 30.0) -> None:
    """
    Wait until every task of a job array has ended, polling ``sacct``.

    A task that ended other than complete, or with failed simulations
    (exit 1) or interrupted ones (exit 2), is logged with its state; its
    simulations stay pending in the status table.
    """
    states = _task_states(job_id)
    while states is None or any(s in _ACTIVE for s, _ in states.values()):
        time.sleep(poll)
        states = _task_states(job_id)
    for task, (state, code) in sorted(states.items()):
        if state == "COMPLETED" or (state == "FAILED" and code in ("1:0", "2:0")):
            continue
        logger.warning(
            "Slurm task %s ended %s (exit %s); see its log", task, state, code
        )


# ---------------------------------------------------------------------------
# A task that is told to stop
# ---------------------------------------------------------------------------


class _Notice:
    """
    Whether SIGUSR1 has arrived.

    While ``armed``, its first arrival raises ``KeyboardInterrupt`` to stop
    the work. Later ones, and any after the work, are only recorded, so
    they cannot cut a cleanup short or end the process.
    """

    def __init__(self) -> None:
        self.received = False
        self.armed = True

    def __bool__(self) -> bool:
        return self.received

    def handle(self, signum: int, frame: object) -> None:
        first = not self.received
        self.received = True
        if first and self.armed:
            raise KeyboardInterrupt


@contextlib.contextmanager
def _usr1_noticed() -> Iterator[_Notice]:
    """Record SIGUSR1 inside the block (:class:`_Notice`); outside the main thread, where handlers cannot be set, nothing is."""
    notice = _Notice()
    if threading.current_thread() is not threading.main_thread():
        yield notice
        return
    previous = signal.signal(signal.SIGUSR1, notice.handle)
    try:
        yield notice
    finally:
        signal.signal(signal.SIGUSR1, previous)


def _run_task(
    project: Project, pending: list[str], work: Callable[[], None]
) -> pd.DataFrame:
    """
    Run a task's receptors and return their status table; requeue the task when Slurm stopped it.

    A task that stops with simulations left is requeued when it got
    SIGUSR1 (the time limit is near) or was preempted.
    """
    with _usr1_noticed() as notice:
        try:
            work()
        except KeyboardInterrupt:
            if not notice:
                raise
        finally:
            notice.armed = False
        table = _status(project, pending)
        unfinished = bool((table["state"] == "pending").any())
        if unfinished and (notice or _preempted()):
            _requeue()
    return table


def _preempted() -> bool:
    """Whether Slurm has selected this job for preemption (``scontrol show job`` gives a ``PreemptTime``)."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return False
    result = subprocess.run(
        ["scontrol", "show", "job", job_id], capture_output=True, text=True, check=False
    )
    match = re.search(r"PreemptTime=(\S+)", result.stdout)
    return match is not None and match.group(1) not in ("None", "Unknown")


def _requeue() -> bool:
    """Put this Slurm task back in the queue with ``scontrol requeue``; return whether it was."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        logger.warning("Told to stop outside a Slurm job; not requeued.")
        return False
    restarts = int(os.environ.get("SLURM_RESTART_COUNT") or 0)
    if restarts >= MAX_REQUEUES:
        logger.warning(
            "Slurm task %s has been requeued %d times; not again. Run the "
            "project again, or raise execution.time.",
            job_id,
            restarts,
        )
        return False
    result = subprocess.run(
        ["scontrol", "requeue", job_id], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        logger.warning("scontrol requeue %s failed: %s", job_id, result.stderr.strip())
        return False
    logger.info(
        "Requeued Slurm task %s; it starts again and skips what finished.", job_id
    )
    return True


__all__ = [
    "MAX_REQUEUES",
    "NOTICE_SECONDS",
    "job_script",
    "resolve_compute_root",
    "run",
    "submit",
    "task_share",
]
