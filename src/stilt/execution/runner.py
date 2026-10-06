"""Running a project: finding the receptors with missing results and handing them to workers."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import submitit

from stilt._paths import absolute
from stilt.execution.config import ExecutionConfig

if TYPE_CHECKING:
    import pandas as pd

    from stilt.project import Project

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The unit of work
# ---------------------------------------------------------------------------


class Batch(submitit.helpers.Checkpointable):
    """
    A batch of receptors of one project, run by one worker.

    Calling it opens the project and runs the receptors with *execution*'s
    settings, ``cpus`` at a time. On Slurm it is one array task. When the task is
    preempted or runs out of time, submitit submits it again
    (:meth:`checkpoint`), and the second run skips the receptors the first
    one finished.

    Parameters
    ----------
    project : str
        Project directory. Its ``config.yaml`` and ``receptors.csv`` must
        already hold the settings and receptors.
    receptor_ids : list of str
        Receptors to run.
    execution : ExecutionConfig
        The run's execution settings, which may differ from the project's
        ``config.yaml``: ``cpus``, ``timeout``, and ``keep_scratch``.
    compute_root : str, optional
        Scratch directory under which HYSPLIT runs.
    skip_existing : bool, default True
        Keep particles and footprints that already exist.
    """

    def __init__(
        self,
        project: str,
        receptor_ids: list[str],
        *,
        execution: ExecutionConfig,
        compute_root: str | None = None,
        skip_existing: bool = True,
    ) -> None:
        self.project = project
        self.receptor_ids = list(receptor_ids)
        self.execution = execution
        self.compute_root = compute_root
        self.skip_existing = skip_existing

    def __call__(self) -> None:
        """Run the batch. The results and failure records are in the output directory."""
        from stilt.project import Project

        from .worker import run_receptors

        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        logging.getLogger("stilt.execution.worker").setLevel(logging.INFO)
        project = Project(self.project)
        run_receptors(
            project,
            self.receptor_ids,
            # Worked out here, in the task, so it is this node's scratch.
            compute_root=resolve_compute_root(project, self.compute_root),
            execution=self.execution,
            skip_existing=self.skip_existing,
        )

    def checkpoint(
        self, *args: Any, **kwargs: Any
    ) -> submitit.helpers.DelayedSubmission:
        """Return this batch to submit again, keeping what finished before the interruption."""
        self.skip_existing = True
        return super().checkpoint(*args, **kwargs)


def split(receptor_ids: list[str], n: int) -> list[list[str]]:
    """Split receptor ids round-robin into at most *n* batches, none empty."""
    n = max(1, min(n, len(receptor_ids)))
    return [receptor_ids[i::n] for i in range(n)]


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


def slurm_parameters(execution: ExecutionConfig, *, job_name: str) -> dict[str, Any]:
    """Return *execution* as the keyword arguments of submitit's Slurm executor."""
    additional = {str(k).replace("_", "-"): v for k, v in execution.slurm.items()}
    # Without this, scontrol cannot requeue a preempted or timed-out task.
    additional.setdefault("requeue", True)
    params: dict[str, Any] = {
        "slurm_job_name": job_name,
        "slurm_cpus_per_task": execution.cpus,
        "slurm_additional_parameters": additional,
    }
    optional = {
        # Minutes, which is the form submitit needs to tell a timeout from a
        # preemption when it decides whether to requeue.
        "slurm_time": execution.time_minutes,
        "slurm_mem": execution.mem,
        "slurm_partition": execution.partition,
        "slurm_account": execution.account,
        "slurm_qos": execution.qos,
        "slurm_array_parallelism": execution.array_parallelism,
        "slurm_setup": execution.setup or None,
    }
    params.update({k: v for k, v in optional.items() if v is not None})
    return params


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
    Return the scratch directory under which HYSPLIT runs for *project*.

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

    Each receptor with missing results runs once: HYSPLIT once for each
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
        Scratch directory under which HYSPLIT runs
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
    RuntimeError
        If a Slurm task failed, was cancelled, or ran out of requeues.
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

        run_receptors(
            project,
            pending,
            compute_root=resolve_compute_root(project, compute_root),
            execution=execution,
            skip_existing=skip_existing,
        )
    return _status(project, pending)


def submit(
    project: Project,
    *,
    receptors: Iterable[str] | None = None,
    execution: ExecutionConfig | None = None,
    skip_existing: bool = True,
    compute_root: str | Path | None = None,
) -> list[submitit.Job[Any]]:
    """
    Submit every simulation of a project that has not finished to Slurm.

    The receptors with missing results are split among ``n_workers`` tasks
    of one job array, and this returns once it is submitted. A task that is
    preempted or runs out of time is submitted again and skips what it
    finished. Logs and submitit's files are in ``slurm/<date_time>_<id>/`` in
    the project.

    Parameters
    ----------
    project, receptors, execution, skip_existing, compute_root
        As for :func:`run`.

    Returns
    -------
    list of submitit.Job
        One per array task, empty when nothing needs to run.
        ``job.wait()``, ``job.result()``, ``job.stdout()``, and
        ``job.cancel()`` follow and control them.

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
        return []
    return _submit(project, pending, execution, skip_existing, compute_root)


def _submit(
    project: Project,
    pending: list[str],
    execution: ExecutionConfig,
    skip_existing: bool,
    compute_root: str | Path | None,
) -> list[submitit.Job[Any]]:
    """Submit *pending* receptors as one job array and return its tasks."""
    # One folder per submission, so a later array never overwrites these.
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    executor = submitit.AutoExecutor(
        folder=project.directory / "slurm" / stamp, cluster="slurm"
    )
    executor.update_parameters(
        **slurm_parameters(
            execution, job_name=f"pystilt-{_project_slug(project.directory)}"
        )
    )
    # A compute root that was not asked for is left to each compute node,
    # whose TMPDIR is its own. One that was is made absolute here, since the
    # task may start in another directory.
    task_root = (
        None
        if compute_root is None
        else str(resolve_compute_root(project, compute_root))
    )
    batches = [
        Batch(
            str(project.directory),
            ids,
            execution=execution,
            compute_root=task_root,
            skip_existing=skip_existing,
        )
        for ids in split(pending, execution.n_workers)
    ]
    with executor.batch():
        jobs = [executor.submit(batch) for batch in batches]
    logger.info(
        "Submitted job: %s (%d tasks)", str(jobs[0].job_id).split("_")[0], len(jobs)
    )
    return jobs


def _wait(jobs: list[submitit.Job[Any]]) -> None:
    """
    Wait until every task has left the queue.

    Raises
    ------
    RuntimeError
        If any task failed, was cancelled, or ran out of requeues. A task
        that was preempted or ran out of time is requeued and counts only by
        how it ends.
    """
    for job in jobs:
        job.wait()
    # Ask each task how it ended. Its state from the scheduler can lag
    # behind a task that has just finished.
    failed = {
        str(job.job_id): error for job in jobs if (error := job.exception()) is not None
    }
    if failed:
        first_id, first_error = next(iter(failed.items()))
        raise RuntimeError(
            f"{len(failed)} of {len(jobs)} Slurm tasks did not complete. Task "
            f"{first_id}: {first_error}\nLogs are in {jobs[0].paths.folder}."
        )


__all__ = [
    "Batch",
    "resolve_compute_root",
    "run",
    "slurm_parameters",
    "split",
    "submit",
]
