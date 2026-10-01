"""Running a project: finding the receptors with missing results and handing them to workers."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import submitit

from stilt.config import ExecutionConfig
from stilt.project import project_slug

if TYPE_CHECKING:
    from stilt.project import Project

    from .worker import ReceptorResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The unit of work
# ---------------------------------------------------------------------------


class Batch(submitit.helpers.Checkpointable):
    """
    A batch of receptors of one project, run by one worker.

    Calling it opens the project and runs the receptors,
    ``cpus`` at a time. On Slurm it is one array task. When the task is
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
    compute_root : str, optional
        Scratch directory under which HYSPLIT runs.
    cpus : int, default 1
        Number of receptors to run at once.
    skip_existing : bool, default True
        Keep particles and footprints that already exist.
    """

    def __init__(
        self,
        project: str,
        receptor_ids: list[str],
        *,
        compute_root: str | None = None,
        cpus: int = 1,
        skip_existing: bool = True,
    ) -> None:
        self.project = project
        self.receptor_ids = list(receptor_ids)
        self.compute_root = compute_root
        self.cpus = cpus
        self.skip_existing = skip_existing

    def __call__(self) -> list[ReceptorResult]:
        """Run the batch and return one result per receptor."""
        from stilt.project import Project

        from .worker import run_receptors

        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        logging.getLogger("stilt.execution.worker").setLevel(logging.INFO)
        return run_receptors(
            Project(self.project),
            self.receptor_ids,
            compute_root=self.compute_root,
            n_cores=self.cpus,
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
        raw = os.path.expandvars(os.path.expanduser(str(compute_root)))
        return Path(raw).resolve()
    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return (Path(tmp_root) / "pystilt" / project.name).resolve()


def _pending(project: Project, skip_existing: bool) -> list[str]:
    """Return the ids of the receptors to run, each once, in project order."""
    sims = project.simulations
    if skip_existing:
        sims = sims.incomplete()
    return list(dict.fromkeys(sims["receptor"]))


def run(
    project: Project,
    *,
    execution: ExecutionConfig | None = None,
    skip_existing: bool = True,
    compute_root: str | Path | None = None,
) -> list[ReceptorResult]:
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
    list of ReceptorResult
        One per receptor that ran.

    Raises
    ------
    RuntimeError
        If a Slurm task failed, was cancelled, or ran out of requeues.
    """
    execution = execution if execution is not None else project.config.execution
    if execution.backend == "slurm":
        jobs = submit(
            project,
            execution=execution,
            skip_existing=skip_existing,
            compute_root=compute_root,
        )
        return _wait(jobs)
    pending = _pending(project, skip_existing)
    if not pending:
        logger.info("run: every simulation is complete; nothing to do")
        return []
    logger.info("run(%s): %d receptors", ", ".join(project.variants), len(pending))
    # In this process, so Ctrl-C and SIGTERM stop the workers cleanly and
    # progress prints as it happens.
    from .worker import run_receptors

    return run_receptors(
        project,
        pending,
        compute_root=compute_root,
        n_cores=execution.cpus,
        skip_existing=skip_existing,
    )


def submit(
    project: Project,
    *,
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
    project, execution, skip_existing, compute_root
        As for :func:`run`.

    Returns
    -------
    list of submitit.Job
        One per array task, empty when nothing needs to run.

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
    pending = _pending(project, skip_existing)
    if not pending:
        logger.info("submit: every simulation is complete; nothing to do")
        return []
    # One folder per submission, so a later array never overwrites these.
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    executor = submitit.AutoExecutor(
        folder=project.directory / "slurm" / stamp, cluster="slurm"
    )
    executor.update_parameters(
        **slurm_parameters(
            execution, job_name=f"pystilt-{project_slug(project.directory)}"
        )
    )
    # A compute root that was not asked for is left to each compute node,
    # whose TMPDIR is its own. One that was is made absolute here, since the
    # task may start in another directory.
    scratch = (
        None
        if compute_root is None
        else str(resolve_compute_root(project, compute_root))
    )
    batches = [
        Batch(
            str(project.directory),
            ids,
            compute_root=scratch,
            cpus=execution.cpus,
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


def _wait(jobs: list[submitit.Job[Any]]) -> list[ReceptorResult]:
    """
    Wait until every task has left the queue and return their receptor results.

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
    return [result for job in jobs for result in job.result()]


__all__ = [
    "Batch",
    "resolve_compute_root",
    "run",
    "slurm_parameters",
    "split",
    "submit",
]
