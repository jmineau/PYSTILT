"""Starting a model's work: saving its inputs and handing receptors to workers."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

import submitit
import yaml

from stilt.config import ExecutionConfig, ModelConfig, RuntimeSettings
from stilt.errors import ConfigValidationError
from stilt.project import project_slug

if TYPE_CHECKING:
    from stilt.model import Model
    from stilt.project import Project
    from stilt.receptors import Receptor

    from .worker import ReceptorResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------


class JobHandle(Protocol):
    """What :func:`run` returns: a handle to the work it started."""

    @property
    def job_id(self) -> str:
        """Identifier of the work, such as a Slurm job id."""
        ...

    @property
    def detached(self) -> bool:
        """Whether the work goes on after this process exits."""
        ...

    def wait(self) -> None:
        """Block until the work has finished."""
        ...


class LocalHandle:
    """Handle to a local run, which has already finished when it is returned."""

    @property
    def job_id(self) -> str:
        """Always ``"local"``, since local runs have no scheduler id."""
        return "local"

    @property
    def detached(self) -> bool:
        """Always False, since local workers stop when this process exits."""
        return False

    def wait(self) -> None:
        """Return at once: the run finished inside :func:`run`."""


class SlurmHandle:
    """
    Handle to a Slurm job array.

    Attributes
    ----------
    jobs : list of submitit.Job
        One job per array task, for anything this handle does not cover
        (``job.stdout()``, ``job.result()``, ``job.cancel()``).
    folder : Path
        Folder holding the tasks' logs and submitit's files.
    """

    def __init__(self, jobs: list[submitit.Job[Any]], folder: Path) -> None:
        self.jobs = jobs
        self.folder = folder

    @property
    def job_id(self) -> str:
        """Id of the job array."""
        return str(self.jobs[0].job_id).split("_")[0]

    @property
    def detached(self) -> bool:
        """Always True, since Slurm jobs run on after this process exits."""
        return True

    def wait(self) -> None:
        """
        Block until every task has left the queue.

        Raises
        ------
        RuntimeError
            If any task failed, was cancelled, or timed out. A task that was
            preempted or ran out of time is requeued and counts only by how
            it ends.
        """
        for job in self.jobs:
            job.wait()
        # Ask each task how it ended. Its state from the scheduler can lag
        # behind a task that has just finished.
        failed = {
            str(job.job_id): error
            for job in self.jobs
            if (error := job.exception()) is not None
        }
        if failed:
            first_id, first_error = next(iter(failed.items()))
            raise RuntimeError(
                f"Slurm job {self.job_id}: {len(failed)} of {len(self.jobs)} "
                f"tasks did not complete. Task {first_id}: {first_error}\n"
                f"Logs are in {self.folder}."
            )


# ---------------------------------------------------------------------------
# The unit of work
# ---------------------------------------------------------------------------


class Batch(submitit.helpers.Checkpointable):
    """
    A batch of receptors of one project, run by one worker.

    Calling it rebuilds the model from the project and runs the receptors,
    ``cpus`` at a time. On Slurm it is one array task. When the task is
    preempted or runs out of time, submitit submits it again
    (:meth:`checkpoint`), and the second run skips the receptors the first
    one finished.

    Parameters
    ----------
    project : str
        Project directory. Its ``config.yaml`` and ``receptors.csv`` must
        already hold the settings and receptors (:func:`register`).
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
        from stilt.model import Model

        from .worker import run_receptors

        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        logging.getLogger("stilt.execution.worker").setLevel(logging.INFO)
        return run_receptors(
            Model(project=self.project),
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
# Running a model
# ---------------------------------------------------------------------------


def resolve_compute_root(
    project: Project, compute_root: str | Path | None = None
) -> Path:
    """
    Return the scratch directory under which HYSPLIT runs for *project*.

    That is *compute_root* when given, else ``PYSTILT_COMPUTE_ROOT``, else
    ``$TMPDIR/pystilt/<project name>``. The path is absolute and resolved, so
    a worker handed it gets the same directory.
    """
    if compute_root is not None:
        raw = os.path.expandvars(os.path.expanduser(str(compute_root)))
        return Path(raw).resolve()
    configured = RuntimeSettings().compute_root
    if configured is not None:
        return configured.expanduser().resolve()
    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return (Path(tmp_root) / "pystilt" / project.name).resolve()


def _save_config(model: Model) -> None:
    """
    Write the model's config to a project that has none.

    An existing ``config.yaml`` is never rewritten. When the model was given
    settings that differ from the file, one of the two is out of date and
    only the user knows which, so this raises. A file that cannot be read
    raises too, rather than being replaced.

    Raises
    ------
    ConfigValidationError
        If the project's ``config.yaml`` holds other settings than the model.
    """
    project = model.project
    if not project.has_config:
        project.save_config(model.config)
        return
    on_disk = project.load_config()
    if on_disk == model.config:
        return
    # Writing fills in what the settings imply (one variant per met), so a
    # config never equals its own round trip through config.yaml. Compare
    # the settings as they would be written.
    as_written = ModelConfig.model_validate(yaml.safe_load(model.config.to_yaml()))
    if as_written != on_disk:
        raise ConfigValidationError(
            f"{project.config_path} holds other settings than this model. Open "
            "the project with Model(project) to use the file, or edit the file."
        )


def register(model: Model, receptors: Iterable[Receptor] | None = None) -> list[str]:
    """
    Save a model's settings and receptors to its project.

    Workers rebuild the model from the project alone, so :func:`run` calls
    this first. ``config.yaml`` is written when the project has none or its
    settings differ from the model's, with only the settings that were set.
    Receptors not yet in ``receptors.csv`` are appended to it.

    Parameters
    ----------
    model : Model
        Model whose inputs to save.
    receptors : iterable of Receptor, optional
        Receptors to add to the project. Defaults to the model's own.

    Returns
    -------
    list of str
        Ids of the receptors registered, including any the project already
        had.
    """
    _save_config(model)
    batch = list(model.receptors) if receptors is None else list(receptors)
    if receptors is not None or not model.receptors.from_project:
        model.project.add_receptors(batch)
    return [r.id for r in batch]


def run(
    model: Model,
    *,
    execution: ExecutionConfig | None = None,
    skip_existing: bool = True,
    wait: bool = True,
    compute_root: str | Path | None = None,
) -> JobHandle:
    """
    Run every simulation of a model that has not finished.

    Saves the settings and receptors to the project (:func:`register`), then
    runs each receptor with missing results. A receptor's worker runs HYSPLIT
    once for each distinct set of transport settings whose particles are
    missing, then calculates the footprint of every variant that has a grid.

    Parameters
    ----------
    model : Model
        Model to run.
    execution : ExecutionConfig, optional
        Where to run and with what resources. Defaults to the ``execution``
        settings of the model's config (this machine, one process, unless
        configured).
    skip_existing : bool, default True
        Skip simulations whose outputs all exist. ``False`` runs every
        simulation again.
    wait : bool, default True
        Block until the work finishes. With ``False`` a Slurm run returns
        once it is submitted. A local run always finishes before this
        returns.
    compute_root : str or Path, optional
        Scratch directory under which HYSPLIT runs
        (:func:`resolve_compute_root`). On Slurm it is resolved on the
        compute node unless given here.

    Returns
    -------
    JobHandle
        Handle to the work.
    """
    execution = execution if execution is not None else model.config.execution

    receptor_ids = register(model)
    if not receptor_ids:
        logger.info("run: no receptors configured — nothing to do")
        return LocalHandle()
    pending = (
        model.simulations.incomplete().receptors if skip_existing else receptor_ids
    )
    if not pending:
        logger.info("run: all simulations already complete — nothing to do")
        return LocalHandle()

    logger.info(
        "run(%s): %d receptors on %s",
        ", ".join(model.variants),
        len(pending),
        execution.backend,
    )
    handle = _dispatch(
        model,
        pending,
        execution,
        compute_root=compute_root,
        skip_existing=skip_existing,
    )
    if wait:
        handle.wait()
    return handle


def _dispatch(
    model: Model,
    pending: list[str],
    execution: ExecutionConfig,
    *,
    compute_root: str | Path | None,
    skip_existing: bool,
) -> JobHandle:
    """Run *pending* here, or submit it to Slurm, and return a handle."""
    if execution.backend == "local":
        # In this process, so Ctrl-C and SIGTERM stop the workers cleanly and
        # progress prints as it happens.
        from .worker import run_receptors

        run_receptors(
            model,
            pending,
            compute_root=compute_root,
            n_cores=execution.cpus,
            skip_existing=skip_existing,
        )
        return LocalHandle()
    return _submit_slurm(
        model.project,
        pending,
        execution,
        # A compute root that was not asked for is left to each compute node,
        # whose TMPDIR is its own. One that was is made absolute here, since
        # the task may start in another directory.
        compute_root=None
        if compute_root is None
        else str(resolve_compute_root(model.project, compute_root)),
        skip_existing=skip_existing,
    )


def _submit_slurm(
    project: Project,
    pending: list[str],
    execution: ExecutionConfig,
    *,
    compute_root: str | None,
    skip_existing: bool,
) -> SlurmHandle:
    """Submit *pending* as one Slurm job array and return its handle."""
    import submitit

    # One folder per submission, so a later array never overwrites these.
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    folder = project.directory / "slurm" / stamp
    executor = submitit.AutoExecutor(folder=folder, cluster="slurm")
    executor.update_parameters(
        **slurm_parameters(execution, job_name=f"pystilt-{project_slug(project.root)}")
    )
    batches = [
        Batch(
            project.root,
            ids,
            compute_root=compute_root,
            cpus=execution.cpus,
            skip_existing=skip_existing,
        )
        for ids in split(pending, execution.n_workers)
    ]
    with executor.batch():
        jobs = [executor.submit(batch) for batch in batches]
    handle = SlurmHandle(jobs, folder)
    logger.info("Submitted job: %s (%d tasks)", handle.job_id, len(jobs))
    return handle


__all__ = [
    "Batch",
    "JobHandle",
    "LocalHandle",
    "SlurmHandle",
    "register",
    "resolve_compute_root",
    "run",
    "slurm_parameters",
    "split",
]
