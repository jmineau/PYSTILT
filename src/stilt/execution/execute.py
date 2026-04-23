"""Single-simulation execution core.

:func:`execute_task` — run one simulation task (trajectory or footprint path).
:func:`execute_batch` — run a list of tasks sequentially or in a process pool.
"""

from __future__ import annotations

import datetime as dt
import logging
import multiprocessing
import signal

from stilt.simulation import Simulation

from .phases import (
    _append_simulation_error_log,
    _failure_status,
    _run_footprint_phase,
    _run_trajectory_phase,
)
from .tasks import SimulationResult, SimulationTask

logger = logging.getLogger(__name__)


def _build_simulation(task: SimulationTask) -> Simulation:
    """Return the compute-local simulation handle for one task."""
    return Simulation(
        directory=task.compute_root / task.sim_id,
        meteorology=task.meteorology,
        receptor=task.receptor,
        params=task.params,
        store=task.storage.store,
    )


def execute_task(task: SimulationTask) -> SimulationResult:
    """Run one STILT simulation task and any requested footprint outputs.

    When ``task.foot_configs`` is ``None``, only trajectories are run.  When
    configs are provided, trajectories are auto-run as needed and all
    footprint products are computed in a single pass so the trajectory file
    is loaded exactly once. Durable outputs are published through
    ``task.storage``. Durable index writes happen separately through an
    explicit index backend.

    Parameters
    ----------
    task : SimulationTask
        Serialisable bundle containing project dir, sim ID, meteorology,
        receptor, STILT params, optional footprint configs, and durable storage.

    Returns
    -------
    SimulationResult
        Typed per-simulation execution result.
    """
    started_at = dt.datetime.now(dt.timezone.utc)

    sim = _build_simulation(task)

    result = (
        _run_footprint_phase(sim, task, started_at=started_at)
        if task.foot_configs
        else _run_trajectory_phase(sim, task, started_at=started_at)
    )
    return result


def _sigterm_handler(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def _init_pool_worker() -> None:
    """Install SIGTERM → KeyboardInterrupt in each pool worker process."""
    signal.signal(signal.SIGTERM, _sigterm_handler)


def _unexpected_task_failure_result(
    task: SimulationTask,
    *,
    error: Exception,
    started_at: dt.datetime,
) -> SimulationResult:
    """Convert one uncaught task exception into a durable worker result."""
    logger.exception(
        "simulation %s failed before worker result normalization: %s",
        task.sim_id,
        error,
    )

    sim: Simulation | None = None
    log_path = None
    traj_present = False
    traj_path = None
    error_traj_path = None

    try:
        sim = _build_simulation(task)
    except Exception:
        logger.exception(
            "simulation %s failed while preparing failure artifacts",
            task.sim_id,
        )
    else:
        _append_simulation_error_log(sim, phase="worker", error=error)
        try:
            task.storage.publish_simulation(sim)
        except Exception:
            logger.exception(
                "simulation %s failed while publishing failure artifacts",
                task.sim_id,
            )
        if sim.log_path.exists():
            log_path = sim.log_path
        traj_present = sim.resolve_output(sim.trajectories_path) is not None
        traj_path = sim.trajectories_path if traj_present else None
        if sim.error_trajectories_path.exists():
            error_traj_path = sim.error_trajectories_path

    finished_at = dt.datetime.now(dt.timezone.utc)
    return SimulationResult(
        sim_id=task.sim_id,
        status=_failure_status(error),
        traj_present=traj_present,
        traj_path=traj_path,
        error_traj_path=error_traj_path,
        log_path=log_path,
        wrote_traj=False,
        error=str(error),
        started_at=started_at,
        finished_at=finished_at,
    )


def _execute_task_result(task: SimulationTask) -> SimulationResult:
    """Run one task and normalize worker interrupts and uncaught exceptions."""
    started_at = dt.datetime.now(dt.timezone.utc)
    try:
        return execute_task(task)
    except KeyboardInterrupt:
        finished_at = dt.datetime.now(dt.timezone.utc)
        return SimulationResult(
            sim_id=task.sim_id,
            status="interrupted",
            wrote_traj=False,
            error="Worker preempted by SIGTERM",
            started_at=started_at,
            finished_at=finished_at,
        )
    except Exception as error:
        return _unexpected_task_failure_result(
            task,
            error=error,
            started_at=started_at,
        )


def _execute_task_guarded(task: SimulationTask) -> SimulationResult:
    """Wrap execute_task to catch SIGTERM and uncaught task exceptions."""
    return _execute_task_result(task)


def execute_batch(
    batch: list[SimulationTask], n_cores: int = 1
) -> list[SimulationResult]:
    """Run a list of simulation tasks sequentially or in a local process pool.

    When Slurm preempts or wall-times the task it sends SIGTERM. The handler
    converts SIGTERM to :exc:`KeyboardInterrupt` so the worker can return an
    interrupted result.

    Parameters
    ----------
    batch : list[SimulationTask]
        Simulations assigned to this worker.
    n_cores : int, optional
        Number of local CPU cores to use. Defaults to 1.

    Returns
    -------
    list[SimulationResult]
        Per-simulation results in batch order.
    """
    if n_cores <= 1:
        signal.signal(signal.SIGTERM, _sigterm_handler)
        results = []
        for task in batch:
            result = _execute_task_result(task)
            results.append(result)
            if result.status == "interrupted":
                break
        return results
    with multiprocessing.Pool(n_cores, initializer=_init_pool_worker) as pool:
        return list(pool.map(_execute_task_guarded, batch))
