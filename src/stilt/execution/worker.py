"""
Worker functions that run one simulation, one receptor, or many receptors.

Workers are handed receptors. :func:`run_receptor` runs every variant of
one receptor, and :func:`run_simulation` runs each one: HYSPLIT where the
particles are missing, then the footprint. Variants with the same transport
settings share one HYSPLIT run. :func:`run_receptors` runs a list of
receptors in this process or a process pool, and :func:`pull_receptors`
takes receptors from the PostgreSQL work queue.
"""

from __future__ import annotations

import logging
import multiprocessing
import signal
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from stilt.errors import ConfigValidationError, SimulationError
from stilt.simulation import Simulation

from .backends.protocol import sigterm_as_interrupt

if TYPE_CHECKING:
    from stilt.model import Model

logger = logging.getLogger(__name__)

Status = Literal["complete", "failed", "error", "interrupted"]

#: Statuses from worst to best, for summarizing a receptor's simulations.
_SEVERITY: tuple[Status, ...] = ("interrupted", "error", "failed", "complete")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """
    Outcome of one simulation run by a worker.

    Attributes
    ----------
    sim_id : str
        Simulation id.
    status : {"complete", "failed", "error", "interrupted"}
        ``failed`` is a HYSPLIT or STILT failure (a :class:`SimulationError`),
        ``error`` any other exception, and ``interrupted`` a stopped worker.
    error : str or None
        Error message, when the simulation did not complete.
    ran_hysplit : bool
        Whether HYSPLIT ran in this call.
    """

    sim_id: str
    status: Status
    error: str | None = None
    ran_hysplit: bool = False


@dataclass(frozen=True, slots=True)
class ReceptorResult:
    """
    Outcome of running every simulation of one receptor.

    Attributes
    ----------
    receptor_id : str
        Receptor id.
    status : {"complete", "failed", "error", "interrupted"}
        Worst status among the receptor's simulations.
    error : str or None
        Error message of the simulation with the worst status.
    simulations : tuple of SimulationResult
        Result of each simulation.
    """

    receptor_id: str
    status: Status
    error: str | None = None
    simulations: tuple[SimulationResult, ...] = field(default_factory=tuple)

    @classmethod
    def summarise(
        cls, receptor_id: str, results: list[SimulationResult]
    ) -> ReceptorResult:
        """Return the receptor result for a list of simulation results."""
        if not results:
            return cls(receptor_id, "complete")
        worst = min(results, key=lambda r: _SEVERITY.index(r.status))
        return cls(receptor_id, worst.status, worst.error, tuple(results))


def _append_error_log(sim: Simulation, *, phase: str, error: BaseException) -> None:
    """Append the error and its traceback to the simulation's log in the output directory."""
    log_path = sim.outputs.ensure_run().log_path(sim.receptor_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    trace = traceback.format_exc()
    lines = [
        "",
        "=== PYSTILT ERROR ===",
        f"Phase: {phase}",
        f"Type: {type(error).__name__}",
        f"Message: {error}",
    ]
    if trace and trace.strip() and trace.strip() != "NoneType: None":
        lines.extend(["", "Traceback:", trace.rstrip()])
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_simulation(
    sim: Simulation, *, skip_existing: bool = True, footprint_stale: bool = False
) -> SimulationResult:
    """
    Run one simulation and write its results to the output directory.

    HYSPLIT runs when the particles are missing, or always when
    ``skip_existing`` is false. When the variant has a grid, the footprint
    is computed if it is missing, and again whenever HYSPLIT ran or
    ``footprint_stale`` says the particles changed, so a footprint always
    matches its particles. An empty footprint is recorded with its reason
    and counts as complete. Errors are caught, written to the log, and
    returned in the result.

    Parameters
    ----------
    sim : Simulation
        Simulation to run.
    skip_existing : bool, default True
        Keep particles and footprints that already exist.
    footprint_stale : bool, default False
        Recompute the footprint even if it exists, because the particles it
        was made from were replaced in this call (by a variant that shares
        them).

    Returns
    -------
    SimulationResult
    """
    phase = "trajectory"
    ran_hysplit = False
    try:
        if not (skip_existing and sim.has_trajectory):
            sim.run_trajectories(write=True)
            ran_hysplit = True

        if sim.makes_footprint and (
            ran_hysplit or footprint_stale or not (skip_existing and sim.has_footprint)
        ):
            phase = "footprint"
            sim.generate_footprint(write=True)
        return SimulationResult(str(sim.id), "complete", ran_hysplit=ran_hysplit)
    except Exception as error:
        logger.exception("simulation %s failed during %s: %s", sim.id, phase, error)
        try:
            _append_error_log(sim, phase=phase, error=error)
        except Exception:
            logger.exception("simulation %s: could not write the failure log", sim.id)
        status = "failed" if isinstance(error, SimulationError) else "error"
        return SimulationResult(str(sim.id), status, error=str(error))


def run_receptor(
    model: Model, receptor_id: str, *, skip_existing: bool = True
) -> ReceptorResult:
    """
    Run every simulation of one receptor.

    Variants with the same transport settings share one HYSPLIT run: the
    first of them runs it, the others reuse the particles and make their own
    footprints. A footprint whose particles were replaced in this call is
    recomputed even with ``skip_existing``. A ``KeyboardInterrupt``, such as
    a preempted job, gives an ``interrupted`` result.

    Parameters
    ----------
    model : Model
        Model the receptor belongs to.
    receptor_id : str
        Receptor to run.
    skip_existing : bool, default True
        Keep particles and footprints that already exist.

    Returns
    -------
    ReceptorResult
    """
    sims = list(model.simulations.sel(receptor=receptor_id))
    results: list[SimulationResult] = []
    reran: set[str] = set()  # transport settings whose HYSPLIT ran in this call
    sim = None
    try:
        for sim in sims:
            key = sim.outputs.settings.hash
            result = run_simulation(
                sim,
                skip_existing=skip_existing or key in reran,
                footprint_stale=key in reran,
            )
            if result.ran_hysplit:
                reran.add(key)
            results.append(result)
    except KeyboardInterrupt:
        label = str(sim.id) if sim is not None else receptor_id
        results.append(SimulationResult(label, "interrupted", error="Worker preempted"))
    return ReceptorResult.summarise(receptor_id, results)


def _log_result(result: ReceptorResult, done: int, total: int) -> None:
    """Log one progress line for a finished receptor."""
    detail = f": {result.error}" if result.error else ""
    logger.info(
        "[%d/%d] %s %s%s", done, total, result.receptor_id, result.status, detail
    )


# -- process pool -------------------------------------------------------------

_POOL_MODEL: Model | None = None
_POOL_SKIP: bool = True


def _raise_interrupt(signum: int, frame: object) -> None:
    """Turn a signal into KeyboardInterrupt so cleanup runs."""
    raise KeyboardInterrupt


def _init_pool_worker(project: str, compute_root: str, skip_existing: bool) -> None:
    """Build the worker process's Model and make SIGTERM raise KeyboardInterrupt."""
    from stilt.model import Model

    global _POOL_MODEL, _POOL_SKIP
    signal.signal(signal.SIGTERM, _raise_interrupt)
    _POOL_MODEL = Model(project=project, compute_root=compute_root)
    _POOL_SKIP = skip_existing


def _pool_run(item: tuple[int, str]) -> tuple[int, ReceptorResult]:
    """Run one receptor in a pool worker, returning its index and result."""
    idx, receptor_id = item
    assert _POOL_MODEL is not None
    return idx, run_receptor(_POOL_MODEL, receptor_id, skip_existing=_POOL_SKIP)


def run_receptors(
    model: Model,
    receptor_ids: list[str],
    *,
    n_cores: int = 1,
    skip_existing: bool = True,
) -> list[ReceptorResult]:
    """
    Run a list of receptors, in this process or in a process pool.

    Pool workers load the model again from ``model.project.root``, so the
    config and receptors must already be saved in the project, as
    :meth:`Model.register` does. A SIGTERM, such as Slurm preemption or the
    end of the job's time limit, stops the batch with an ``interrupted``
    result.

    Parameters
    ----------
    model : Model
        Model the receptors belong to.
    receptor_ids : list of str
        Receptors to run.
    n_cores : int, default 1
        Number of worker processes. 1 runs in this process.
    skip_existing : bool, default True
        Keep trajectories and footprints that already exist.

    Returns
    -------
    list of ReceptorResult
        One result per receptor, in input order. After an interruption, only
        the receptors that finished.
    """
    if not receptor_ids:
        return []

    if n_cores <= 1:
        results: list[ReceptorResult] = []
        with sigterm_as_interrupt():
            for i, receptor_id in enumerate(receptor_ids, 1):
                result = run_receptor(model, receptor_id, skip_existing=skip_existing)
                results.append(result)
                _log_result(result, i, len(receptor_ids))
                if result.status == "interrupted":
                    break
        return results

    ordered: dict[int, ReceptorResult] = {}
    pool = multiprocessing.Pool(
        n_cores,
        initializer=_init_pool_worker,
        initargs=(model.project.root, str(model.compute_root), skip_existing),
    )
    with sigterm_as_interrupt():
        try:
            for idx, result in pool.imap_unordered(
                _pool_run, list(enumerate(receptor_ids))
            ):
                ordered[idx] = result
                _log_result(result, len(ordered), len(receptor_ids))
                if result.status == "interrupted":
                    pool.terminate()
                    break
            else:
                # Normal completion: let workers exit cleanly. terminate() would
                # SIGTERM idle workers, whose handler raises KeyboardInterrupt.
                pool.close()
        except KeyboardInterrupt:
            # Preempted or Ctrl-C: stop the workers and hand back what finished.
            pool.terminate()
        finally:
            pool.join()
    return [ordered[i] for i in sorted(ordered)]


# -- pull mode ----------------------------------------------------------------


def pull_receptors(
    model: Model,
    follow: bool = False,
    poll_interval: float = 10.0,
    *,
    skip_existing: bool = True,
) -> None:
    """
    Run receptors from the model's work queue until it is empty.

    Each receptor is claimed so that no other worker runs it at the same
    time, and its result is recorded in the queue.

    Parameters
    ----------
    model : Model
        Model with a work queue (set ``PYSTILT_DB_URL``).
    follow : bool, default False
        Keep waiting for new work when the queue is empty.
    poll_interval : float, default 10.0
        Seconds to wait after finding the queue empty. The wait doubles on
        each empty check, up to 60 s.
    skip_existing : bool, default True
        Keep trajectories and footprints that already exist.
    """
    queue = model.queue
    if queue is None:
        raise ConfigValidationError(
            "Pull-mode workers require a Postgres work queue. "
            "Configure it via PYSTILT_DB_URL."
        )
    idle_sleep = max(poll_interval, 0.1)
    max_idle_sleep = min(60.0, max(idle_sleep, poll_interval * 8))
    while True:
        with queue.claim_one() as claim:
            if claim is None:
                if not follow:
                    return
                time.sleep(idle_sleep)
                idle_sleep = min(idle_sleep * 2.0, max_idle_sleep)
                continue
            idle_sleep = max(poll_interval, 0.1)
            claim.record(
                run_receptor(model, claim.receptor_id, skip_existing=skip_existing)
            )


__all__ = [
    "ReceptorResult",
    "SimulationResult",
    "pull_receptors",
    "run_receptor",
    "run_receptors",
    "run_simulation",
]
