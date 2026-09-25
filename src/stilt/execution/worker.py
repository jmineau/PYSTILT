"""
Worker-side execution: run one simulation, one receptor, or many receptors.

:func:`run_simulation` runs one :class:`~stilt.simulation.Simulation` end to
end (HYSPLIT where a trajectory is missing, then the footprint) and publishes
its outputs. The unit of work handed to workers is a **receptor**:
:func:`run_receptor` runs every variant of one receptor, transport variants
before the derived ones that rasterize their particles.
:func:`run_receptors` runs a list of receptor ids for a model, inline or in
one process pool. :func:`pull_receptors` drains a Postgres work queue.
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

Status = Literal["complete", "complete-empty", "failed", "error", "interrupted"]

#: Worst outcome first, for summarising a receptor's simulations.
_SEVERITY: tuple[Status, ...] = (
    "interrupted",
    "error",
    "failed",
    "complete-empty",
    "complete",
)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """
    Outcome of one worker-run simulation.

    ``failed`` is a STILT/HYSPLIT failure (a :class:`SimulationError`);
    ``error`` is any other exception; ``interrupted`` means the worker was
    preempted. Everything else about the run is readable from its outputs.
    """

    sim_id: str
    status: Status
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReceptorResult:
    """
    Outcome of running every simulation of one receptor.

    ``status`` is the worst of the simulations' statuses; ``simulations``
    holds each one.
    """

    receptor_id: str
    status: Status
    error: str | None = None
    simulations: tuple[SimulationResult, ...] = field(default_factory=tuple)

    @classmethod
    def summarise(
        cls, receptor_id: str, results: list[SimulationResult]
    ) -> ReceptorResult:
        """Fold simulation results into one receptor result."""
        if not results:
            return cls(receptor_id, "complete")
        worst = min(results, key=lambda r: _SEVERITY.index(r.status))
        return cls(receptor_id, worst.status, worst.error, tuple(results))


def _append_error_log(sim: Simulation, *, phase: str, error: BaseException) -> None:
    """Append a PYSTILT error section to the simulation log."""
    sim.log_path.parent.mkdir(parents=True, exist_ok=True)
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
    with sim.log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_simulation(sim: Simulation, *, skip_existing: bool = True) -> SimulationResult:
    """
    Run one simulation and publish its outputs.

    HYSPLIT runs when the trajectory is missing (or always, without
    ``skip_existing``); a derived simulation reads its parent's trajectory
    instead. The footprint is rasterized when one is configured. An empty
    footprint writes a marker so the outcome is durable and skip-existing
    treats it as done.

    Parameters
    ----------
    sim
        The simulation handle to run.
    skip_existing
        Skip outputs that already exist (trajectory parquet, footprint
        netCDF or empty marker).
    """
    phase = "trajectory"
    try:
        if sim.is_derived:
            if sim.trajectories is None:
                raise SimulationError(
                    f"{sim.id} derives from {sim.parent.id}, which has no trajectory"  # type: ignore[union-attr]
                )
        elif not (skip_existing and sim.has_trajectory):
            sim.run_trajectories(write=True)

        status: Status = "complete"
        if sim.footprint_config is not None:
            phase = "footprint"
            if skip_existing and sim.has_footprint:
                if sim.resolve(sim.footprint_path) is None:
                    status = "complete-empty"
            else:
                foot = sim.generate_footprint(write=True)
                if foot.is_empty:
                    sim.write_empty_footprint_marker()
                    status = "complete-empty"
                else:
                    sim.clear_empty_footprint_marker()
        sim.publish()
        return SimulationResult(str(sim.id), status)
    except Exception as error:
        logger.exception("simulation %s failed during %s: %s", sim.id, phase, error)
        try:
            _append_error_log(sim, phase=phase, error=error)
            sim.publish()
        except Exception:
            logger.exception("simulation %s: could not publish failure log", sim.id)
        status = "failed" if isinstance(error, SimulationError) else "error"
        return SimulationResult(str(sim.id), status, error=str(error))


def run_receptor(
    model: Model, receptor_id: str, *, skip_existing: bool = True
) -> ReceptorResult:
    """
    Run every simulation of one receptor, transport variants first.

    Derived variants come last so the trajectory they rasterize exists. A
    preemption (``KeyboardInterrupt``) is normalised into an ``interrupted``
    result.
    """
    sims = list(model.simulations.sel(receptor=receptor_id))
    ordered = [s for s in sims if not s.is_derived] + [s for s in sims if s.is_derived]
    results: list[SimulationResult] = []
    try:
        for sim in ordered:
            results.append(run_simulation(sim, skip_existing=skip_existing))
    except KeyboardInterrupt:
        results.append(
            SimulationResult(str(sim.id), "interrupted", error="Worker preempted")
        )
    return ReceptorResult.summarise(receptor_id, results)


# -- process pool -------------------------------------------------------------

_POOL_MODEL: Model | None = None
_POOL_SKIP: bool = True


def _raise_interrupt(signum: int, frame: object) -> None:
    """Turn a signal into KeyboardInterrupt so cleanup runs."""
    raise KeyboardInterrupt


def _init_pool_worker(project: str, compute_root: str, skip_existing: bool) -> None:
    """Build one Model per worker process and translate SIGTERM to interrupt."""
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
    skip_existing: bool | None = None,
) -> list[ReceptorResult]:
    """
    Run a list of receptor ids for *model*, inline or in a process pool.

    Pool workers rebuild the model from ``model.project.root``, so the
    project's inputs must already be persisted (``Model.register()`` does
    this; ``Model.run()`` calls it). A SIGTERM (Slurm preemption or
    wall-time) is turned into an ``interrupted`` result and stops the batch.

    Parameters
    ----------
    model
        The model the receptors belong to.
    receptor_ids
        Receptor ids to run.
    n_cores
        Worker processes. ``1`` runs inline in this process.
    skip_existing
        Skip outputs that already exist. Defaults to ``config.skip_existing``.

    Returns
    -------
    list[ReceptorResult]
        One result per id, in input order (truncated after an interruption).
    """
    skip = model.config.skip_existing if skip_existing is None else skip_existing
    if not receptor_ids:
        return []

    if n_cores <= 1:
        results: list[ReceptorResult] = []
        with sigterm_as_interrupt():
            for receptor_id in receptor_ids:
                result = run_receptor(model, receptor_id, skip_existing=skip)
                results.append(result)
                if result.status == "interrupted":
                    break
        return results

    ordered: dict[int, ReceptorResult] = {}
    pool = multiprocessing.Pool(
        n_cores,
        initializer=_init_pool_worker,
        initargs=(model.project.root, str(model.compute_root), skip),
    )
    with sigterm_as_interrupt():
        try:
            for idx, result in pool.imap_unordered(
                _pool_run, list(enumerate(receptor_ids))
            ):
                ordered[idx] = result
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
    skip_existing: bool | None = None,
) -> None:
    """
    Drain the model's Postgres work queue through atomic claims.

    Parameters
    ----------
    model
        A model with a configured queue (``PYSTILT_DB_URL``).
    follow
        Keep polling when the queue is empty (long-lived worker).
    poll_interval
        Base sleep between empty polls; backs off up to 60 s.
    skip_existing
        Skip outputs that already exist. Defaults to ``config.skip_existing``.
    """
    queue = model.queue
    if queue is None:
        raise ConfigValidationError(
            "Pull-mode workers require a Postgres work queue. "
            "Configure it via PYSTILT_DB_URL."
        )
    skip = model.config.skip_existing if skip_existing is None else skip_existing

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
            claim.record(run_receptor(model, claim.receptor_id, skip_existing=skip))


__all__ = [
    "ReceptorResult",
    "SimulationResult",
    "pull_receptors",
    "run_receptor",
    "run_receptors",
    "run_simulation",
]
