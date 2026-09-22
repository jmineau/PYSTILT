"""
Worker-side execution: run one simulation, or many, for a model.

:func:`run_simulation` runs one :class:`~stilt.simulation.Simulation` end to
end (trajectory, then every requested footprint) and publishes its outputs.
:func:`run_simulations` runs a list of ids for a model, inline or in one
process pool. :func:`pull_simulations` drains a Postgres work queue.
"""

from __future__ import annotations

import logging
import multiprocessing
import signal
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from stilt.config import FootprintConfig
from stilt.errors import ConfigValidationError, SimulationError
from stilt.simulation import Simulation

from .backends.protocol import sigterm_as_interrupt

if TYPE_CHECKING:
    from stilt.model import Model

logger = logging.getLogger(__name__)

Status = Literal["complete", "complete-empty", "failed", "error", "interrupted"]


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


def _footprint_targets(
    footprints: Mapping[str, FootprintConfig],
) -> list[tuple[str, str, FootprintConfig, bool]]:
    """Return ``(base_name, stored_name, config, is_error)`` in execution order."""
    targets = []
    for name, config in footprints.items():
        targets.append((name, name, config, False))
        if config.error:
            targets.append((name, f"{name}_error", config, True))
    return targets


def run_simulation(
    sim: Simulation,
    footprints: Mapping[str, FootprintConfig] | None = None,
    *,
    skip_existing: bool = True,
) -> SimulationResult:
    """
    Run one simulation and publish its outputs.

    With no footprints, only the trajectory is produced. With footprints, the
    trajectory is run as needed and every footprint is computed in one pass
    (the trajectory is loaded once). An empty footprint writes a marker so the
    outcome is durable and skip-existing treats it as done.

    Parameters
    ----------
    sim
        The simulation handle to run.
    footprints
        Named footprint configs to produce.
    skip_existing
        Skip footprints that already exist (netCDF or empty marker).
    """
    phase = "trajectory"
    try:
        if not footprints:
            sim.run_trajectories(write=True)
            sim.publish()
            return SimulationResult(str(sim.id), "complete")

        statuses: dict[str, str] = {}
        for base_name, stored_name, config, is_error in _footprint_targets(footprints):
            phase = f"footprint:{stored_name}"
            if skip_existing and sim.has_footprint(stored_name):
                statuses[stored_name] = (
                    "complete"
                    if sim.resolve(sim.footprint_path(stored_name)) is not None
                    else "complete-empty"
                )
                continue
            foot = sim.generate_footprint(base_name, config, write=True, error=is_error)
            if foot is None or foot.is_empty:
                sim.write_empty_footprint_marker(stored_name)
                statuses[stored_name] = "complete-empty"
            else:
                sim.clear_empty_footprint_marker(stored_name)
                statuses[stored_name] = "complete"
        sim.publish()
        status: Status = (
            "complete" if "complete" in statuses.values() else "complete-empty"
        )
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


def _run_one(model: Model, sim_id: str, skip_existing: bool) -> SimulationResult:
    """Run one id for *model*, normalising preemption into a result."""
    try:
        return run_simulation(
            model.simulation(sim_id),
            model.config.footprints,
            skip_existing=skip_existing,
        )
    except KeyboardInterrupt:
        return SimulationResult(sim_id, "interrupted", error="Worker preempted")


# -- process pool -------------------------------------------------------------

_POOL_MODEL: Model | None = None
_POOL_SKIP: bool = True


def _raise_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def _init_pool_worker(project: str, compute_root: str, skip_existing: bool) -> None:
    """Build one Model per worker process and translate SIGTERM to interrupt."""
    from stilt.model import Model

    global _POOL_MODEL, _POOL_SKIP
    signal.signal(signal.SIGTERM, _raise_interrupt)
    _POOL_MODEL = Model(project=project, compute_root=compute_root)
    _POOL_SKIP = skip_existing


def _pool_run(item: tuple[int, str]) -> tuple[int, SimulationResult]:
    idx, sim_id = item
    assert _POOL_MODEL is not None
    return idx, _run_one(_POOL_MODEL, sim_id, _POOL_SKIP)


def run_simulations(
    model: Model,
    sim_ids: list[str],
    *,
    n_cores: int = 1,
    skip_existing: bool | None = None,
) -> list[SimulationResult]:
    """
    Run a list of simulation ids for *model*, inline or in a process pool.

    Pool workers rebuild the model from ``model.project.root``, so the
    project's inputs must already be persisted (``Model.register()`` does
    this; ``Model.run()`` calls it). A SIGTERM (Slurm preemption or
    wall-time) is turned into an ``interrupted`` result and stops the batch.

    Parameters
    ----------
    model
        The model the simulations belong to.
    sim_ids
        Simulation ids to run.
    n_cores
        Worker processes. ``1`` runs inline in this process.
    skip_existing
        Skip outputs that already exist. Defaults to ``config.skip_existing``.

    Returns
    -------
    list[SimulationResult]
        One result per id, in input order (truncated after an interruption).
    """
    skip = model.config.skip_existing if skip_existing is None else skip_existing
    if not sim_ids:
        return []

    if n_cores <= 1:
        results: list[SimulationResult] = []
        with sigterm_as_interrupt():
            for sim_id in sim_ids:
                result = _run_one(model, sim_id, skip)
                results.append(result)
                if result.status == "interrupted":
                    break
        return results

    ordered: dict[int, SimulationResult] = {}
    pool = multiprocessing.Pool(
        n_cores,
        initializer=_init_pool_worker,
        initargs=(model.project.root, str(model.compute_root), skip),
    )
    with sigterm_as_interrupt():
        try:
            for idx, result in pool.imap_unordered(_pool_run, list(enumerate(sim_ids))):
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


def pull_simulations(
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
            claim.record(_run_one(model, claim.sim_id, skip))


__all__ = [
    "SimulationResult",
    "pull_simulations",
    "run_simulation",
    "run_simulations",
]
