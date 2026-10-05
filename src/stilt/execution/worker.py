"""
Worker functions that run one receptor, or many receptors.

Workers are handed receptors. :func:`run_receptor` runs every variant of
one receptor. Variants with the same transport settings share one set of
particles, so it groups them: HYSPLIT runs once per group where the
particles are missing (:func:`run_particles`), then each variant's
footprint is made from the particles in memory (:func:`make_footprint`).
A failure is recorded with the simulation and the worker goes on.
:func:`run_receptors` runs a list of receptors in this process or a
process pool. A :class:`~stilt.Simulation` itself runs nothing; these
functions write through its output directory.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import multiprocessing
import shutil
import signal
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import xarray as xr

from stilt.exceptions import EmptyFootprint, SimulationError
from stilt.footprint import calculate
from stilt.meteorology import Met
from stilt.output import Footprints, Particles
from stilt.simulation import Simulation
from stilt.transport import get_model

if TYPE_CHECKING:
    from stilt.project import Project

logger = logging.getLogger(__name__)


def _raise_interrupt(signum: int, frame: object) -> None:
    """Turn a signal into KeyboardInterrupt so cleanup runs."""
    raise KeyboardInterrupt


@contextlib.contextmanager
def _sigterm_as_interrupt():
    """
    Make SIGTERM raise ``KeyboardInterrupt`` inside the ``with`` block.

    Slurm sends SIGTERM on preemption or when a job reaches its time limit.
    Python's default action ends the process without running ``finally``
    blocks, so pool workers would be left running.

    It does nothing when SIGTERM is already handled, as it is inside a task
    submitit started: there the scheduler's signals belong to submitit, which
    requeues a preempted task. Signal handlers can only be set from the main
    thread, so in any other thread this does nothing either.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous = signal.getsignal(signal.SIGTERM)
    if previous is not signal.SIG_DFL:
        yield
        return

    signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


Status = Literal["complete", "failed", "error", "interrupted"]
#: The step of a simulation that failed: its particles or its footprint.
Step = Literal["particles", "footprint"]


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
        A failure is also recorded with the simulation
        (:attr:`stilt.Simulation.failure`).
    error : str or None
        Error message, when the simulation did not complete.
    """

    sim_id: str
    status: Status
    error: str | None = None


# -- failure records ---------------------------------------------------------


def _failed(sim: Simulation, step: Step, error: Exception) -> SimulationResult:
    """
    Record why *sim* failed at *step* and return its result.

    The record goes in the logs of the folder whose result failed
    (:meth:`stilt.output.Particles.record_failure`): a particles failure
    covers every variant on those particles, a footprint failure is the
    variant's own. ``reason`` is the error's short cause, or its class
    for an error without one. An expected failure (a
    :class:`~stilt.exceptions.SimulationError`) logs one line; any other
    error logs, and records, its traceback.
    """
    expected = isinstance(error, SimulationError)
    reason = getattr(error, "reason", None) or type(error).__name__
    if expected:
        logger.warning(
            "simulation %s failed during %s (%s): %s", sim.id, step, reason, error
        )
    else:
        logger.exception("simulation %s failed during %s: %s", sim.id, step, error)
    record: dict[str, Any] = {
        "step": step,
        "reason": str(reason),
        "message": str(error),
        "time": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    }
    if not expected:
        record["traceback"] = traceback.format_exc()
    try:
        _folder(sim, step).record_failure(sim.receptor.id, record)
    except Exception:
        logger.exception("simulation %s: could not record the failure", sim.id)
    status: Status = "failed" if expected else "error"
    return SimulationResult(str(sim.id), status, error=str(error))


def _succeeded(sim: Simulation, step: Step) -> SimulationResult:
    """Remove *sim*'s failure record for *step*, whose result is now written, and return its result."""
    _folder(sim, step).clear_failure(sim.receptor.id)
    return SimulationResult(str(sim.id), "complete")


def _folder(sim: Simulation, step: Step) -> Particles | Footprints:
    """Return the output folder of *sim*'s result for *step*."""
    if step == "particles":
        return sim.output.particles(sim.variant)
    return sim.output.footprints(sim.variant)


def run_particles(
    sim: Simulation,
    *,
    met: Met,
    workdir: Path,
    keep_scratch: bool = False,
    timeout: int | None = None,
) -> pd.DataFrame:
    """
    Run the transport model for a simulation and write its particles to the output directory.

    The model the settings name (HYSPLIT) runs in *workdir*, on scratch.
    The log is copied into the output directory whether the run succeeds or
    fails. The working directory is then removed, unless the run failed or
    *keep_scratch* is set, in which case it is copied under the output
    directory's ``scratch/`` first.

    Parameters
    ----------
    sim : Simulation
        What to run.
    met : Met
        Meteorology for the run.
    workdir : Path
        Scratch directory to run in. Created here.
    keep_scratch : bool, default False
        Keep the working directory of a successful run too.
    timeout : int, optional
        Time limit for the transport model run, in seconds.

    Returns
    -------
    pandas.DataFrame
        The particles, also written to the output directory.

    Raises
    ------
    SimulationError
        As the transport model raises it (with a ``reason``), or with
        ``reason`` ``NO_PARTICLE_DATA`` when the model wrote no particles.
        :class:`~stilt.exceptions.MeteorologyError` when the met files are
        missing.
    """
    params = sim.variant.transport
    model = get_model(sim.variant.model.name)
    folder = sim.output.particles(sim.variant)
    rid = sim.receptor.id
    # The model runs in an empty directory. A job stopped partway can leave
    # this simulation's directory behind; it is PYSTILT's own, so clear it.
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    scratch_log = workdir / "stilt.log"
    succeeded = False
    try:
        result = model.run(sim.receptor, params, met, workdir, timeout=timeout)
        if result.particles.empty:
            raise SimulationError(
                "The transport model wrote no particles.", reason="NO_PARTICLE_DATA"
            )
        folder.write(sim.receptor, result.particles, result.met_files)
        succeeded = True
        return result.particles
    finally:
        if scratch_log.exists():
            folder.write_log(rid, scratch_log.read_text())
        _finish_scratch(
            workdir, folder.scratch_path(rid), keep=keep_scratch or not succeeded
        )


def _finish_scratch(workdir: Path, kept: Path, *, keep: bool) -> None:
    """
    Copy *workdir* to *kept* when *keep*, then remove it.

    An empty directory is not kept: a run that failed before writing
    anything, such as on missing meteorology, has nothing to look at.
    """
    if not workdir.exists():
        return
    if keep and any(workdir.iterdir()):
        shutil.rmtree(kept, ignore_errors=True)
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(workdir, kept, symlinks=True)
    shutil.rmtree(workdir, ignore_errors=True)


def make_footprint(
    sim: Simulation,
    particles: pd.DataFrame,
    *,
    directory: Path | None = None,
) -> xr.DataArray | None:
    """
    Calculate a simulation's footprint from *particles* and write it to its folder.

    The settings are the variant's own. A relative file name in a
    transform's settings starts from *directory*, the project's. When no particle reaches the grid,
    an empty footprint is recorded with the reason and ``None`` is returned.
    :meth:`stilt.Simulation.generate_footprint` makes footprints with other
    settings, without writing them.

    Raises
    ------
    ValueError
        If the variant has no grid.
    """
    feet = sim.output.footprints(sim.variant)
    try:
        foot = calculate(
            particles,
            sim.receptor,
            feet.config,
            name=sim.variant.name,
            directory=directory,
            geometry_hash=sim.variant.geometry_hash,
        )
    except EmptyFootprint as error:
        feet.write_empty(sim.receptor, error.reason, name=sim.variant.name)
        return None
    feet.write(foot)
    return foot


def run_receptor(
    project: Project,
    receptor_id: str,
    *,
    compute_root: Path,
    skip_existing: bool = True,
) -> list[SimulationResult]:
    """
    Run every simulation of one receptor, and return their results.

    Variants with the same transport settings share one set of particles,
    so they run as a group: HYSPLIT runs once for the group when the
    particles are missing (or always, without ``skip_existing``), and each
    variant's footprint is made from those particles in memory. A footprint
    is made again whenever its particles were, so it always matches them.
    A failure is recorded with the simulation (:attr:`stilt.Simulation.failure`)
    and does not stop the others; a failed HYSPLIT run fails every variant
    of its group. A ``KeyboardInterrupt``, such as a preempted job, gives an
    ``interrupted`` result.

    Parameters
    ----------
    project : Project
        Project the receptor belongs to.
    receptor_id : str
        Receptor to run.
    compute_root : Path
        Scratch directory under which HYSPLIT runs, as
        :func:`~stilt.execution.resolve_compute_root` returns it.
    skip_existing : bool, default True
        Keep particles and footprints that already exist.

    Returns
    -------
    list of SimulationResult
        One per variant, in config order. After an interruption, the
        finished ones and then one ``interrupted``.
    """
    sims = [project.simulation(receptor_id, variant) for variant in project.variants]
    groups: dict[str, list[Simulation]] = {}
    for sim in sims:
        groups.setdefault(sim.variant.particles_hash, []).append(sim)
    results: dict[str, SimulationResult] = {}
    try:
        for group in groups.values():
            results.update(
                _run_group(
                    project,
                    group,
                    compute_root=compute_root,
                    skip_existing=skip_existing,
                )
            )
    except KeyboardInterrupt:
        stopped = next((s for s in sims if str(s.id) not in results), None)
        label = str(stopped.id) if stopped is not None else receptor_id
        done = [results[str(s.id)] for s in sims if str(s.id) in results]
        return [*done, SimulationResult(label, "interrupted", error="Worker preempted")]
    return [results[str(s.id)] for s in sims]


def _run_group(
    project: Project,
    sims: list[Simulation],
    *,
    compute_root: Path,
    skip_existing: bool,
) -> dict[str, SimulationResult]:
    """Run the simulations of one receptor that share particles: HYSPLIT at most once, then each footprint."""
    first = sims[0]
    particles: pd.DataFrame | None = None
    rerun = not (skip_existing and first.has_particles)
    if rerun:
        try:
            particles = run_particles(
                first,
                met=project.mets[first.variant.met],
                workdir=compute_root / first.id,
                keep_scratch=project.config.execution.keep_scratch,
                timeout=project.config.execution.timeout,
            )
        except Exception as error:
            failed = _failed(first, "particles", error)
            return {
                str(sim.id): SimulationResult(str(sim.id), failed.status, failed.error)
                for sim in sims
            }
        _succeeded(first, "particles")

    results: dict[str, SimulationResult] = {}
    for sim in sims:
        if not sim.makes_footprint or (
            not rerun and skip_existing and sim.has_footprint
        ):
            results[str(sim.id)] = SimulationResult(str(sim.id), "complete")
            continue
        try:
            if particles is None:
                particles = first.particles  # read once for the group
            make_footprint(sim, particles, directory=project.directory)
        except Exception as error:
            results[str(sim.id)] = _failed(sim, "footprint", error)
            continue
        results[str(sim.id)] = _succeeded(sim, "footprint")
    return results


def _log_result(
    receptor_id: str, results: list[SimulationResult], done: int, total: int
) -> None:
    """Log one progress line for a finished receptor: complete, or its first problem."""
    problem = next((r for r in results if r.status != "complete"), None)
    if problem is None:
        logger.info("[%d/%d] %s complete", done, total, receptor_id)
    else:
        logger.info(
            "[%d/%d] %s %s: %s", done, total, receptor_id, problem.status, problem.error
        )


def _interrupted(results: list[SimulationResult]) -> bool:
    """Return whether a receptor's run was stopped."""
    return any(r.status == "interrupted" for r in results)


# -- process pool -------------------------------------------------------------

_POOL_PROJECT: Project | None = None
_POOL_COMPUTE_ROOT: Path | None = None
_POOL_SKIP: bool = True


def _init_pool_worker(project: str, compute_root: str, skip_existing: bool) -> None:
    """Open the worker process's Project and make SIGTERM raise KeyboardInterrupt."""
    from stilt.project import Project

    global _POOL_PROJECT, _POOL_COMPUTE_ROOT, _POOL_SKIP
    signal.signal(signal.SIGTERM, _raise_interrupt)
    _POOL_PROJECT = Project(project)
    _POOL_COMPUTE_ROOT = Path(compute_root)
    _POOL_SKIP = skip_existing


def _pool_run(item: tuple[int, str]) -> tuple[int, list[SimulationResult]]:
    """Run one receptor in a pool worker, returning its index and result."""
    idx, receptor_id = item
    assert _POOL_PROJECT is not None and _POOL_COMPUTE_ROOT is not None
    return idx, run_receptor(
        _POOL_PROJECT,
        receptor_id,
        compute_root=_POOL_COMPUTE_ROOT,
        skip_existing=_POOL_SKIP,
    )


def run_receptors(
    project: Project,
    receptor_ids: list[str],
    *,
    compute_root: Path,
    n_cores: int = 1,
    skip_existing: bool = True,
) -> list[SimulationResult]:
    """
    Run a list of receptors, in this process or in a process pool.

    Pool workers open the project again from its directory. A SIGTERM,
    such as Slurm preemption or the end of the job's time limit, stops the batch with an ``interrupted``
    result.

    Parameters
    ----------
    project : Project
        Project the receptors belong to.
    receptor_ids : list of str
        Receptors to run.
    compute_root : Path
        Scratch directory under which HYSPLIT runs, as
        :func:`~stilt.execution.resolve_compute_root` returns it.
    n_cores : int, default 1
        Number of worker processes. 1 runs in this process.
    skip_existing : bool, default True
        Keep particles and footprints that already exist.

    Returns
    -------
    list of SimulationResult
        The results of every simulation, receptor by receptor in input
        order. After an interruption, only the receptors that finished.
    """
    if not receptor_ids:
        return []

    if n_cores <= 1:
        results: list[SimulationResult] = []
        with _sigterm_as_interrupt():
            for i, receptor_id in enumerate(receptor_ids, 1):
                done = run_receptor(
                    project,
                    receptor_id,
                    compute_root=compute_root,
                    skip_existing=skip_existing,
                )
                results.extend(done)
                _log_result(receptor_id, done, i, len(receptor_ids))
                if _interrupted(done):
                    break
        return results

    ordered: dict[int, list[SimulationResult]] = {}
    pool = multiprocessing.Pool(
        n_cores,
        initializer=_init_pool_worker,
        initargs=(str(project.directory), str(compute_root), skip_existing),
    )
    with _sigterm_as_interrupt():
        try:
            for idx, done in pool.imap_unordered(
                _pool_run, list(enumerate(receptor_ids))
            ):
                ordered[idx] = done
                _log_result(receptor_ids[idx], done, len(ordered), len(receptor_ids))
                if _interrupted(done):
                    pool.terminate()
                    break
            else:
                # Normal completion: let workers exit cleanly. terminate() would
                # SIGTERM idle workers, whose handler raises KeyboardInterrupt.
                pool.close()
        except KeyboardInterrupt:
            # Preempted or Ctrl-C: stop the workers and hand back what finished.
            pool.terminate()
        except SystemExit:
            # A requeued Slurm task exits here; its workers must not outlive it.
            pool.terminate()
            raise
        finally:
            pool.join()
    return [result for i in sorted(ordered) for result in ordered[i]]


__all__ = [
    "SimulationResult",
    "make_footprint",
    "run_particles",
    "run_receptor",
    "run_receptors",
]
