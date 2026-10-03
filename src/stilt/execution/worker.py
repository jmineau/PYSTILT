"""
Worker functions that run one simulation, one receptor, or many receptors.

Workers are handed receptors. :func:`run_receptor` runs every variant of
one receptor, and :func:`run_simulation` runs each one: HYSPLIT where the
particles are missing (:func:`run_particles`), then the footprint
(:func:`make_footprint`). Variants with the same transport settings share
one HYSPLIT run. :func:`run_receptors` runs a list of receptors in this
process or a process pool. A :class:`~stilt.Simulation` itself runs
nothing; these functions write through its output directory.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import shutil
import signal
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import xarray as xr

from stilt.exceptions import (
    EmptyFootprint,
    EmptyParticleOutputError,
    SimulationError,
)
from stilt.footprint import calculate
from stilt.meteorology import Met
from stilt.particles import prepare
from stilt.simulation import Simulation
from stilt.transforms import TransformContext
from stilt.transport import get_model

from .runner import resolve_compute_root

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
    phase : {"particles", "footprint"} or None
        Step that failed, when the simulation did not complete.
    """

    sim_id: str
    status: Status
    error: str | None = None
    ran_hysplit: bool = False
    phase: str | None = None


def _append_error_log(sim: Simulation, *, phase: str, error: BaseException) -> None:
    """Append the error and its traceback to the simulation's log in the output directory."""
    folder = sim.output.particles(sim.variant.name, sim.variant.transport)
    log_path = folder.log_path(sim.receptor.id)
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
    MeteorologyError, HYSPLITTimeoutError, HYSPLITFailureError,
    NoParticleOutputError, EmptyParticleOutputError
        As the HYSPLIT driver and the particle reader raise them.
    """
    params = sim.variant.transport
    model = get_model(params.model.name)
    folder = sim.output.particles(sim.variant.name, sim.variant.transport)
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
            raise EmptyParticleOutputError(f"HYSPLIT wrote no particles for {sim.id}")
        particles = prepare(result.particles, sim.receptor, params)
        folder.write(sim.receptor, particles, params, result.met_files)
        succeeded = True
        return particles
    finally:
        if scratch_log.exists():
            folder.write_log(rid, scratch_log.read_text())
        _finish_scratch(
            workdir, folder.scratch_path(rid), keep=keep_scratch or not succeeded
        )


def _finish_scratch(workdir: Path, kept: Path, *, keep: bool) -> None:
    """Copy *workdir* to *kept* when *keep*, then remove it."""
    if not workdir.exists():
        return
    if keep:
        shutil.rmtree(kept, ignore_errors=True)
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(workdir, kept, symlinks=True)
    shutil.rmtree(workdir, ignore_errors=True)


def make_footprint(
    sim: Simulation,
    particles: pd.DataFrame,
    *,
    context: TransformContext,
) -> xr.DataArray | None:
    """
    Calculate a simulation's footprint from *particles* and write it to its folder.

    The settings are the variant's own. When no particle reaches the grid,
    an empty footprint is recorded with the reason and ``None`` is returned.
    :meth:`stilt.Simulation.generate_footprint` makes footprints with other
    settings, without writing them.

    Raises
    ------
    TypeError
        If the variant has no grid.
    """
    config = sim.variant.footprint
    if config is None:
        raise TypeError(f"{sim.id} has no footprint settings (no grid).")
    folder = sim.output.particles(sim.variant.name, sim.variant.transport)
    feet = folder.footprints(config, name=sim.variant.name)
    try:
        foot = calculate(
            particles, sim.receptor, config, name=sim.variant.name, context=context
        )
    except EmptyFootprint as error:
        feet.write_empty(sim.receptor, error.reason, name=sim.variant.name)
        return None
    feet.write(foot)
    return foot


def run_simulation(
    sim: Simulation,
    *,
    met: Met,
    compute_root: Path,
    project_dir: Path | None = None,
    keep_scratch: bool = False,
    timeout: int | None = None,
    skip_existing: bool = True,
    footprint_stale: bool = False,
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
    met : Met
        Meteorology for the run.
    compute_root : Path
        Scratch root; HYSPLIT runs in ``compute_root / sim.id``.
    project_dir : Path, optional
        Directory that relative file names in transform settings are taken
        from.
    keep_scratch : bool, default False
        Keep every run's working directory under the output directory.
    timeout : int, optional
        Time limit for one HYSPLIT run, in seconds.
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
    phase = "particles"
    ran_hysplit = False
    try:
        particles: pd.DataFrame | None = None
        if not (skip_existing and sim.has_particles):
            particles = run_particles(
                sim,
                met=met,
                workdir=compute_root / sim.id,
                keep_scratch=keep_scratch,
                timeout=timeout,
            )
            ran_hysplit = True

        if sim.makes_footprint and (
            ran_hysplit or footprint_stale or not (skip_existing and sim.has_footprint)
        ):
            phase = "footprint"
            if particles is None:
                particles = sim.particles
            context = TransformContext(
                receptor=sim.receptor, variant=sim.variant.name, directory=project_dir
            )
            make_footprint(sim, particles, context=context)
        return SimulationResult(str(sim.id), "complete", ran_hysplit=ran_hysplit)
    except Exception as error:
        logger.exception("simulation %s failed during %s: %s", sim.id, phase, error)
        try:
            _append_error_log(sim, phase=phase, error=error)
        except Exception:
            logger.exception("simulation %s: could not write the failure log", sim.id)
        status = "failed" if isinstance(error, SimulationError) else "error"
        return SimulationResult(str(sim.id), status, error=str(error), phase=phase)


def run_receptor(
    project: Project,
    receptor_id: str,
    *,
    compute_root: Path,
    skip_existing: bool = True,
) -> list[SimulationResult]:
    """
    Run every simulation of one receptor, and return their results.

    Variants with the same transport settings share one HYSPLIT run: the
    first of them runs it, the others reuse the particles and make their own
    footprints. A footprint whose particles were replaced in this call is
    recomputed even with ``skip_existing``. A ``KeyboardInterrupt``, such as
    a preempted job, gives an ``interrupted`` result.

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
        One per variant, in config order. After an interruption, the last
        one is ``interrupted``.
    """
    sims = [project.simulation(receptor_id, variant) for variant in project.variants]
    results: list[SimulationResult] = []
    reran: set[str] = set()  # transport settings whose HYSPLIT ran in this call
    failed: dict[str, SimulationResult] = {}  # ... and whose HYSPLIT failed
    sim = None
    try:
        for sim in sims:
            key = sim.variant.transport.hash
            if key in failed:
                # Variants with these transport settings share one HYSPLIT
                # run, and it already failed. Running it again fails the same way.
                first = failed[key]
                results.append(
                    SimulationResult(
                        str(sim.id), first.status, error=first.error, phase="particles"
                    )
                )
                continue
            result = run_simulation(
                sim,
                met=project.mets[sim.variant.met],
                compute_root=compute_root,
                project_dir=project.directory,
                keep_scratch=project.config.execution.keep_scratch,
                timeout=project.config.execution.timeout,
                skip_existing=skip_existing or key in reran,
                footprint_stale=key in reran,
            )
            if result.ran_hysplit:
                reran.add(key)
            elif result.phase == "particles":
                failed[key] = result
            results.append(result)
    except KeyboardInterrupt:
        label = str(sim.id) if sim is not None else receptor_id
        results.append(SimulationResult(label, "interrupted", error="Worker preempted"))
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
    compute_root: str | Path | None = None,
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
    compute_root : str or Path, optional
        Scratch directory under which HYSPLIT runs
        (:func:`~stilt.execution.resolve_compute_root`).
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
    scratch = resolve_compute_root(project, compute_root)

    if n_cores <= 1:
        results: list[SimulationResult] = []
        with _sigterm_as_interrupt():
            for i, receptor_id in enumerate(receptor_ids, 1):
                done = run_receptor(
                    project,
                    receptor_id,
                    compute_root=scratch,
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
        initargs=(str(project.directory), str(scratch), skip_existing),
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
    "run_receptor",
    "run_receptors",
    "run_simulation",
    "run_particles",
    "make_footprint",
]
