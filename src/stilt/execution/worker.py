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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import xarray as xr

from stilt.exceptions import EmptyFootprint, SimulationError
from stilt.execution.config import ExecutionConfig
from stilt.footprint import calc_footprint
from stilt.meteorology import Met
from stilt.output import Kind
from stilt.simulation import Simulation
from stilt.transport import run_model

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


#: The step of a simulation that failed: its particles or its footprint.
Step = Literal["particles", "footprint"]

#: The output folder each step writes to.
_KIND: dict[Step, Kind] = {"particles": "particles", "footprint": "footprints"}


# -- failure records ---------------------------------------------------------


def _failed(sim: Simulation, step: Step, error: Exception) -> str:
    """
    Record why *sim* failed at *step*, and return one line saying so.

    The record goes in the logs of the folder whose result failed
    (:meth:`stilt.output.Output.record_failure`): a particles failure
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
            "simulation %s failed during %s (%s): %s", sim, step, reason, error
        )
    else:
        logger.exception("simulation %s failed during %s: %s", sim, step, error)
    record: dict[str, Any] = {
        "step": step,
        "reason": str(reason),
        "message": str(error),
        "time": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    }
    if not expected:
        record["traceback"] = traceback.format_exc()
    try:
        sim.output.record_failure(_KIND[step], sim.variant, sim.receptor.id, record)
    except Exception:
        logger.exception("simulation %s: could not record the failure", sim)
    return f"{sim.variant.name} failed during {step} ({reason}): {error}"


def _succeeded(sim: Simulation, step: Step) -> None:
    """Remove *sim*'s failure record for *step*, whose result is now written."""
    sim.output.clear_failure(_KIND[step], sim.variant, sim.receptor.id)


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

    The model the settings name (HYSPLIT) runs in *workdir*.
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
        Directory to run in. Created here.
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
    output, rid = sim.output, sim.receptor.id
    # The model runs in an empty directory. A job stopped partway can leave
    # this simulation's directory behind; it is PYSTILT's own, so clear it.
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    log = workdir / "stilt.log"
    succeeded = False
    try:
        result = run_model(
            sim.variant.model.name,
            sim.receptor,
            sim.variant.transport,
            met,
            workdir,
            timeout=timeout,
        )
        output.write_particles(
            sim.variant, sim.receptor, result.particles, result.met_files
        )
        succeeded = True
        return result.particles
    finally:
        if log.exists():
            output.write_log(sim.variant, rid, log.read_text())
        # An empty directory is not kept: a run that failed before writing
        # anything, such as on missing meteorology, has nothing to look at.
        if (keep_scratch or not succeeded) and any(workdir.iterdir()):
            output.keep_workdir(sim.variant, rid, workdir)
        shutil.rmtree(workdir, ignore_errors=True)


def make_footprint(sim: Simulation, particles: pd.DataFrame) -> xr.DataArray | None:
    """
    Calculate a simulation's footprint from *particles* and write it to its folder.

    The settings are the variant's own. A relative file name in a
    transform's settings starts from the project directory
    (``sim.directory``). When no particle reaches the grid,
    an empty footprint is recorded with the reason and ``None`` is returned.
    :meth:`stilt.Simulation.calc_footprint` makes footprints with other
    settings, without writing them.

    Raises
    ------
    ValueError
        If the variant has no grid.
    """
    config = sim.variant.footprint
    if config is None or config.grid is None:
        raise ValueError(f"Variant {sim.variant.name!r} makes no footprints (no grid).")
    try:
        foot = calc_footprint(
            particles,
            sim.receptor,
            config.grid,
            smooth_factor=config.smooth_factor,
            time_integrate=config.time_integrate,
            transforms=config.transforms,
            name=sim.variant.name,
            directory=sim.directory,
            geometry_hash=sim.variant.geometry_hash,
        )
    except EmptyFootprint as error:
        sim.output.write_empty_footprint(sim.variant, sim.receptor, error.reason)
        return None
    sim.output.write_footprint(sim.variant, foot)
    return foot


def run_receptor(
    project: Project,
    receptor_id: str,
    *,
    compute_root: Path,
    execution: ExecutionConfig | None = None,
    skip_existing: bool = True,
) -> list[str]:
    """
    Run every simulation of one receptor, and return what did not complete.

    Variants with the same transport settings share one set of particles,
    so they run as a group: HYSPLIT runs once for the group when the
    particles are missing (or always, without ``skip_existing``), and each
    variant's footprint is made from those particles in memory. A footprint
    is made again whenever its particles were, so it always matches them.
    A failure is recorded with the simulation (:attr:`stilt.Simulation.failure`)
    and does not stop the others; a failed HYSPLIT run fails every variant
    of its group. A ``KeyboardInterrupt``, such as a preempted job, stops
    the receptor and is raised again.

    Parameters
    ----------
    project : Project
        Project the receptor belongs to.
    receptor_id : str
        Receptor to run.
    compute_root : Path
        Scratch directory under which HYSPLIT runs, as
        :func:`~stilt.execution.resolve_compute_root` returns it.
    execution : ExecutionConfig, optional
        Execution settings, for ``timeout`` and ``keep_scratch``. Defaults
        to the project's.
    skip_existing : bool, default True
        Keep particles and footprints that already exist.

    Returns
    -------
    list of str
        One line per failure, empty when every simulation is complete. The
        failures are also recorded in the output directory, which is where
        :meth:`stilt.Project.status` reads them.
    """
    execution = execution if execution is not None else project.config.execution
    groups: dict[str, list[Simulation]] = {}
    for variant in project.variants:
        sim = project.simulation(receptor_id, variant)
        groups.setdefault(sim.variant.particles_hash, []).append(sim)
    problems: list[str] = []
    for group in groups.values():
        problems += _run_group(
            project,
            group,
            compute_root=compute_root,
            execution=execution,
            skip_existing=skip_existing,
        )
    return problems


def _run_group(
    project: Project,
    sims: list[Simulation],
    *,
    compute_root: Path,
    execution: ExecutionConfig,
    skip_existing: bool,
) -> list[str]:
    """Run the simulations of one receptor that share particles: HYSPLIT at most once, then each footprint."""
    first = sims[0]
    particles: pd.DataFrame | None = None
    rerun = not (skip_existing and first.has_particles)
    if rerun:
        try:
            particles = run_particles(
                first,
                met=project.mets[first.variant.met],
                workdir=compute_root / first.receptor.id / first.variant.name,
                keep_scratch=execution.keep_scratch,
                timeout=execution.timeout,
            )
        except Exception as error:
            return [_failed(first, "particles", error)]
        _succeeded(first, "particles")

    problems: list[str] = []
    for sim in sims:
        if not sim.makes_footprint or (
            not rerun and skip_existing and sim.has_footprint
        ):
            continue
        try:
            if particles is None:
                particles = first.particles  # read once for the group
            make_footprint(sim, particles)
        except Exception as error:
            problems.append(_failed(sim, "footprint", error))
            continue
        _succeeded(sim, "footprint")
    return problems


def _log_progress(receptor_id: str, problems: list[str], done: int, total: int) -> None:
    """Log one progress line for a finished receptor: complete, or its first problem."""
    if problems:
        logger.info("[%d/%d] %s: %s", done, total, receptor_id, problems[0])
    else:
        logger.info("[%d/%d] %s complete", done, total, receptor_id)


# -- process pool -------------------------------------------------------------

_POOL_PROJECT: Project | None = None
_POOL_COMPUTE_ROOT: Path | None = None
_POOL_EXECUTION: ExecutionConfig | None = None
_POOL_SKIP: bool = True


def _init_pool_worker(
    project: str, compute_root: str, execution: ExecutionConfig, skip_existing: bool
) -> None:
    """Open the worker process's Project and make SIGTERM raise KeyboardInterrupt."""
    from stilt.project import Project

    global _POOL_PROJECT, _POOL_COMPUTE_ROOT, _POOL_EXECUTION, _POOL_SKIP
    signal.signal(signal.SIGTERM, _raise_interrupt)
    _POOL_PROJECT = Project(project)
    _POOL_COMPUTE_ROOT = Path(compute_root)
    _POOL_EXECUTION = execution
    _POOL_SKIP = skip_existing


def _pool_run(item: tuple[int, str]) -> tuple[int, list[str] | None]:
    """Run one receptor in a pool worker, returning its index and problems, or ``None`` when it was stopped."""
    idx, receptor_id = item
    assert _POOL_PROJECT is not None and _POOL_COMPUTE_ROOT is not None
    try:
        return idx, run_receptor(
            _POOL_PROJECT,
            receptor_id,
            compute_root=_POOL_COMPUTE_ROOT,
            execution=_POOL_EXECUTION,
            skip_existing=_POOL_SKIP,
        )
    except KeyboardInterrupt:
        # A pool task must return; the parent stops the pool when it sees this.
        return idx, None


def run_receptors(
    project: Project,
    receptor_ids: list[str],
    *,
    compute_root: Path,
    execution: ExecutionConfig | None = None,
    skip_existing: bool = True,
) -> None:
    """
    Run a list of receptors, in this process or in a process pool.

    Pool workers open the project again from its directory. A SIGTERM,
    such as Slurm preemption or the end of the job's time limit, stops the
    batch. What finished is in the output directory, with a failure record
    for each simulation that failed.

    Parameters
    ----------
    project : Project
        Project the receptors belong to.
    receptor_ids : list of str
        Receptors to run.
    compute_root : Path
        Scratch directory under which HYSPLIT runs, as
        :func:`~stilt.execution.resolve_compute_root` returns it.
    execution : ExecutionConfig, optional
        Execution settings: ``cpus`` is the number of worker processes (1
        runs in this process), and ``timeout`` and ``keep_scratch`` apply
        to each run. Defaults to the project's.
    skip_existing : bool, default True
        Keep particles and footprints that already exist.
    """
    if not receptor_ids:
        return
    execution = execution if execution is not None else project.config.execution
    total = len(receptor_ids)

    if execution.cpus <= 1:
        with _sigterm_as_interrupt():
            for i, receptor_id in enumerate(receptor_ids, 1):
                try:
                    problems = run_receptor(
                        project,
                        receptor_id,
                        compute_root=compute_root,
                        execution=execution,
                        skip_existing=skip_existing,
                    )
                except KeyboardInterrupt:
                    logger.info("stopped at %s; the rest did not run", receptor_id)
                    return
                _log_progress(receptor_id, problems, i, total)
        return

    pool = multiprocessing.Pool(
        execution.cpus,
        initializer=_init_pool_worker,
        initargs=(str(project.directory), str(compute_root), execution, skip_existing),
    )
    done = 0
    with _sigterm_as_interrupt():
        try:
            for idx, problems in pool.imap_unordered(
                _pool_run, list(enumerate(receptor_ids))
            ):
                if problems is None:
                    logger.info("stopped at %s", receptor_ids[idx])
                    pool.terminate()
                    break
                done += 1
                _log_progress(receptor_ids[idx], problems, done, total)
            else:
                # Normal completion: let workers exit cleanly. terminate() would
                # SIGTERM idle workers, whose handler raises KeyboardInterrupt.
                pool.close()
        except KeyboardInterrupt:
            # Preempted or Ctrl-C: stop the workers; what finished is written.
            pool.terminate()
        except SystemExit:
            # A requeued Slurm task exits here; its workers must not outlive it.
            pool.terminate()
            raise
        finally:
            pool.join()


__all__ = [
    "make_footprint",
    "run_particles",
    "run_receptor",
    "run_receptors",
]
