"""Worker entry points for STILT simulation execution.

This module provides the :func:`run_worker` function that each executor
dispatches to compute trajectories and footprints for a single simulation,
the :func:`pull_worker_loop` function that drives the pull-based
claim/run/release cycle, the :func:`push_worker_loop` helper used by
chunk-based executors, plus the :func:`_run_batch` helper used internally
by both worker paths.
"""

from __future__ import annotations

import datetime as dt
import logging
import multiprocessing
import os
import signal
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from stilt.artifacts import (
    ArtifactStore,
    clear_empty_footprint_marker,
    write_empty_footprint_marker,
)
from stilt.config import FootprintConfig, STILTParams
from stilt.errors import SimulationError
from stilt.meteorology import MetStream
from stilt.receptor import Receptor
from stilt.repositories import (
    ArtifactSummary,
    QueueRepository,
    SimulationAttempt,
    SimulationClaim,
    StateRepository,
)
from stilt.simulation import SimID, Simulation

logger = logging.getLogger(__name__)


def _append_simulation_error_log(
    sim: Simulation,
    *,
    phase: str,
    error: BaseException,
) -> None:
    """Append a durable PYSTILT error section to one simulation log."""
    log_path = sim.log_path
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


class SimulationTask(BaseModel):
    """Serialisable bundle of everything a worker needs to run one simulation.

    Passed as the single argument to :func:`run_worker` so it can be pickled
    and shipped to a remote executor (local subprocess, Slurm task, Kubernetes Job)
    without needing shared state.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    compute_root: Path
    sim_id: SimID
    meteorology: MetStream
    receptor: Receptor
    params: STILTParams
    foot_configs: dict[str, FootprintConfig] | None = None
    artifact_store: ArtifactStore
    repository: StateRepository
    claim: SimulationClaim | None = None
    dispatch: Literal["push", "pull"] = "pull"


@dataclass
class SimulationResult:
    """Typed result contract returned by one worker-run simulation."""

    sim_id: SimID
    status: Literal["complete", "complete-empty", "failed", "error", "interrupted"]
    traj_path: Path | None = None
    error_traj_path: Path | None = None
    log_path: Path | None = None
    wrote_traj: bool = False
    foot_paths: list[Path] = field(default_factory=list)
    empty_footprints: list[str] = field(default_factory=list)
    footprint_statuses: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def _build_artifact_summary(
    sim: Simulation,
    footprint_statuses: dict[str, str] | None = None,
) -> ArtifactSummary:
    """Build a lightweight durable artifact summary from a simulation handle."""
    return ArtifactSummary(
        traj_present=sim._artifact_path(sim.trajectories_path) is not None,
        error_traj_present=sim._artifact_path(sim.error_trajectories_path) is not None,
        log_present=sim._artifact_path(sim.log_path) is not None,
        footprints=dict(footprint_statuses or {}),
    )


def _record_attempt(
    repository: QueueRepository,
    sim_id: SimID,
    started_at: dt.datetime,
    outcome: str,
    claim_token: str | None = None,
    *,
    terminal: bool | None = None,
    error: str | None = None,
) -> None:
    """Append one execution attempt record for a simulation run."""
    repository.record_attempt(
        SimulationAttempt(
            attempt_id=uuid.uuid4().hex,
            sim_id=str(sim_id),
            claim_token=claim_token,
            started_at=started_at,
            finished_at=dt.datetime.now(dt.timezone.utc),
            outcome=outcome,
            terminal=(outcome == "failed" if terminal is None else terminal),
            error=error,
        )
    )


def _claim_token(claim: SimulationClaim | None) -> str | None:
    """Return the claim token for attempt history, if one exists."""
    return claim.claim_token if claim is not None else None


def _default_worker_id() -> str:
    """Return a stable-ish worker identifier for queue claims."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _claim_is_current(
    repository: QueueRepository,
    claim: SimulationClaim | None,
) -> bool:
    """Return True when no claim is attached or the claim still owns the sim."""
    if claim is None:
        return True
    return repository.claim_is_current(claim.sim_id, claim.claim_token)


def _current_runtime_repository(
    repository: QueueRepository | None,
    claim: SimulationClaim | None,
) -> QueueRepository | None:
    """Return the queue repository only when this worker still owns the claim."""
    if repository is None or not _claim_is_current(repository, claim):
        return None
    return repository


def _update_runtime_state(
    repository: QueueRepository | None,
    claim: SimulationClaim | None,
    *,
    sim_id: SimID,
    update: Callable[[QueueRepository], None],
    started_at: dt.datetime | None = None,
    outcome: str | None = None,
    terminal: bool | None = None,
    error: str | None = None,
) -> None:
    """Apply a runtime state update only when the active claim is still current."""
    active_repository = _current_runtime_repository(repository, claim)
    if active_repository is None:
        return

    update(active_repository)

    if outcome is None:
        return
    if started_at is None:
        raise ValueError("started_at is required when recording an attempt.")

    _record_attempt(
        active_repository,
        sim_id,
        started_at,
        outcome,
        claim_token=_claim_token(claim),
        terminal=terminal,
        error=error,
    )


def _start_claim_heartbeat(
    repository: QueueRepository,
    claim: SimulationClaim | None,
) -> tuple[threading.Event | None, threading.Thread | None]:
    """Start a background heartbeat thread for an active lease claim."""
    if claim is None:
        return None, None

    lease_ttl = max(
        (claim.expires_at - claim.heartbeat_at).total_seconds(),
        1.0,
    )
    interval = max(1.0, lease_ttl / 3.0)
    stop_event = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop_event.wait(interval):
            if not repository.heartbeat_claim(
                claim.sim_id,
                claim.claim_token,
                lease_ttl=lease_ttl,
            ):
                return

    thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    thread.start()
    return stop_event, thread


def chunk_path(chunk_dir: str | Path, index: int) -> Path:
    """Return the canonical chunk-file path for one worker shard."""
    return Path(chunk_dir) / f"task_{index}.txt"


def _read_chunk(path: str | Path) -> list[str]:
    """Read one chunk file into a list of simulation ids."""
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_worker(args: SimulationTask) -> SimulationResult:
    """Run one STILT simulation task and any requested footprint outputs.

    When ``args.foot_configs`` is ``None``, only trajectories are run.  When
    configs are provided, trajectories are auto-run as needed and all
    footprint products are computed in a single pass so the trajectory file
    is loaded exactly once. Durable artifacts are published through
    ``args.artifact_store`` and queue state is updated directly in the
    repository.

    Parameters
    ----------
    args : SimulationTask
        Serialisable bundle containing project dir, sim ID, meteorology,
        receptor, STILT params, optional footprint configs, artifact store,
        and repository.

    Returns
    -------
    SimulationResult
        Typed per-simulation execution result.
    """
    started_at = dt.datetime.now(dt.timezone.utc)

    # Build the simulation object
    sim = Simulation(
        directory=args.compute_root / args.sim_id,
        meteorology=args.meteorology,
        receptor=args.receptor,
        params=args.params,
        artifact_store=args.artifact_store,
    )

    # Determine whether we need a runtime repo (do we need to track live queue state?)
    track_runtime = args.dispatch == "pull"
    runtime_repository = (
        cast(QueueRepository, args.repository) if track_runtime else None
    )
    (
        heartbeat_stop,
        heartbeat_thread,
    ) = (  # Start a heartbeat thread if we have a live claim to maintain, else no-op
        _start_claim_heartbeat(runtime_repository, args.claim)
        if runtime_repository is not None
        else (None, None)
    )

    try:
        # --- Trajectories Only (no footprints) ---

        if not args.foot_configs:
            try:
                # Run the trajectories and publish durable artifacts
                sim.run_trajectories(write=True)
                args.artifact_store.publish_simulation(sim)

                summary = _build_artifact_summary(sim)

                def _mark_complete(repository: QueueRepository) -> None:
                    repository.mark_trajectory_complete(str(sim.id))
                    repository.record_artifacts(str(sim.id), summary)

                _update_runtime_state(
                    runtime_repository,
                    args.claim,
                    sim_id=sim.id,
                    update=_mark_complete,
                    started_at=started_at,
                    outcome="complete",
                )
                return SimulationResult(
                    sim_id=sim.id,
                    status="complete",
                    traj_path=sim.trajectories_path,
                    error_traj_path=(
                        sim.error_trajectories_path
                        if sim.error_trajectories_path.exists()
                        else None
                    ),
                    log_path=sim.log_path if sim.log_path.exists() else None,
                    wrote_traj=True,
                )
            except Exception as e:
                outcome = "failed" if isinstance(e, SimulationError) else "error"

                _append_simulation_error_log(sim, phase="trajectory", error=e)
                args.artifact_store.publish_simulation(sim)

                summary = _build_artifact_summary(sim)

                def _mark_failed(
                    repository: QueueRepository,
                    _e: str = str(e),
                    _summary: ArtifactSummary = summary,
                ) -> None:
                    repository.mark_trajectory_failed(str(sim.id), _e)
                    repository.record_artifacts(str(sim.id), _summary)

                _update_runtime_state(
                    runtime_repository,
                    args.claim,
                    sim_id=sim.id,
                    update=_mark_failed,
                    started_at=started_at,
                    outcome=outcome,
                    terminal=True,
                    error=str(e),
                )
                return SimulationResult(
                    sim_id=sim.id,
                    status=outcome,
                    error=str(e),
                    log_path=sim.log_path if sim.log_path.exists() else None,
                    wrote_traj=False,
                )

        # --- Trajectories + Footprints ---

        traj_path = sim.trajectories_path
        wrote_traj = False
        foot_paths: list[Path] = []
        empty_footprints: list[str] = []
        footprint_statuses: dict[str, str] = {}
        completed_footprints: list[str] = []
        first_name = next(iter(args.foot_configs))
        current_name = first_name
        try:
            traj_existed = sim._artifact_path(traj_path) is not None
            traj_marked_complete = False
            for name, fc in args.foot_configs.items():
                current_name = name
                foot = sim.generate_footprint(name, fc, write=True)

                if (
                    not traj_marked_complete
                    and sim._artifact_path(traj_path) is not None
                ):
                    traj_marked_complete = True
                    wrote_traj = not traj_existed and traj_path.exists()

                if foot is None:
                    sim.footprint_path(name).unlink(missing_ok=True)
                    write_empty_footprint_marker(sim.directory, name, sim_id=sim.id)
                    empty_footprints.append(name)
                    footprint_statuses[name] = "complete-empty"
                else:
                    clear_empty_footprint_marker(sim.directory, name, sim_id=sim.id)
                    completed_footprints.append(name)
                    foot_paths.append(sim.footprint_path(name))
                    footprint_statuses[name] = "complete"

                if fc.error:
                    error_name = f"{name}_error"
                    current_name = error_name
                    error_foot = sim.generate_footprint(
                        name, fc, write=True, error=True
                    )
                    if error_foot is None:
                        sim.footprint_path(error_name).unlink(missing_ok=True)
                        write_empty_footprint_marker(
                            sim.directory,
                            error_name,
                            sim_id=sim.id,
                        )
                        empty_footprints.append(error_name)
                        footprint_statuses[error_name] = "complete-empty"
                    else:
                        clear_empty_footprint_marker(
                            sim.directory,
                            error_name,
                            sim_id=sim.id,
                        )
                        completed_footprints.append(error_name)
                        foot_paths.append(sim.footprint_path(error_name))
                        footprint_statuses[error_name] = "complete"

            if not traj_marked_complete and sim._artifact_path(traj_path) is not None:
                wrote_traj = not traj_existed and traj_path.exists()

            args.artifact_store.publish_simulation(sim)

            summary = _build_artifact_summary(sim, footprint_statuses)

            def _mark_footprints_complete(repository: QueueRepository) -> None:
                if sim._artifact_path(traj_path) is not None:
                    repository.mark_trajectory_complete(str(sim.id))
                for name in completed_footprints:
                    repository.mark_footprint_complete(str(sim.id), name)
                for name in empty_footprints:
                    repository.mark_footprint_empty(str(sim.id), name)
                repository.record_artifacts(str(sim.id), summary)

            _update_runtime_state(
                runtime_repository,
                args.claim,
                sim_id=sim.id,
                update=_mark_footprints_complete,
                started_at=started_at,
                outcome="complete" if foot_paths else "complete-empty",
            )

            return SimulationResult(
                sim_id=sim.id,
                status="complete" if foot_paths else "complete-empty",
                traj_path=traj_path if wrote_traj else None,
                error_traj_path=(
                    sim.error_trajectories_path
                    if sim.error_trajectories_path.exists()
                    else None
                ),
                log_path=sim.log_path if sim.log_path.exists() else None,
                wrote_traj=wrote_traj,
                foot_paths=foot_paths,
                empty_footprints=empty_footprints,
                footprint_statuses=footprint_statuses,
            )
        except Exception as e:
            _append_simulation_error_log(
                sim,
                phase=f"footprint:{current_name}",
                error=e,
            )
            args.artifact_store.publish_simulation(sim)
            footprint_statuses[current_name] = "failed"
            summary = _build_artifact_summary(sim, footprint_statuses)

            def _mark_footprint_failed(
                repository: QueueRepository,
                _e: str = str(e),
                _name: str = current_name,
                _summary: ArtifactSummary = summary,
            ) -> None:
                repository.mark_footprint_failed(str(sim.id), _name, _e)
                if traj_path.exists():
                    repository.mark_trajectory_complete(str(sim.id))
                else:
                    repository.mark_trajectory_failed(str(sim.id), _e)
                repository.record_artifacts(str(sim.id), _summary)

            _update_runtime_state(
                runtime_repository,
                args.claim,
                sim_id=sim.id,
                update=_mark_footprint_failed,
                started_at=started_at,
                outcome="error",
                terminal=True,
                error=str(e),
            )
            return SimulationResult(
                sim_id=sim.id,
                status="error",
                error=str(e),
                error_traj_path=(
                    sim.error_trajectories_path
                    if sim.error_trajectories_path.exists()
                    else None
                ),
                log_path=sim.log_path if sim.log_path.exists() else None,
                wrote_traj=False,
                foot_paths=[],
            )
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)


def _sigterm_handler(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def _init_pool_worker() -> None:
    """Install SIGTERM → KeyboardInterrupt in each pool worker process."""
    signal.signal(signal.SIGTERM, _sigterm_handler)


def _run_batch(batch: list[SimulationTask], n_cores: int = 1) -> list[SimulationResult]:
    """Run a list of simulations from either push chunks or the pull queue.

    Called by the worker entrypoints with a batch of simulation tasks.
    Runs them sequentially (``n_cores=1``) or in a local multiprocessing pool
    (``n_cores>1``).

    When Slurm preempts or wall-times the task it sends SIGTERM.  The handler
    converts SIGTERM to :exc:`KeyboardInterrupt` so the in-flight simulation
    can call ``mark_trajectory_failed()`` before the process exits.

    Parameters
    ----------
    batch : list[SimulationTask]
        Simulations assigned to this array task.
    n_cores : int, optional
        Number of local CPU cores to use within this task.  Defaults to 1.

    Returns
    -------
    list[SimulationResult]
        Per-simulation results in batch order.
    """
    if n_cores <= 1:
        signal.signal(signal.SIGTERM, _sigterm_handler)
        results = []
        for args in batch:
            try:
                results.append(run_worker(args))
            except KeyboardInterrupt:
                runtime_repository = (
                    cast(QueueRepository, args.repository)
                    if args.dispatch == "pull"
                    else None
                )
                interrupt_message = "Worker preempted by SIGTERM"
                interrupted_at = dt.datetime.now(dt.timezone.utc)

                def _mark_interrupted(
                    repository: QueueRepository,
                    _sid: SimID = args.sim_id,
                    _msg: str = interrupt_message,
                ) -> None:
                    repository.mark_trajectory_failed(str(_sid), _msg)

                _update_runtime_state(
                    runtime_repository,
                    args.claim,
                    sim_id=args.sim_id,
                    update=_mark_interrupted,
                    started_at=interrupted_at,
                    outcome="interrupted",
                    error=interrupt_message,
                )
                break
        return results
    with multiprocessing.Pool(n_cores, initializer=_init_pool_worker) as pool:
        return list(pool.map(_run_worker_guarded, batch))


def _run_worker_guarded(args: SimulationTask) -> SimulationResult:
    """Wrap run_worker to catch KeyboardInterrupt from SIGTERM in pool workers."""
    try:
        return run_worker(args)
    except KeyboardInterrupt:
        runtime_repository = (
            cast(QueueRepository, args.repository) if args.dispatch == "pull" else None
        )
        interrupt_message = "Worker preempted by SIGTERM"
        interrupted_at = dt.datetime.now(dt.timezone.utc)

        def _mark_interrupted(repository: QueueRepository) -> None:
            repository.mark_trajectory_failed(str(args.sim_id), interrupt_message)

        _update_runtime_state(
            runtime_repository,
            args.claim,
            sim_id=args.sim_id,
            update=_mark_interrupted,
            started_at=interrupted_at,
            outcome="interrupted",
            error=interrupt_message,
        )
        return SimulationResult(
            sim_id=args.sim_id, status="interrupted", wrote_traj=False
        )


def _run_transactional_claim_loop(
    model: object,
    *,
    follow: bool,
    poll_interval: float,
    lease_ttl: float,
) -> None:
    """Run one-at-a-time workers through a backend transactional claim context."""
    worker_id = _default_worker_id()
    idle_sleep = max(poll_interval, 0.1)
    max_idle_sleep = max(idle_sleep, min(60.0, max(lease_ttl / 3.0, idle_sleep)))
    while True:
        model.repository.reclaim_expired_claims()  # type: ignore[attr-defined]
        with model.repository.begin_claim_uow(  # type: ignore[attr-defined]
            worker_id=worker_id,
            lease_ttl=lease_ttl,
        ) as uow:
            if uow is None:
                if follow:
                    time.sleep(idle_sleep)
                    idle_sleep = min(idle_sleep * 2.0, max_idle_sleep)
                    continue
                return

            idle_sleep = max(poll_interval, 0.1)
            run_args = model._build_run_args(uow.claim.sim_id)  # type: ignore[attr-defined]
            if run_args is None:
                uow.release()
                if follow:
                    time.sleep(idle_sleep)
                    idle_sleep = min(idle_sleep * 2.0, max_idle_sleep)
                    continue
                return

            transactional_args = run_args.model_copy(
                update={
                    "claim": uow.claim,
                    "repository": cast(QueueRepository, uow.repository),
                    "dispatch": "pull",
                }
            )
            run_worker(transactional_args)


def pull_worker_loop(
    model: object,
    n_cores: int = 1,
    follow: bool = False,
    poll_interval: float = 10.0,
    lease_ttl: float = 1800.0,
) -> None:
    """Drain pending simulations from the model's repository.

    Each iteration claims up to *n_cores* simulations, builds
    :class:`SimulationTask` via ``model._build_run_args(sim_id)``, releases any
    claims where the sim was skipped, then hands the rest to
    :func:`_run_batch`.

    Parameters
    ----------
    model : Model
        A :class:`~stilt.model.Model` instance (typed as ``object`` to avoid a
        circular import; the duck-typed contract is ``repository`` + ``_build_run_args``).
    n_cores : int, optional
        Number of simulations to claim and run per iteration.  Defaults to 1.
    follow : bool, optional
        If ``False`` (default), exit when the queue is empty (batch mode).
        If ``True``, keep polling for new work (streaming mode).
    poll_interval : float, optional
        Seconds to sleep between polls when *follow* is ``True`` and the queue
        is empty.  Defaults to 10.
    lease_ttl : float, optional
        Seconds for claim leases to live before expiring.  Defaults to 1800 (30 minutes).
    """
    if n_cores == 1 and hasattr(model.repository, "begin_claim_uow"):  # type: ignore[attr-defined]
        _run_transactional_claim_loop(
            model,
            follow=follow,
            poll_interval=poll_interval,
            lease_ttl=lease_ttl,
        )
        return

    worker_id = _default_worker_id()
    idle_sleep = max(poll_interval, 0.1)
    max_idle_sleep = max(idle_sleep, min(60.0, max(lease_ttl / 3.0, idle_sleep)))
    while True:
        model.repository.reclaim_expired_claims()  # type: ignore[attr-defined]
        claims = model.repository.claim_pending_claims(  # type: ignore[attr-defined]
            n=n_cores,
            worker_id=worker_id,
            lease_ttl=lease_ttl,
        )
        if not claims:
            if follow:
                time.sleep(idle_sleep)
                idle_sleep = min(idle_sleep * 2.0, max_idle_sleep)
                continue
            return
        idle_sleep = max(poll_interval, 0.1)

        # Build SimulationTask objects; release claims for sims that should be skipped.
        batch: list[SimulationTask] = []
        released: list[SimulationClaim] = []
        for claim in claims:
            run_args = model._build_run_args(claim.sim_id)  # type: ignore[attr-defined]
            if run_args is None:
                released.append(claim)
            else:
                batch.append(
                    run_args.model_copy(update={"claim": claim, "dispatch": "pull"})
                )
        if released:
            model.repository.release_claims(released)  # type: ignore[attr-defined]

        if batch:
            _run_batch(batch, n_cores=n_cores)

        # If every claimed sim was skipped (all released, nothing run) and we
        # are not in follow mode, treat the queue as effectively drained and
        # exit.  Without this guard, a sim that permanently returns None from
        # _build_run_args (e.g. already complete per skip_existing) would be
        # claimed → released → claimed forever.
        if not batch and not follow:
            return


def push_worker_loop(
    model: object,
    chunk_path: str | Path,
    n_cores: int = 1,
) -> None:
    """Run one immutable chunk shard without queue claims or heartbeats."""
    batch: list[SimulationTask] = []
    for sim_id in _read_chunk(chunk_path):
        run_args = model._build_run_args(sim_id)  # type: ignore[attr-defined]
        if run_args is None:
            continue
        batch.append(run_args.model_copy(update={"claim": None, "dispatch": "push"}))
    if batch:
        _run_batch(batch, n_cores=n_cores)
