from __future__ import annotations

import datetime as dt
import logging
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from stilt.config import FootprintConfig
from stilt.errors import SimulationError
from stilt.simulation import Simulation

from .tasks import SimulationResult, SimulationTask

logger = logging.getLogger(__name__)
FootprintTerminalStatus = Literal["complete", "complete-empty"]


def _failure_status(error: BaseException) -> Literal["failed", "error"]:
    """Classify one execution error into a stable worker result status."""
    return "failed" if isinstance(error, SimulationError) else "error"


def _finished_result(result: SimulationResult) -> SimulationResult:
    """Ensure one worker result has a finished timestamp."""
    if result.finished_at is not None:
        return result
    result.finished_at = dt.datetime.now(dt.timezone.utc)
    return result


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


def _log_simulation_exception(
    sim: Simulation,
    *,
    phase: str,
    error: BaseException,
) -> None:
    """Emit one simulation failure to stderr and the per-sim log."""
    logger.exception(
        "simulation %s failed during %s: %s",
        sim.id,
        phase,
        error,
    )
    _append_simulation_error_log(sim, phase=phase, error=error)


def _existing_footprint_status(
    sim: Simulation,
    footprint_name: str,
) -> FootprintTerminalStatus | None:
    """Return the durable terminal state already present for one footprint."""
    if sim.resolve_output(sim.footprint_path(footprint_name)) is not None:
        return "complete"
    if sim.resolve_output(sim.files.empty_footprint_path(footprint_name)) is not None:
        return "complete-empty"
    return None


def _trajectory_present(sim: Simulation) -> bool:
    """Return whether a durable trajectory output already exists."""
    return sim.resolve_output(sim.trajectories_path) is not None


def _result_log_path(sim: Simulation) -> Path | None:
    """Return the simulation log path when the log exists locally."""
    return sim.log_path if sim.log_path.exists() else None


def _result_error_traj_path(sim: Simulation) -> Path | None:
    """Return the error-trajectory path when present locally."""
    return sim.error_trajectories_path if sim.error_trajectories_path.exists() else None


def _update_written_traj(
    sim: Simulation,
    *,
    traj_existed: bool,
    wrote_traj: bool,
) -> bool:
    """Return whether this phase newly wrote the trajectory parquet."""
    if not wrote_traj and _trajectory_present(sim):
        return not traj_existed and sim.trajectories_path.exists()
    return wrote_traj


def _record_footprint_status(
    footprint_name: str,
    status: FootprintTerminalStatus,
    *,
    empty_footprints: list[str],
    footprint_statuses: dict[str, str],
) -> None:
    """Record one footprint status on the outgoing worker result."""
    footprint_statuses[footprint_name] = status
    if status == "complete-empty":
        empty_footprints.append(footprint_name)


def _footprint_targets(
    foot_configs: dict[str, FootprintConfig],
) -> Iterator[tuple[str, str, FootprintConfig, bool]]:
    """Yield logical footprint outputs in execution order.

    Returns tuples of ``(base_name, stored_name, config, is_error_output)``.
    """
    for name, foot_config in foot_configs.items():
        yield name, name, foot_config, False
        if foot_config.error:
            yield name, f"{name}_error", foot_config, True


def _footprint_result_status(
    footprint_statuses: dict[str, str],
) -> FootprintTerminalStatus:
    """Collapse per-footprint states into one result status."""
    if any(status == "complete" for status in footprint_statuses.values()):
        return "complete"
    return "complete-empty"


def _materialize_footprint_output(
    sim: Simulation,
    *,
    base_name: str,
    stored_name: str,
    foot_config: FootprintConfig,
    is_error_output: bool,
    skip_existing: bool,
) -> tuple[FootprintTerminalStatus, Path | None]:
    """Ensure one footprint output exists durably and return its terminal state.

    Returns the terminal status plus the output path when this call produced a
    new non-empty NetCDF footprint.
    """
    if skip_existing:
        existing = _existing_footprint_status(sim, stored_name)
        if existing is not None:
            return existing, None

    foot = sim.generate_footprint(
        base_name,
        foot_config,
        write=True,
        error=is_error_output,
    )
    if foot is None:
        sim.files.write_empty_footprint_marker(stored_name)
        return "complete-empty", None
    if foot.is_empty:
        sim.files.write_empty_footprint_marker(stored_name)
        return "complete-empty", None

    sim.files.clear_empty_footprint_marker(stored_name)
    return "complete", sim.footprint_path(stored_name)


def _run_trajectory_phase(
    sim: Simulation,
    task: SimulationTask,
    *,
    started_at: dt.datetime,
) -> SimulationResult:
    """Run the trajectory-only execution path."""
    try:
        sim.run_trajectories(write=True)
        task.storage.publish_simulation(sim)

        return _finished_result(
            SimulationResult(
                sim_id=sim.id,
                status="complete",
                traj_present=True,
                traj_path=sim.trajectories_path,
                error_traj_path=_result_error_traj_path(sim),
                log_path=_result_log_path(sim),
                wrote_traj=True,
                started_at=started_at,
            )
        )
    except Exception as error:
        _log_simulation_exception(sim, phase="trajectory", error=error)
        task.storage.publish_simulation(sim)

        return _finished_result(
            SimulationResult(
                sim_id=sim.id,
                status=_failure_status(error),
                traj_present=_trajectory_present(sim),
                error=str(error),
                log_path=_result_log_path(sim),
                wrote_traj=False,
                started_at=started_at,
                error_traj_path=_result_error_traj_path(sim),
            )
        )


def _run_footprint_phase(
    sim: Simulation,
    task: SimulationTask,
    *,
    started_at: dt.datetime,
) -> SimulationResult:
    """Run trajectory-plus-footprint execution."""
    foot_configs = task.foot_configs
    if foot_configs is None:
        raise ValueError("foot_configs are required for footprint execution.")

    traj_path = sim.trajectories_path
    wrote_traj = False
    foot_paths: list[Path] = []
    empty_footprints: list[str] = []
    footprint_statuses: dict[str, str] = {}
    current_name = next(iter(foot_configs))

    try:
        traj_existed = _trajectory_present(sim)
        for base_name, stored_name, foot_config, is_error_output in _footprint_targets(
            foot_configs
        ):
            current_name = stored_name
            status, output_path = _materialize_footprint_output(
                sim,
                base_name=base_name,
                stored_name=stored_name,
                foot_config=foot_config,
                is_error_output=is_error_output,
                skip_existing=task.skip_existing,
            )
            if output_path is not None:
                foot_paths.append(output_path)
            _record_footprint_status(
                stored_name,
                status,
                empty_footprints=empty_footprints,
                footprint_statuses=footprint_statuses,
            )
            wrote_traj = _update_written_traj(
                sim,
                traj_existed=traj_existed,
                wrote_traj=wrote_traj,
            )

        task.storage.publish_simulation(sim)

        return _finished_result(
            SimulationResult(
                sim_id=sim.id,
                status=_footprint_result_status(footprint_statuses),
                traj_present=_trajectory_present(sim),
                traj_path=traj_path if wrote_traj else None,
                error_traj_path=_result_error_traj_path(sim),
                log_path=_result_log_path(sim),
                wrote_traj=wrote_traj,
                foot_paths=foot_paths,
                empty_footprints=empty_footprints,
                footprint_statuses=footprint_statuses,
                started_at=started_at,
            )
        )
    except Exception as error:
        _log_simulation_exception(
            sim,
            phase=f"footprint:{current_name}",
            error=error,
        )
        task.storage.publish_simulation(sim)
        footprint_statuses[current_name] = "failed"

        return _finished_result(
            SimulationResult(
                sim_id=sim.id,
                status=_failure_status(error),
                traj_present=_trajectory_present(sim),
                error=str(error),
                traj_path=traj_path if _trajectory_present(sim) else None,
                error_traj_path=_result_error_traj_path(sim),
                log_path=_result_log_path(sim),
                wrote_traj=False,
                foot_paths=[],
                footprint_statuses=footprint_statuses,
                started_at=started_at,
            )
        )
