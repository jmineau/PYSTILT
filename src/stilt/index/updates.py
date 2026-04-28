"""Shared transition helpers for output index backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .protocol import COMPLETE_FOOTPRINT_STATUSES, OutputSummary

if TYPE_CHECKING:
    from stilt.execution import SimulationResult


@dataclass(frozen=True, slots=True)
class IndexUpdate:
    """One fully resolved row update for a output simulation index."""

    trajectory_status: str
    error: str | None
    summary: OutputSummary


def output_summary_from_result(result: SimulationResult) -> OutputSummary:
    """Project one worker result onto output-presence flags."""
    return OutputSummary(
        traj_present=result.traj_present,
        error_traj_present=result.error_traj_path is not None,
        log_present=result.log_path is not None,
        footprints=dict(result.footprint_statuses),
    )


def index_update_from_result(result: SimulationResult) -> IndexUpdate:
    """Resolve trajectory status, error text, and output summary for one result."""
    summary = output_summary_from_result(result)
    if result.status == "interrupted":
        return IndexUpdate(
            trajectory_status="pending",
            error=result.error,
            summary=summary,
        )
    if any(status == "failed" for status in summary.footprints.values()):
        return IndexUpdate(
            trajectory_status="failed",
            error=result.error or "footprint failed",
            summary=summary,
        )
    if result.status in COMPLETE_FOOTPRINT_STATUSES:
        return IndexUpdate(
            trajectory_status="complete" if summary.traj_present else "pending",
            error=None,
            summary=summary,
        )
    return IndexUpdate(
        trajectory_status="failed",
        error=result.error or "simulation failed",
        summary=summary,
    )


def index_update_from_summary(summary: OutputSummary) -> IndexUpdate:
    """Resolve trajectory status for one output scan summary."""
    if summary.traj_present:
        return IndexUpdate(
            trajectory_status="complete",
            error=None,
            summary=summary,
        )
    if summary.error_traj_present or summary.log_present:
        return IndexUpdate(
            trajectory_status="failed",
            error=None,
            summary=summary,
        )
    return IndexUpdate(
        trajectory_status="pending",
        error=None,
        summary=summary,
    )
