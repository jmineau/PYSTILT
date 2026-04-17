"""Repository protocol and shared helpers."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from stilt.receptor import Receptor
from stilt.simulation import SimID

COMPLETE_FOOTPRINT_STATUSES = frozenset({"complete", "complete-empty"})


def _normalize_footprint_names(
    footprint_names: list[str] | None = None,
) -> list[str]:
    """Return sorted unique footprint names for durable repository storage."""
    return sorted(set(footprint_names or []))


@dataclass(frozen=True, slots=True)
class ArtifactSummary:
    """Lightweight durable artifact presence summary for one simulation."""

    traj_present: bool = False
    error_traj_present: bool = False
    log_present: bool = False
    footprints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SimulationClaim:
    """Ephemeral claim/lease record for one in-flight simulation."""

    sim_id: str
    claim_token: str
    worker_id: str
    claimed_at: dt.datetime
    heartbeat_at: dt.datetime
    expires_at: dt.datetime


@dataclass(frozen=True, slots=True)
class SimulationAttempt:
    """Append-only execution attempt history for one simulation."""

    attempt_id: str
    sim_id: str
    claim_token: str | None
    started_at: dt.datetime
    finished_at: dt.datetime | None
    outcome: str
    terminal: bool = False
    error: str | None = None


def latest_attempt_is_terminal(
    attempt: SimulationAttempt | None,
) -> bool:
    """Return True when the latest recorded attempt is terminal-failed."""
    return attempt is not None and attempt.terminal


def simulation_complete(
    summary: ArtifactSummary,
    requested_footprints: list[str],
) -> bool:
    """Return True when all requested durable outputs are present."""
    return summary.traj_present and all(
        summary.footprints.get(name) in COMPLETE_FOOTPRINT_STATUSES
        for name in requested_footprints
    )


def simulation_pending(
    summary: ArtifactSummary,
    requested_footprints: list[str],
    *,
    active_claim: bool,
    latest_attempt: SimulationAttempt | None,
) -> bool:
    """Return True when a simulation still needs work and is claimable."""
    return (
        not simulation_complete(summary, requested_footprints)
        and not active_claim
        and not latest_attempt_is_terminal(latest_attempt)
    )


def trajectory_status_from_state(
    summary: ArtifactSummary,
    *,
    active_claim: bool,
    latest_attempt: SimulationAttempt | None,
) -> str:
    """Return compatibility trajectory status derived from status-light state."""
    if active_claim:
        return "running"
    if summary.traj_present:
        return "complete"
    if latest_attempt_is_terminal(latest_attempt):
        return "failed"
    return "pending"


@runtime_checkable
class SimulationCatalog(Protocol):
    """Simulation identity and receptor catalog operations."""

    def register_many(
        self,
        pairs: list[tuple[str, Receptor]],
        batch_id: str | None = None,
        footprint_names: list[str] | None = None,
    ) -> None:
        """Register one or more simulations and their receptor geometry."""
        ...

    def all_sim_ids(self) -> list[str]:
        """Return all registered simulation identifiers."""
        ...

    def has(self, sim_id: SimID | str) -> bool:
        """Return True when a simulation id is registered."""
        ...

    def count(self) -> int:
        """Return the number of registered simulations."""
        ...

    def get_receptor(self, sim_id: SimID | str) -> Receptor:
        """Load the stored receptor geometry for one simulation."""
        ...


@runtime_checkable
class ArtifactStateStore(Protocol):
    """Durable artifact state mutation operations."""

    def sync(self) -> None:
        """Flush deferred state updates when a backend batches writes."""
        ...

    def mark_trajectory_complete(self, sim_id: str) -> None:
        """Record that a trajectory artifact completed successfully."""
        ...

    def mark_trajectory_failed(self, sim_id: str, error: str = "") -> None:
        """Record a terminal trajectory failure for one simulation."""
        ...

    def mark_footprint_complete(self, sim_id: str, name: str) -> None:
        """Record that one named footprint completed successfully."""
        ...

    def mark_footprint_empty(self, sim_id: str, name: str) -> None:
        """Record that one named footprint completed but was empty."""
        ...

    def mark_footprint_failed(self, sim_id: str, name: str, error: str = "") -> None:
        """Record a terminal failure for one named footprint."""
        ...

    def reset_runtime_state(self) -> None:
        """Clear transient running/claim state without rescanning durable outputs."""
        ...

    def rebuild(self) -> None:
        """Fully reconstruct repository state from durable simulation artifacts."""
        ...

    def reset_to_pending(self, sim_ids: list[str]) -> None:
        """Reset one or more simulations to pending trajectory state."""
        ...

    def clear_footprints(
        self, sim_ids: list[str], names: list[str] | None = None
    ) -> None:
        """Clear all or selected footprint statuses for the given simulations."""
        ...

    def record_artifacts(self, sim_id: str, summary: ArtifactSummary) -> None:
        """Persist a durable artifact summary for one simulation."""
        ...

    def record_artifacts_many(self, pairs: list[tuple[str, ArtifactSummary]]) -> None:
        """Persist artifact summaries for many simulations in one transaction."""
        ...


@runtime_checkable
class ArtifactStatusQuery(Protocol):
    """Artifact status and summary query operations."""

    def completed_trajectories(self) -> list[str]:
        """Return simulation ids whose trajectory artifacts are complete."""
        ...

    def completed_simulations(self) -> list[str]:
        """Return simulation ids whose requested outputs are fully complete."""
        ...

    def pending_trajectories(self) -> list[str]:
        """Return simulation ids still eligible for trajectory work."""
        ...

    def traj_status(self, sim_id: SimID | str) -> str | None:
        """Return the compatibility trajectory status for one simulation."""
        ...

    def bulk_traj_status(
        self,
        sim_ids: list[str] | None = None,
    ) -> dict[str, str | None]:
        """Return compatibility trajectory statuses for many simulations."""
        ...

    def footprint_status(self, sim_id: SimID | str, name: str) -> str | None:
        """Return the durable status for one named footprint."""
        ...

    def bulk_footprint_status(
        self,
        name: str,
        sim_ids: list[str] | None = None,
    ) -> dict[str, str | None]:
        """Return durable status for one named footprint across many simulations."""
        ...

    def footprint_completed(self, sim_id: SimID | str, name: str) -> bool:
        """Return True when one named footprint reached a complete state."""
        ...

    def bulk_footprint_completed(self, names: list[str]) -> set[str]:
        """Return sim_ids where every listed footprint name is complete."""
        ...

    def artifact_summary(self, sim_id: SimID | str) -> ArtifactSummary:
        """Return the durable artifact summary for one simulation."""
        ...

    def to_dataframe(self) -> pd.DataFrame:
        """Return the repository state as a tabular dataframe."""
        ...


@runtime_checkable
class BatchStore(Protocol):
    """Submitted batch progress query operations."""

    def batch_progress(self, batch_id: str) -> tuple[int, int]:
        """Return completed-vs-total counts for one submitted batch."""
        ...

    def all_batches(self) -> list[tuple[str, int, int]]:
        """Return summary progress for every known batch."""
        ...


@runtime_checkable
class QueueStore(Protocol):
    """Live pull-worker claim and attempt coordination operations."""

    def claim_pending_claims(
        self,
        n: int = 1,
        worker_id: str = "legacy",
        lease_ttl: float = 1800.0,
    ) -> list[SimulationClaim]:
        """Claim up to ``n`` pending simulations and return the lease records."""
        ...

    def claim_pending(
        self,
        n: int = 1,
        worker_id: str = "legacy",
        lease_ttl: float = 1800.0,
    ) -> list[str]:
        """Claim up to ``n`` pending simulations and return only their ids."""
        ...

    def release_claim(self, sim_ids: list[str]) -> None:
        """Release one or more claimed simulations back to pending state."""
        ...

    def release_claims(self, claims: list[SimulationClaim]) -> None:
        """Release one or more explicit claim records."""
        ...

    def heartbeat_claim(
        self,
        sim_id: str,
        claim_token: str,
        lease_ttl: float = 1800.0,
    ) -> bool:
        """Refresh one claim lease and return True when it is still current."""
        ...

    def claim_is_current(self, sim_id: str, claim_token: str) -> bool:
        """Return True when the given claim token still owns the simulation."""
        ...

    def reclaim_expired_claims(self) -> list[str]:
        """Return expired claims to pending state and list their sim ids."""
        ...

    def list_claims(self) -> list[SimulationClaim]:
        """Return all currently stored claim/lease records."""
        ...

    def upsert_claim(self, claim: SimulationClaim) -> None:
        """Insert or replace one claim record."""
        ...

    def delete_claim(self, sim_id: str, claim_token: str | None = None) -> None:
        """Delete a claim by simulation id, optionally guarding by claim token."""
        ...

    def list_attempts(
        self, sim_id: SimID | str | None = None
    ) -> list[SimulationAttempt]:
        """Return attempt history, optionally filtered to one simulation."""
        ...

    def record_attempt(self, attempt: SimulationAttempt) -> None:
        """Append or upsert one execution attempt record."""
        ...


@runtime_checkable
class StateRepository(
    SimulationCatalog,
    ArtifactStateStore,
    ArtifactStatusQuery,
    BatchStore,
    Protocol,
):
    """Composite durable repository interface used by existing callers."""


@runtime_checkable
class QueueRepository(StateRepository, QueueStore, Protocol):
    """Composite durable repository plus live queue coordination."""
