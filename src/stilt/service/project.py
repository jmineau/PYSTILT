"""Thin service-oriented wrappers over the STILT queue/runtime API."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from stilt.executors.factory import resolve_dispatch
from stilt.model import Model
from stilt.receptor import Receptor
from stilt.repositories import (
    QueueRepository,
    SimulationAttempt,
    SimulationClaim,
)
from stilt.workers import pull_worker_loop

if TYPE_CHECKING:
    from stilt.executors import Executor, JobHandle


@dataclass(frozen=True, slots=True)
class QueueStatus:
    """High-level simulation queue summary for one STILT project.

    These counts are project-level simulation counts, not per-footprint counts.
    ``completed`` means the simulation has reached terminal success for the
    full requested artifact contract tracked by the repository.
    """

    project: str
    total: int
    completed: int
    running: int
    pending: int
    failed: int


@dataclass(frozen=True, slots=True)
class BatchStatus:
    """Progress summary for one submitted simulation batch."""

    batch_id: str
    completed: int
    total: int

    @property
    def is_complete(self) -> bool:
        return self.total > 0 and self.completed == self.total

    @property
    def percent_complete(self) -> float:
        if self.total <= 0:
            return 0.0
        return 100.0 * self.completed / self.total


def summarize_queue(
    model: Model,
    *,
    now: dt.datetime | None = None,
) -> QueueStatus:
    """Return the project-level simulation queue summary for *model*."""
    if _should_refresh_durable_state(model):
        model.repository.rebuild()
    repo = model.repository
    current = now or dt.datetime.now(dt.timezone.utc)
    total = repo.count()
    completed = len(repo.completed_simulations())
    pending = len(repo.pending_trajectories())
    running = sum(1 for claim in repo.list_claims() if claim.expires_at >= current)
    failed = max(total - completed - pending - running, 0)
    return QueueStatus(
        project=model.project,
        total=total,
        completed=completed,
        running=running,
        pending=pending,
        failed=failed,
    )


class Service:
    """Named service-facing facade over :class:`stilt.model.Model`.

    ``Service`` does not define a second orchestration system. It wraps the
    same repository-backed submission, worker, and status APIs already used by
    :class:`~stilt.model.Model` and the CLI, but gives always-on and cloud
    workflows a smaller queue-oriented surface.
    """

    def __init__(
        self,
        project: str | Path | None = None,
        *,
        output_dir: str | Path | None = None,
        compute_root: str | Path | None = None,
        model: Model | None = None,
        **model_kwargs,
    ) -> None:
        if model is not None and (
            project is not None
            or output_dir is not None
            or compute_root is not None
            or model_kwargs
        ):
            raise ValueError(
                "Pass either an existing model or model construction args, not both."
            )
        self.model = model or Model(
            project=project,
            output_dir=output_dir,
            compute_root=compute_root,
            **model_kwargs,
        )

    @property
    def repository(self) -> QueueRepository:
        """Underlying repository used for queue/service state."""
        return self.model.repository

    def submit(
        self,
        receptors: list[Receptor] | None = None,
        batch_id: str | None = None,
    ) -> list[str]:
        """Register simulations without starting any workers.

        When ``receptors`` is omitted, the wrapped model uses its configured or
        on-disk receptor set.
        """
        return self.model.submit(receptors=receptors, batch_id=batch_id)

    def run(
        self,
        executor: Executor | None = None,
        *,
        skip_existing: bool | None = None,
        wait: bool = True,
        batch_id: str | None = None,
    ) -> JobHandle:
        """Register work and launch the configured executor."""
        return self.model.run(
            executor=executor,
            skip_existing=skip_existing,
            wait=wait,
            batch_id=batch_id,
        )

    def drain(
        self,
        cpus: int = 1,
        *,
        follow: bool = False,
        poll_interval: float = 10.0,
        lease_ttl: float = 1800.0,
    ) -> None:
        """Drain pending simulations from the project's repository."""
        pull_worker_loop(
            self.model,
            n_cores=cpus,
            follow=follow,
            poll_interval=poll_interval,
            lease_ttl=lease_ttl,
        )

    def serve(
        self,
        cpus: int = 1,
        *,
        poll_interval: float = 10.0,
        lease_ttl: float = 1800.0,
    ) -> None:
        """Run long-lived queue workers that keep polling for new work."""
        self.drain(
            cpus=cpus,
            follow=True,
            poll_interval=poll_interval,
            lease_ttl=lease_ttl,
        )

    def status(self) -> QueueStatus:
        """Return the current project-level simulation queue summary."""
        return summarize_queue(self.model)

    def batch_status(self, batch_id: str) -> BatchStatus:
        """Return the progress summary for one submitted batch."""
        if _should_refresh_durable_state(self.model):
            self.repository.rebuild()
        completed, total = self.repository.batch_progress(batch_id)
        return BatchStatus(batch_id=batch_id, completed=completed, total=total)

    def batches(self) -> list[BatchStatus]:
        """Return all known batches ordered by repository policy."""
        if _should_refresh_durable_state(self.model):
            self.repository.rebuild()
        return [
            BatchStatus(batch_id=batch_id, completed=completed, total=total)
            for batch_id, completed, total in self.repository.all_batches()
        ]

    def active_claims(
        self,
        *,
        include_expired: bool = False,
        now: dt.datetime | None = None,
    ) -> list[SimulationClaim]:
        """Return active claim rows, optionally including expired leases."""
        claims = self.repository.list_claims()
        if include_expired:
            return claims
        current = now or dt.datetime.now(dt.timezone.utc)
        return [claim for claim in claims if claim.expires_at >= current]

    def attempts(self, sim_id: str | None = None) -> list[SimulationAttempt]:
        """Return recorded execution attempts, optionally filtered by sim_id."""
        return self.repository.list_attempts(sim_id)


def _should_refresh_durable_state(model: Model) -> bool:
    """Return True when status should rebuild durable state before reading it."""
    try:
        return resolve_dispatch(model.config.execution or {}) == "push"
    except FileNotFoundError:
        return True
