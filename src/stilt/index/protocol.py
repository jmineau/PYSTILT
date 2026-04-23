"""Durable simulation index types for PYSTILT.

The index layer tracks one row per simulation and sits between:

- queue registration APIs such as ``Model.register_pending()``,
- worker result recording,
- read paths such as ``model.status()`` and collection-level queries.

`OutputSummary` is the light durable presence summary for one simulation.
`IndexCounts` is the cheap aggregate view over a whole index or one scene.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from stilt.receptor import Receptor
from stilt.simulation import SimID

if TYPE_CHECKING:
    from stilt.execution import SimulationResult

COMPLETE_FOOTPRINT_STATUSES = frozenset({"complete", "complete-empty"})


def _normalize_footprint_names(
    footprint_names: list[str] | None = None,
) -> list[str]:
    """Return sorted unique footprint names for durable index storage."""
    return sorted(set(footprint_names or []))


def _normalize_registration_pairs(
    pairs: str | tuple[str, Receptor] | list[tuple[str, Receptor]],
    receptor: Receptor | None = None,
) -> list[tuple[str, Receptor]]:
    """Normalize one-or-many register inputs to a list of pairs."""
    if isinstance(pairs, str):
        if receptor is None:
            raise TypeError("register() requires a receptor when called with sim_id")
        return [(pairs, receptor)]
    if isinstance(pairs, tuple):
        if receptor is not None:
            raise TypeError(
                "register() accepts either (sim_id, receptor) or a list of pairs"
            )
        sim_id, receptor = pairs
        return [(sim_id, receptor)]
    if receptor is not None:
        raise TypeError(
            "register() accepts either sim_id plus receptor or a list of pairs"
        )
    return list(pairs)


@dataclass(frozen=True, slots=True)
class OutputSummary:
    """Lightweight durable output presence summary for one simulation."""

    traj_present: bool = False
    error_traj_present: bool = False
    log_present: bool = False
    footprints: dict[str, str] = field(default_factory=dict)

    def footprint_complete(self, name: str) -> bool:
        """Return whether one named footprint has reached a terminal state."""
        return self.footprints.get(name) in COMPLETE_FOOTPRINT_STATUSES

    def footprints_complete(self, names: Iterable[str]) -> bool:
        """Return whether all requested footprints have reached terminal states."""
        return all(self.footprint_complete(name) for name in names)

    def outputs_complete(self, footprint_names: Iterable[str]) -> bool:
        """Return whether trajectory and all requested footprints are complete."""
        return self.traj_present and self.footprints_complete(footprint_names)

    def pending_footprints(self, names: Iterable[str]) -> list[str]:
        """Return configured footprint names that still need work."""
        return [name for name in names if not self.footprint_complete(name)]

    def needs_work(
        self,
        footprint_names: Iterable[str],
        *,
        skip_existing: bool,
    ) -> bool:
        """Return whether this simulation still requires worker execution."""
        if not skip_existing:
            return True
        targets = list(footprint_names)
        if targets:
            return not self.outputs_complete(targets)
        return not self.traj_present


@dataclass(frozen=True, slots=True)
class IndexCounts:
    """Cheap aggregate counts for one durable simulation index view."""

    total: int = 0
    completed: int = 0
    running: int = 0
    pending: int = 0
    failed: int = 0


@runtime_checkable
class SimulationIndex(Protocol):
    """Durable simulation registry surface for model, CLI, and workers."""

    def record(self, result: SimulationResult) -> None: ...

    def register(
        self,
        pairs: str | tuple[str, Receptor] | list[tuple[str, Receptor]],
        receptor: Receptor | None = None,
        footprint_names: list[str] | None = None,
        scene_id: str | None = None,
    ) -> None: ...

    def sim_ids(self) -> list[str]: ...

    def has(self, sim_id: SimID | str) -> bool: ...

    def count(self) -> int: ...

    def counts(self, scene_id: str | None = None) -> IndexCounts: ...

    def scene_counts(self) -> dict[str, IndexCounts]: ...

    def receptors_for(self, sim_ids: list[str]) -> dict[str, Receptor]: ...

    def reset_to_pending(
        self,
        sim_ids: list[str],
        *,
        clear_outputs: bool = False,
    ) -> None: ...

    def pending_trajectories(self) -> list[str]: ...

    def summaries(
        self,
        sim_ids: list[str] | None = None,
    ) -> dict[str, OutputSummary]: ...

    def rebuild(self) -> None:
        """Rebuild durable index rows by rescanning durable outputs."""
        ...


__all__ = [
    "OutputSummary",
    "COMPLETE_FOOTPRINT_STATUSES",
    "IndexCounts",
    "SimulationIndex",
]
