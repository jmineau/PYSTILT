"""Shared simulation/output selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from stilt.errors import ConfigValidationError
from stilt.receptors import Receptor
from stilt.simulation import SimID
from stilt.storage import Storage

if TYPE_CHECKING:
    from stilt.manifest import Manifest


def _normalized_time_range(
    time_range: tuple | None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if time_range is None:
        return None
    return cast(pd.Timestamp, pd.Timestamp(time_range[0])), cast(
        pd.Timestamp, pd.Timestamp(time_range[1])
    )


def _sim_id_matches(
    sim_id: str,
    *,
    mets: set[str],
    time_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    location_ids: set[str] | None = None,
) -> bool:
    sid = SimID(sim_id)
    if sid.met not in mets:
        return False
    if time_range is not None:
        start, end = time_range
        timestamp = pd.Timestamp(sid.time)
        if timestamp < start or timestamp > end:
            return False
    return location_ids is None or sid.location in location_ids


def filter_ids(
    sim_ids: list[str],
    *,
    mets: set[str],
    time_range: tuple | None = None,
    location_ids: set[str] | None = None,
) -> list[str]:
    """Return registered simulation identifiers that match query filters."""
    if not sim_ids:
        return []
    query_time = _normalized_time_range(time_range)
    return [
        sim_id
        for sim_id in sim_ids
        if _sim_id_matches(
            sim_id,
            mets=mets,
            time_range=query_time,
            location_ids=location_ids,
        )
    ]


def resolve_mets(
    available_mets: Iterable[str],
    mets: str | list[str] | None,
) -> set[str]:
    """Resolve a met filter against an available-name set."""
    available = set(available_mets)
    if mets is None:
        return available
    requested = {mets} if isinstance(mets, str) else set(mets)
    missing = sorted(requested - available)
    if missing:
        raise ConfigValidationError(f"Unknown met name(s): {missing}")
    return requested


def _receptor_matches(
    receptor: Receptor,
    *,
    time_range: tuple | None = None,
    location_ids: set[str] | None = None,
) -> bool:
    query_time = _normalized_time_range(time_range)
    if query_time is not None:
        start, end = query_time
        timestamp = pd.Timestamp(receptor.time)
        if timestamp < start or timestamp > end:
            return False
    return location_ids is None or receptor.id.location in location_ids


def _candidate_ids(
    receptors: Iterable[Receptor],
    *,
    mets: Iterable[str],
    time_range: tuple | None = None,
    location_ids: set[str] | None = None,
) -> list[str]:
    return [
        str(SimID.from_parts(met_name, receptor))
        for met_name in mets
        for receptor in receptors
        if _receptor_matches(
            receptor,
            time_range=time_range,
            location_ids=location_ids,
        )
    ]


def matching_ids(
    registry: Manifest,
    *,
    receptors: Iterable[Receptor],
    configured_mets: Iterable[str] | None,
    registered: bool,
    mets: str | list[str] | None = None,
    time_range: tuple | None = None,
    location_ids: set[str] | None = None,
) -> list[str]:
    """Return registered (manifest) or candidate (receptors×mets) simulation IDs."""
    if registered:
        sim_ids = registry.sim_ids()
        available_mets = (
            set(configured_mets)
            if configured_mets is not None
            else {SimID(sim_id).met for sim_id in sim_ids}
        )
        return filter_ids(
            sim_ids,
            mets=resolve_mets(available_mets, mets),
            time_range=time_range,
            location_ids=location_ids,
        )
    return _candidate_ids(
        receptors,
        mets=resolve_mets(configured_mets or [], mets),
        time_range=time_range,
        location_ids=location_ids,
    )


def output_paths(
    storage: Storage,
    sim_ids: list[str],
    *,
    local_path: Callable[[str], Path],
) -> list[Path]:
    """Resolve, by key, the local-accessible output paths that exist."""
    resolved_paths: list[Path] = []
    for sim_id in sim_ids:
        resolved = storage.resolve(sim_id, local_path(sim_id))
        if resolved is not None:
            resolved_paths.append(resolved)
    return resolved_paths


def missing_ids(
    sim_ids: list[str],
    *,
    present: Callable[[str], bool],
) -> list[str]:
    """Return simulation IDs whose output is not present (checked by key)."""
    return [sim_id for sim_id in sim_ids if not present(sim_id)]


__all__ = [
    "filter_ids",
    "matching_ids",
    "missing_ids",
    "output_paths",
    "resolve_mets",
]
