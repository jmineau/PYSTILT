"""Shared simulation/output selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

import pandas as pd

from stilt.errors import ConfigValidationError
from stilt.index import OutputSummary, SimulationIndex
from stilt.receptor import Receptor
from stilt.simulation import SimID
from stilt.storage import Storage


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
    index: SimulationIndex,
    *,
    receptors: Iterable[Receptor],
    configured_mets: Iterable[str] | None,
    registered: bool,
    mets: str | list[str] | None = None,
    time_range: tuple | None = None,
    location_ids: set[str] | None = None,
) -> list[str]:
    """Return registered or candidate simulation IDs that match query filters."""
    if registered:
        sim_ids = index.sim_ids()
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
    index: SimulationIndex,
    sim_ids: list[str],
    *,
    present: Callable[[OutputSummary], bool],
    local_path: Callable[[str], Path],
) -> list[Path]:
    """Resolve matching durable output paths for one simulation-id query."""
    found = index.summaries(sim_ids)
    resolved_paths: list[Path] = []
    for sim_id in sim_ids:
        if not present(found.get(sim_id, OutputSummary())):
            continue
        resolved = storage.resolve(sim_id, local_path(sim_id))
        if resolved is not None:
            resolved_paths.append(resolved)
    return resolved_paths


def missing_ids(
    index: SimulationIndex,
    sim_ids: list[str],
    *,
    present: Callable[[OutputSummary], bool],
) -> list[str]:
    """Return simulation IDs whose output summaries do not satisfy ``present``."""
    found = index.summaries(sim_ids)
    return [
        sim_id for sim_id in sim_ids if not present(found.get(sim_id, OutputSummary()))
    ]


__all__ = [
    "filter_ids",
    "matching_ids",
    "missing_ids",
    "output_paths",
    "resolve_mets",
]
