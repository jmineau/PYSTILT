"""
The single definition of simulation completion.

A STILT simulation is *complete* when every output it is configured to produce is
present in the store. "What it must produce" is :func:`expected_artifacts`; "what
is present" is read **by key** from the store — never by listing. Every execution
path (local, HPC, queue) shares this one definition, so completion can never
disagree between paths.

The error *trajectory* is required only when the wind-error params are set
(``error_enabled``); error *footprints* are never required.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from stilt.storage.files import ProjectFiles

if TYPE_CHECKING:
    from stilt.config import ModelConfig
    from stilt.storage import Storage

TRAJECTORY = "trajectory"
ERROR_TRAJECTORY = "error_trajectory"
_FOOTPRINT_PREFIX = "footprint:"


@dataclass(frozen=True, slots=True)
class StatusCounts:
    """Aggregate completion counts for one project or scene."""

    total: int = 0
    completed: int = 0
    running: int = 0
    pending: int = 0
    failed: int = 0


def footprint_artifact(name: str) -> str:
    """Return the artifact id for one named footprint."""
    return f"{_FOOTPRINT_PREFIX}{name}"


def expected_artifacts(
    footprint_names: Iterable[str],
    *,
    error_enabled: bool,
) -> frozenset[str]:
    """
    Return the artifacts a complete simulation must produce.

    Always the main trajectory plus one entry per configured footprint; the error
    trajectory is included only when ``error_enabled`` is True.
    """
    artifacts = {TRAJECTORY}
    if error_enabled:
        artifacts.add(ERROR_TRAJECTORY)
    artifacts.update(footprint_artifact(name) for name in footprint_names)
    return frozenset(artifacts)


def expected_for_config(config: ModelConfig) -> frozenset[str]:
    """Return the expected artifacts for one model config."""
    return expected_artifacts(config.footprints, error_enabled=config.error_enabled)


def _artifact_present(
    artifact: str,
    sim_id: str,
    files,
    storage: Storage,
) -> bool:
    """Return whether one artifact id already exists in the store, by key."""
    if artifact == TRAJECTORY:
        return storage.exists(sim_id, files.trajectory_path)
    if artifact == ERROR_TRAJECTORY:
        return storage.exists(sim_id, files.error_trajectory_path)
    if artifact.startswith(_FOOTPRINT_PREFIX):
        name = artifact[len(_FOOTPRINT_PREFIX) :]
        # An empty-footprint marker is a legitimate terminal outcome.
        return storage.exists(sim_id, files.footprint_path(name)) or storage.exists(
            sim_id, files.empty_footprint_path(name)
        )
    return False


def _check_order(artifact: str) -> int:
    """Order checks so the cheapest discriminator (the trajectory) runs first."""
    return {TRAJECTORY: 0, ERROR_TRAJECTORY: 1}.get(artifact, 2)


def present_artifacts(
    sim_id: str,
    expected: Collection[str],
    storage: Storage,
) -> set[str]:
    """
    Return which *expected* artifacts already exist in the store, by key.

    Only the requested artifacts are probed, so this costs one existence check
    per expected output (two for footprints, which may be empty markers) — never
    a directory listing.
    """
    files = ProjectFiles(storage.output_dir).simulation(sim_id)
    return {
        artifact
        for artifact in expected
        if _artifact_present(artifact, sim_id, files, storage)
    }


def is_complete(
    sim_id: str,
    expected: Collection[str],
    storage: Storage,
) -> bool:
    """
    Return whether one simulation has produced all of its *expected* outputs.

    ``expected`` is computed once by the caller (e.g. :func:`expected_for_config`)
    and reused across many simulations. Short-circuits on the first missing
    artifact, checking the trajectory first so the many incomplete simulations
    cost a single existence check.
    """
    files = ProjectFiles(storage.output_dir).simulation(sim_id)
    ordered = sorted(expected, key=_check_order)
    return all(
        _artifact_present(artifact, sim_id, files, storage) for artifact in ordered
    )
