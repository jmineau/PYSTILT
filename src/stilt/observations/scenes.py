"""Helpers for grouping observations into scenes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .observation import Observation


@dataclass(slots=True)
class Scene:
    """A logical grouping of related observations, typically sensor-defined."""

    id: str
    sensor: str
    observations: list[Observation]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_id(self) -> str:
        """Default batch grouping to use when submitting this scene to Model."""
        return self.id

    @property
    def observation_ids(self) -> list[str | None]:
        """Observation identifiers in the same order as ``observations``."""
        return [obs.observation_id for obs in self.observations]


def make_scene(
    observations: Iterable[Observation],
    *,
    scene_id: str | None = None,
    sensor: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Scene:
    """Build one scene from an ordered collection of observations."""
    items = sorted(list(observations), key=lambda obs: obs.time)
    if not items:
        raise ValueError("At least one observation is required to build a scene.")
    resolved_sensor = sensor or items[0].sensor
    resolved_id = scene_id or f"{resolved_sensor}-{items[0].time:%Y%m%d%H%M%S}"
    return Scene(
        id=resolved_id,
        sensor=resolved_sensor,
        observations=items,
        metadata=dict(metadata or {}),
    )


def group_scenes_by_key(
    observations: Iterable[Observation],
    *,
    key: Callable[[Observation], object],
    scene_prefix: str | None = None,
    metadata_factory: Callable[[object], dict[str, Any]] | None = None,
) -> list[Scene]:
    """Group observations by an arbitrary key function."""
    grouped: dict[object, list[Observation]] = defaultdict(list)
    for observation in observations:
        grouped[key(observation)].append(observation)

    scenes: list[Scene] = []
    for group_key, members in sorted(
        grouped.items(), key=lambda item: min(obs.time for obs in item[1])
    ):
        first = min(members, key=lambda obs: obs.time)
        prefix = scene_prefix or first.sensor
        group_str = str(group_key)
        metadata = (
            metadata_factory(group_key)
            if metadata_factory is not None
            else {"group_key": group_str}
        )
        scenes.append(
            make_scene(
                members,
                scene_id=f"{prefix}-{group_str}",
                sensor=first.sensor,
                metadata=metadata,
            )
        )
    return scenes


def group_scenes_by_time_gap(
    observations: Iterable[Observation],
    *,
    max_gap: str | pd.Timedelta,
    scene_prefix: str | None = None,
) -> list[Scene]:
    """Group temporally adjacent observations into scenes."""
    items = sorted(list(observations), key=lambda obs: obs.time)
    if not items:
        return []

    max_gap = pd.to_timedelta(max_gap)
    groups: list[list[Observation]] = [[items[0]]]
    for observation in items[1:]:
        previous = groups[-1][-1]
        if (pd.Timestamp(observation.time) - pd.Timestamp(previous.time)) <= max_gap:
            groups[-1].append(observation)
        else:
            groups.append([observation])

    scenes: list[Scene] = []
    for idx, group in enumerate(groups, start=1):
        prefix = scene_prefix or group[0].sensor
        scenes.append(
            make_scene(
                group,
                scene_id=f"{prefix}-{group[0].time:%Y%m%d%H%M%S}-g{idx}",
                metadata={"grouping": "time_gap", "max_gap": str(max_gap)},
            )
        )
    return scenes


def group_scenes_by_swath(
    observations: Iterable[Observation],
    *,
    scene_prefix: str | None = None,
) -> list[Scene]:
    """Group observations by ``HorizontalGeometry.swath``."""

    def _swath_key(observation: Observation) -> object:
        geometry = observation.geometry
        if geometry is None or geometry.swath is None:
            raise ValueError(
                "All observations must define geometry.swath for swath grouping."
            )
        return geometry.swath

    return group_scenes_by_key(
        observations,
        key=_swath_key,
        scene_prefix=scene_prefix,
        metadata_factory=lambda swath: {"grouping": "swath", "swath": swath},
    )


def group_scenes_by_metadata(
    observations: Iterable[Observation],
    *,
    key: str,
    scene_prefix: str | None = None,
) -> list[Scene]:
    """Group observations by a metadata field."""

    def _metadata_key(observation: Observation) -> object:
        if key not in observation.metadata:
            raise ValueError(
                f"All observations must define metadata[{key!r}] for scene grouping."
            )
        return observation.metadata[key]

    return group_scenes_by_key(
        observations,
        key=_metadata_key,
        scene_prefix=scene_prefix,
        metadata_factory=lambda value: {
            "grouping": "metadata",
            "key": key,
            "value": value,
        },
    )
