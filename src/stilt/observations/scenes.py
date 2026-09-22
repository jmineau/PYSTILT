"""
Scenes: the observations from one overpass, handled as a unit.

X-STILT works one satellite overpass at a time: select the soundings from
it, build a receptor for each, run them, evaluate them together. A
:class:`Scene` is that group with a name and whatever metadata the overpass
carries. It has no durable state; the simulations it produced are found the
usual way, by receptor time and location.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from stilt.receptors import Receptor

from .observation import Observation


def _time(obs: Observation) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(obs.time))


@dataclass(frozen=True, slots=True)
class Scene:
    """
    A named group of observations, typically one overpass.

    Parameters
    ----------
    id
        Label for the group (for example ``"oco2-202301151830"``).
    observations
        The member observations; stored in time order.
    metadata
        Anything shared by the group: orbit, swath, site, selection settings.
    """

    id: str
    observations: tuple[Observation, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("A Scene needs at least one observation.")
        object.__setattr__(
            self, "observations", tuple(sorted(self.observations, key=_time))
        )

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.observations)

    @property
    def time(self) -> pd.Timestamp:
        """Time of the first observation."""
        return _time(self.observations[0])

    @property
    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last observation times."""
        return self.time, _time(self.observations[-1])

    def receptors(self, build: Callable[[Observation], Receptor]) -> list[Receptor]:
        """
        Build one receptor per observation with *build*.

        *build* is any callable from an observation to a receptor: a built-in
        such as :func:`~stilt.observations.build_slant_receptor`,
        ``functools.partial(build_column_receptor, bottom=0, top=3000)``, or
        your own function.
        """
        return [build(obs) for obs in self.observations]


def _scene_id(prefix: str, members: Sequence[Observation], label: object) -> str:
    return f"{prefix if prefix else members[0].sensor + '-'}{label}"


def group_observations(
    observations: Iterable[Observation],
    key: Callable[[Observation], Hashable],
    *,
    prefix: str = "",
) -> list[Scene]:
    """
    Group observations into scenes by an arbitrary key.

    Scenes are ordered by their first observation time and named
    ``f"{prefix}{key}"``; ``prefix`` defaults to the first member's sensor
    name plus a hyphen. The key is kept in ``scene.metadata["key"]``.
    """
    groups: dict[Hashable, list[Observation]] = defaultdict(list)
    for obs in observations:
        groups[key(obs)].append(obs)
    scenes = [
        Scene(
            id=_scene_id(prefix, members, group_key),
            observations=tuple(members),
            metadata={"key": group_key},
        )
        for group_key, members in groups.items()
    ]
    return sorted(scenes, key=lambda s: s.time)


def group_by_overpass(
    observations: Iterable[Observation],
    *,
    max_gap: str | pd.Timedelta = "30min",
    prefix: str = "",
) -> list[Scene]:
    """
    Split observations into overpasses wherever consecutive times differ by more than *max_gap*.

    This is the X-STILT overpass finder: soundings from one satellite pass
    are seconds apart, passes are hours apart. Scenes are named
    ``f"{prefix}{YYYYMMDDHHMM}"`` from their first observation.
    """
    items = sorted(observations, key=_time)
    if not items:
        return []
    gap = pd.Timedelta(max_gap)
    groups: list[list[Observation]] = [[items[0]]]
    for obs in items[1:]:
        if _time(obs) - _time(groups[-1][-1]) <= gap:
            groups[-1].append(obs)
        else:
            groups.append([obs])
    return [
        Scene(
            id=_scene_id(prefix, members, f"{_time(members[0]):%Y%m%d%H%M}"),
            observations=tuple(members),
            metadata={"max_gap": str(gap)},
        )
        for members in groups
    ]


__all__ = ["Scene", "group_by_overpass", "group_observations"]
