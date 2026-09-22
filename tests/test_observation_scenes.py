"""Tests for Scene and the observation grouping helpers."""

from functools import partial

import pandas as pd
import pytest

from stilt.observations import (
    Observation,
    Scene,
    build_column_receptor,
    build_point_receptor,
    group_by_overpass,
    group_observations,
)
from stilt.receptors import ColumnReceptor, PointReceptor


def _obs(
    time: str,
    *,
    sensor: str = "tropomi",
    observation_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Observation:
    return Observation(
        sensor=sensor,
        species="xco2",
        time=time,
        latitude=40.7,
        longitude=-111.9,
        altitude=30.0,
        observation_id=observation_id,
        metadata=metadata or {},
    )


# -- Scene ------------------------------------------------------------------------


def test_scene_sorts_observations_and_exposes_times():
    late = _obs("2023-01-01 12:10:00", observation_id="b")
    early = _obs("2023-01-01 12:00:00", observation_id="a")

    scene = Scene(id="s", observations=(late, early))

    assert [o.observation_id for o in scene] == ["a", "b"]
    assert len(scene) == 2
    assert scene.time == pd.Timestamp("2023-01-01 12:00:00")
    assert scene.time_range == (
        pd.Timestamp("2023-01-01 12:00:00"),
        pd.Timestamp("2023-01-01 12:10:00"),
    )


def test_scene_requires_at_least_one_observation():
    with pytest.raises(ValueError, match="at least one"):
        Scene(id="empty", observations=())


def test_scene_receptors_maps_any_builder():
    scene = Scene(
        id="s", observations=(_obs("2023-01-01 12:00:00"), _obs("2023-01-01 12:01:00"))
    )

    points = scene.receptors(build_point_receptor)
    columns = scene.receptors(partial(build_column_receptor, bottom=0.0, top=3000.0))

    def custom(obs: Observation) -> PointReceptor:
        return PointReceptor(obs.time, obs.longitude, obs.latitude, 5.0)

    customs = scene.receptors(custom)

    assert all(isinstance(r, PointReceptor) for r in points)
    assert [r.altitude for r in points] == [30.0, 30.0]
    assert all(isinstance(r, ColumnReceptor) for r in columns)
    assert [r.altitude for r in customs] == [5.0, 5.0]


# -- group_by_overpass ------------------------------------------------------------


def test_group_by_overpass_splits_on_gap_and_names_by_first_time():
    observations = [
        _obs("2023-01-01 12:20:00", observation_id="c"),
        _obs("2023-01-01 12:00:00", observation_id="a"),
        _obs("2023-01-01 12:04:00", observation_id="b"),
    ]

    scenes = group_by_overpass(observations, max_gap="10min")

    assert [s.id for s in scenes] == ["tropomi-202301011200", "tropomi-202301011220"]
    assert [[o.observation_id for o in s] for s in scenes] == [["a", "b"], ["c"]]
    assert scenes[0].metadata == {"max_gap": "0 days 00:10:00"}


def test_group_by_overpass_default_gap_and_prefix():
    observations = [_obs("2023-01-01 12:00:00"), _obs("2023-01-01 12:25:00")]

    [scene] = group_by_overpass(observations, prefix="slc-")

    assert scene.id == "slc-202301011200"
    assert len(scene) == 2


def test_group_by_overpass_empty():
    assert group_by_overpass([]) == []


# -- group_observations -----------------------------------------------------------


def test_group_observations_by_metadata_key_orders_by_time():
    observations = [
        _obs("2023-01-01 13:00:00", observation_id="c", metadata={"orbit": "002"}),
        _obs("2023-01-01 12:00:00", observation_id="a", metadata={"orbit": "001"}),
        _obs("2023-01-01 12:01:00", observation_id="b", metadata={"orbit": "001"}),
    ]

    scenes = group_observations(observations, key=lambda o: o.metadata["orbit"])

    assert [s.id for s in scenes] == ["tropomi-001", "tropomi-002"]
    assert [[o.observation_id for o in s] for s in scenes] == [["a", "b"], ["c"]]
    assert scenes[0].metadata == {"key": "001"}
