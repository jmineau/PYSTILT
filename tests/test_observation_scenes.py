"""Tests for observation scene-grouping helpers."""

from stilt.observations import (
    HorizontalGeometry,
    Observation,
    group_scenes_by_metadata,
    group_scenes_by_swath,
    group_scenes_by_time_gap,
    make_scene,
)


def _obs(
    *,
    time: str,
    sensor: str = "tropomi",
    observation_id: str | None = None,
    swath: int | None = None,
    metadata: dict[str, object] | None = None,
) -> Observation:
    geometry = None
    if swath is not None:
        geometry = HorizontalGeometry(
            kind="swath_cell",
            center_longitude=-111.9,
            center_latitude=40.7,
            swath=swath,
            resolution_km=(7.0, 5.0),
        )
    return Observation(
        sensor=sensor,
        species="xco2",
        time=time,
        latitude=40.7,
        longitude=-111.9,
        observation_id=observation_id,
        geometry=geometry,
        metadata=metadata or {},
    )


def test_make_scene_sorts_observations_and_uses_default_id():
    obs_late = _obs(time="2023-01-01 12:10:00", observation_id="b")
    obs_early = _obs(time="2023-01-01 12:00:00", observation_id="a")

    scene = make_scene([obs_late, obs_early])

    assert scene.id == "tropomi-20230101120000"
    assert scene.observation_ids == ["a", "b"]


def test_group_scenes_by_time_gap_splits_when_gap_exceeds_threshold():
    observations = [
        _obs(time="2023-01-01 12:00:00", observation_id="a"),
        _obs(time="2023-01-01 12:04:00", observation_id="b"),
        _obs(time="2023-01-01 12:20:00", observation_id="c"),
    ]

    scenes = group_scenes_by_time_gap(observations, max_gap="5min")

    assert len(scenes) == 2
    assert scenes[0].observation_ids == ["a", "b"]
    assert scenes[1].observation_ids == ["c"]


def test_group_scenes_by_swath_groups_shared_swaths():
    observations = [
        _obs(time="2023-01-01 12:00:00", observation_id="a", swath=1),
        _obs(time="2023-01-01 12:01:00", observation_id="b", swath=2),
        _obs(time="2023-01-01 12:02:00", observation_id="c", swath=1),
    ]

    scenes = group_scenes_by_swath(observations, scene_prefix="oco")

    assert len(scenes) == 2
    assert scenes[0].id == "oco-1"
    assert scenes[0].observation_ids == ["a", "c"]
    assert scenes[0].metadata["swath"] == 1


def test_group_scenes_by_metadata_groups_on_requested_key():
    observations = [
        _obs(
            time="2023-01-01 12:00:00",
            observation_id="a",
            metadata={"orbit": "001"},
        ),
        _obs(
            time="2023-01-01 12:01:00",
            observation_id="b",
            metadata={"orbit": "002"},
        ),
        _obs(
            time="2023-01-01 12:02:00",
            observation_id="c",
            metadata={"orbit": "001"},
        ),
    ]

    scenes = group_scenes_by_metadata(observations, key="orbit", scene_prefix="oco")

    assert len(scenes) == 2
    assert scenes[0].id == "oco-001"
    assert scenes[0].observation_ids == ["a", "c"]
    assert scenes[0].metadata["key"] == "orbit"
