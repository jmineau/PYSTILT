"""Tests for observation selection helpers."""

import pytest

import stilt.observations.selection as selection
from stilt.observations import (
    HorizontalGeometry,
    Observation,
    filter_observations,
    jitter_observation,
)


def test_jitter_observation_regular_polygon_returns_point_observations():
    observation = Observation(
        sensor="tropomi",
        species="ch4",
        time="2023-01-01 12:00:00",
        latitude=40.75,
        longitude=-111.85,
        observation_id="obs-1",
        geometry=HorizontalGeometry(
            kind="polygon",
            center_longitude=-111.85,
            center_latitude=40.75,
            vertices=[
                (-111.9, 40.7),
                (-111.8, 40.7),
                (-111.8, 40.8),
                (-111.9, 40.8),
            ],
        ),
    )

    jittered = jitter_observation(observation, n=4, method="regular")

    assert len(jittered) == 4
    assert all(
        obs.geometry is not None and obs.geometry.kind == "point" for obs in jittered
    )
    assert jittered[0].observation_id == "obs-1-j1"
    assert jittered[0].metadata["parent_observation_id"] == "obs-1"


def test_jitter_observation_random_swath_cell_uses_resolution_when_vertices_absent():
    observation = Observation(
        sensor="tropomi",
        species="no2",
        time="2023-01-01 12:00:00",
        latitude=40.75,
        longitude=-111.85,
        geometry=HorizontalGeometry(
            kind="swath_cell",
            center_longitude=-111.85,
            center_latitude=40.75,
            resolution_km=(7.0, 5.0),
            orientation_deg=20.0,
        ),
    )

    jittered = jitter_observation(observation, n=3, method="random", seed=7)

    assert len(jittered) == 3
    assert all(
        obs.geometry is not None and obs.geometry.kind == "point" for obs in jittered
    )


def test_jitter_observation_requires_geometry():
    observation = Observation(
        sensor="tower",
        species="co2",
        time="2023-01-01 12:00:00",
        latitude=40.75,
        longitude=-111.85,
    )

    try:
        jitter_observation(observation, n=2)
    except ValueError as exc:
        assert "geometry" in str(exc)
    else:
        raise AssertionError("Expected jitter_observation to require geometry.")


def test_jitter_observation_requires_pyproj_for_synthesized_geometry(monkeypatch):
    observation = Observation(
        sensor="tropomi",
        species="no2",
        time="2023-01-01 12:00:00",
        latitude=40.75,
        longitude=-111.85,
        geometry=HorizontalGeometry(
            kind="swath_cell",
            center_longitude=-111.85,
            center_latitude=40.75,
            resolution_km=(7.0, 5.0),
        ),
    )

    def _missing_pyproj(_geometry):
        raise ImportError("Synthetic observation geometry requires pyproj.")

    monkeypatch.setattr(selection, "_projection_transforms", _missing_pyproj)

    with pytest.raises(ImportError, match="pyproj"):
        jitter_observation(observation, n=2)


def test_filter_observations_preserves_order_and_filters_by_sensor_species_and_id():
    observations = [
        Observation(
            sensor="tower",
            species="co2",
            time="2023-01-01 12:00:00",
            latitude=40.75,
            longitude=-111.85,
            observation_id="a",
        ),
        Observation(
            sensor="oco2",
            species="xco2",
            time="2023-01-01 12:05:00",
            latitude=40.76,
            longitude=-111.84,
            observation_id="b",
        ),
        Observation(
            sensor="tower",
            species="ch4",
            time="2023-01-01 12:10:00",
            latitude=40.77,
            longitude=-111.83,
            observation_id="c",
        ),
    ]

    filtered = filter_observations(
        observations,
        sensors="tower",
        species=("co2", "ch4"),
        observation_ids=("c", "a"),
    )

    assert [obs.observation_id for obs in filtered] == ["a", "c"]


def test_filter_observations_filters_by_time_metadata_quality_and_predicate():
    observations = [
        Observation(
            sensor="tropomi",
            species="xch4",
            time="2023-01-01 12:00:00",
            latitude=40.75,
            longitude=-111.85,
            observation_id="a",
            metadata={"orbit": "001"},
            quality={"good": True, "qa_value": 1},
        ),
        Observation(
            sensor="tropomi",
            species="xch4",
            time="2023-01-01 12:05:00",
            latitude=40.76,
            longitude=-111.84,
            observation_id="b",
            metadata={"orbit": "002"},
            quality={"good": True, "qa_value": 0},
        ),
        Observation(
            sensor="tropomi",
            species="xch4",
            time="2023-01-01 12:10:00",
            latitude=40.77,
            longitude=-111.83,
            observation_id="c",
            metadata={"orbit": "001"},
            quality={"good": False, "qa_value": 1},
        ),
    ]

    filtered = filter_observations(
        observations,
        time_range=("2023-01-01 12:00:00", "2023-01-01 12:06:00"),
        metadata={"orbit": "002"},
        quality={"good": True},
        predicate=lambda obs: obs.longitude > -111.85,
    )

    assert [obs.observation_id for obs in filtered] == ["b"]
