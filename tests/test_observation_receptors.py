"""Tests for observation-to-receptor conversion helpers."""

import pytest

from stilt.observations import (
    LineOfSight,
    Observation,
    ViewingGeometry,
    build_column_receptor,
    build_multipoint_receptor,
    build_point_receptor,
    build_slant_receptor,
)


def test_build_point_receptor_uses_observation_altitude():
    observation = Observation(
        sensor="tower",
        species="co2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        altitude=30.0,
    )

    receptor = build_point_receptor(observation)

    assert receptor.kind == "point"
    assert receptor.altitude == 30.0
    assert receptor.altitude_ref == "agl"


def test_build_point_receptor_altitude_override_wins():
    observation = Observation(
        sensor="tower",
        species="co2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        altitude=30.0,
    )

    receptor = build_point_receptor(observation, altitude=100.0)

    assert receptor.altitude == 100.0


def test_build_point_receptor_requires_altitude():
    observation = Observation(
        sensor="tower",
        species="co2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )

    with pytest.raises(ValueError, match="altitude"):
        build_point_receptor(observation)


def test_build_column_receptor():
    observation = Observation(
        sensor="tccon",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )

    receptor = build_column_receptor(observation, bottom=10.0, top=1500.0)

    assert receptor.kind == "column"
    assert receptor.bottom == 10.0
    assert receptor.top == 1500.0
    assert receptor.altitude_ref == "agl"


def test_build_multipoint_receptor():
    observation = Observation(
        sensor="slant",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )

    receptor = build_multipoint_receptor(
        observation,
        points=[
            (-111.9, 40.7, 100.0),
            (-111.8, 40.8, 500.0),
            (-111.7, 40.9, 900.0),
        ],
    )

    assert receptor.kind == "multipoint"
    assert len(receptor) == 3


def test_build_slant_receptor():
    observation = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        viewing=ViewingGeometry(
            viewing_zenith_angle=30.0,
            viewing_azimuth_angle=90.0,
        ),
        line_of_sight=LineOfSight(
            start_altitude=0.0,
            end_altitude=2000.0,
            count=5,
            altitude_ref="msl",
            anchor_altitude=0.0,
        ),
    )

    receptor = build_slant_receptor(observation)

    assert receptor.kind == "multipoint"
    assert len(receptor) == 5
    assert receptor.longitudes[0] == -111.9
    assert receptor.latitudes[0] == 40.7
    assert receptor.altitude_ref == "msl"
    assert receptor.altitudes[-1] == pytest.approx(2000.0)
    assert receptor.longitudes[-1] > receptor.longitudes[0]


def test_build_slant_receptor_requires_viewing_geometry():
    observation = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        line_of_sight=LineOfSight(
            start_altitude=0.0,
            end_altitude=1000.0,
            count=3,
        ),
    )

    with pytest.raises(ValueError, match="Observation.viewing"):
        build_slant_receptor(observation)


def test_build_slant_receptor_auto_assigns_anchor_from_observation_altitude():
    observation = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        altitude=1000.0,
        altitude_ref="msl",
        viewing=ViewingGeometry(
            viewing_zenith_angle=30.0,
            viewing_azimuth_angle=90.0,
        ),
        line_of_sight=LineOfSight(
            altitude_levels=[1000.0, 2000.0],
            altitude_ref="msl",
        ),
    )

    receptor = build_slant_receptor(observation)

    assert receptor.altitudes.tolist() == [1000.0, 2000.0]
    assert receptor.longitudes[0] == pytest.approx(observation.longitude)
    assert receptor.latitudes[0] == pytest.approx(observation.latitude)
    assert receptor.longitudes[1] > receptor.longitudes[0]


def test_build_slant_receptor_auto_clips_msl_below_observation_surface():
    observation = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        altitude=1200.0,
        altitude_ref="msl",
        viewing=ViewingGeometry(
            viewing_zenith_angle=20.0,
            viewing_azimuth_angle=0.0,
        ),
        line_of_sight=LineOfSight(
            altitude_levels=[500.0, 1200.0, 1500.0],
            altitude_ref="msl",
        ),
    )

    receptor = build_slant_receptor(observation)

    assert receptor.altitudes.tolist() == [1200.0, 1500.0]


def test_build_slant_receptor_explicit_model_top_clips_upper_levels():
    observation = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        altitude=1000.0,
        altitude_ref="msl",
        viewing=ViewingGeometry(
            viewing_zenith_angle=20.0,
            viewing_azimuth_angle=180.0,
        ),
        line_of_sight=LineOfSight(
            start_altitude=1000.0,
            end_altitude=4000.0,
            count=4,
            altitude_ref="msl",
        ),
    )

    receptor = build_slant_receptor(observation, model_top_altitude=2500.0)

    assert receptor.altitudes.tolist() == [1000.0, 2000.0]
