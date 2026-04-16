"""Tests for the column-sensor implementation."""

import pytest

from stilt.observations import ColumnSensor, LineOfSight, Observation, ViewingGeometry


def test_column_sensor_make_observation_requires_species_without_single_default():
    sensor = ColumnSensor(mode="vertical", bottom=50.0, top=1500.0)

    with pytest.raises(ValueError, match="requires `species=`"):
        sensor.make_observation(
            time="2023-01-01 12:00:00",
            latitude=40.7,
            longitude=-111.9,
        )


def test_column_sensor_vertical_builds_column_receptor():
    observation = Observation(
        sensor="tccon",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )

    sensor = ColumnSensor(mode="vertical", bottom=50.0, top=1500.0)
    receptor = sensor.build_receptor(observation)

    assert receptor.kind == "column"
    assert receptor.bottom == 50.0
    assert receptor.top == 1500.0


def test_column_sensor_vertical_can_use_msl_altitudes():
    observation = Observation(
        sensor="tccon",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )

    sensor = ColumnSensor(
        mode="vertical", bottom=1500.0, top=6000.0, altitude_ref="msl"
    )
    receptor = sensor.build_receptor(observation)

    assert receptor.altitude_ref == "msl"
    assert receptor.bottom == 1500.0
    assert receptor.top == 6000.0


def test_column_sensor_vertical_requires_bounds():
    observation = Observation(
        sensor="tccon",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )

    sensor = ColumnSensor(mode="vertical")

    with pytest.raises(ValueError, match="bottom and top"):
        sensor.build_receptor(observation)


def test_column_sensor_slant_builds_multipoint_receptor():
    observation = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        viewing=ViewingGeometry(
            viewing_zenith_angle=20.0,
            viewing_azimuth_angle=135.0,
        ),
        line_of_sight=LineOfSight(
            start_altitude=1000.0,
            end_altitude=3000.0,
            count=5,
            altitude_ref="msl",
            anchor_altitude=1000.0,
        ),
    )

    sensor = ColumnSensor(mode="slant")
    receptor = sensor.build_receptor(observation)

    assert receptor.kind == "multipoint"
    assert len(receptor) == 5
    assert receptor.altitude_ref == "msl"


def test_column_sensor_slant_requires_los_definition():
    observation = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        viewing=ViewingGeometry(
            viewing_zenith_angle=20.0,
            viewing_azimuth_angle=135.0,
        ),
    )

    sensor = ColumnSensor(mode="slant")

    with pytest.raises(ValueError, match="line_of_sight"):
        sensor.build_receptor(observation)
