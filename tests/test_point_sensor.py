"""Tests for the point-sensor implementation."""

import pytest

from stilt.observations import HorizontalGeometry, Observation, PointSensor
from stilt.receptors import PointReceptor


def test_point_sensor_make_observation_sets_sensor_and_default_species():
    sensor = PointSensor(name="tower", supported_species=("co2",))

    observation = sensor.make_observation(
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        altitude=25.0,
    )

    assert observation.sensor == "tower"
    assert observation.species == "co2"
    assert observation.altitude == 25.0


def test_point_sensor_make_observation_rejects_unsupported_species():
    sensor = PointSensor(name="tower", supported_species=("co2",))

    with pytest.raises(ValueError, match="does not support species"):
        sensor.make_observation(
            species="ch4",
            time="2023-01-01 12:00:00",
            latitude=40.7,
            longitude=-111.9,
        )


def test_point_sensor_uses_observation_altitude_when_present():
    observation = Observation(
        sensor="tower",
        species="co2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        altitude=25.0,
    )

    sensor = PointSensor(name="tower", supported_species=("co2",))
    receptor = sensor.build_receptor(observation)

    assert isinstance(receptor, PointReceptor)
    assert receptor.altitude == 25.0


def test_point_sensor_default_height_overrides_missing_altitude():
    observation = Observation(
        sensor="train",
        species="co2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )

    sensor = PointSensor(name="train", default_height=4.0)
    receptor = sensor.build_receptor(observation)

    assert isinstance(receptor, PointReceptor)
    assert receptor.altitude == 4.0


def test_point_sensor_rejects_non_point_geometry():
    observation = Observation(
        sensor="satellite",
        species="xco2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
        geometry=HorizontalGeometry(
            kind="swath_cell",
            center_longitude=-111.9,
            center_latitude=40.7,
            resolution_km=(2.0, 7.0),
        ),
    )

    sensor = PointSensor()

    with pytest.raises(ValueError, match="point geometry"):
        sensor.build_receptor(observation)
