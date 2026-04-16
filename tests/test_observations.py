"""Tests for observation-domain models and sensor interfaces."""

import pandas as pd

from stilt.observations import (
    BaseSensor,
    HorizontalGeometry,
    Observation,
    Scene,
    VerticalOperator,
    ViewingGeometry,
)
from stilt.receptor import Receptor


def test_observation_normalizes_timestamp_and_keeps_geometry():
    geometry = HorizontalGeometry(
        kind="swath_cell",
        center_longitude=-111.9,
        center_latitude=40.7,
        across_track_index=12,
        along_track_index=34,
        swath=2,
        resolution_km=(2.0, 7.0),
    )
    viewing = ViewingGeometry(
        solar_zenith_angle=45.0,
        viewing_zenith_angle=18.0,
        relative_azimuth_angle=132.0,
    )
    operator = VerticalOperator(
        mode="ak",
        levels=[0.0, 1000.0, 2000.0],
        values=[0.1, 0.6, 0.3],
    )

    obs = Observation(
        sensor="oco2",
        species="xco2",
        time="2023-01-01 12:34:56",
        latitude=40.7,
        longitude=-111.9,
        geometry=geometry,
        viewing=viewing,
        operator=operator,
        observation_id="sound-1",
    )

    assert isinstance(obs.time, pd.Timestamp)
    assert obs.geometry is geometry
    assert obs.viewing is viewing
    assert obs.operator is operator


def test_scene_batch_id_defaults_to_scene_id():
    obs = Observation(
        sensor="tccon",
        species="xco2",
        time="2023-01-01T00:00:00Z",
        latitude=40.7,
        longitude=-111.9,
        observation_id="obs-1",
    )
    scene = Scene(id="my-scene", sensor="tccon", observations=[obs])

    assert scene.batch_id == "my-scene"
    assert scene.observation_ids == ["obs-1"]


def test_base_sensor_groups_scene_and_allows_receptor_building():
    obs = Observation(
        sensor="tower",
        species="co2",
        time="2023-01-01 12:00:00",
        latitude=40.7,
        longitude=-111.9,
    )

    class TowerSensor(BaseSensor):
        name = "tower"
        supported_species = ("co2",)

        def build_receptor(self, observation: Observation) -> Receptor:
            return Receptor(
                time=observation.time,
                longitude=observation.longitude,
                latitude=observation.latitude,
                altitude=30.0,
            )

    sensor = TowerSensor()

    receptor = sensor.build_receptor(obs)
    scenes = sensor.group_scenes([obs])

    assert receptor.altitude == 30.0
    assert scenes[0].sensor == "tower"
    assert scenes[0].observations == [obs]
