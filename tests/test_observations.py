"""Tests for observation-domain models."""

import pandas as pd

from stilt.observations import (
    HorizontalGeometry,
    Observation,
    ViewingGeometry,
)
from stilt.transforms import AveragingKernel


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
    viewing = ViewingGeometry(zenith_angle=18.0, azimuth_angle=132.0)
    kernel = AveragingKernel(
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
        transforms=[kernel],
        observation_id="sound-1",
    )

    assert isinstance(obs.time, pd.Timestamp)
    assert obs.geometry is geometry
    assert obs.viewing is viewing
    assert obs.transforms[0] is kernel
