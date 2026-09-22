"""Tests for slant line-of-sight geometry and the slant receptor builder."""

import math

import numpy as np
import pytest

from stilt.observations import (
    Observation,
    ViewingGeometry,
    build_slant_receptor,
    slant_points,
)
from stilt.receptors import ColumnReceptor, MultiPointReceptor

METRES_PER_DEG_LAT = math.radians(1.0) * 6_371_000.0


def _observation(**kwargs) -> Observation:
    base = dict(
        sensor="em27",
        species="xch4",
        time="2023-07-15 18:00:00",
        latitude=40.7,
        longitude=-111.9,
        altitude=1300.0,
        altitude_ref="msl",
        viewing=ViewingGeometry(zenith_angle=30.0, azimuth_angle=90.0),
    )
    base.update(kwargs)
    return Observation(**base)


# -- slant_points ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("azimuth", "d_east", "d_north"),
    [(0.0, 0.0, 1.0), (90.0, 1.0, 0.0), (180.0, 0.0, -1.0), (270.0, -1.0, 0.0)],
)
def test_slant_points_azimuth_is_clockwise_from_north(azimuth, d_east, d_north):
    """Higher points lie in the azimuth direction: the ground-to-sun/satellite bearing."""
    zenith = 60.0
    dz = 1000.0

    points = slant_points(
        -111.9, 40.7, [1300.0, 1300.0 + dz], zenith=zenith, azimuth=azimuth
    )

    horizontal = dz * math.tan(math.radians(zenith))  # 1732 m
    metres_per_deg_lon = METRES_PER_DEG_LAT * math.cos(math.radians(40.7))
    (lon0, lat0, alt0), (lon1, lat1, alt1) = points
    assert (lon0, lat0, alt0) == (-111.9, 40.7, 1300.0)
    assert alt1 == 1300.0 + dz
    assert lat1 - lat0 == pytest.approx(
        d_north * horizontal / METRES_PER_DEG_LAT, abs=1e-9
    )
    assert lon1 - lon0 == pytest.approx(
        d_east * horizontal / metres_per_deg_lon, abs=1e-9
    )


def test_slant_points_zenith_zero_is_vertical():
    points = slant_points(
        -111.9, 40.7, np.linspace(0, 3000, 4), zenith=0.0, azimuth=135.0
    )

    assert [(lon, lat) for lon, lat, _ in points] == [(-111.9, 40.7)] * 4
    assert [alt for _, _, alt in points] == [0.0, 1000.0, 2000.0, 3000.0]


def test_slant_points_anchor_below_and_above():
    """Points below the anchor are displaced away from the azimuth, points above toward it."""
    points = slant_points(
        -111.9, 40.7, [500.0, 1000.0, 1500.0], zenith=45.0, azimuth=0.0, anchor=1000.0
    )

    (_, lat_lo, _), (lon_mid, lat_mid, _), (_, lat_hi, _) = points
    assert (lon_mid, lat_mid) == (-111.9, 40.7)
    assert lat_lo - lat_mid == pytest.approx(-500.0 / METRES_PER_DEG_LAT)
    assert lat_hi - lat_mid == pytest.approx(500.0 / METRES_PER_DEG_LAT)


def test_slant_points_accepts_numpy_and_returns_floats():
    points = slant_points(
        -111.9, 40.7, np.array([0.0, 100.0]), zenith=10.0, azimuth=45.0
    )

    assert len(points) == 2
    assert all(isinstance(v, float) for point in points for v in point)


def test_slant_points_rejects_empty_and_bad_zenith():
    with pytest.raises(ValueError, match="at least one altitude"):
        slant_points(-111.9, 40.7, [], zenith=10.0, azimuth=0.0)
    with pytest.raises(ValueError, match="zenith"):
        slant_points(-111.9, 40.7, [0.0, 1.0], zenith=90.0, azimuth=0.0)


# -- ViewingGeometry ---------------------------------------------------------------


def test_viewing_geometry_rejects_zenith_out_of_range():
    with pytest.raises(ValueError, match="zenith_angle"):
        ViewingGeometry(zenith_angle=95.0, azimuth_angle=0.0)
    with pytest.raises(ValueError, match="zenith_angle"):
        ViewingGeometry(zenith_angle=-1.0, azimuth_angle=0.0)


# -- build_slant_receptor ----------------------------------------------------------


def test_build_slant_receptor():
    observation = _observation()

    receptor = build_slant_receptor(observation, np.linspace(1300.0, 4300.0, 5))

    assert isinstance(receptor, MultiPointReceptor)
    assert len(receptor) == 5
    assert receptor.altitude_ref == "msl"
    assert receptor.longitudes[0] == observation.longitude
    assert receptor.latitudes[0] == observation.latitude
    assert receptor.altitudes[-1] == pytest.approx(4300.0)
    assert receptor.longitudes[-1] > receptor.longitudes[0]  # azimuth 90: leans east


def test_build_slant_receptor_matches_slant_points():
    observation = _observation()
    altitudes = [1300.0, 2300.0, 3300.0]

    receptor = build_slant_receptor(observation, altitudes)
    expected = slant_points(
        observation.longitude,
        observation.latitude,
        altitudes,
        zenith=30.0,
        azimuth=90.0,
    )

    assert isinstance(receptor, MultiPointReceptor)
    assert receptor.longitudes.tolist() == [p[0] for p in expected]
    assert receptor.latitudes.tolist() == [p[1] for p in expected]
    assert receptor.altitudes.tolist() == [p[2] for p in expected]


def test_build_slant_receptor_anchors_at_observation_altitude():
    """Samples above the anchor are displaced; a sample at the anchor is at the origin."""
    observation = _observation(altitude=1300.0)

    receptor = build_slant_receptor(observation, [1000.0, 1300.0, 2300.0])

    assert isinstance(receptor, MultiPointReceptor)
    assert receptor.longitudes[1] == pytest.approx(observation.longitude)
    assert receptor.longitudes[0] < observation.longitude < receptor.longitudes[2]


def test_build_slant_receptor_requires_viewing_geometry():
    observation = _observation(viewing=None)

    with pytest.raises(ValueError, match="Observation.viewing"):
        build_slant_receptor(observation, [1300.0, 2300.0])


def test_build_slant_receptor_requires_observation_altitude():
    observation = _observation(altitude=None)

    with pytest.raises(ValueError, match="Observation.altitude"):
        build_slant_receptor(observation, [1300.0, 2300.0])


def test_build_slant_receptor_warns_on_agl():
    observation = _observation(altitude=0.0, altitude_ref="agl")

    with pytest.warns(UserWarning, match="AGL"):
        receptor = build_slant_receptor(observation, [0.0, 1000.0])

    assert receptor.altitude_ref == "agl"


def test_build_slant_receptor_vertical_two_sample_path_is_a_column():
    observation = _observation(
        viewing=ViewingGeometry(zenith_angle=0.0, azimuth_angle=0.0)
    )

    receptor = build_slant_receptor(observation, [1300.0, 4300.0])

    assert isinstance(receptor, ColumnReceptor)
    assert (receptor.bottom, receptor.top) == (1300.0, 4300.0)
