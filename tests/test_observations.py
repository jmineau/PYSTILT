"""Tests for stilt.observations: slant geometry, overpass grouping, selection, jitter."""

import math

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from stilt.observations import (
    group_by_overpass,
    jitter_points,
    select_observations_spatial,
    slant_points,
)
from stilt.receptors import ColumnReceptor, MultiPointReceptor, Receptor

METRES_PER_DEG_LAT = math.radians(1.0) * 6_371_000.0


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


def test_slant_points_make_a_multipoint_receptor():
    """The documented two-liner: slant_points + Receptor.from_points."""
    points = slant_points(
        -111.9, 40.7, np.linspace(1300.0, 4300.0, 5), zenith=30.0, azimuth=90.0
    )
    receptor = Receptor.from_points("2023-07-15 18:00", points, altitude_ref="msl")

    assert isinstance(receptor, MultiPointReceptor)
    assert len(receptor) == 5
    assert receptor.altitude_ref == "msl"
    assert receptor.longitudes[-1] > receptor.longitudes[0]  # azimuth 90: leans east

    vertical = slant_points(-111.9, 40.7, [1300.0, 4300.0], zenith=0.0, azimuth=0.0)
    column = Receptor.from_points("2023-07-15 18:00", vertical, altitude_ref="msl")
    assert isinstance(column, ColumnReceptor)
    assert (column.bottom, column.top) == (1300.0, 4300.0)


# -- group_by_overpass ------------------------------------------------------------


def test_group_by_overpass_splits_on_gap_and_labels_by_first_time():
    times = pd.Series(
        ["2023-01-01 12:20:00", "2023-01-01 12:00:00", "2023-01-01 12:04:00"],
        index=[7, 8, 9],
    )

    labels = group_by_overpass(times, max_gap="10min")

    assert labels.index.tolist() == [7, 8, 9]
    assert labels.tolist() == ["202301011220", "202301011200", "202301011200"]


def test_group_by_overpass_default_gap_and_groupby():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2023-01-01 12:00", "2023-01-01 12:25", "2023-01-01 14:00"]
            ),
            "value": [1.0, 2.0, 3.0],
        }
    )

    df["overpass"] = group_by_overpass(df["time"])
    groups = {k: g["value"].tolist() for k, g in df.groupby("overpass")}

    assert groups == {"202301011200": [1.0, 2.0], "202301011400": [3.0]}


def test_group_by_overpass_accepts_plain_sequences_and_empty():
    labels = group_by_overpass(["2023-01-01 12:00", "2023-01-02 12:00"])
    assert labels.tolist() == ["202301011200", "202301021200"]
    assert group_by_overpass([]).empty


def test_group_by_overpass_rejects_nat():
    with pytest.raises(ValueError, match="NaT"):
        group_by_overpass(["2023-01-01 12:00", None])


# -- select_observations_spatial ---------------------------------------------------

# A regular 5 x 5 grid of soundings over a domain centred on SLC.
_LONS = np.repeat([-112.5, -112.0, -111.5, -111.0, -110.5], 5)
_LATS = np.tile([40.0, 40.5, 41.0, 41.5, 42.0], 5)
_SITE = dict(site_longitude=-111.85, site_latitude=40.77)
_DOMAIN = dict(domain_lon_range=(-112.5, -110.5), domain_lat_range=(40.0, 42.0))


def _select(lons=_LONS, lats=_LATS, **overrides):
    kwargs = dict(
        near_field_dlon=0.5,
        near_field_dlat=0.5,
        near_field_cols=3,
        near_field_rows=3,
        background_cols=2,
        background_rows=2,
        **_SITE,
        **_DOMAIN,
    )
    kwargs.update(overrides)
    return select_observations_spatial(lons, lats, **kwargs)


def test_select_returns_unique_indices_sorted_by_latitude():
    selected = _select()

    assert selected.dtype.kind == "i"
    assert len(selected) == len(set(selected.tolist())) > 0
    assert all(0 <= i < len(_LONS) for i in selected)
    assert _LATS[selected].tolist() == sorted(_LATS[selected].tolist())


def test_select_background_grid_reaches_domain_corners():
    selected = set(_select(near_field_cols=0, near_field_rows=0).tolist())

    corners = {0, 4, 20, 24}  # (lon, lat) grid corners in _LONS/_LATS order
    assert corners <= selected


def test_select_near_field_favours_site_adjacent_soundings():
    selected = _select(
        near_field_dlon=0.2,
        near_field_dlat=0.2,
        background_cols=0,
        background_rows=0,
    )

    assert np.all(np.abs(_LONS[selected] - _SITE["site_longitude"]) < 1.0)
    assert np.all(np.abs(_LATS[selected] - _SITE["site_latitude"]) < 1.0)


def test_select_empty_and_single():
    assert _select(lons=[], lats=[]).tolist() == []
    assert _select(lons=[-111.85], lats=[40.77]).tolist() == [0]


def test_select_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        _select(lons=[0.0, 1.0], lats=[0.0])


# -- jitter_points ---------------------------------------------------------------

_PIXEL = [(-111.9, 40.7), (-111.8, 40.7), (-111.8, 40.8), (-111.9, 40.8)]


def test_jitter_points_regular_grid_inside_polygon():
    points = jitter_points(_PIXEL, 4)

    assert len(points) == 4
    outline = Polygon(_PIXEL)
    assert all(outline.covers(Point(p)) for p in points)
    assert points == [
        (-111.875, 40.725),
        (-111.825, 40.725),
        (-111.875, 40.775),
        (-111.825, 40.775),
    ]


def test_jitter_points_random_is_seeded_and_accepts_shapely():
    outline = Polygon(_PIXEL)

    a = jitter_points(outline, 3, method="random", seed=7)
    b = jitter_points(outline, 3, method="random", seed=7)

    assert a == b
    assert len(a) == 3
    assert all(outline.covers(Point(p)) for p in a)


def test_jitter_points_rejects_bad_inputs():
    with pytest.raises(ValueError, match="n must be"):
        jitter_points(_PIXEL, 0)
    with pytest.raises(ValueError, match="Unknown jitter method"):
        jitter_points(_PIXEL, 2, method="hexagonal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="zero-area"):
        jitter_points([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], 2)
