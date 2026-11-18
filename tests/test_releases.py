"""Tests for stilt.releases."""

import datetime as dt
import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from stilt.releases import (
    ColumnRelease,
    MultiPointRelease,
    PointRelease,
    Release,
    read_releases,
    write_releases,
)
from stilt.releases.models import _format_coord
from stilt.releases.table import releases_to_csv

# ---------------------------------------------------------------------------
# _format_coord
# ---------------------------------------------------------------------------


def test_format_coord_integer_float():
    assert _format_coord(5.0) == "5"


def test_format_coord_negative_integer_float():
    assert _format_coord(-114.0) == "-114"


def test_format_coord_zero():
    assert _format_coord(0.0) == "0"


def test_format_coord_fractional():
    assert _format_coord(-111.85) == "-111.85"


def test_format_coord_small_positive():
    assert _format_coord(0.5) == "0.5"


# ---------------------------------------------------------------------------
# location_id - PointRelease
# ---------------------------------------------------------------------------


def test_location_id_point_basic():
    r = PointRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    assert r.location_id == "-111.85_40.77_5"


def test_location_id_point_integer_coords():
    r = PointRelease(
        time="202301011200", longitude=-112.0, latitude=40.0, altitude=10.0
    )
    assert r.location_id == "-112_40_10"


def test_location_id_point_fractional_height():
    r = PointRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=2.5
    )
    assert r.location_id == "-111.85_40.77_2.5"


# ---------------------------------------------------------------------------
# location_id - ColumnRelease
# ---------------------------------------------------------------------------


def test_location_id_column_spells_out_bottom_and_top():
    r = ColumnRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, bottom=5.0, top=50.0
    )
    assert r.location_id == "-111.85_40.77_X5-50"


def test_location_id_column_integer_coords():
    r = ColumnRelease(
        time="202301011200", longitude=-112.0, latitude=40.0, bottom=0.0, top=3000.0
    )
    assert r.location_id == "-112_40_X0-3000"


def test_columns_that_differ_only_in_top_get_different_ids():
    """Two columns at one place and time once shared an id and result files (#105)."""
    low, high = (
        ColumnRelease(
            time="202101150600", longitude=-112, latitude=40.5, bottom=0, top=top
        )
        for top in (3000, 5000)
    )
    assert low.id == "202101150600_-112_40.5_X0-3000"
    assert high.id != low.id


# ---------------------------------------------------------------------------
# location_id - altitude reference (#105)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "agl_id"),
    [
        (
            lambda ref: PointRelease(
                time="202101150600",
                longitude=-112,
                latitude=40.5,
                altitude=100,
                altitude_ref=ref,
            ),
            "202101150600_-112_40.5_100",
        ),
        (
            lambda ref: ColumnRelease(
                time="202101150600",
                longitude=-112,
                latitude=40.5,
                bottom=0,
                top=3000,
                altitude_ref=ref,
            ),
            "202101150600_-112_40.5_X0-3000",
        ),
        (
            lambda ref: MultiPointRelease(
                time="202101150600",
                longitudes=[-112, -112.1],
                latitudes=[40.5, 40.6],
                altitudes=[100, 200],
                altitude_ref=ref,
            ),
            None,
        ),
    ],
    ids=["point", "column", "multipoint"],
)
def test_msl_and_agl_releases_get_different_ids(make, agl_id):
    """Releases that differ only in altitude reference once shared an id (#105)."""
    agl, msl = make("agl"), make("msl")
    assert msl.id == agl.id + "msl"
    assert not agl.id.endswith("msl")
    if agl_id is not None:
        assert agl.id == agl_id


# ---------------------------------------------------------------------------
# location_id - MultiPointRelease (SHA-256 hash, order-independent)
# ---------------------------------------------------------------------------


def test_location_id_multipoint_stable():
    lons = [-111.85, -111.86, -111.84]
    lats = [40.77, 40.78, 40.76]
    alts = [5, 5, 5]
    r1 = MultiPointRelease(
        time="202301011200", longitudes=lons, latitudes=lats, altitudes=alts
    )
    r2 = MultiPointRelease(
        time="202301011200", longitudes=lons, latitudes=lats, altitudes=alts
    )
    assert r1.location_id == r2.location_id


def test_location_id_multipoint_order_independent():
    r_a = MultiPointRelease(
        time="202301011200",
        longitudes=[-111.85, -111.86],
        latitudes=[40.77, 40.78],
        altitudes=[5, 5],
    )
    r_b = MultiPointRelease(
        time="202301011200",
        longitudes=[-111.86, -111.85],
        latitudes=[40.78, 40.77],
        altitudes=[5, 5],
    )
    assert r_a.location_id == r_b.location_id


def test_location_id_multipoint_starts_with_multi():
    r = MultiPointRelease(
        time="202301011200",
        longitudes=[-111.85, -111.86],
        latitudes=[40.77, 40.78],
        altitudes=[5, 5],
    )
    assert r.location_id.startswith("multi_")


def test_location_id_multipoint_hash_length():
    r = MultiPointRelease(
        time="202301011200",
        longitudes=[-111.85, -111.86],
        latitudes=[40.77, 40.78],
        altitudes=[5, 5],
    )
    hash_part = r.location_id.replace("multi_", "")
    assert len(hash_part) == 10
    assert all(c in "0123456789abcdef" for c in hash_part)


def test_location_id_multipoint_matches_spec():
    pts = [(-111.85, 40.77, 5), (-111.86, 40.78, 5)]
    pts_sorted = sorted(pts)
    canonical = json.dumps(
        [[round(lon, 5), round(lat, 5), int(zagl)] for lon, lat, zagl in pts_sorted],
        separators=(",", ":"),
    )
    expected_hash = hashlib.sha256(canonical.encode()).hexdigest()[:10]
    r = MultiPointRelease(
        time="202301011200",
        longitudes=[-111.85, -111.86],
        latitudes=[40.77, 40.78],
        altitudes=[5, 5],
    )
    assert r.location_id == f"multi_{expected_hash}"


def test_location_id_multipoint_differs_for_different_points():
    r_a = MultiPointRelease(
        time="202301011200",
        longitudes=[-111.85, -111.86],
        latitudes=[40.77, 40.78],
        altitudes=[5, 5],
    )
    r_b = MultiPointRelease(
        time="202301011200",
        longitudes=[-111.85, -111.87],
        latitudes=[40.77, 40.79],
        altitudes=[5, 5],
    )
    assert r_a.location_id != r_b.location_id


# ---------------------------------------------------------------------------
# PointRelease
# ---------------------------------------------------------------------------


def test_release_id_format(point_release):
    assert point_release.id == "202301011200_-111.85_40.77_5"


def test_release_points_point(point_release):
    assert len(point_release.coords()) == 1


def test_release_points_column(column_release):
    assert len(column_release.coords()) == 2


def test_release_points_multipoint(multipoint_release):
    assert len(multipoint_release.coords()) == 3


def test_release_coordinates(point_release):
    assert point_release.longitude == -111.85
    assert point_release.latitude == 40.77
    assert point_release.altitude == 5.0
    assert point_release.altitude_ref == "agl"


def test_release_column_top_bottom(column_release):
    assert column_release.bottom == 5.0
    assert column_release.top == 50.0


def test_point_has_no_bottom():
    r = PointRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    with pytest.raises(AttributeError):
        _ = r.bottom
    with pytest.raises(AttributeError):
        _ = r.top


def test_column_has_no_altitude():
    r = ColumnRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, bottom=5.0, top=50.0
    )
    with pytest.raises(AttributeError):
        _ = r.altitude


def test_release_time_iso_string():
    r = PointRelease(
        time="2023-01-01T12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2023, 1, 1, 12, 0)


def test_release_time_compact_string():
    r = PointRelease(
        time="202301011200",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2023, 1, 1, 12, 0)


def test_release_time_aware_input_normalizes_to_naive_utc():
    r = PointRelease(
        time="2023-01-01T05:00:00-07:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2023, 1, 1, 12, 0)
    assert r.time.tzinfo is None


@pytest.mark.parametrize("text", ["05/07/2023 18:00", "07/15/2023 18:00", "now"])
def test_release_time_text_that_is_not_iso_or_compact_is_refused(text):
    """05/07/2023 was read as 7 May, and refused only when the day was above 12 (#440)."""
    with pytest.raises(ValueError, match="is not ISO 8601"):
        PointRelease(time=text, longitude=-111.85, latitude=40.77, altitude=5.0)


def test_read_releases_refuses_a_day_first_time_and_names_the_row():
    text = "time,lon,lat,z\n2023-07-15 18:00,-111.8,40.7,10\n05/07/2023 18:00,-111.8,40.7,10\n"
    with pytest.raises(ValueError, match="row 1: time is '05/07/2023 18:00'"):
        read_releases(io.StringIO(text))


def test_read_releases_reads_iso_and_compact_times_with_or_without_an_offset():
    text = (
        "time,lon,lat,z\n"
        "2023-07-15 18:00,-111.8,40.7,10\n"
        "202307151900,-111.8,40.7,10\n"
        "2023-07-15T14:00:00-06:00,-111.8,40.7,10\n"
        "2023-07-15T21:00Z,-111.8,40.7,10\n"
    )
    times = [r.time for r in read_releases(io.StringIO(text))]
    assert times == [dt.datetime(2023, 7, 15, h) for h in (18, 19, 20, 21)]


def test_read_releases_refuses_an_empty_time():
    with pytest.raises(ValueError, match="row 0: time is empty"):
        read_releases(io.StringIO("time,lon,lat,z\n,-111.8,40.7,10\n"))


def test_release_time_with_nanoseconds_is_cut_to_microseconds():
    # satellite readers give nanosecond times (OCO-2: float seconds since 1970)
    r = PointRelease(
        time=pd.Timestamp("2026-02-08T19:37:37.282009602"),
        longitude=-94.91,
        latitude=29.22,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2026, 2, 8, 19, 37, 37, 282009)


def test_release_equality():
    r1 = PointRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    r2 = PointRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    assert r1 == r2


def test_release_inequality_time():
    r1 = PointRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    r2 = PointRelease(
        time="202301011300", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    assert r1 != r2


def test_release_inequality_across_types():
    r1 = PointRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    r2 = ColumnRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, bottom=5.0, top=50.0
    )
    assert r1 != r2


def test_release_init_rejects_missing_time():
    with pytest.raises(ValueError, match="'time' must be provided"):
        PointRelease(time=None, longitude=-111.85, latitude=40.77, altitude=5.0)


def test_release_init_parses_iso_and_compact_strings():
    r_iso = PointRelease(
        time="2023-01-01T12:00:00", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    r_compact = PointRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    expected = dt.datetime(2023, 1, 1, 12, 0)
    assert r_iso.time == expected
    assert r_compact.time == expected


def test_release_eq_non_release_is_false(point_release):
    assert (point_release == object()) is False


def test_release_points_property_yields_points(point_release):
    items = list(point_release.points)
    assert len(items) == 1
    assert hasattr(items[0], "x") and items[0].x == pytest.approx(-111.85)
    assert items[0].y == pytest.approx(40.77)
    assert items[0].z == pytest.approx(5.0)


def test_release_lazy_geometry_point(point_release):
    from shapely import Point

    assert isinstance(point_release.geometry, Point)


def test_release_lazy_geometry_column(column_release):
    from shapely import LineString

    assert isinstance(column_release.geometry, LineString)


def test_release_lazy_geometry_multipoint(multipoint_release):
    from shapely import MultiPoint

    assert isinstance(multipoint_release.geometry, MultiPoint)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_longitude_out_of_range_raises():
    with pytest.raises(ValueError, match="longitude"):
        PointRelease(time="202301011200", longitude=200.0, latitude=40.77, altitude=5.0)


def test_latitude_out_of_range_raises():
    with pytest.raises(ValueError, match="latitude"):
        PointRelease(
            time="202301011200", longitude=-111.85, latitude=-95.0, altitude=5.0
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("longitude", np.nan, "longitude must be a number"),
        ("latitude", np.nan, "latitude must be a number"),
        ("longitude", np.inf, "longitude must be a number"),
        ("altitude", np.nan, "altitude must be a finite number"),
        ("altitude", np.inf, "altitude must be a finite number"),
    ],
)
def test_a_point_with_a_nan_or_infinite_coordinate_is_refused(field, value, message):
    """A NaN passed the range checks and failed later at ``.id``, naming no field (#440)."""
    point = {"longitude": -111.85, "latitude": 40.77, "altitude": 5.0}
    with pytest.raises(ValueError, match=message):
        PointRelease(time="202301011200", **{**point, field: value})


@pytest.mark.parametrize(("bottom", "top"), [(np.nan, 100.0), (0.0, np.inf)])
def test_a_column_with_a_nan_or_infinite_height_is_refused(bottom, top):
    with pytest.raises(ValueError, match="altitude must be a finite number"):
        ColumnRelease(
            time="202301011200",
            longitude=-111.85,
            latitude=40.77,
            bottom=bottom,
            top=top,
        )


def test_a_multipoint_release_with_a_nan_point_is_refused():
    with pytest.raises(ValueError, match="longitude must be a number"):
        MultiPointRelease(
            time="202301011200",
            longitudes=[-111.0, np.nan],
            latitudes=[40.0, 40.5],
            altitudes=[10.0, 20.0],
        )


def test_a_releases_file_with_an_empty_longitude_names_the_row():
    text = (
        "time,lon,lat,z\n2023-07-15 18:00,-111.8,40.7,10\n2023-07-15 18:00,,40.7,10\n"
    )
    with pytest.raises(ValueError, match="row 1: longitude must be a number"):
        read_releases(io.StringIO(text))


def test_negative_agl_altitude_raises():
    with pytest.raises(ValueError, match="AGL altitudes"):
        PointRelease(
            time="202301011200", longitude=-111.85, latitude=40.77, altitude=-1.0
        )


def test_negative_msl_altitude_is_allowed():
    r = PointRelease(
        time="202301011200",
        longitude=-111.85,
        latitude=40.77,
        altitude=-50.0,
        altitude_ref="msl",
    )
    assert r.altitude == -50.0
    assert r.altitude_ref == "msl"


def test_column_bottom_ge_top_raises():
    with pytest.raises(ValueError, match="bottom"):
        ColumnRelease(
            time="202301011200", longitude=-111.85, latitude=40.77, bottom=50.0, top=5.0
        )


def test_column_bottom_eq_top_raises():
    with pytest.raises(ValueError, match="bottom"):
        ColumnRelease(
            time="202301011200", longitude=-111.85, latitude=40.77, bottom=5.0, top=5.0
        )


def test_multipoint_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        MultiPointRelease(
            time="202301011200",
            longitudes=[-111.85, -111.86],
            latitudes=[40.77],
            altitudes=[5.0, 5.0],
        )


# ---------------------------------------------------------------------------
# isinstance type checks
# ---------------------------------------------------------------------------


def test_isinstance_point(point_release):
    assert isinstance(point_release, PointRelease)


def test_isinstance_column(column_release):
    assert isinstance(column_release, ColumnRelease)


def test_isinstance_multipoint(multipoint_release):
    assert isinstance(multipoint_release, MultiPointRelease)


# ---------------------------------------------------------------------------
# Release.from_points
# ---------------------------------------------------------------------------


def test_release_from_points_single_makes_point():
    r = Release.from_points("202301011200", [(-111.85, 40.77, 5)])
    assert isinstance(r, PointRelease)
    assert r.location_id == "-111.85_40.77_5"


def test_release_from_points_two_same_xy_makes_column():
    r = Release.from_points("202301011200", [(-111.85, 40.77, 5), (-111.85, 40.77, 50)])
    assert isinstance(r, ColumnRelease)
    assert r.location_id.endswith("_X5-50")


def test_release_from_points_two_different_xy_makes_multipoint():
    r = Release.from_points("202301011200", [(-111.85, 40.77, 5), (-111.86, 40.78, 5)])
    assert isinstance(r, MultiPointRelease)


def test_release_from_points_empty_raises():
    with pytest.raises(ValueError, match="least one"):
        Release.from_points("202301011200", [])


def test_release_from_points_column_sorts_bottom_top():
    r = Release.from_points("202301011200", [(-111.85, 40.77, 50), (-111.85, 40.77, 5)])
    assert isinstance(r, ColumnRelease)
    assert r.bottom == 5.0
    assert r.top == 50.0


# ---------------------------------------------------------------------------
# Release.from_dict round-trips
# ---------------------------------------------------------------------------


def test_release_from_dict_point_round_trip():
    r = PointRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    r2 = Release.from_dict(r.to_dict())
    assert isinstance(r2, PointRelease)
    assert r2 == r


def test_release_from_dict_column_round_trip():
    r = ColumnRelease(
        time="202301011200", longitude=-111.85, latitude=40.77, bottom=5.0, top=50.0
    )
    r2 = Release.from_dict(r.to_dict())
    assert isinstance(r2, ColumnRelease)
    assert r2 == r


def test_release_from_dict_multipoint_round_trip():
    r = MultiPointRelease(
        time="202301011200",
        longitudes=[-111.85, -111.86],
        latitudes=[40.77, 40.78],
        altitudes=[5.0, 5.0],
    )
    r2 = Release.from_dict(r.to_dict())
    assert isinstance(r2, MultiPointRelease)
    assert r2 == r


# ---------------------------------------------------------------------------
# read_releases
# ---------------------------------------------------------------------------


def test_read_releases_missing_required_columns_raises(tmp_path):
    csv = tmp_path / "releases.csv"
    csv.write_text("time,lat,lon\n2023-01-01 12:00:00,40.77,-111.85\n")
    with pytest.raises(ValueError, match="must contain columns"):
        read_releases(csv)


def test_read_releases_basic(tmp_path):
    csv = tmp_path / "releases.csv"
    csv.write_text("time,lati,long,zagl\n2023-01-01 12:00:00,40.77,-111.85,5.0\n")
    releases = read_releases(csv)
    assert len(releases) == 1
    assert isinstance(releases[0], PointRelease)
    assert releases[0].latitude == 40.77
    assert releases[0].longitude == -111.85
    assert releases[0].altitude == 5.0
    assert releases[0].altitude_ref == "agl"


def test_read_releases_multiple_rows(tmp_path):
    csv = tmp_path / "releases.csv"
    csv.write_text(
        "time,lati,long,zagl\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0\n"
        "2023-01-01 13:00:00,40.78,-111.86,5.0\n"
    )
    releases = read_releases(csv)
    assert len(releases) == 2


def test_read_releases_multipoint_via_group(tmp_path):
    csv = tmp_path / "releases.csv"
    csv.write_text(
        "time,lati,long,zagl,group\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0,0\n"
        "2023-01-01 12:00:00,40.78,-111.86,5.0,0\n"
        "2023-01-01 13:00:00,40.79,-111.87,5.0,1\n"
    )
    releases = read_releases(csv)
    assert len(releases) == 2
    assert isinstance(releases[0], MultiPointRelease)
    assert len(releases[0].coords()) == 2


def test_read_releases_refuses_the_old_group_column_name(tmp_path):
    """``r_idx`` is not read: a file that has it is told to say ``group``."""
    csv = tmp_path / "releases.csv"
    csv.write_text(
        "time,lati,long,zagl,r_idx\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0,0\n"
        "2023-01-01 12:00:00,40.78,-111.86,5.0,0\n"
    )
    with pytest.raises(ValueError, match="r_idx.*Rename it to 'group'"):
        read_releases(csv)


def test_read_releases_group_with_mixed_times_raises(tmp_path):
    csv = tmp_path / "releases.csv"
    csv.write_text(
        "time,lati,long,zagl,group\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0,0\n"
        "2023-01-01 13:00:00,40.78,-111.86,5.0,0\n"
    )
    with pytest.raises(ValueError, match="same release time"):
        read_releases(csv)


def test_read_releases_alt_column_names(tmp_path):
    csv = tmp_path / "releases.csv"
    csv.write_text(
        "time,latitude,longitude,altitude\n2023-01-01 12:00:00,40.77,-111.85,5.0\n"
    )
    releases = read_releases(csv)
    assert len(releases) == 1
    assert releases[0].latitude == 40.77


def test_read_releases_zmsl_infers_msl(tmp_path):
    csv = tmp_path / "releases.csv"
    csv.write_text("time,lati,long,zmsl\n2023-01-01 12:00:00,40.77,-111.85,1500.0\n")
    releases = read_releases(csv)
    assert len(releases) == 1
    assert releases[0].altitude_ref == "msl"


# ---------------------------------------------------------------------------
# MultiPointRelease - duplicate horizontal locations are rejected
# ---------------------------------------------------------------------------


def test_multipoint_rejects_stacked_heights_at_one_location():
    # HYSPLIT chains same-lat/lon release locations into one vertical line
    # source and releases only between the last two heights.
    with pytest.raises(ValueError, match="distinct horizontal"):
        MultiPointRelease(
            time="202508110440",
            longitudes=[-76.68869] * 5,
            latitudes=[37.76396] * 5,
            altitudes=[100.0, 200.0, 300.0, 400.0, 500.0],
        )


def test_multipoint_rejects_nonconsecutive_duplicate_location():
    with pytest.raises(ValueError, match="distinct horizontal"):
        MultiPointRelease(
            time="202301011200",
            longitudes=[-111.85, -111.86, -111.85],
            latitudes=[40.77, 40.78, 40.77],
            altitudes=[100.0, 200.0, 300.0],
        )


def test_read_releases_stacked_group_reports_its_group(tmp_path):
    csv = tmp_path / "releases.csv"
    csv.write_text(
        "group,time,longitude,latitude,altitude,altitude_ref\n"
        + "".join(
            f"0,2025-08-11 04:40:50,-76.688690186,37.763957977,{z},agl\n"
            for z in (100.0, 200.0, 300.0, 400.0, 500.0)
        )
    )
    with pytest.raises(ValueError, match="group=0.*distinct horizontal"):
        read_releases(csv)


# ---------------------------------------------------------------------------
# releases_to_csv / write_releases round trips
# ---------------------------------------------------------------------------


def test_releases_to_csv_header_and_rows(point_release, column_release):
    text = releases_to_csv([point_release, column_release])
    lines = text.splitlines()

    assert lines[0] == "group,time,longitude,latitude,altitude,altitude_ref"
    # One row per constituent point: 1 (point) + 2 (column).
    assert len(lines) == 1 + 1 + 2
    assert [line.split(",")[0] for line in lines[1:]] == ["0", "1", "1"]
    assert all(line.endswith(",agl") for line in lines[1:])


def test_releases_to_csv_empty_has_only_header():
    assert releases_to_csv([]).splitlines() == [
        "group,time,longitude,latitude,altitude,altitude_ref"
    ]


def test_write_releases_round_trip_point(tmp_path, point_release):
    path = write_releases([point_release], tmp_path / "releases.csv")
    assert path == tmp_path / "releases.csv"
    assert read_releases(path) == [point_release]


def test_write_releases_round_trip_column(tmp_path, column_release):
    loaded = read_releases(write_releases([column_release], tmp_path / "r.csv"))
    assert loaded == [column_release]
    assert isinstance(loaded[0], ColumnRelease)


def test_write_releases_round_trip_multipoint(tmp_path, multipoint_release):
    loaded = read_releases(write_releases([multipoint_release], tmp_path / "r.csv"))
    assert loaded == [multipoint_release]
    assert isinstance(loaded[0], MultiPointRelease)


def test_write_releases_round_trip_mixed_preserves_order_and_types(
    tmp_path, point_release, column_release, multipoint_release
):
    original = [multipoint_release, point_release, column_release]
    loaded = read_releases(write_releases(original, tmp_path / "r.csv"))

    assert loaded == original
    assert [type(r) for r in loaded] == [
        MultiPointRelease,
        PointRelease,
        ColumnRelease,
    ]


def test_write_releases_round_trip_preserves_msl_reference(tmp_path):
    original = [
        PointRelease(
            time="202301011200",
            longitude=-111.85,
            latitude=40.77,
            altitude=1500.0,
            altitude_ref="msl",
        ),
        ColumnRelease(
            time="202301011200",
            longitude=-111.85,
            latitude=40.77,
            bottom=1300.0,
            top=1800.0,
            altitude_ref="msl",
        ),
    ]
    loaded = read_releases(write_releases(original, tmp_path / "r.csv"))
    assert loaded == original
    assert all(r.altitude_ref == "msl" for r in loaded)


def test_write_releases_creates_parent_directories(tmp_path, point_release):
    path = write_releases([point_release], tmp_path / "nested" / "dir" / "r.csv")
    assert path.is_file()
    assert read_releases(path) == [point_release]


def test_read_releases_keeps_extra_columns_as_attrs(tmp_path):
    csv = tmp_path / "releases.csv"
    csv.write_text(
        "time,lati,long,zagl,group,Scene,note\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0,0,A,\n"
        "2023-01-01 12:00:00,40.78,-111.86,5.0,0,A,\n"
        "2023-01-01 13:00:00,40.79,-111.87,5.0,1,B,hello\n"
    )
    multi, point = read_releases(csv)
    assert isinstance(multi, MultiPointRelease)
    assert multi.attrs == {"Scene": "A", "note": None}
    assert point.attrs == {"Scene": "B", "note": "hello"}


def test_release_attrs_round_trip_through_csv(tmp_path, point_release, column_release):
    from stilt.releases.table import append_releases_csv, releases_to_csv

    labelled = point_release.model_copy(update={"attrs": {"scene": "A"}})
    text = releases_to_csv([labelled, column_release])
    assert text.splitlines()[0].endswith(",altitude_ref,scene")
    back = read_releases(io.StringIO(text))
    assert back[0].attrs == {"scene": "A"}
    assert back[1].attrs == {"scene": None}

    extra = PointRelease(
        time="2023-02-01 00:00",
        longitude=-111.0,
        latitude=40.0,
        altitude=1.0,
        attrs={"scene": "C", "ignored": 1},
    )
    grown = read_releases(io.StringIO(append_releases_csv(text, [extra])))
    assert grown[-1].attrs == {"scene": "C"}


# ---------------------------------------------------------------------------
# Releases as frozen models
# ---------------------------------------------------------------------------


def test_releases_are_frozen_values(point_release):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="frozen"):
        point_release.altitude = 20.0
    twin = PointRelease(
        time=point_release.time,
        longitude=point_release.longitude,
        latitude=point_release.latitude,
        altitude=point_release.altitude,
    )
    assert twin == point_release and hash(twin) == hash(point_release)
    assert len({twin, point_release}) == 1


def test_attrs_do_not_count_for_equality_or_the_dict(point_release):
    labelled = point_release.model_copy(update={"attrs": {"site": "WBB"}})
    assert labelled == point_release
    assert hash(labelled) == hash(point_release)
    assert "attrs" not in labelled.to_dict()
    assert labelled.id == point_release.id


def test_unknown_field_is_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="height"):
        Release.from_dict(
            {
                "kind": "point",
                "time": "2023-01-01T12:00:00",
                "longitude": -111.85,
                "latitude": 40.77,
                "altitude": 5,
                "height": 5,
            }
        )


def test_from_dict_needs_the_kind_key():
    """A dict without ``kind``, such as the ``type`` form older versions wrote, is refused."""
    old = {
        "type": "ColumnRelease",
        "time": "2023-01-01T12:00:00",
        "longitude": -111.85,
        "latitude": 40.77,
        "bottom": 0.0,
        "top": 3000.0,
        "altitude_ref": "agl",
    }
    with pytest.raises(ValueError, match="'kind'"):
        Release.from_dict(old)
    r = Release.from_dict(
        {**{k: v for k, v in old.items() if k != "type"}, "kind": "column"}
    )
    assert isinstance(r, ColumnRelease) and r.top == 3000.0
    assert Release.from_dict(r.to_dict()) == r


def test_parse_release_id_splits_time_and_location(
    point_release, column_release, multipoint_release
):
    from stilt.releases import parse_release_id

    for r in (point_release, column_release, multipoint_release):
        assert parse_release_id(r.id) == (r.time, r.location_id)
    for bad in ("nope", "202301011200", "202301011200_", "202313011200_-111_40_5"):
        with pytest.raises(ValueError):
            parse_release_id(bad)


# ---------------------------------------------------------------------------
# Multipoint ids and altitude (#50)
# ---------------------------------------------------------------------------


def _multi(alts, **kwargs):
    return MultiPointRelease(
        time="2023-01-01 12:00",
        longitudes=[-111.85, -111.86],
        latitudes=[40.77, 40.78],
        altitudes=alts,
        **kwargs,
    )


def test_multipoint_ids_tell_sub_metre_heights_apart():
    """Heights under a metre apart used to hash the same and share result files."""
    low, high = _multi([10.2, 500.0]), _multi([10.9, 500.0])
    assert low != high
    assert low.id != high.id
    assert _multi([10.2, 500.0]).id == low.id


def test_multipoint_id_of_whole_metre_heights_is_unchanged():
    """Whole-metre heights keep the id earlier versions gave them."""
    import hashlib
    import json

    r = _multi([4.0, 4.0])
    legacy = json.dumps(
        sorted([[-111.85, 40.77, 4], [-111.86, 40.78, 4]]), separators=(",", ":")
    )
    assert r.location_id == "multi_" + hashlib.sha256(legacy.encode()).hexdigest()[:10]


def test_releases_that_share_an_id_are_refused():
    """Two releases closer than the id resolves would overwrite each other."""
    from stilt.releases.table import check_distinct_ids

    a, b = _multi([10.001, 500.0]), _multi([10.004, 500.0])
    assert a != b and a.id == b.id
    with pytest.raises(ValueError, match="share the id"):
        check_distinct_ids([a, b])
    check_distinct_ids([a, _multi([10.001, 500.0])])  # the same release twice is fine

    text = releases_to_csv([a, b])
    with pytest.raises(ValueError, match="share the id"):
        read_releases(io.StringIO(text))


# ---------------------------------------------------------------------------
# The release table
# ---------------------------------------------------------------------------


def test_frame_has_a_row_per_release_point(
    point_release, column_release, multipoint_release
):
    from stilt.releases import releases_from_frame, releases_to_frame

    releases = [
        point_release.model_copy(update={"attrs": {"site": "WBB"}}),
        column_release,
        multipoint_release,
    ]
    frame = releases_to_frame(releases)
    assert list(frame.columns) == [
        "group",
        "time",
        "longitude",
        "latitude",
        "altitude",
        "altitude_ref",
        "site",
    ]
    assert len(frame) == 1 + 2 + len(multipoint_release.coords())
    assert frame["group"].tolist()[:3] == [0, 1, 1]
    assert str(frame["time"].dtype).startswith("datetime64")

    back = releases_from_frame(frame)
    assert back == releases
    assert back[0].attrs == {"site": "WBB"}
    assert releases_to_frame([]).empty


def test_frame_accepts_the_alternate_column_names():
    from stilt.releases import releases_from_frame

    frame = pd.DataFrame(
        {
            "Time": ["2023-01-01 12:00"],
            "LON": [-111.85],
            "lati": [40.77],
            "zmsl": [1500.0],
            "Scene": ["A"],
        }
    )
    [r] = releases_from_frame(frame)
    assert isinstance(r, PointRelease)
    assert (r.longitude, r.latitude, r.altitude) == (-111.85, 40.77, 1500.0)
    assert r.altitude_ref == "msl"
    assert r.attrs == {"Scene": "A"}


# ---------------------------------------------------------------------------
# Appending keeps the altitude reference (#51)
# ---------------------------------------------------------------------------


def test_append_msl_release_to_plain_altitude_file_keeps_its_reference():
    """An MSL release appended to a file with no reference column read back as AGL."""
    from stilt.releases.table import append_releases_csv

    text = "time,longitude,latitude,altitude\n2023-01-01 12:00:00,-111.85,40.77,5\n"
    msl = PointRelease(
        time="2023-01-02 12:00",
        longitude=-111.9,
        latitude=40.7,
        altitude=1500.0,
        altitude_ref="msl",
    )

    grown = append_releases_csv(text, [msl])

    assert grown.splitlines()[0] == "time,longitude,latitude,altitude,altitude_ref"
    assert grown.splitlines()[1] == "2023-01-01 12:00:00,-111.85,40.77,5,agl"
    first, second = read_releases(io.StringIO(grown))
    assert first.altitude_ref == "agl"
    assert second == msl and second.altitude_ref == "msl"


def test_append_agl_release_leaves_a_plain_file_without_the_column():
    from stilt.releases.table import append_releases_csv

    text = "time,lon,lat,z,site\n2023-01-01 12:00:00,-111.85,40.77,5,WBB\n"
    agl = PointRelease(
        time="2023-01-02 12:00",
        longitude=-111.9,
        latitude=40.7,
        altitude=10.0,
        attrs={"site": "UOU"},
    )

    grown = append_releases_csv(text, [agl])

    assert grown.splitlines()[0] == "time,lon,lat,z,site"
    assert grown.splitlines()[1] == "2023-01-01 12:00:00,-111.85,40.77,5,WBB"
    assert grown.splitlines()[2] == "2023-01-02 12:00:00,-111.9,40.7,10.0,UOU"


def test_append_to_a_zagl_file_still_refuses_an_msl_release():
    from stilt.releases.table import append_releases_csv

    text = "time,long,lati,zagl\n2023-01-01 12:00:00,-111.85,40.77,5\n"
    msl = PointRelease(
        time="2023-01-02 12:00",
        longitude=-111.9,
        latitude=40.7,
        altitude=1500.0,
        altitude_ref="msl",
    )
    with pytest.raises(ValueError, match="altitudes are agl"):
        append_releases_csv(text, [msl])


# ---------------------------------------------------------------------------
# Reading a table without building releases
# ---------------------------------------------------------------------------


def test_release_rows_ids_match_the_releases(
    point_release, column_release, multipoint_release
):
    """The table's ids come from the same rules as release.id, for every kind."""
    from stilt.releases.table import (
        release_rows,
        releases_from_rows,
        releases_to_frame,
    )

    sub_metre = MultiPointRelease(
        time="2023-01-01 13:00",
        longitudes=[-111.85, -111.86],
        latitudes=[40.77, 40.78],
        altitudes=[10.25, 500.0],
        altitude_ref="msl",
    )
    msl_point = point_release.model_copy(update={"altitude_ref": "msl"})
    msl_column = column_release.model_copy(update={"altitude_ref": "msl"})
    releases = [
        point_release,
        column_release,
        multipoint_release,
        sub_metre,
        msl_point,
        msl_column,
    ]

    rows = release_rows(releases_to_frame(releases))

    first = rows.drop_duplicates("release")
    assert first["release"].tolist() == [r.id for r in releases]
    assert first["kind"].tolist() == [
        "point",
        "column",
        "multipoint",
        "multipoint",
        "point",
        "column",
    ]
    assert first["location"].tolist() == [r.location_id for r in releases]
    assert releases_from_rows(rows) == releases


def test_release_rows_column_id_does_not_depend_on_row_order():
    """A column listed top row first gets the id its release has."""
    from stilt.releases.table import release_rows

    frame = pd.DataFrame(
        {
            "group": [1, 1],
            "time": ["2021-01-15 06:00", "2021-01-15 06:00"],
            "longitude": [-112.0, -112.0],
            "latitude": [40.5, 40.5],
            "altitude": [3000.0, 0.0],
        }
    )
    rows = release_rows(frame)
    assert set(rows["release"]) == {"202101150600_-112_40.5_X0-3000"}


def test_release_rows_checks_every_row_without_building(tmp_path):
    from stilt.releases.table import release_rows

    def table(**columns):
        base = {
            "time": ["2023-01-01 12:00"],
            "longitude": [-111.85],
            "latitude": [40.77],
            "altitude": [5.0],
        }
        return pd.DataFrame({**base, **columns})

    with pytest.raises(ValueError, match="row 0: latitude must be a number within"):
        release_rows(table(latitude=[95.0]))
    with pytest.raises(ValueError, match="AGL altitudes must be >= 0"):
        release_rows(table(altitude=[-1.0]))
    with pytest.raises(ValueError, match="column named \\['kind'\\]"):
        release_rows(table(kind=["tower"]))
    column = pd.DataFrame(
        {
            "group": [7, 7],
            "time": ["2023-01-01 12:00"] * 2,
            "longitude": [-111.85] * 2,
            "latitude": [40.77] * 2,
            "altitude": [5.0, 5.0],
        }
    )
    with pytest.raises(ValueError, match="group=7: 'bottom' must be less than 'top'"):
        release_rows(column)


def test_a_release_listed_twice_is_kept_once(point_release):
    from stilt.releases.table import release_rows, releases_to_frame

    rows = release_rows(releases_to_frame([point_release, point_release]))

    assert rows["release"].tolist() == [point_release.id]


def test_many_releases_listed_twice_are_checked_quickly():
    """Each repeated id was built through a mask over the whole table: 31 s here (#191)."""
    import time

    from stilt.releases.table import release_rows

    once = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=10_000, freq="h"),
            "longitude": -111.85,
            "latitude": 40.77,
            "altitude": 5.0,
        }
    )
    start = time.perf_counter()
    rows = release_rows(pd.concat([once, once], ignore_index=True))
    assert time.perf_counter() - start < 10
    assert rows["release"].is_unique and len(rows) == 10_000


def test_a_repeated_id_among_many_still_names_two_different_releases():
    from stilt.releases.table import release_rows, releases_to_frame

    a, b = _multi([10.001, 500.0]), _multi([10.004, 500.0])
    points = [
        PointRelease(
            time=dt.datetime(2024, 1, 1, h),
            longitude=-111.85,
            latitude=40.77,
            altitude=5,
        )
        for h in range(3)
    ]
    frame = releases_to_frame([*points, a, *points, b])

    with pytest.raises(ValueError, match="share the id"):
        release_rows(frame)


def test_releases_from_checked_rows_are_not_checked_again(
    monkeypatch, point_release, column_release, multipoint_release
):
    """The table runs a release's checks once; building from its rows skips them."""
    import stilt.releases.models as models
    from stilt.releases.table import (
        release_rows,
        releases_from_rows,
        releases_to_frame,
    )

    releases = [point_release, column_release, multipoint_release]
    rows = release_rows(releases_to_frame(releases))

    def refuse(*args, **kwargs):
        raise AssertionError("a checked row was checked again")

    monkeypatch.setattr(models, "point_errors", refuse)
    built = releases_from_rows(rows)

    assert built == releases
    assert [r.id for r in built] == [r.id for r in releases]
    assert [r.geometry.geom_type for r in built] == [
        "Point",
        "LineString",
        "MultiPoint",
    ]


def _multipoint_location_as_first_written(lons, lats, alts, altitude_ref):
    """The multipoint id as it was written before #148, kept to check the faster one."""

    def hash_altitude(alt):
        rounded = round(alt, 2)
        return int(rounded) if rounded == int(rounded) else rounded

    points = sorted(zip(lons, lats, alts, strict=True))
    canonical = json.dumps(
        [
            [round(float(lon), 5), round(float(lat), 5), hash_altitude(float(alt))]
            for lon, lat, alt in points
        ],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:10]
    return f"multi_{digest}{'msl' if altitude_ref == 'msl' else ''}"


def test_multipoint_ids_are_the_same_as_first_written():
    """Ids name result files, so the faster id must give the same bytes."""
    from stilt.releases.models import _multipoint_location, _multipoint_locations

    rng = np.random.default_rng(148)
    groups, lons, lats, alts, refs = [], [], [], [], []
    for g in range(300):
        n = int(rng.integers(2, 40))
        lon = -112 + rng.uniform(-1, 1, n)
        lat = 40.7 + rng.uniform(-1, 1, n)
        alt = rng.choice([rng.uniform(0, 3000, n), np.round(rng.uniform(0, 3000, n))])
        if g % 3 == 0:
            # Values on and next to the rounding halves, and repeated points.
            lon = np.round(lon, 5) + 0.000005 * rng.choice([-1, 1], n)
            alt = np.round(alt, 2) + 0.005
            lon[1:] = np.where(rng.random(n - 1) < 0.2, lon[0], lon[1:])
        groups += [g] * n
        lons += lon.tolist()
        lats += lat.tolist()
        alts += alt.tolist()
        refs.append("msl" if g % 2 else "agl")

    found = _multipoint_locations(np.array(groups), lons, lats, alts, refs)
    g = np.array(groups)
    for group, location in found.items():
        mine = g == group
        args = (
            np.array(lons)[mine].tolist(),
            np.array(lats)[mine].tolist(),
            np.array(alts)[mine].tolist(),
            refs[group],
        )
        assert location == _multipoint_location_as_first_written(*args)
        assert location == _multipoint_location(*args)


@pytest.mark.parametrize("name", ["alt_ref", "height_reference", "ref"])
def test_a_misnamed_altitude_ref_column_is_refused(name):
    """Kept as a label, it left 1500 m above sea level as 1500 m above ground (#440)."""
    text = f"time,lon,lat,z,{name}\n2023-07-15 18:00,-111.8,40.7,1500,MSL\n"
    with pytest.raises(ValueError, match=f"'{name}'.*Did you mean 'altitude_ref'"):
        read_releases(io.StringIO(text))


def test_a_label_column_that_is_not_a_reference_is_kept():
    text = "time,lon,lat,z,site,altitude_ref,note\n2023-07-15 18:00,-111.8,40.7,1500,WBB,msl,agl\n"
    (release,) = read_releases(io.StringIO(text))
    assert release.altitude_ref == "msl"
    assert release.attrs == {"site": "WBB", "note": "agl"}


def test_a_table_of_numeric_times_is_refused():
    from stilt.releases.table import release_rows

    frame = pd.DataFrame(
        {
            "time": [202307151800],
            "longitude": [-111.8],
            "latitude": [40.7],
            "altitude": [10.0],
        }
    )
    with pytest.raises(ValueError, match="row 0: time must be ISO 8601.*the number"):
        release_rows(frame)


def test_a_table_of_datetimes_with_a_missing_one_names_the_row():
    from stilt.releases.table import release_rows

    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2023-07-15 18:00", None]),
            "longitude": [-111.8, -111.7],
            "latitude": [40.7, 40.7],
            "altitude": [10.0, 10.0],
        }
    )
    with pytest.raises(ValueError, match="row 1: time is empty"):
        release_rows(frame)
