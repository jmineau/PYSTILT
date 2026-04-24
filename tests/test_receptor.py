"""Tests for stilt.receptor - Receptor, read_receptors."""

import datetime as dt
import hashlib
import json

import pytest

from stilt.receptor import Receptor, _format_coord, read_receptors

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
# location_id - point
# ---------------------------------------------------------------------------


def test_location_id_point_basic():
    r = Receptor("202301011200", -111.85, 40.77, 5.0)
    assert r.id.location == "-111.85_40.77_5"


def test_location_id_point_integer_coords():
    r = Receptor("202301011200", -112.0, 40.0, 10.0)
    assert r.id.location == "-112_40_10"


def test_location_id_point_fractional_height():
    r = Receptor("202301011200", -111.85, 40.77, 2.5)
    assert r.id.location == "-111.85_40.77_2.5"


# ---------------------------------------------------------------------------
# location_id - column
# ---------------------------------------------------------------------------


def test_location_id_column_ends_with_X():
    r = Receptor("202301011200", -111.85, 40.77, [5.0, 50.0])
    assert r.id.location == "-111.85_40.77_X"


def test_location_id_column_integer_coords():
    r = Receptor("202301011200", -112.0, 40.0, [5.0, 50.0])
    assert r.id.location == "-112_40_X"


# ---------------------------------------------------------------------------
# location_id - multipoint (SHA-256 hash, order-independent)
# ---------------------------------------------------------------------------


def test_location_id_multipoint_stable():
    """Same points always produce the same hash."""
    lons = [-111.85, -111.86, -111.84]
    lats = [40.77, 40.78, 40.76]
    hgts = [5, 5, 5]
    r1 = Receptor("202301011200", lons, lats, hgts)
    r2 = Receptor("202301011200", lons, lats, hgts)
    assert r1.id.location == r2.id.location


def test_location_id_multipoint_order_independent():
    """Hash must not depend on insertion order."""
    r_a = Receptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    r_b = Receptor("202301011200", [-111.86, -111.85], [40.78, 40.77], [5, 5])
    assert r_a.id.location == r_b.id.location


def test_location_id_multipoint_starts_with_multi():
    r = Receptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    assert r.id.location.startswith("multi_")


def test_location_id_multipoint_hash_length():
    """Hash portion is exactly 10 hex chars."""
    r = Receptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    hash_part = r.id.location.replace("multi_", "")
    assert len(hash_part) == 10
    assert all(c in "0123456789abcdef" for c in hash_part)


def test_location_id_multipoint_matches_spec():
    """Verify hash matches the canonical JSON spec independently."""
    pts = [(-111.85, 40.77, 5), (-111.86, 40.78, 5)]
    pts_sorted = sorted(pts)
    canonical = json.dumps(
        [[round(lon, 5), round(lat, 5), int(zagl)] for lon, lat, zagl in pts_sorted],
        separators=(",", ":"),
    )
    expected_hash = hashlib.sha256(canonical.encode()).hexdigest()[:10]
    r = Receptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    assert r.id.location == f"multi_{expected_hash}"


def test_location_id_multipoint_differs_for_different_points():
    r_a = Receptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    r_b = Receptor("202301011200", [-111.85, -111.87], [40.77, 40.79], [5, 5])
    assert r_a.id.location != r_b.id.location


# ---------------------------------------------------------------------------
# Kind inference
# ---------------------------------------------------------------------------


def test_single_point_makes_point():
    r = Receptor("202301011200", -111.85, 40.77, 5)
    assert r.kind == "point"


def test_two_heights_same_xy_makes_column():
    r = Receptor("202301011200", -111.85, 40.77, [5, 50])
    assert r.kind == "column"


def test_two_different_xy_makes_multipoint():
    r = Receptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    assert r.kind == "multipoint"


# ---------------------------------------------------------------------------
# from_column
# ---------------------------------------------------------------------------


def test_from_column_kind():
    r = Receptor.from_column("202301011200", -111.85, 40.77, bottom=5.0, top=50.0)
    assert r.kind == "column"


def test_from_column_top_bottom():
    r = Receptor.from_column("202301011200", -111.85, 40.77, bottom=5.0, top=50.0)
    assert r.bottom == 5.0
    assert r.top == 50.0


def test_from_column_id():
    r = Receptor.from_column("202301011200", -111.85, 40.77, bottom=5.0, top=50.0)
    assert r.id.location == "-111.85_40.77_X"


def test_from_column_bottom_ge_top_raises():
    with pytest.raises(ValueError, match="bottom"):
        Receptor.from_column("202301011200", -111.85, 40.77, bottom=50.0, top=5.0)


def test_from_column_bottom_eq_top_raises():
    with pytest.raises(ValueError, match="bottom"):
        Receptor.from_column("202301011200", -111.85, 40.77, bottom=5.0, top=5.0)


# ---------------------------------------------------------------------------
# from_points
# ---------------------------------------------------------------------------


def test_from_points_single_makes_point():
    r = Receptor.from_points("202301011200", [(-111.85, 40.77, 5)])
    assert r.kind == "point"
    assert r.id.location == "-111.85_40.77_5"


def test_from_points_two_same_xy_makes_column():
    r = Receptor.from_points(
        "202301011200", [(-111.85, 40.77, 5), (-111.85, 40.77, 50)]
    )
    assert r.kind == "column"
    assert r.id.location.endswith("_X")


def test_from_points_two_different_xy_makes_multipoint():
    r = Receptor.from_points("202301011200", [(-111.85, 40.77, 5), (-111.86, 40.78, 5)])
    assert r.kind == "multipoint"


def test_from_points_empty_raises():
    with pytest.raises(ValueError, match="least one"):
        Receptor.from_points("202301011200", [])


# ---------------------------------------------------------------------------
# Receptor
# ---------------------------------------------------------------------------


def test_receptor_kind_point(point_receptor):
    assert point_receptor.kind == "point"


def test_receptor_kind_column(column_receptor):
    assert column_receptor.kind == "column"


def test_receptor_kind_multipoint(multipoint_receptor):
    assert multipoint_receptor.kind == "multipoint"


def test_receptor_id_format(point_receptor):
    assert point_receptor.id == "202301011200_-111.85_40.77_5"


def test_receptor_len_point(point_receptor):
    assert len(point_receptor) == 1


def test_receptor_len_column(column_receptor):
    assert len(column_receptor) == 2


def test_receptor_len_multipoint(multipoint_receptor):
    assert len(multipoint_receptor) == 3


def test_receptor_coordinates(point_receptor):
    assert point_receptor.longitude == -111.85
    assert point_receptor.latitude == 40.77
    assert point_receptor.altitude == 5.0
    assert point_receptor.altitude_ref == "agl"


def test_receptor_column_top_bottom(column_receptor):
    assert column_receptor.bottom == 5.0
    assert column_receptor.top == 50.0


def test_receptor_point_bottom_is_none(point_receptor):
    with pytest.raises(AttributeError):
        _ = point_receptor.bottom
    with pytest.raises(AttributeError):
        _ = point_receptor.top


def test_receptor_time_iso_string():
    r = Receptor(
        time="2023-01-01T12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2023, 1, 1, 12, 0)


def test_receptor_time_compact_string():
    r = Receptor(
        time="202301011200",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2023, 1, 1, 12, 0)


def test_receptor_time_aware_input_normalizes_to_naive_utc():
    r = Receptor(
        time="2023-01-01T05:00:00-07:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2023, 1, 1, 12, 0)
    assert r.time.tzinfo is None


def test_receptor_scalar_altitude_makes_point():
    r = Receptor(time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0)
    assert r.kind == "point"


def test_receptor_two_altitudes_make_column():
    """Scalar lon/lat + 2-element altitude list -> column receptor."""
    r = Receptor(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=[5.0, 50.0]
    )
    assert r.kind == "column"


def test_receptor_list_coords_makes_multipoint():
    r = Receptor(
        time="202301011200",
        longitude=[-111.85, -111.86],
        latitude=[40.77, 40.78],
        altitude=[5.0, 5.0],
    )
    assert r.kind == "multipoint"
    assert len(r) == 2


def test_receptor_equality():
    r1 = Receptor(time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0)
    r2 = Receptor(time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0)
    assert r1 == r2


def test_receptor_inequality_time():
    r1 = Receptor(time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0)
    r2 = Receptor(time="202301011300", longitude=-111.85, latitude=40.77, altitude=5.0)
    assert r1 != r2


def test_receptor_init_rejects_missing_time():
    with pytest.raises(ValueError, match="'time' must be provided"):
        Receptor(time=None, longitude=-111.85, latitude=40.77, altitude=5.0)


def test_receptor_init_parses_iso_and_compact_strings():
    r_iso = Receptor(
        time="2023-01-01T12:00:00", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    r_compact = Receptor(
        time="202301011200", longitude=-111.85, latitude=40.77, altitude=5.0
    )

    expected = dt.datetime(2023, 1, 1, 12, 0)
    assert r_iso.time == expected
    assert r_compact.time == expected


def test_receptor_geometry_and_vertical_flag(point_receptor, column_receptor):
    assert len(point_receptor.points) == 1
    assert point_receptor.kind != "column"
    assert column_receptor.kind == "column"


def test_receptor_eq_non_receptor_is_false(point_receptor):
    assert (point_receptor == object()) is False


def test_receptor_points_property_yields_points(point_receptor):
    """points property yields shapely Points regardless of receptor kind."""
    items = list(point_receptor.points)
    assert len(items) == 1
    assert hasattr(items[0], "x") and items[0].x == pytest.approx(-111.85)
    assert items[0].y == pytest.approx(40.77)
    assert items[0].z == pytest.approx(5.0)


def test_receptor_lazy_geometry_point(point_receptor):
    """geometry property lazily creates shapely objects."""
    from shapely import Point

    assert isinstance(point_receptor.geometry, Point)


def test_receptor_lazy_geometry_column(column_receptor):
    from shapely import LineString

    assert isinstance(column_receptor.geometry, LineString)


def test_receptor_lazy_geometry_multipoint(multipoint_receptor):
    from shapely import MultiPoint

    assert isinstance(multipoint_receptor.geometry, MultiPoint)


# ---------------------------------------------------------------------------
# Public coordinate arrays
# ---------------------------------------------------------------------------


def test_public_arrays_point(point_receptor):
    import numpy as np

    assert isinstance(point_receptor.longitudes, np.ndarray)
    assert point_receptor.longitudes[0] == pytest.approx(-111.85)
    assert point_receptor.latitudes[0] == pytest.approx(40.77)
    assert point_receptor.altitudes[0] == pytest.approx(5.0)


def test_public_arrays_multipoint(multipoint_receptor):
    assert len(multipoint_receptor.longitudes) == 3
    assert len(multipoint_receptor.latitudes) == 3
    assert len(multipoint_receptor.altitudes) == 3
    assert len(multipoint_receptor.altitudes) == 3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_longitude_out_of_range_raises():
    with pytest.raises(ValueError, match="longitude"):
        Receptor("202301011200", 200.0, 40.77, 5.0)


def test_latitude_out_of_range_raises():
    with pytest.raises(ValueError, match="latitude"):
        Receptor("202301011200", -111.85, -95.0, 5.0)


def test_negative_agl_altitude_raises():
    with pytest.raises(ValueError, match="AGL altitudes"):
        Receptor("202301011200", -111.85, 40.77, -1.0)


def test_negative_msl_altitude_is_allowed():
    receptor = Receptor(
        "202301011200",
        -111.85,
        40.77,
        altitude=-50.0,
        altitude_ref="msl",
    )
    assert receptor.altitude == -50.0
    assert receptor.altitude_ref == "msl"


def test_bottom_raises_for_non_column(point_receptor):
    with pytest.raises(AttributeError):
        _ = point_receptor.bottom


def test_top_raises_for_non_column(point_receptor):
    with pytest.raises(AttributeError):
        _ = point_receptor.top


def test_altitude_raises_for_column(column_receptor):
    with pytest.raises(AttributeError):
        _ = column_receptor.altitude


def test_read_receptors_missing_required_columns_raises(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text("time,lat,lon\n2023-01-01 12:00:00,40.77,-111.85\n")

    with pytest.raises(ValueError, match="must contain columns"):
        read_receptors(csv)


# ---------------------------------------------------------------------------
# read_receptors
# ---------------------------------------------------------------------------


def test_read_receptors_basic(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text("time,lati,long,zagl\n2023-01-01 12:00:00,40.77,-111.85,5.0\n")
    receptors = read_receptors(csv)
    assert len(receptors) == 1
    assert receptors[0].latitude == 40.77
    assert receptors[0].longitude == -111.85
    assert receptors[0].altitude == 5.0
    assert receptors[0].altitude_ref == "agl"


def test_read_receptors_multiple_rows(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text(
        "time,lati,long,zagl\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0\n"
        "2023-01-01 13:00:00,40.78,-111.86,5.0\n"
    )
    receptors = read_receptors(csv)
    assert len(receptors) == 2


def test_read_receptors_multipoint_via_r_idx(tmp_path):
    """Rows sharing a r_idx id are assembled into a single multipoint receptor."""
    csv = tmp_path / "receptors.csv"
    csv.write_text(
        "time,lati,long,zagl,r_idx\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0,0\n"
        "2023-01-01 12:00:00,40.78,-111.86,5.0,0\n"
        "2023-01-01 13:00:00,40.79,-111.87,5.0,1\n"
    )
    receptors = read_receptors(csv)
    assert len(receptors) == 2
    assert receptors[0].kind == "multipoint"
    assert len(receptors[0]) == 2


def test_read_receptors_alt_column_names(tmp_path):
    """latitude/longitude/altitude column aliases are accepted."""
    csv = tmp_path / "receptors.csv"
    csv.write_text(
        "time,latitude,longitude,altitude\n2023-01-01 12:00:00,40.77,-111.85,5.0\n"
    )
    receptors = read_receptors(csv)
    assert len(receptors) == 1
    assert receptors[0].latitude == 40.77


def test_read_receptors_zmsl_infers_msl(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text("time,lati,long,zmsl\n2023-01-01 12:00:00,40.77,-111.85,1500.0\n")

    receptors = read_receptors(csv)

    assert len(receptors) == 1
    assert receptors[0].altitude == 1500.0
    assert receptors[0].altitude_ref == "msl"
