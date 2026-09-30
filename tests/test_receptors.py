"""Tests for stilt.receptors."""

import datetime as dt
import hashlib
import io
import json

import pandas as pd
import pytest

from stilt.receptors import (
    ColumnReceptor,
    MultiPointReceptor,
    PointReceptor,
    Receptor,
    _format_coord,
    read_receptors,
    receptors_to_csv,
    write_receptors,
)

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
# location_id - PointReceptor
# ---------------------------------------------------------------------------


def test_location_id_point_basic():
    r = PointReceptor("202301011200", -111.85, 40.77, 5.0)
    assert r.location_id == "-111.85_40.77_5"


def test_location_id_point_integer_coords():
    r = PointReceptor("202301011200", -112.0, 40.0, 10.0)
    assert r.location_id == "-112_40_10"


def test_location_id_point_fractional_height():
    r = PointReceptor("202301011200", -111.85, 40.77, 2.5)
    assert r.location_id == "-111.85_40.77_2.5"


# ---------------------------------------------------------------------------
# location_id - ColumnReceptor
# ---------------------------------------------------------------------------


def test_location_id_column_ends_with_X():
    r = ColumnReceptor("202301011200", -111.85, 40.77, 5.0, 50.0)
    assert r.location_id == "-111.85_40.77_X"


def test_location_id_column_integer_coords():
    r = ColumnReceptor("202301011200", -112.0, 40.0, 5.0, 50.0)
    assert r.location_id == "-112_40_X"


# ---------------------------------------------------------------------------
# location_id - MultiPointReceptor (SHA-256 hash, order-independent)
# ---------------------------------------------------------------------------


def test_location_id_multipoint_stable():
    lons = [-111.85, -111.86, -111.84]
    lats = [40.77, 40.78, 40.76]
    alts = [5, 5, 5]
    r1 = MultiPointReceptor("202301011200", lons, lats, alts)
    r2 = MultiPointReceptor("202301011200", lons, lats, alts)
    assert r1.location_id == r2.location_id


def test_location_id_multipoint_order_independent():
    r_a = MultiPointReceptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    r_b = MultiPointReceptor("202301011200", [-111.86, -111.85], [40.78, 40.77], [5, 5])
    assert r_a.location_id == r_b.location_id


def test_location_id_multipoint_starts_with_multi():
    r = MultiPointReceptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    assert r.location_id.startswith("multi_")


def test_location_id_multipoint_hash_length():
    r = MultiPointReceptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
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
    r = MultiPointReceptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    assert r.location_id == f"multi_{expected_hash}"


def test_location_id_multipoint_differs_for_different_points():
    r_a = MultiPointReceptor("202301011200", [-111.85, -111.86], [40.77, 40.78], [5, 5])
    r_b = MultiPointReceptor("202301011200", [-111.85, -111.87], [40.77, 40.79], [5, 5])
    assert r_a.location_id != r_b.location_id


# ---------------------------------------------------------------------------
# PointReceptor
# ---------------------------------------------------------------------------


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


def test_point_has_no_bottom():
    r = PointReceptor("202301011200", -111.85, 40.77, 5.0)
    with pytest.raises(AttributeError):
        _ = r.bottom
    with pytest.raises(AttributeError):
        _ = r.top


def test_column_has_no_altitude():
    r = ColumnReceptor("202301011200", -111.85, 40.77, 5.0, 50.0)
    with pytest.raises(AttributeError):
        _ = r.altitude


def test_receptor_time_iso_string():
    r = PointReceptor(
        time="2023-01-01T12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2023, 1, 1, 12, 0)


def test_receptor_time_compact_string():
    r = PointReceptor(
        time="202301011200",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2023, 1, 1, 12, 0)


def test_receptor_time_aware_input_normalizes_to_naive_utc():
    r = PointReceptor(
        time="2023-01-01T05:00:00-07:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    assert r.time == dt.datetime(2023, 1, 1, 12, 0)
    assert r.time.tzinfo is None


def test_receptor_equality():
    r1 = PointReceptor("202301011200", -111.85, 40.77, 5.0)
    r2 = PointReceptor("202301011200", -111.85, 40.77, 5.0)
    assert r1 == r2


def test_receptor_inequality_time():
    r1 = PointReceptor("202301011200", -111.85, 40.77, 5.0)
    r2 = PointReceptor("202301011300", -111.85, 40.77, 5.0)
    assert r1 != r2


def test_receptor_inequality_across_types():
    r1 = PointReceptor("202301011200", -111.85, 40.77, 5.0)
    r2 = ColumnReceptor("202301011200", -111.85, 40.77, 5.0, 50.0)
    assert r1 != r2


def test_receptor_init_rejects_missing_time():
    with pytest.raises(ValueError, match="'time' must be provided"):
        PointReceptor(time=None, longitude=-111.85, latitude=40.77, altitude=5.0)


def test_receptor_init_parses_iso_and_compact_strings():
    r_iso = PointReceptor("2023-01-01T12:00:00", -111.85, 40.77, 5.0)
    r_compact = PointReceptor("202301011200", -111.85, 40.77, 5.0)
    expected = dt.datetime(2023, 1, 1, 12, 0)
    assert r_iso.time == expected
    assert r_compact.time == expected


def test_receptor_eq_non_receptor_is_false(point_receptor):
    assert (point_receptor == object()) is False


def test_receptor_points_property_yields_points(point_receptor):
    items = list(point_receptor.points)
    assert len(items) == 1
    assert hasattr(items[0], "x") and items[0].x == pytest.approx(-111.85)
    assert items[0].y == pytest.approx(40.77)
    assert items[0].z == pytest.approx(5.0)


def test_receptor_lazy_geometry_point(point_receptor):
    from shapely import Point

    assert isinstance(point_receptor.geometry, Point)


def test_receptor_lazy_geometry_column(column_receptor):
    from shapely import LineString

    assert isinstance(column_receptor.geometry, LineString)


def test_receptor_lazy_geometry_multipoint(multipoint_receptor):
    from shapely import MultiPoint

    assert isinstance(multipoint_receptor.geometry, MultiPoint)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_longitude_out_of_range_raises():
    with pytest.raises(ValueError, match="longitude"):
        PointReceptor("202301011200", 200.0, 40.77, 5.0)


def test_latitude_out_of_range_raises():
    with pytest.raises(ValueError, match="latitude"):
        PointReceptor("202301011200", -111.85, -95.0, 5.0)


def test_negative_agl_altitude_raises():
    with pytest.raises(ValueError, match="AGL altitudes"):
        PointReceptor("202301011200", -111.85, 40.77, -1.0)


def test_negative_msl_altitude_is_allowed():
    r = PointReceptor(
        "202301011200", -111.85, 40.77, altitude=-50.0, altitude_ref="msl"
    )
    assert r.altitude == -50.0
    assert r.altitude_ref == "msl"


def test_column_bottom_ge_top_raises():
    with pytest.raises(ValueError, match="bottom"):
        ColumnReceptor("202301011200", -111.85, 40.77, bottom=50.0, top=5.0)


def test_column_bottom_eq_top_raises():
    with pytest.raises(ValueError, match="bottom"):
        ColumnReceptor("202301011200", -111.85, 40.77, bottom=5.0, top=5.0)


def test_multipoint_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        MultiPointReceptor("202301011200", [-111.85, -111.86], [40.77], [5.0, 5.0])


# ---------------------------------------------------------------------------
# isinstance type checks
# ---------------------------------------------------------------------------


def test_isinstance_point(point_receptor):
    assert isinstance(point_receptor, PointReceptor)


def test_isinstance_column(column_receptor):
    assert isinstance(column_receptor, ColumnReceptor)


def test_isinstance_multipoint(multipoint_receptor):
    assert isinstance(multipoint_receptor, MultiPointReceptor)


# ---------------------------------------------------------------------------
# Receptor.from_points
# ---------------------------------------------------------------------------


def test_receptor_from_points_single_makes_point():
    r = Receptor.from_points("202301011200", [(-111.85, 40.77, 5)])
    assert isinstance(r, PointReceptor)
    assert r.location_id == "-111.85_40.77_5"


def test_receptor_from_points_two_same_xy_makes_column():
    r = Receptor.from_points(
        "202301011200", [(-111.85, 40.77, 5), (-111.85, 40.77, 50)]
    )
    assert isinstance(r, ColumnReceptor)
    assert r.location_id.endswith("_X")


def test_receptor_from_points_two_different_xy_makes_multipoint():
    r = Receptor.from_points("202301011200", [(-111.85, 40.77, 5), (-111.86, 40.78, 5)])
    assert isinstance(r, MultiPointReceptor)


def test_receptor_from_points_empty_raises():
    with pytest.raises(ValueError, match="least one"):
        Receptor.from_points("202301011200", [])


def test_receptor_from_points_column_sorts_bottom_top():
    r = Receptor.from_points(
        "202301011200", [(-111.85, 40.77, 50), (-111.85, 40.77, 5)]
    )
    assert isinstance(r, ColumnReceptor)
    assert r.bottom == 5.0
    assert r.top == 50.0


# ---------------------------------------------------------------------------
# Receptor.from_dict round-trips
# ---------------------------------------------------------------------------


def test_receptor_from_dict_point_round_trip():
    r = PointReceptor("202301011200", -111.85, 40.77, 5.0)
    r2 = Receptor.from_dict(r.to_dict())
    assert isinstance(r2, PointReceptor)
    assert r2 == r


def test_receptor_from_dict_column_round_trip():
    r = ColumnReceptor("202301011200", -111.85, 40.77, 5.0, 50.0)
    r2 = Receptor.from_dict(r.to_dict())
    assert isinstance(r2, ColumnReceptor)
    assert r2 == r


def test_receptor_from_dict_multipoint_round_trip():
    r = MultiPointReceptor(
        "202301011200", [-111.85, -111.86], [40.77, 40.78], [5.0, 5.0]
    )
    r2 = Receptor.from_dict(r.to_dict())
    assert isinstance(r2, MultiPointReceptor)
    assert r2 == r


# ---------------------------------------------------------------------------
# read_receptors
# ---------------------------------------------------------------------------


def test_read_receptors_missing_required_columns_raises(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text("time,lat,lon\n2023-01-01 12:00:00,40.77,-111.85\n")
    with pytest.raises(ValueError, match="must contain columns"):
        read_receptors(csv)


def test_read_receptors_basic(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text("time,lati,long,zagl\n2023-01-01 12:00:00,40.77,-111.85,5.0\n")
    receptors = read_receptors(csv)
    assert len(receptors) == 1
    assert isinstance(receptors[0], PointReceptor)
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
    csv = tmp_path / "receptors.csv"
    csv.write_text(
        "time,lati,long,zagl,r_idx\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0,0\n"
        "2023-01-01 12:00:00,40.78,-111.86,5.0,0\n"
        "2023-01-01 13:00:00,40.79,-111.87,5.0,1\n"
    )
    receptors = read_receptors(csv)
    assert len(receptors) == 2
    assert isinstance(receptors[0], MultiPointReceptor)
    assert len(receptors[0]) == 2


def test_read_receptors_r_idx_group_mixed_times_raises(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text(
        "time,lati,long,zagl,r_idx\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0,0\n"
        "2023-01-01 13:00:00,40.78,-111.86,5.0,0\n"
    )
    with pytest.raises(ValueError, match="same release time"):
        read_receptors(csv)


def test_read_receptors_alt_column_names(tmp_path):
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
    assert receptors[0].altitude_ref == "msl"


# ---------------------------------------------------------------------------
# MultiPointReceptor - duplicate horizontal locations are rejected
# ---------------------------------------------------------------------------


def test_multipoint_rejects_stacked_heights_at_one_location():
    # HYSPLIT chains same-lat/lon starting locations into one vertical line
    # source and releases only between the last two heights.
    with pytest.raises(ValueError, match="distinct horizontal"):
        MultiPointReceptor(
            "202508110440",
            [-76.68869] * 5,
            [37.76396] * 5,
            [100.0, 200.0, 300.0, 400.0, 500.0],
        )


def test_multipoint_rejects_nonconsecutive_duplicate_location():
    with pytest.raises(ValueError, match="distinct horizontal"):
        MultiPointReceptor(
            "202301011200",
            [-111.85, -111.86, -111.85],
            [40.77, 40.78, 40.77],
            [100.0, 200.0, 300.0],
        )


def test_read_receptors_stacked_group_reports_r_idx(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text(
        "r_idx,time,longitude,latitude,altitude,altitude_ref\n"
        + "".join(
            f"0,2025-08-11 04:40:50,-76.688690186,37.763957977,{z},agl\n"
            for z in (100.0, 200.0, 300.0, 400.0, 500.0)
        )
    )
    with pytest.raises(ValueError, match="r_idx=0.*distinct horizontal"):
        read_receptors(csv)


# ---------------------------------------------------------------------------
# receptors_to_csv / write_receptors round trips
# ---------------------------------------------------------------------------


def test_receptors_to_csv_header_and_rows(point_receptor, column_receptor):
    text = receptors_to_csv([point_receptor, column_receptor])
    lines = text.splitlines()

    assert lines[0] == "r_idx,time,longitude,latitude,altitude,altitude_ref"
    # One row per constituent point: 1 (point) + 2 (column).
    assert len(lines) == 1 + 1 + 2
    assert [line.split(",")[0] for line in lines[1:]] == ["0", "1", "1"]
    assert all(line.endswith(",agl") for line in lines[1:])


def test_receptors_to_csv_empty_has_only_header():
    assert receptors_to_csv([]).splitlines() == [
        "r_idx,time,longitude,latitude,altitude,altitude_ref"
    ]


def test_write_receptors_round_trip_point(tmp_path, point_receptor):
    path = write_receptors([point_receptor], tmp_path / "receptors.csv")
    assert path == tmp_path / "receptors.csv"
    assert read_receptors(path) == [point_receptor]


def test_write_receptors_round_trip_column(tmp_path, column_receptor):
    loaded = read_receptors(write_receptors([column_receptor], tmp_path / "r.csv"))
    assert loaded == [column_receptor]
    assert isinstance(loaded[0], ColumnReceptor)


def test_write_receptors_round_trip_multipoint(tmp_path, multipoint_receptor):
    loaded = read_receptors(write_receptors([multipoint_receptor], tmp_path / "r.csv"))
    assert loaded == [multipoint_receptor]
    assert isinstance(loaded[0], MultiPointReceptor)


def test_write_receptors_round_trip_mixed_preserves_order_and_types(
    tmp_path, point_receptor, column_receptor, multipoint_receptor
):
    original = [multipoint_receptor, point_receptor, column_receptor]
    loaded = read_receptors(write_receptors(original, tmp_path / "r.csv"))

    assert loaded == original
    assert [type(r) for r in loaded] == [
        MultiPointReceptor,
        PointReceptor,
        ColumnReceptor,
    ]


def test_write_receptors_round_trip_preserves_msl_reference(tmp_path):
    original = [
        PointReceptor("202301011200", -111.85, 40.77, 1500.0, altitude_ref="msl"),
        ColumnReceptor(
            "202301011200", -111.85, 40.77, 1300.0, 1800.0, altitude_ref="msl"
        ),
    ]
    loaded = read_receptors(write_receptors(original, tmp_path / "r.csv"))
    assert loaded == original
    assert all(r.altitude_ref == "msl" for r in loaded)


def test_write_receptors_creates_parent_directories(tmp_path, point_receptor):
    path = write_receptors([point_receptor], tmp_path / "nested" / "dir" / "r.csv")
    assert path.is_file()
    assert read_receptors(path) == [point_receptor]


def test_read_receptors_keeps_extra_columns_as_attrs(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text(
        "time,lati,long,zagl,r_idx,Scene,note\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0,0,A,\n"
        "2023-01-01 12:00:00,40.78,-111.86,5.0,0,A,\n"
        "2023-01-01 13:00:00,40.79,-111.87,5.0,1,B,hello\n"
    )
    multi, point = read_receptors(csv)
    assert isinstance(multi, MultiPointReceptor)
    assert multi.attrs == {"Scene": "A", "note": None}
    assert point.attrs == {"Scene": "B", "note": "hello"}


def test_receptor_attrs_round_trip_through_csv(
    tmp_path, point_receptor, column_receptor
):
    from stilt.receptors import append_receptors_csv, receptors_to_csv

    labelled = point_receptor.model_copy(update={"attrs": {"scene": "A"}})
    text = receptors_to_csv([labelled, column_receptor])
    assert text.splitlines()[0].endswith(",altitude_ref,scene")
    back = read_receptors(io.StringIO(text))
    assert back[0].attrs == {"scene": "A"}
    assert back[1].attrs == {"scene": None}

    extra = PointReceptor(
        time="2023-02-01 00:00",
        longitude=-111.0,
        latitude=40.0,
        altitude=1.0,
        attrs={"scene": "C", "ignored": 1},
    )
    grown = read_receptors(io.StringIO(append_receptors_csv(text, [extra])))
    assert grown[-1].attrs == {"scene": "C"}


# ---------------------------------------------------------------------------
# Receptors as frozen models
# ---------------------------------------------------------------------------


def test_receptors_are_frozen_values(point_receptor):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="frozen"):
        point_receptor.altitude = 20.0
    twin = PointReceptor(
        time=point_receptor.time,
        longitude=point_receptor.longitude,
        latitude=point_receptor.latitude,
        altitude=point_receptor.altitude,
    )
    assert twin == point_receptor and hash(twin) == hash(point_receptor)
    assert len({twin, point_receptor}) == 1


def test_attrs_do_not_count_for_equality_or_the_dict(point_receptor):
    labelled = point_receptor.model_copy(update={"attrs": {"site": "WBB"}})
    assert labelled == point_receptor
    assert hash(labelled) == hash(point_receptor)
    assert "attrs" not in labelled.to_dict()
    assert labelled.id == point_receptor.id


def test_unknown_field_is_rejected():
    with pytest.raises(TypeError, match="height"):
        Receptor.from_dict(
            {
                "kind": "point",
                "time": "2023-01-01T12:00:00",
                "longitude": -111.85,
                "latitude": 40.77,
                "altitude": 5,
                "height": 5,
            }
        )


def test_from_dict_reads_the_dict_earlier_versions_stored():
    """Particle and footprint files written before the `kind` field still load."""
    old = {
        "type": "ColumnReceptor",
        "time": "2023-01-01T12:00:00",
        "longitude": -111.85,
        "latitude": 40.77,
        "bottom": 0.0,
        "top": 3000.0,
        "altitude_ref": "agl",
    }
    r = Receptor.from_dict(old)
    assert isinstance(r, ColumnReceptor) and r.top == 3000.0
    assert Receptor.from_dict(r.to_dict()) == r
    with pytest.raises(ValueError, match="Unknown receptor type"):
        Receptor.from_dict({**old, "type": "BoxReceptor"})
    with pytest.raises(ValueError, match="'kind'"):
        Receptor.from_dict({k: v for k, v in old.items() if k != "type"})


def test_parse_receptor_id_splits_time_and_location(
    point_receptor, column_receptor, multipoint_receptor
):
    from stilt.receptors import parse_receptor_id

    for r in (point_receptor, column_receptor, multipoint_receptor):
        assert parse_receptor_id(r.id) == (r.time, r.location_id)
    for bad in ("nope", "202301011200", "202313011200_-111_40_5", "202301011200_a_b"):
        with pytest.raises(ValueError):
            parse_receptor_id(bad)


# ---------------------------------------------------------------------------
# Multipoint ids and altitude (#50)
# ---------------------------------------------------------------------------


def _multi(alts, **kwargs):
    return MultiPointReceptor(
        "2023-01-01 12:00", [-111.85, -111.86], [40.77, 40.78], alts, **kwargs
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


def test_receptors_that_share_an_id_are_refused():
    """Two receptors closer than the id resolves would overwrite each other."""
    from stilt.receptors import check_distinct_ids

    a, b = _multi([10.001, 500.0]), _multi([10.004, 500.0])
    assert a != b and a.id == b.id
    with pytest.raises(ValueError, match="share the id"):
        check_distinct_ids([a, b])
    check_distinct_ids([a, _multi([10.001, 500.0])])  # the same receptor twice is fine

    text = receptors_to_csv([a, b])
    with pytest.raises(ValueError, match="share the id"):
        read_receptors(io.StringIO(text))


# ---------------------------------------------------------------------------
# The receptor table
# ---------------------------------------------------------------------------


def test_frame_has_a_row_per_release_point(
    point_receptor, column_receptor, multipoint_receptor
):
    from stilt.receptors import receptors_from_frame, receptors_to_frame

    receptors = [
        point_receptor.model_copy(update={"attrs": {"site": "WBB"}}),
        column_receptor,
        multipoint_receptor,
    ]
    frame = receptors_to_frame(receptors)
    assert list(frame.columns) == [
        "r_idx",
        "time",
        "longitude",
        "latitude",
        "altitude",
        "altitude_ref",
        "site",
    ]
    assert len(frame) == 1 + 2 + len(multipoint_receptor)
    assert frame["r_idx"].tolist()[:3] == [0, 1, 1]
    assert str(frame["time"].dtype).startswith("datetime64")

    back = receptors_from_frame(frame)
    assert back == receptors
    assert back[0].attrs == {"site": "WBB"}
    assert receptors_to_frame([]).empty


def test_frame_accepts_the_alternate_column_names():
    from stilt.receptors import receptors_from_frame

    frame = pd.DataFrame(
        {
            "Time": ["2023-01-01 12:00"],
            "LON": [-111.85],
            "lati": [40.77],
            "zmsl": [1500.0],
            "Scene": ["A"],
        }
    )
    [r] = receptors_from_frame(frame)
    assert isinstance(r, PointReceptor)
    assert (r.longitude, r.latitude, r.altitude) == (-111.85, 40.77, 1500.0)
    assert r.altitude_ref == "msl"
    assert r.attrs == {"Scene": "A"}


# ---------------------------------------------------------------------------
# Appending keeps the altitude reference (#51)
# ---------------------------------------------------------------------------


def test_append_msl_receptor_to_plain_altitude_file_keeps_its_reference():
    """An MSL receptor appended to a file with no reference column read back as AGL."""
    from stilt.receptors import append_receptors_csv

    text = "time,longitude,latitude,altitude\n2023-01-01 12:00:00,-111.85,40.77,5\n"
    msl = PointReceptor("2023-01-02 12:00", -111.9, 40.7, 1500.0, altitude_ref="msl")

    grown = append_receptors_csv(text, [msl])

    assert grown.splitlines()[0] == "time,longitude,latitude,altitude,altitude_ref"
    assert grown.splitlines()[1] == "2023-01-01 12:00:00,-111.85,40.77,5,agl"
    first, second = read_receptors(io.StringIO(grown))
    assert first.altitude_ref == "agl"
    assert second == msl and second.altitude_ref == "msl"


def test_append_agl_receptor_leaves_a_plain_file_without_the_column():
    from stilt.receptors import append_receptors_csv

    text = "time,lon,lat,z,site\n2023-01-01 12:00:00,-111.85,40.77,5,WBB\n"
    agl = PointReceptor("2023-01-02 12:00", -111.9, 40.7, 10.0, attrs={"site": "UOU"})

    grown = append_receptors_csv(text, [agl])

    assert grown.splitlines()[0] == "time,lon,lat,z,site"
    assert grown.splitlines()[1] == "2023-01-01 12:00:00,-111.85,40.77,5,WBB"
    assert grown.splitlines()[2] == "2023-01-02 12:00:00,-111.9,40.7,10.0,UOU"


def test_append_to_a_zagl_file_still_refuses_an_msl_receptor():
    from stilt.receptors import append_receptors_csv

    text = "time,long,lati,zagl\n2023-01-01 12:00:00,-111.85,40.77,5\n"
    msl = PointReceptor("2023-01-02 12:00", -111.9, 40.7, 1500.0, altitude_ref="msl")
    with pytest.raises(ValueError, match="altitudes are agl"):
        append_receptors_csv(text, [msl])
