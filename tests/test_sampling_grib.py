"""Tests for sampling pressure from GFS GRIB files, on small synthetic files."""

import math

import numpy as np
import pandas as pd
import pytest
from runtraj.spatial import Grid

from stilt.footprint import FootprintConfig, add_foot, footprint_particles
from stilt.sampling import is_grib, sample_grib_pressure

pytest.importorskip("eccodes")
pytest.importorskip("cfgrib")

from .fixtures.grib import write_era5, write_gfs


def points(zagl, time="2023-06-01 12:00", lon=-111.0, lat=41.0):
    """Rows at *zagl* meters above ground."""
    zagl = np.atleast_1d(np.asarray(zagl, dtype=float))
    return pd.DataFrame(
        {"lon": lon, "lat": lat, "zagl": zagl, "time": pd.Timestamp(time)}
    )


@pytest.fixture
def gfs(tmp_path):
    """Two files three hours apart, the second with a surface pressure of 940 hPa."""
    return [
        write_gfs(tmp_path / "a.grb2", "2023-06-01 12:00"),
        write_gfs(tmp_path / "b.grb2", "2023-06-01 15:00", sp=94000.0),
    ]


def test_is_grib(gfs, tmp_path):
    other = tmp_path / "met.arl"
    other.write_bytes(b"0123456789")
    assert is_grib(gfs[0])
    assert not is_grib(other)
    assert not is_grib(tmp_path / "missing")


def test_ground_is_the_surface_pressure(gfs):
    np.testing.assert_allclose(sample_grib_pressure(points(0.0), gfs), [950.0])


def test_ln_pressure_is_linear_in_height_through_the_levels(gfs):
    pres = sample_grib_pressure(points([500.0, 1000.0, 1500.0, 250.0]), gfs)

    # The ground is at 500 m, so these are 1000, 1500, 2000, and 750 m high.
    expected = [
        900.0,
        math.exp((math.log(900) + math.log(800)) / 2),
        800.0,
        math.exp((math.log(950) + math.log(900)) / 2),
    ]
    np.testing.assert_allclose(pres, expected)


def test_level_below_the_ground_is_not_used(tmp_path):
    # The 1000 hPa level at 100 m is under the 500 m ground; 1000 hPa there
    # would give a pressure above the surface's at 200 m above ground.
    path = write_gfs(tmp_path / "a.grb2", "2023-06-01 12:00")

    pres = sample_grib_pressure(points(200.0), [path])

    assert 900.0 < pres[0] < 950.0


def test_above_the_highest_level_is_nan(gfs):
    assert np.isnan(sample_grib_pressure(points(1600.0), gfs)[0])


def test_no_height_is_nan(gfs):
    assert np.isnan(sample_grib_pressure(points(np.nan), gfs)[0])


def test_linear_in_time_between_files(gfs):
    pres = sample_grib_pressure(points(0.0, time="2023-06-01 13:00"), gfs)

    np.testing.assert_allclose(
        pres, [math.exp(math.log(950) * 2 / 3 + math.log(940) / 3)]
    )


def test_time_outside_the_files(gfs):
    with pytest.raises(ValueError, match="outside the GRIB files"):
        sample_grib_pressure(points(0.0, time="2023-06-01 16:00"), gfs)


def test_two_files_for_one_time(tmp_path):
    a = write_gfs(tmp_path / "a.grb2", "2023-06-01 12:00")
    b = write_gfs(tmp_path / "b.grb2", "2023-06-01 12:00")

    with pytest.raises(ValueError, match="both for"):
        sample_grib_pressure(points(0.0), [a, b])


def test_bilinear_across_a_global_grid_wraps(tmp_path):
    # Columns at longitudes 0, 90, 180, 270 with surface pressure 1000, 990,
    # 980, 970 hPa; a point at 315 is halfway from 270 back to 0.
    sp = np.tile([100000.0, 99000.0, 98000.0, 97000.0], 3)
    path = write_gfs(tmp_path / "a.grb2", "2023-06-01 12:00", sp=sp, global_grid=True)

    pres = sample_grib_pressure(points(0.0, lon=-45.0, lat=40.0), [path])

    # Fields are bilinear across the grid, before the logarithm.
    np.testing.assert_allclose(pres, [985.0])


def test_add_foot_divides_by_the_air_in_the_grib_column(gfs):
    particles = points([100.0, 900.0]).assign(particle=[1, 2], age=-60, mlht=1000.0)

    foot = add_foot(particles, gfs, footprint_depth=0.5)["foot"]

    # 500 m deep: from 950 hPa at the ground to 900 hPa at 500 m.
    air = 100 * (950.0 - 900.0) / 9.80665
    np.testing.assert_allclose(foot, [0.02897 * 3600 / air, 0.0])


def test_add_foot_with_grib_needs_a_mixed_layer_of_its_own(gfs):
    particles = points([100.0]).assign(particle=1, age=-60)

    with pytest.raises(ValueError, match="With GRIB met files"):
        add_foot(particles, gfs, footprint_depth=0.5)


def test_era5_model_levels_isothermal(tmp_path):
    # An isothermal dry column: p = sp exp(-g z / (R T)) exactly, ln p being
    # linear in height inside each layer.
    path = write_era5(tmp_path / "EA23060112", "2023-06-01 12:00")
    zagl = np.array([0.0, 1000.0, 3000.0])

    pres = sample_grib_pressure(points(zagl), [path])

    np.testing.assert_allclose(
        pres, 1000.0 * np.exp(-9.80665 * zagl / (287.06 * 250.0)), rtol=1e-9
    )


def test_era5_moisture_thickens_the_layers(tmp_path):
    dry = write_era5(tmp_path / "dry", "2023-06-01 12:00")
    wet = write_era5(tmp_path / "wet", "2023-06-01 12:00", q=0.01)

    # Moist air is lighter, so pressure falls more slowly with height.
    assert (
        sample_grib_pressure(points(1000.0), [wet])[0]
        > sample_grib_pressure(points(1000.0), [dry])[0]
    )


def test_era5_above_the_lowest_half_level_below_the_top(tmp_path):
    path = write_era5(tmp_path / "EA23060112", "2023-06-01 12:00")
    # The top half level (p = 0) is left out, so the column ends at 200 hPa.
    top = 287.06 * 250.0 / 9.80665 * np.log(1000.0 / 200.0)
    assert np.isnan(sample_grib_pressure(points(top + 100.0), [path])[0])


def test_footprint_stage_computes_foot_from_grib_met(gfs):
    # A run that wrote positions and its own mixed layer, on GRIB met, as
    # FLEXPART's do: the footprint stage computes foot from the GRIB files.
    particles = points([100.0, 900.0]).assign(particle=[1, 2], age=-60, mlht=1000.0)
    config = FootprintConfig(
        grid=Grid(xmin=-112, xmax=-110, ymin=40, ymax=42, xres=0.1, yres=0.1)
    )

    out = footprint_particles(particles, config, gfs)

    air = 100 * (950.0 - 900.0) / 9.80665
    np.testing.assert_allclose(out["foot"], [0.02897 * 3600 / air, 0.0])
