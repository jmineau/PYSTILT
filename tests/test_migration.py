"""
Tests for reading STILT-R and X-STILT ``by-id`` output (``stilt.migration``).

The sample in ``fixtures/stilt_r_by_id`` was written by STILT-R's own code
(``saveRDS`` of the list ``simulation_step`` saves, and ``calc_footprint``)
from made-up particles: ``fixtures/r_helpers/write_by_id.r`` says how, and
writes it again with ``Rscript write_by_id.r <STILT_R_DIR> <folder>``.
"""

from __future__ import annotations

import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt import ColumnReceptor, MultiPointReceptor, PointReceptor, calc_footprint
from stilt.migration import _axis, _particles, _receptor, read_stilt_r

SAMPLE = Path(__file__).parent / "fixtures" / "stilt_r_by_id"
POINT = "201507050000_-111.85_40.77_5"
COLUMN = "201507051800_-111.9_40.7_X"

pytest.importorskip("rdata")


def _read(by_id: Path = SAMPLE) -> dict[str, tuple]:
    """Return every simulation of a by-id folder, by its STILT-R simulation id."""
    sims = list(read_stilt_r(by_id))
    names = sorted(p.name for p in by_id.iterdir() if p.is_dir())
    return dict(zip(names, sims, strict=True))


def _raw_foot(name: str) -> xr.DataArray:
    """Return a sample footprint as STILT-R wrote it, read with xarray alone."""
    with xr.open_dataset(SAMPLE / name / f"{name}_foot.nc", decode_times=False) as ds:
        return ds["foot"].load()


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    """Return a copy of the sample by-id folder, to change."""
    return Path(shutil.copytree(SAMPLE, tmp_path / "by-id"))


def test_point_receptor_is_the_one_stilt_r_ran():
    receptor, _, _ = _read()[POINT]
    assert receptor == PointReceptor(
        time=dt.datetime(2015, 7, 5), longitude=-111.85, latitude=40.77, altitude=5
    )
    assert receptor.id == POINT


def test_receptor_coordinates_are_the_15_digits_hysplit_ran():
    # R's paste keeps 15 significant digits, in CONTROL and in the folder name.
    stored = {
        "run_time": pd.DatetimeIndex(["2023-06-25 21:00"]),
        "long": np.array([-112.04393884275244]),
        "lati": np.array([40.745448298385512]),
        "zagl": np.array([5.0]),
    }
    receptor = _receptor(stored, Path("x_traj.rds"))
    assert receptor.id == "202306252100_-112.043938842752_40.7454482983855_5"


def test_slant_points_stored_as_lists_are_a_multipoint_receptor():
    # X-STILT's slant columns keep lati and long as R lists of vectors.
    stored = {
        "run_time": pd.DatetimeIndex(["2020-07-01 20:00"]),
        "long": [np.array([-112.0, -112.01, -112.02])],
        "lati": [np.array([40.5, 40.51, 40.52])],
        "zagl": np.array([100.0, 1000.0, 2000.0]),
    }
    receptor = _receptor(stored, Path("x_traj.rds"))
    assert isinstance(receptor, MultiPointReceptor)
    assert receptor.longitudes == (-112.0, -112.01, -112.02)
    assert receptor.altitudes == (100.0, 1000.0, 2000.0)


def test_a_receptor_that_is_not_stilt_r_s_raises():
    path = Path("x_traj.rds")
    with pytest.raises(ValueError, match="has no long, zagl"):
        _receptor({"run_time": pd.DatetimeIndex(["2020-07-01"]), "lati": 40.5}, path)
    stored = {
        "run_time": pd.DatetimeIndex(["2020-07-01"]),
        "long": np.array([-112.0, -112.1]),
        "lati": np.array([40.5, 40.6, 40.7]),
        "zagl": np.array([5.0]),
    }
    with pytest.raises(ValueError, match="have 2, 3, and 1 values"):
        _receptor(stored, path)


def test_particles_without_a_schema_column_raise():
    receptor, particles, _ = _read()[POINT]
    stilt_r = particles.drop(columns=["time", "zagl"]).rename(
        columns={"particle": "indx", "age": "time", "lon": "long", "lat": "lati"}
    )
    with pytest.raises(ValueError, match="x_traj.rds: .*'zagl'"):
        _particles(stilt_r, receptor, Path("x_traj.rds"))


def test_an_rds_without_particles_raises(copy):
    import rdata

    rdata.write_rds(copy / POINT / f"{POINT}_traj.rds", {"file": "x", "params": {}})
    with pytest.raises(ValueError, match="not a STILT-R trajectory file"):
        _read(copy)


def test_column_receptor_spans_the_heights_and_keeps_seconds():
    receptor, _, _ = _read()[COLUMN]
    assert receptor == ColumnReceptor(
        time=dt.datetime(2015, 7, 5, 18, 0, 30),
        longitude=-111.9,
        latitude=40.7,
        bottom=0,
        top=1000,
    )


def test_particle_columns_get_pystilt_names():
    _, particles, _ = _read()[POINT]
    for stilt_r in ("indx", "long", "lati", "xhgt"):
        assert stilt_r not in particles.columns
    assert {"particle", "age", "lon", "lat", "zagl", "foot"} <= set(particles.columns)
    assert "foot_no_hnf_dilution" in particles.columns
    assert sorted(particles["particle"].unique()) == list(range(1, 11))
    # STILT-R's time is minutes since release, negative backward.
    assert particles["age"].min() == -120 and particles["age"].max() == -1
    assert particles["particle"].dtype == "float64"
    assert particles["age"].dtype == "float64"


def test_time_is_the_receptor_time_plus_age():
    receptor, particles, _ = _read()[COLUMN]
    expected = pd.Timestamp(receptor.time) + pd.to_timedelta(particles["age"], "min")
    pd.testing.assert_series_equal(particles["time"], expected, check_names=False)
    assert particles["time"].max() == pd.Timestamp("2015-07-05 17:59:30")


def test_column_release_height_is_stilt_r_xhgt():
    _, particles, _ = _read()[COLUMN]
    # 10 particles spread over 0 to 1000 m: slab centers 50, 150, ... 950.
    expected = (particles["particle"] - 0.5) * 100
    np.testing.assert_allclose(particles["release_height"], expected)


def test_hourly_footprint_layers_are_stamped_as_stilt_r_stamps_them():
    _, _, foot = _read()[POINT]
    raw = _raw_foot(POINT)
    stamps = pd.to_datetime(raw["time"].to_numpy(), unit="s")
    assert list(stamps) == [
        pd.Timestamp("2015-07-04 22:00"),
        pd.Timestamp("2015-07-04 23:00"),
    ]
    np.testing.assert_array_equal(foot["time"].to_numpy(), stamps.to_numpy())
    assert foot.stilt.config.time_integrate is False


def test_footprint_values_are_the_file_values_on_its_grid():
    _, _, foot = _read()[POINT]
    raw = _raw_foot(POINT).transpose("time", "lat", "lon")
    assert foot.dims == ("time", "lat", "lon")
    np.testing.assert_array_equal(foot.to_numpy(), raw.to_numpy().astype("float64"))
    np.testing.assert_allclose(foot["lon"], raw["lon"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(foot["lat"], raw["lat"], rtol=0, atol=1e-9)
    grid = foot.stilt.grid
    assert (grid.xmin, grid.xmax, grid.ymin, grid.ymax) == (-112.2, -111.6, 40.6, 41.0)
    assert (grid.xres, grid.yres) == (0.02, 0.025)


def test_time_integrated_footprint_is_one_layer_at_the_receptor_time():
    receptor, _, foot = _read()[COLUMN]
    assert foot.sizes["time"] == 1
    assert foot["time"].to_numpy()[0] == np.datetime64(receptor.time)
    assert foot.stilt.config.time_integrate is True
    assert foot.stilt.receptor == receptor


@pytest.mark.parametrize("name", [POINT, COLUMN])
def test_footprint_matches_pystilt_footprint_of_the_particles(name):
    # STILT-R made its footprint from these particles; PYSTILT's matches
    # STILT-R's at rtol=1e-7. This holds only when the columns, ages,
    # receptor time, grid, and layer stamps are all read right.
    receptor, particles, foot = _read()[name]
    config = foot.stilt.config
    mine = calc_footprint(
        particles,
        receptor,
        config.grid,
        smooth_factor=config.smooth_factor,
        time_integrate=config.time_integrate,
    )
    assert mine is not None
    xr.testing.assert_allclose(mine, foot, rtol=1e-7, atol=1e-12)


def test_a_folder_without_a_footprint_gives_none(copy):
    (copy / POINT / f"{POINT}_foot.nc").unlink()
    _, particles, foot = _read(copy)[POINT]
    assert foot is None
    assert not particles.empty


def test_other_footprint_files_are_not_read(copy):
    # X-STILT writes footprints at other resolutions beside the main one.
    (copy / COLUMN / f"{COLUMN}_0.1x0.1_foot.nc").write_bytes(b"not netcdf")
    _, _, foot = _read(copy)[COLUMN]
    assert foot is not None


def test_a_footprint_without_particles_is_skipped_with_a_warning(copy):
    (copy / POINT / f"{POINT}_traj.rds").unlink()
    with pytest.warns(UserWarning, match="footprint but no _traj.rds"):
        sims = list(read_stilt_r(copy))
    assert [r.id for r, _, _ in sims] == ["201507051800_-111.9_40.7_X0-1000"]


def test_layers_that_are_not_whole_hours_raise(copy):
    path = copy / POINT / f"{POINT}_foot.nc"
    with xr.open_dataset(path, decode_times=False) as ds:
        shifted = ds.load()
    shifted["time"] = shifted["time"] + 1800
    path.unlink()
    shifted.to_netcdf(path)
    with pytest.raises(ValueError, match="not whole hours"):
        _read(copy)


def test_a_projected_footprint_raises(copy):
    path = copy / POINT / f"{POINT}_foot.nc"
    with xr.open_dataset(path, decode_times=False) as ds:
        projected = ds.load().rename({"lon": "x", "lat": "y"})
    path.unlink()
    projected.to_netcdf(path)
    with pytest.raises(ValueError, match="longitude/latitude grid"):
        _read(copy)


def test_uneven_footprint_cells_raise(copy):
    path = copy / POINT / f"{POINT}_foot.nc"
    with xr.open_dataset(path, decode_times=False) as ds:
        uneven = ds.load()
    lon = uneven["lon"].to_numpy().copy()
    lon[3] += 0.005
    uneven = uneven.assign_coords(lon=lon)
    path.unlink()
    uneven.to_netcdf(path)
    with pytest.raises(ValueError, match="not evenly spaced"):
        _read(copy)


def test_an_axis_of_one_cell_raises():
    with pytest.raises(ValueError, match="two lon cells"):
        _axis(np.array([-112.0]), "lon", Path("x_foot.nc"))


def test_a_folder_that_is_not_by_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="by-id"):
        read_stilt_r(tmp_path)
    with pytest.raises(FileNotFoundError, match="not a folder"):
        read_stilt_r(tmp_path / "missing")


def test_without_rdata_the_error_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdata", None)
    with pytest.raises(ImportError, match=r"pystilt\[stilt-r\]"):
        read_stilt_r(SAMPLE)
