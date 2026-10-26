"""
Tests for reading STILT-R and X-STILT ``by-id`` output (``stilt.migration``).

The sample project in ``fixtures/stilt_r_project`` was written by STILT-R's
own code (``write_control``, ``saveRDS`` of the list ``simulation_step``
saves, and ``calc_footprint`` with ``smooth_factor`` 0.5) from made-up
particles, with the ``params.RDS`` rslurm keeps beside ``out/``:
``fixtures/r_helpers/write_by_id.r`` says how, and writes it again with
``Rscript write_by_id.r <STILT_R_DIR> <folder>``.
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
from stilt.migration import (
    _axis,
    _particles,
    _receptor,
    _rslurm_params,
    _smooth_factor,
    read_stilt_r,
)

PROJECT = Path(__file__).parent / "fixtures" / "stilt_r_project"
SAMPLE = PROJECT / "out" / "by-id"
PARAMS = PROJECT / "_rslurm_stilt_r_project" / "params.RDS"
POINT = "201507050000_-111.85_40.77_5"
COLUMN = "201507051800_-111.9_40.7_X"

rdata = pytest.importorskip("rdata")


def _read(by_id: Path = SAMPLE, **kwargs) -> dict[str, tuple]:
    """Return every simulation of a by-id folder, by its STILT-R simulation id."""
    sims = list(read_stilt_r(by_id, **kwargs))
    names = sorted(p.name for p in by_id.iterdir() if p.is_dir())
    return dict(zip(names, sims, strict=True))


def _raw_foot(name: str) -> xr.DataArray:
    """Return a sample footprint as STILT-R wrote it, read with xarray alone."""
    with xr.open_dataset(SAMPLE / name / f"{name}_foot.nc", decode_times=False) as ds:
        return ds["foot"].load()


def _write_points(control: Path, points: list[str]) -> None:
    """Replace the release points of a CONTROL file with *points* (``"lat lon height"``)."""
    lines = control.read_text().splitlines()
    n = int(lines[1])
    control.write_text(
        "\n".join([lines[0], str(len(points)), *points, *lines[2 + n :]])
    )


def _set_n_hours(control: Path, n_hours: int) -> None:
    """Replace the run length of a CONTROL file."""
    lines = control.read_text().splitlines()
    lines[2 + int(lines[1])] = str(n_hours)
    control.write_text("\n".join(lines))


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    """Return the by-id folder of a copy of the sample project, to change."""
    return (
        Path(shutil.copytree(PROJECT, tmp_path / "stilt_r_project")) / "out" / "by-id"
    )


@pytest.fixture
def bare(tmp_path: Path) -> Path:
    """Return a copy of the sample by-id folder without rslurm's params.RDS."""
    return Path(shutil.copytree(SAMPLE, tmp_path / "project" / "out" / "by-id"))


def test_point_receptor_is_the_one_stilt_r_ran():
    receptor, _, _ = _read()[POINT]
    assert receptor == PointReceptor(
        time=dt.datetime(2015, 7, 5), longitude=-111.85, latitude=40.77, altitude=5
    )
    assert receptor.id == POINT


def test_receptor_coordinates_are_the_numbers_in_control(copy):
    # The landfill receptor: HYSPLIT read these digits from CONTROL.
    _write_points(copy / POINT / "CONTROL", ["40.7454482983855 -112.043938842752 5"])
    receptor, _, _ = _read(copy)[POINT]
    assert (receptor.latitude, receptor.longitude) == (
        40.7454482983855,
        -112.043938842752,
    )
    assert receptor.location_id == "-112.043938842752_40.7454482983855_5"


def test_receptor_coordinates_do_not_come_from_the_rds(copy):
    # The .rds keeps the receptor as run_stilt.r gave it; CONTROL is what ran.
    _write_points(copy / POINT / "CONTROL", ["40.78 -111.86 10"])
    receptor, _, _ = _read(copy)[POINT]
    assert receptor.coords() == [(40.78, -111.86, 10.0)]


def test_slant_points_in_control_are_a_multipoint_receptor(copy):
    # X-STILT's slant columns release from a point per height, each its own place.
    points = ["40.5 -112.0 100", "40.51 -112.01 1000", "40.52 -112.02 2000"]
    _write_points(copy / POINT / "CONTROL", points)
    receptor, _, _ = _read(copy)[POINT]
    assert isinstance(receptor, MultiPointReceptor)
    assert receptor.longitudes == (-112.0, -112.01, -112.02)
    assert receptor.altitudes == (100.0, 1000.0, 2000.0)


def test_a_folder_without_control_raises_naming_it(copy):
    (copy / COLUMN / "CONTROL").unlink()
    with pytest.raises(FileNotFoundError, match=f"1 folders .* first .*{COLUMN}"):
        read_stilt_r(copy)


def test_a_control_that_cannot_be_read_raises(copy):
    (copy / POINT / "CONTROL").write_text("15 07 05 00 00\n1\n")
    with pytest.raises(ValueError, match=f"{POINT}/CONTROL cannot be read"):
        _read(copy)


def test_a_receptor_without_its_time_raises():
    with pytest.raises(ValueError, match="has no run_time"):
        _receptor({"lati": 40.5}, None, Path("x_traj.rds"))


def test_particles_without_a_schema_column_raise():
    receptor, particles, _ = _read()[POINT]
    stilt_r = particles.drop(columns=["time", "zagl"]).rename(
        columns={"particle": "indx", "age": "time", "lon": "long", "lat": "lati"}
    )
    with pytest.raises(ValueError, match="x_traj.rds: .*'zagl'"):
        _particles(stilt_r, receptor, Path("x_traj.rds"))


def test_an_rds_without_particles_raises(copy):
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
    assert config.smooth_factor == 0.5
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


def test_params_rds_columns_are_read_past_its_functions():
    # rslurm's table holds the before_trajec and before_footprint functions,
    # which rdata cannot convert; the columns needed are read alone.
    path, values = _rslurm_params(SAMPLE, rdata)
    assert path == PARAMS.resolve()
    assert values == {"smooth_factor": {0.5}, "n_hours": {-2.0}, "numpar": {10.0}}


def test_smooth_factor_comes_from_params_rds():
    for _, _, foot in read_stilt_r(SAMPLE):
        assert foot.stilt.config.smooth_factor == 0.5


def test_smooth_factor_given_without_params_rds(bare):
    receptor, particles, foot = _read(bare, smooth_factor=0.5)[POINT]
    assert foot.stilt.config.smooth_factor == 0.5
    mine = calc_footprint(particles, receptor, foot.stilt.grid, smooth_factor=0.5)
    xr.testing.assert_allclose(mine, foot, rtol=1e-7, atol=1e-12)


def test_a_footprint_without_a_smooth_factor_raises_saying_how(bare):
    with pytest.raises(ValueError, match=r"params.RDS .* smooth_factor=1"):
        _read(bare)


def test_runs_without_footprints_need_no_smooth_factor(bare):
    for foot in bare.glob("*/*_foot.nc"):
        foot.unlink()
    sims = _read(bare)
    assert all(foot is None for _, _, foot in sims.values())


def test_smooth_factor_given_and_in_params_rds_must_agree():
    assert _read(smooth_factor=0.5)[POINT][2].stilt.config.smooth_factor == 0.5
    with pytest.raises(
        ValueError, match="smooth_factor=1 was given, but .* records 0.5"
    ):
        _read(smooth_factor=1)


def test_params_rds_of_another_run_length_raises(copy):
    _set_n_hours(copy / POINT / "CONTROL", -24)
    with pytest.raises(ValueError, match="disagree: it gives n_hours -2, .* give -24"):
        _read(copy)


def test_params_rds_of_another_numpar_raises():
    rslurm = (PARAMS, {"smooth_factor": {1.0}, "n_hours": {-2.0}, "numpar": {10.0}})
    foot = Path("x_foot.nc")
    assert _smooth_factor(foot, -2, {"numpar": [10.0]}, rslurm, None) == 1
    with pytest.raises(ValueError, match="numpar 10, and the run's files give 1000"):
        _smooth_factor(foot, -2, {"numpar": [1000.0]}, rslurm, None)


def test_params_rds_with_several_smooth_factors_raises():
    values = {"smooth_factor": {1.0, 2.0}, "n_hours": {-2.0}, "numpar": {10.0}}
    with pytest.raises(ValueError, match="gives smooth_factor 1, 2; one value"):
        _smooth_factor(Path("x_foot.nc"), -2, {}, (PARAMS, values), None)


def test_params_rds_that_is_not_a_table_raises(copy):
    rdata.write_rds(
        copy.parents[1] / "_rslurm_stilt_r_project" / "params.RDS", {"a": 1}
    )
    with pytest.raises(ValueError, match="not a table of STILT-R runs"):
        read_stilt_r(copy)
