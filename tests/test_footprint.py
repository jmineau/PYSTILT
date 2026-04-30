"""Tests for stilt.footprint helpers and aggregation."""

import builtins
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.config import FootprintConfig, Grid, VerticalOperatorTransformSpec
from stilt.footprint import (
    Footprint,
    _calc_digits,
    _cf_grid_mapping_attrs,
    _interpolate_early_timesteps,
    _interpolation_times,
    _make_gauss_kernel,
)
from stilt.receptors import PointReceptor


def _make_footprint(
    xres: float = 0.1, yres: float = 0.1, n_times: int = 1
) -> Footprint:
    receptor_time = dt.datetime(2023, 1, 1, 12)
    receptor = PointReceptor(
        time=receptor_time,
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    grid = Grid(
        xmin=-114.0,
        xmax=-113.8,
        ymin=39.0,
        ymax=39.2,
        xres=xres,
        yres=yres,
    )
    config = FootprintConfig(grid=grid)

    lons = np.array([-113.95, -113.85])
    lats = np.array([39.05, 39.15])
    times = [receptor_time + pd.Timedelta(hours=i) for i in range(n_times)]

    data = xr.DataArray(
        np.zeros((n_times, len(lats), len(lons))),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"units": "ppm (umol-1 m2 s)"},
    )
    return Footprint(receptor=receptor, config=config, data=data, name="slv")


def test_calc_digits_values():
    assert _calc_digits(0.01) == 3
    assert _calc_digits(0.1) == 2
    assert _calc_digits(1.0) == 0


def test_calc_digits_invalid():
    with pytest.raises(ValueError):
        _calc_digits(0.0)


def test_make_gauss_kernel_sigma_zero():
    k = _make_gauss_kernel((0.1, 0.1), sigma=0)
    assert k.shape == (1, 1)
    assert k[0, 0] == pytest.approx(1.0)


def test_cf_grid_mapping_attrs_without_pyproj_uses_conservative_longlat_fallback(
    monkeypatch,
):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyproj":
            raise ImportError("pyproj unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    attrs = _cf_grid_mapping_attrs("+proj=longlat")

    assert attrs == {
        "proj4_params": "+proj=longlat",
        "grid_mapping_name": "latitude_longitude",
    }


def test_aggregate_returns_dataframe():
    foot = _make_footprint(n_times=1)
    t0 = pd.Timestamp("2023-01-01 12:00")
    foot.data.loc[t0, 39.05, -113.95] = 1e-4

    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.aggregate(coords=[(-113.95, 39.05)], time_bins=bins)

    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0, 0] == pytest.approx(1e-4)


def test_aggregate_multiple_bins():
    foot = _make_footprint(n_times=3)
    t0 = pd.Timestamp("2023-01-01 12:00")
    t1 = t0 + pd.Timedelta(hours=1)
    t2 = t0 + pd.Timedelta(hours=2)

    foot.data.loc[t0, 39.05, -113.95] = 1e-4
    foot.data.loc[t1, 39.05, -113.95] = 2e-4
    foot.data.loc[t2, 39.05, -113.95] = 3e-4

    bins = pd.interval_range(start=t0, periods=3, freq="1h", closed="left")
    result = foot.aggregate(coords=[(-113.95, 39.05)], time_bins=bins)

    assert list(result.columns) == list(bins.left)
    assert result.iloc[0, 0] == pytest.approx(1e-4)
    assert result.iloc[0, 1] == pytest.approx(2e-4)
    assert result.iloc[0, 2] == pytest.approx(3e-4)


def test_netcdf_roundtrip_preserves_name(tmp_path):
    foot = _make_footprint(n_times=1)
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    path = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    foot.to_netcdf(path)

    loaded = Footprint.from_netcdf(path)
    assert loaded.name == "slv"
    assert loaded.grid.xres == pytest.approx(0.1)


def test_from_netcdf_forwards_chunks_to_xarray(tmp_path, monkeypatch):
    foot = _make_footprint(n_times=1)
    path = tmp_path / "chunked_foot.nc"
    foot.to_netcdf(path)
    seen_kwargs = {}
    real_open_dataset = xr.open_dataset

    def fake_open_dataset(path_arg, **kwargs):
        seen_kwargs.update(kwargs)
        return real_open_dataset(path_arg)

    monkeypatch.setattr("stilt.footprint.xr.open_dataset", fake_open_dataset)

    loaded = Footprint.from_netcdf(path, chunks={"time": 1})

    assert loaded.name == "slv"
    assert seen_kwargs["chunks"] == {"time": 1}


def test_netcdf_writes_cf_grid_mapping_and_coordinates(tmp_path):
    foot = _make_footprint(n_times=1)
    path = tmp_path / "cf_foot.nc"

    foot.to_netcdf(path)

    ds = xr.open_dataset(path)
    try:
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert "crs" in ds
        assert ds["crs"].attrs["grid_mapping_name"] == "latitude_longitude"
        assert ds["foot"].attrs["grid_mapping"] == "crs"
        assert ds["lon"].attrs["standard_name"] == "longitude"
        assert ds["lon"].attrs["units"] == "degrees_east"
        assert ds["lat"].attrs["standard_name"] == "latitude"
        assert ds["lat"].attrs["units"] == "degrees_north"
        assert ds["time"].attrs["standard_name"] == "time"
        assert "receptor" in ds.attrs
        assert "receptor_time" not in ds.coords
        assert "receptor_longitude" not in ds.coords
        assert "receptor_latitude" not in ds.coords
        assert "receptor_altitude" not in ds.coords
    finally:
        ds.close()


def test_netcdf_roundtrip_prefers_stored_name_attr(tmp_path):
    foot = _make_footprint(n_times=1)
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    original = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    renamed = sim_dir / "202301011200_-111.85_40.77_5_wrong_foot.nc"
    foot.to_netcdf(original)

    ds = xr.open_dataset(original)
    ds.load()
    ds.close()
    ds.attrs["name"] = "stored"
    ds.to_netcdf(renamed)

    loaded = Footprint.from_netcdf(renamed)
    assert loaded.name == "stored"


def test_netcdf_roundtrip_preserves_transform_specs(tmp_path):
    foot = _make_footprint(n_times=1)
    foot.config = FootprintConfig(
        grid=foot.grid,
        transforms=[
            VerticalOperatorTransformSpec(
                kind="vertical_operator",
                mode="ak",
                levels=[0.0, 1000.0],
                values=[0.1, 0.9],
                coordinate="xhgt",
            )
        ],
    )
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    path = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    foot.to_netcdf(path)

    loaded = Footprint.from_netcdf(path)

    assert len(loaded.config.transforms) == 1
    transform = loaded.config.transforms[0]
    assert isinstance(transform, VerticalOperatorTransformSpec)
    assert transform.mode == "ak"


def test_netcdf_roundtrip_preserves_empty_metadata(tmp_path, point_receptor):
    particles = _particles_in_domain()
    particles["long"] = 0.0
    particles["lati"] = 0.0
    foot = Footprint.calculate(
        particles, receptor=point_receptor, config=_foot_config()
    )
    path = tmp_path / "empty_foot.nc"

    foot.to_netcdf(path)
    loaded = Footprint.from_netcdf(path)

    assert loaded.is_empty is True
    assert loaded.empty_reason == "outside_domain"
    assert float(loaded.data.sum()) == pytest.approx(0.0)


def test_netcdf_roundtrip_no_name(tmp_path):
    """Footprint with no name (unnamed) roundtrips as empty string."""
    foot = _make_footprint(n_times=1)
    foot.name = ""
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    path = sim_dir / "202301011200_-111.85_40.77_5_foot.nc"
    foot.to_netcdf(path)
    loaded = Footprint.from_netcdf(path)
    assert loaded.name == ""


def test_netcdf_roundtrip_with_timezone_aware_time(tmp_path):
    receptor_time = pd.Timestamp("2023-01-01 12:00:00+00:00")
    receptor = PointReceptor(
        time=receptor_time,
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.8, ymin=39.0, ymax=39.2, xres=0.1, yres=0.1)
    )
    data = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=["time", "lat", "lon"],
        coords={
            "time": [receptor_time],
            "lat": np.array([39.05, 39.15]),
            "lon": np.array([-113.95, -113.85]),
        },
        attrs={"units": "ppm (umol-1 m2 s)"},
    )
    foot = Footprint(receptor=receptor, config=config, data=data, name="slv")

    path = tmp_path / "timezone_aware_foot.nc"
    foot.to_netcdf(path)

    loaded = Footprint.from_netcdf(path)
    assert tuple(loaded.data.dims) == ("time", "lat", "lon")
    assert loaded.data.shape == (1, 2, 2)
    assert float(loaded.data.sum()) == pytest.approx(4.0)
    # Time coord and receptor.time must come back as naive UTC.
    assert loaded.receptor.time.tzinfo is None
    loaded_time = pd.Timestamp(loaded.data.time.values[0])
    assert loaded_time.tzinfo is None
    assert loaded_time == pd.Timestamp("2023-01-01 12:00:00")


def test_time_range_single_timestep():
    foot = _make_footprint(n_times=1)
    start, stop = foot.time_range
    assert stop == start
    assert isinstance(start, dt.datetime)


def test_time_range_multiple_timesteps():
    foot = _make_footprint(n_times=3)
    start, stop = foot.time_range
    assert stop > start


def test_integrate_over_time_no_bounds():
    foot = _make_footprint(n_times=3)
    t0 = pd.Timestamp("2023-01-01 12:00")
    foot.data.loc[t0, 39.05, -113.95] = 1e-4
    result = foot.integrate_over_time()
    assert result.shape == (len(foot.data.lat), len(foot.data.lon))
    assert float(result.sel(lat=39.05, lon=-113.95)) == pytest.approx(1e-4)


def test_integrate_over_time_with_bounds():
    foot = _make_footprint(n_times=3)
    t0 = pd.Timestamp("2023-01-01 12:00")
    t1 = t0 + pd.Timedelta(hours=1)
    t2 = t0 + pd.Timedelta(hours=2)
    foot.data.loc[t0, 39.05, -113.95] = 1e-4
    foot.data.loc[t2, 39.05, -113.95] = 3e-4
    # Restrict to [t0, t1] - only t0 included
    result = foot.integrate_over_time(start=t0.to_pydatetime(), end=t1.to_pydatetime())
    assert float(result.sel(lat=39.05, lon=-113.95)) == pytest.approx(1e-4)


def test_aggregate_zero_values_in_domain():
    """Grid coordinate within domain but with all-zero values returns zeros."""
    foot = _make_footprint(n_times=1)
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    # -113.95 and 39.05 are valid cell centers in the footprint
    result = foot.aggregate(coords=[(-113.95, 39.05)], time_bins=bins)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# Footprint.calculate()
# ---------------------------------------------------------------------------


def _particles_in_domain(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create synthetic particle data within [-114, -113] x [39, 40]."""
    rng = np.random.default_rng(seed)
    times = [-60] * n + [-120] * n
    indx = list(range(1, n + 1)) * 2
    return pd.DataFrame(
        {
            "time": times,
            "indx": indx,
            "long": rng.uniform(-113.9, -113.1, n * 2),
            "lati": rng.uniform(39.1, 39.9, n * 2),
            "zagl": rng.uniform(5, 100, n * 2),
            "foot": rng.uniform(1e-6, 1e-4, n * 2),
        }
    )


def _foot_config(xres=0.1, yres=0.1):
    return FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=xres, yres=yres)
    )


def test_calculate_returns_footprint_instance(point_receptor):
    particles = _particles_in_domain()
    foot = Footprint.calculate(
        particles, receptor=point_receptor, config=_foot_config()
    )
    assert foot is not None
    assert isinstance(foot, Footprint)


def test_calculate_dims_are_time_lat_lon(point_receptor):
    particles = _particles_in_domain()
    foot = Footprint.calculate(
        particles, receptor=point_receptor, config=_foot_config()
    )
    assert foot is not None
    assert tuple(foot.data.dims) == ("time", "lat", "lon")


def test_calculate_returns_explicit_empty_footprint_when_particles_outside_domain(
    point_receptor,
):
    """All particles outside the domain should return an explicit empty footprint."""
    particles = _particles_in_domain()
    particles["long"] = 0.0  # far outside [-114, -113]
    particles["lati"] = 0.0
    config = _foot_config()
    result = Footprint.calculate(particles, receptor=point_receptor, config=config)
    assert isinstance(result, Footprint)
    assert result.is_empty is True
    assert result.empty_reason == "outside_domain"
    assert float(result.data.sum()) == pytest.approx(0.0)


def test_calculate_assigns_name(point_receptor):
    particles = _particles_in_domain()
    foot = Footprint.calculate(
        particles, receptor=point_receptor, config=_foot_config(), name="test"
    )
    assert foot is not None
    assert foot.name == "test"


def test_calculate_time_integrate_collapses_to_single_timestep(point_receptor):
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1),
        time_integrate=True,
    )
    particles = _particles_in_domain()
    foot = Footprint.calculate(particles, receptor=point_receptor, config=config)
    assert foot is not None
    assert len(foot.data.time) == 1


def test_calculate_nonnegative_foot_values(point_receptor):
    """Footprint values should be non-negative."""
    particles = _particles_in_domain()
    foot = Footprint.calculate(
        particles, receptor=point_receptor, config=_foot_config()
    )
    assert foot is not None
    assert float(foot.data.values.min()) >= 0.0


def test_calculate_smooth_factor_zero(point_receptor):
    """smooth_factor=0 is equivalent to no smoothing (identity kernel)."""
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1),
        smooth_factor=0.0,
    )
    particles = _particles_in_domain()
    foot = Footprint.calculate(particles, receptor=point_receptor, config=config)
    assert foot is not None


def test_calculate_grid_property(point_receptor):
    particles = _particles_in_domain()
    config = _foot_config()
    foot = Footprint.calculate(particles, receptor=point_receptor, config=config)
    assert foot is not None
    assert foot.grid.xres == pytest.approx(0.1)


def test_make_gauss_kernel_normalized():
    """Kernel values sum to 1.0."""
    k = _make_gauss_kernel((0.1, 0.1), sigma=0.5)
    assert k.sum() == pytest.approx(1.0, rel=1e-6)


def test_make_gauss_kernel_odd_shape():
    """Kernel shape must be odd in both dimensions."""
    k = _make_gauss_kernel((0.1, 0.1), sigma=0.3)
    assert k.shape[0] % 2 == 1
    assert k.shape[1] % 2 == 1


def test_interpolation_times_match_r_stilt_schedule():
    times = _interpolation_times(-1)

    assert times[0] == pytest.approx(0.0)
    assert times[np.where(np.isclose(times, -10.0))[0][0]] == pytest.approx(-10.0)
    assert times[np.where(np.isclose(times, -20.0))[0][0]] == pytest.approx(-20.0)
    assert times[-1] == pytest.approx(-100.0)
    assert len(times) == 311


def test_interpolate_early_timesteps_preserves_window_foot_sums():
    particles = pd.DataFrame(
        {
            "time": [-5.0, -50.0, -120.0, -5.0, -50.0, -120.0],
            "indx": [1, 1, 1, 2, 2, 2],
            "long": [-113.0, -114.0, -115.0, -112.0, -113.5, -115.0],
            "lati": [39.0, 40.0, 41.0, 39.5, 40.5, 41.5],
            "foot": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0],
        }
    )
    original_atime = np.abs(particles["time"])
    original_sums = [
        particles.loc[original_atime <= 10, "foot"].sum(),
        particles.loc[(original_atime > 10) & (original_atime <= 20), "foot"].sum(),
        particles.loc[(original_atime > 20) & (original_atime <= 100), "foot"].sum(),
    ]

    interpolated = _interpolate_early_timesteps(
        particles, xres=0.01, yres=0.01, time_sign=-1
    )

    assert len(interpolated) > len(particles)
    atime = np.abs(interpolated["time"])
    interpolated_sums = [
        interpolated.loc[atime <= 10, "foot"].sum(),
        interpolated.loc[(atime > 10) & (atime <= 20), "foot"].sum(),
        interpolated.loc[(atime > 20) & (atime <= 100), "foot"].sum(),
    ]
    assert interpolated_sums == pytest.approx(original_sums)
    assert interpolated[["long", "lati", "foot"]].isna().sum().sum() == 0
