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
    _compute_kernel_bandwidths,
    _grid_cell_starts,
    _interpolate_early_timesteps,
    _interpolation_times,
    _make_gauss_kernel,
    _project_particles_to_crs,
    _wrap_antimeridian_longitudes,
)
from stilt.receptors import PointReceptor
from stilt.trajectory import calc_plume_dilution


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


def test_grid_cell_starts_use_complete_half_open_cells():
    starts = _grid_cell_starts(0.0, 1.0, 0.3)

    np.testing.assert_allclose(starts, [0.0, 0.3, 0.6])


def test_grid_cell_starts_keep_decimal_boundary_cell():
    starts = _grid_cell_starts(-113.0, -111.0, 0.01)

    assert len(starts) == 200
    assert starts[0] == pytest.approx(-113.0)
    assert starts[-1] == pytest.approx(-111.01)


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


def test_calculate_irregular_grid_uses_complete_cells(point_receptor):
    config = FootprintConfig(
        grid=Grid(
            xmin=-114.0,
            xmax=-113.0,
            ymin=39.0,
            ymax=40.0,
            xres=0.3,
            yres=0.4,
        ),
        smooth_factor=0.0,
    )
    particles = _particles_in_domain()

    foot = Footprint.calculate(particles, receptor=point_receptor, config=config)

    np.testing.assert_allclose(foot.data.lon.values, [-113.85, -113.55, -113.25])
    np.testing.assert_allclose(foot.data.lat.values, [39.2, 39.6])
    assert foot.data.shape == (2, 2, 3)


def test_calculate_empty_irregular_grid_uses_complete_cells(point_receptor):
    config = FootprintConfig(
        grid=Grid(
            xmin=-114.0,
            xmax=-113.0,
            ymin=39.0,
            ymax=40.0,
            xres=0.3,
            yres=0.4,
        ),
        smooth_factor=0.0,
    )
    particles = _particles_in_domain()
    particles["long"] = 0.0
    particles["lati"] = 0.0

    foot = Footprint.calculate(particles, receptor=point_receptor, config=config)

    assert foot.is_empty is True
    np.testing.assert_allclose(foot.data.lon.values, [-113.85, -113.55, -113.25])
    np.testing.assert_allclose(foot.data.lat.values, [39.2, 39.6])
    assert foot.data.shape == (2, 2, 3)


def test_calculate_non_square_resolution_is_finite(point_receptor):
    config = FootprintConfig(
        grid=Grid(
            xmin=-114.0,
            xmax=-113.0,
            ymin=39.0,
            ymax=40.0,
            xres=0.01,
            yres=0.05,
        )
    )
    particles = _particles_in_domain()

    foot = Footprint.calculate(particles, receptor=point_receptor, config=config)

    assert foot.data.sizes["lon"] == 100
    assert foot.data.sizes["lat"] == 20
    assert np.isfinite(foot.data.values).all()
    assert float(foot.data.values.min()) >= 0.0


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


def test_make_gauss_kernel_symmetric():
    """
    For equal x/y resolution, the kernel is symmetric under both axis flips and
    transposition.  An asymmetric kernel would create directional bias — footprints
    would incorrectly favour one compass direction over another.
    """
    k = _make_gauss_kernel((0.01, 0.01), sigma=0.3)
    np.testing.assert_array_equal(
        k, k.T, err_msg="kernel must be symmetric under transpose"
    )
    np.testing.assert_array_equal(
        k, k[::-1, :], err_msg="kernel must be symmetric about row axis"
    )
    np.testing.assert_array_equal(
        k, k[:, ::-1], err_msg="kernel must be symmetric about col axis"
    )


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


def test_interpolate_early_timesteps_matches_r_na_omit_with_extra_columns():
    particles = pd.DataFrame(
        {
            "time": [-5.0, -50.0, -120.0, -5.0, -50.0, -120.0],
            "indx": [1, 1, 1, 2, 2, 2],
            "long": [-113.0, -114.0, -115.0, -112.0, -113.5, -115.0],
            "lati": [39.0, 40.0, 41.0, 39.5, 40.5, 41.5],
            "zagl": [5.0, 6.0, 7.0, 5.0, 6.0, 7.0],
            "foot": [1.0, 2.0, 4.0, 3.0, 5.0, 7.0],
        }
    )

    interpolated = _interpolate_early_timesteps(
        particles, xres=0.01, yres=0.01, time_sign=-1
    )

    expected = particles.sort_values(
        ["indx", "time"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(interpolated, expected, check_dtype=False)


# ---------------------------------------------------------------------------
# Mathematical invariants — no R required
#
# These properties must hold from pure math regardless of R-STILT agreement.
# They are the foundation of using PYSTILT footprints in linear inversion:
#   concentration = sum(footprint * flux)
# If the footprint is not linear in the particle sensitivity values, or if
# smoothing is not mass-conservative, that inversion is invalid.
# ---------------------------------------------------------------------------


def _interior_particles(n: int = 40, seed: int = 55) -> pd.DataFrame:
    """Particles well inside the domain, no t=0 receptor row."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": [-60.0] * n,
            "indx": [float(i + 1) for i in range(n)],
            "long": rng.uniform(-113.8, -113.2, n),
            "lati": rng.uniform(39.2, 39.8, n),
            "zagl": [5.0] * n,
            "foot": rng.uniform(1e-5, 1e-4, n),
        }
    )


def _interior_config(smooth_factor: float = 0.0) -> FootprintConfig:
    return FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1),
        smooth_factor=smooth_factor,
    )


def test_calculate_linearity_in_foot_values(point_receptor):
    """
    Scaling all particle foot values by a constant scales the output by the same
    factor.

    This is the foundational property of Bayesian inversion: concentration =
    integral(footprint * flux). If the footprint is not linear in the particle
    foot values, that integral is invalid.  All operations in Footprint.calculate
    are linear in foot (bincount, Gaussian convolution, division by n_particles),
    so the output must scale exactly.
    """
    particles = _interior_particles()
    config = _interior_config(smooth_factor=1.0)

    foot_1x = Footprint.calculate(particles, receptor=point_receptor, config=config)

    particles_2x = particles.copy()
    particles_2x["foot"] = particles_2x["foot"] * 2.0
    foot_2x = Footprint.calculate(particles_2x, receptor=point_receptor, config=config)

    np.testing.assert_allclose(
        foot_2x.data.values,
        foot_1x.data.values * 2.0,
        rtol=1e-10,
        err_msg="Footprint must scale linearly with particle foot values",
    )


def test_calculate_total_equals_normalized_input_sum_at_zero_smooth(point_receptor):
    """
    With smooth_factor=0, total footprint = sum(in-domain foot) / n_particles.

    STILT normalizes by particle count so that the footprint is intensive (per
    particle).  The 1×1 identity kernel does not move any sensitivity between
    cells, so the grid sum must exactly equal the un-normalized particle sum
    divided by the ensemble size.
    """
    particles = _interior_particles()
    n = particles["indx"].nunique()
    config = _interior_config(smooth_factor=0.0)

    foot = Footprint.calculate(particles, receptor=point_receptor, config=config)

    expected = float(particles["foot"].sum()) / n
    assert float(foot.data.values.sum()) == pytest.approx(expected, rel=1e-10)


def test_calculate_gaussian_smoothing_preserves_total_sensitivity(point_receptor):
    """
    Gaussian smoothing does not create or destroy total footprint sensitivity.

    The kernel sums to 1 (verified separately in test_make_gauss_kernel_normalized)
    and particles are placed well inside the domain so the Gaussian tails do not
    spill outside the grid boundary.  Any loss of total sensitivity from smoothing
    would silently bias flux inversion toward underestimating emissions.
    """
    particles = _interior_particles()
    config_0 = _interior_config(smooth_factor=0.0)
    config_s = _interior_config(smooth_factor=1.0)

    foot_0 = Footprint.calculate(particles, receptor=point_receptor, config=config_0)
    foot_s = Footprint.calculate(particles, receptor=point_receptor, config=config_s)

    total_0 = float(foot_0.data.values.sum())
    total_s = float(foot_s.data.values.sum())

    assert total_s == pytest.approx(total_0, rel=1e-5), (
        f"Smoothing changed total footprint: {total_0:.6g} → {total_s:.6g} "
        f"(Δ = {abs(total_s - total_0) / total_0:.2e})"
    )


def test_calculate_reproducible(point_receptor):
    """
    Calling Footprint.calculate twice with identical inputs returns identical arrays.

    Statefulness bugs (e.g. a mutable module-level cache that accumulates across
    calls) would cause different runs of the same simulation to diverge silently.
    """
    particles = _interior_particles()
    config = _interior_config(smooth_factor=1.0)

    foot1 = Footprint.calculate(particles, receptor=point_receptor, config=config)
    foot2 = Footprint.calculate(particles, receptor=point_receptor, config=config)

    np.testing.assert_array_equal(
        foot1.data.values,
        foot2.data.values,
        err_msg="Footprint.calculate must be deterministic — identical inputs must produce identical outputs",
    )


def test_calculate_time_integrate_equals_sum_of_time_slices(point_receptor):
    """
    time_integrate=True must equal summing the per-time-step footprint.

    If the collapsed footprint were computed differently from the sum of slices,
    daily-average footprints used in Bayesian inversion would silently differ from
    the sum of the hourly footprints researchers expect.
    """
    particles = _particles_in_domain()
    grid = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)

    foot_ti = Footprint.calculate(
        particles,
        receptor=point_receptor,
        config=FootprintConfig(grid=grid, time_integrate=True),
    )
    foot_no = Footprint.calculate(
        particles,
        receptor=point_receptor,
        config=FootprintConfig(grid=grid, time_integrate=False),
    )

    np.testing.assert_allclose(
        foot_ti.data.values.squeeze(),
        foot_no.data.sum("time").values,
        rtol=1e-10,
        err_msg="time_integrate=True must equal the sum over all individual time slices",
    )


def test_calculate_smooth_zero_assigns_exact_cells(point_receptor):
    """
    With smooth_factor=0, each particle's foot goes entirely into the one cell it
    falls in — no neighbouring cells receive any spillover.

    This tests the 1×1 identity kernel path (sigma=0 → _make_gauss_kernel returns
    [[1.0]]).  A bug here would mean that the ``permute.f90``-equivalent scatter
    operation distributes sensitivity to wrong cells, corrupting the spatial pattern
    of all no-smooth footprints.
    """
    grid = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)
    config = FootprintConfig(grid=grid, smooth_factor=0.0)

    # 10 identical particles exactly at the centre of cell (start=-113.9, centre=-113.85)
    n = 10
    foot_val = 1e-4
    particles = pd.DataFrame(
        {
            "time": [-60.0] * n,
            "indx": [float(i + 1) for i in range(n)],
            "long": [-113.85] * n,
            "lati": [39.05] * n,
            "zagl": [5.0] * n,
            "foot": [foot_val] * n,
        }
    )

    foot = Footprint.calculate(particles, receptor=point_receptor, config=config)

    # total = sum(foot) / n_particles = (n * foot_val) / n = foot_val
    assert float(foot.data.values.sum()) == pytest.approx(foot_val, rel=1e-10)

    # Exactly one non-zero cell across all time layers
    nonzero_count = int((foot.data.values > 0).sum())
    assert nonzero_count == 1, (
        f"smooth_factor=0: expected exactly 1 non-zero cell, got {nonzero_count}"
    )


def test_concentration_reconstruction_from_known_footprint(point_receptor):
    """
    c = Σ foot[i,j] * q[i,j] recovers the analytically expected concentration.

    This is the fundamental identity that Bayesian flux inversion relies on:
    a receptor concentration enhancement equals the dot product of the footprint
    sensitivity matrix with the surface flux field.  If this identity is broken
    — by a normalization error, wrong cell assignment, or unit mismatch — every
    inferred emission estimate is wrong, silently.

    Setup (smooth_factor=0 so values are exact, no Gaussian spread):
      - Cluster A: 10 particles at cell centre (-113.85°, 39.05°), foot = 2e-4
      - Cluster B: 10 particles at cell centre (-113.35°, 39.55°), foot = 3e-4
      - 20 total unique particles (n_particles = 20)

    Analytical footprint values:
      F_A = n_A * foot_A / n_particles = 10 * 2e-4 / 20 = 1e-4  ppm/(μmol m⁻² s⁻¹)
      F_B = n_B * foot_B / n_particles = 10 * 3e-4 / 20 = 1.5e-4 ppm/(μmol m⁻² s⁻¹)

    Applied flux field (non-zero only at the two cluster cells):
      q_A = 5.0  μmol m⁻² s⁻¹   (roughly a moderate CH₄ surface source)
      q_B = 8.0  μmol m⁻² s⁻¹

    Expected concentration:
      c = F_A * q_A + F_B * q_B = 1e-4 * 5 + 1.5e-4 * 8 = 1.7e-3 ppm ≈ 1.7 ppb
    """
    grid = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)
    config = FootprintConfig(grid=grid, smooth_factor=0.0)

    n_a, n_b = 10, 10
    n_total = n_a + n_b
    foot_a, foot_b = 2e-4, 3e-4

    particles = pd.DataFrame(
        {
            "time": [-60.0] * n_total,
            "indx": [float(i + 1) for i in range(n_total)],
            "long": [-113.85] * n_a + [-113.35] * n_b,
            "lati": [39.05] * n_a + [39.55] * n_b,
            "zagl": [5.0] * n_total,
            "foot": [foot_a] * n_a + [foot_b] * n_b,
        }
    )

    foot = Footprint.calculate(particles, receptor=point_receptor, config=config)

    # Verify the footprint cell values are exactly what the formula predicts.
    expected_fa = n_a * foot_a / n_total  # 1e-4
    expected_fb = n_b * foot_b / n_total  # 1.5e-4

    f_a = float(foot.data.sel(lon=-113.85, lat=39.05, method="nearest").sum())
    f_b = float(foot.data.sel(lon=-113.35, lat=39.55, method="nearest").sum())

    assert f_a == pytest.approx(expected_fa, rel=1e-10), (
        f"Cell A footprint: expected {expected_fa:.3e}, got {f_a:.3e}"
    )
    assert f_b == pytest.approx(expected_fb, rel=1e-10), (
        f"Cell B footprint: expected {expected_fb:.3e}, got {f_b:.3e}"
    )

    # Apply flux field (non-zero at the two cluster cells only).
    q_a, q_b = 5.0, 8.0
    flux = np.zeros_like(foot.data.values)

    lons = foot.data.lon.values
    lats = foot.data.lat.values
    lon_a = int(np.argmin(np.abs(lons - (-113.85))))
    lat_a = int(np.argmin(np.abs(lats - 39.05)))
    lon_b = int(np.argmin(np.abs(lons - (-113.35))))
    lat_b = int(np.argmin(np.abs(lats - 39.55)))

    flux[:, lat_a, lon_a] = q_a
    flux[:, lat_b, lon_b] = q_b

    # c = F_A * q_A + F_B * q_B
    c_computed = float((foot.data.values * flux).sum())
    c_expected = expected_fa * q_a + expected_fb * q_b  # 1.7e-3 ppm

    assert c_computed == pytest.approx(c_expected, rel=1e-10), (
        f"Concentration reconstruction: c = {c_computed:.4g} ppm, "
        f"expected {c_expected:.4g} ppm  (~{c_expected * 1e3:.2f} ppb)"
    )


def test_hnf_correction_invariants():
    """
    Mathematical invariants of the HNF near-field plume-dilution correction.

    The HNF correction replaces the HYSPLIT-raw foot with a Gaussian near-field
    value (0.02897 / (plume * dens) * samt * 60) when the plume has not yet
    grown to fill the mixing layer.  This can be larger or smaller than the raw
    value.  The invariants that must always hold:

    1. Corrected foot is always positive.
    2. When plume >= pbl_mixing, foot is left unchanged (identity path).
    3. The raw values are preserved in `foot_no_hnf_dilution`.
    """
    rng = np.random.default_rng(77)
    n = 50
    raw_foot = rng.uniform(1e-5, 1e-3, n)
    # Force some particles to have large plume (sigma >> mlht) so the identity
    # path is exercised.  Large sigw + long time → large plume.
    sigw = np.concatenate(
        [
            rng.uniform(0.01, 0.5, n // 2),  # small sigma → near-field path
            rng.uniform(5.0, 20.0, n // 2),  # large sigma → identity path
        ]
    )
    particles = pd.DataFrame(
        {
            "time": np.concatenate(
                [
                    rng.uniform(-0.5, -0.1, n // 2),  # short time → small plume
                    rng.uniform(-6.0, -5.0, n // 2),  # long time → large plume
                ]
            ),
            "indx": [float(i + 1) for i in range(n)],
            "long": [-112.0] * n,
            "lati": [40.5] * n,
            "zagl": [5.0] * n,
            "foot": raw_foot,
            "mlht": [500.0] * n,  # pbl_mixing = 0.5 * 500 = 250 m
            "dens": [1.2] * n,
            "samt": [60.0] * n,
            "sigw": sigw,
            "tlgr": [100.0] * n,
        }
    )

    result = calc_plume_dilution(particles.copy(), r_zagl=5.0, veght=0.5)

    # Invariant 1: corrected foot is always positive.
    assert np.all(result["foot"].values > 0), (
        "HNF-corrected foot values must be positive"
    )

    # Invariant 2: identity path — particles where plume >= pbl_mixing
    # are unchanged.  Reconstruct plume = r_zagl + sigma to find them.
    abs_time_s = np.abs(particles["time"] * 60)
    tlgr = particles["tlgr"]
    sigma = (
        particles["samt"]
        * np.sqrt(2)
        * particles["sigw"]
        * np.sqrt(tlgr * abs_time_s + tlgr**2 * np.exp(-abs_time_s / tlgr) - 1)
    )
    plume = 5.0 + sigma  # r_zagl=5 + sigma (single timestep, cumsum is sigma itself)
    pbl_mixing = 0.5 * particles["mlht"]
    identity_mask = (plume >= pbl_mixing).to_numpy()

    np.testing.assert_allclose(
        result.loc[identity_mask, "foot"].values,
        raw_foot[identity_mask],
        rtol=1e-12,
        err_msg="Particles with plume >= pbl_mixing must have foot unchanged",
    )

    # Invariant 3: raw values preserved in foot_no_hnf_dilution.
    np.testing.assert_array_equal(
        result["foot_no_hnf_dilution"].values,
        raw_foot,
        err_msg="foot_no_hnf_dilution must equal the original foot values",
    )


# ---------------------------------------------------------------------------
# Python-only helper tests: branch coverage for paths the live-R fidelity
# tests cannot reliably target. These run in CI on every Python version
# without needing R, Rscript, or HYSPLIT.
# ---------------------------------------------------------------------------


def test_wrap_antimeridian_longitudes_global_branch():
    """xdist == 0 (global 360° grid) anchors to [-180, 180] without wrapping."""
    p = pd.DataFrame({"long": [-179.0, 0.0, 179.0]})
    out, xmin, xmax, wrapped = _wrap_antimeridian_longitudes(p, xmin=-180.0, xmax=180.0)
    assert xmin == -180.0
    assert xmax == 180.0
    assert wrapped is False
    # Particle longitudes must be unchanged in the global branch.
    np.testing.assert_array_equal(out["long"].values, p["long"].values)


def test_wrap_antimeridian_longitudes_crossing_branch():
    """xmax < xmin (dateline crossing) rotates longitudes into [0, 360)."""
    p = pd.DataFrame({"long": [179.0, -179.0, 170.0, -170.0]})
    out, xmin, xmax, wrapped = _wrap_antimeridian_longitudes(p, xmin=170.0, xmax=-170.0)
    assert wrapped is True
    # Bounds wrap to 170, 190 in [0, 360) space.
    assert xmin == pytest.approx(170.0)
    assert xmax == pytest.approx(190.0)
    expected = np.array([179.0, 181.0, 170.0, 190.0])
    np.testing.assert_allclose(out["long"].values, expected)


def test_wrap_antimeridian_longitudes_partial_wrap_branch():
    """xmax > 180 (partial wrap, e.g. xmin=170, xmax=200) also rotates."""
    p = pd.DataFrame({"long": [175.0, -175.0]})
    out, xmin, xmax, wrapped = _wrap_antimeridian_longitudes(p, xmin=170.0, xmax=200.0)
    assert wrapped is True
    assert xmin == pytest.approx(170.0)
    assert xmax == pytest.approx(200.0)
    np.testing.assert_allclose(out["long"].values, np.array([175.0, 185.0]))


def test_wrap_antimeridian_longitudes_no_wrap_branch():
    """Standard CONUS domain (xmin=-113, xmax=-111) returns particles unchanged."""
    p = pd.DataFrame({"long": [-112.5, -111.5]})
    out, xmin, xmax, wrapped = _wrap_antimeridian_longitudes(
        p, xmin=-113.0, xmax=-111.0
    )
    assert wrapped is False
    assert xmin == -113.0
    assert xmax == -111.0
    np.testing.assert_array_equal(out["long"].values, p["long"].values)


def test_project_particles_to_crs_raises_on_missing_pyproj(monkeypatch):
    """Non-longlat path surfaces a clear ImportError when pyproj is absent."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyproj":
            raise ImportError("pyproj missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = pd.DataFrame({"long": [-112.0], "lati": [40.0]})
    with pytest.raises(ImportError, match="pyproj"):
        _project_particles_to_crs(
            p,
            projection="+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs",
            xmin=-113.0,
            xmax=-111.0,
            ymin=39.0,
            ymax=41.0,
        )


def test_project_particles_to_crs_rejects_invalid_proj_string():
    """An unparseable proj4 string surfaces from pyproj as a parse error."""
    from pyproj.exceptions import CRSError

    p = pd.DataFrame({"long": [-112.0], "lati": [40.0]})
    with pytest.raises(CRSError):
        _project_particles_to_crs(
            p,
            projection="+proj=nope-not-a-projection",
            xmin=-113.0,
            xmax=-111.0,
            ymin=39.0,
            ymax=41.0,
        )


def test_compute_kernel_bandwidths_single_particle_returns_zero_sigma():
    """
    Zero-variance edge case: a single particle has var(long)=var(lati)=NaN,
    which R's na.omit() drops. PYSTILT's helper returns w=0 (identity kernel)
    instead, so a one-particle trajectory still produces a valid (degenerate)
    footprint rather than crashing.
    """
    p = pd.DataFrame(
        {
            "indx": [1.0, 1.0],
            "rtime": [-1.0, -2.0],
            "time": [-1.0, -2.0],
            "long": [-112.0, -112.0],
            "lati": [40.5, 40.5],
            "foot": [1e-3, 1e-3],
        }
    )
    kernel_df, w = _compute_kernel_bandwidths(p, smooth_factor=1.0, is_longlat=True)
    np.testing.assert_array_equal(w, np.zeros_like(w))
    assert len(kernel_df) == len(w)


def test_compute_kernel_bandwidths_two_coincident_particles_returns_zero_sigma():
    """Two particles at identical positions have varsum=0 ⇒ w=0."""
    p = pd.DataFrame(
        {
            "indx": [1.0, 1.0, 2.0, 2.0],
            "rtime": [-1.0, -2.0, -1.0, -2.0],
            "time": [-1.0, -2.0, -1.0, -2.0],
            "long": [-112.0] * 4,
            "lati": [40.5] * 4,
            "foot": [1e-3] * 4,
        }
    )
    kernel_df, w = _compute_kernel_bandwidths(p, smooth_factor=1.0, is_longlat=True)
    np.testing.assert_array_equal(w, np.zeros_like(w))
