"""Tests for stilt.footprint helpers and aggregation."""

import builtins
import datetime as dt
import json
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.config import FootprintConfig, Grid
from stilt.exceptions import EmptyFootprint
from stilt.footprint import calculate, read_footprint
from stilt.footprint.gridding import (
    _compute_kernel_bandwidths,
    _interpolate_early_timesteps,
    _interpolation_times,
    _make_gauss_kernel,
    _project_particles_to_crs,
    _wrap_antimeridian_longitudes,
)
from stilt.footprint.io import _describe
from stilt.footprint.targets import Mesh, Zones
from stilt.particles import calc_plume_dilution
from stilt.receptors import PointReceptor
from stilt.spatial import _grid_cell_starts
from stilt.transforms import AveragingKernel


def _make_footprint(
    xres: float = 0.1, yres: float = 0.1, n_times: int = 1
) -> xr.DataArray:
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
    return _describe(data, receptor, config, "slv")


def test_grid_cell_starts_use_complete_half_open_cells():
    starts = _grid_cell_starts(0.0, 1.0, 0.3)

    np.testing.assert_allclose(starts, [0.0, 0.3, 0.6])


def test_grid_cell_starts_keep_decimal_boundary_cell():
    starts = _grid_cell_starts(-113.0, -111.0, 0.01)

    assert len(starts) == 200
    assert starts[0] == pytest.approx(-113.0)
    assert starts[-1] == pytest.approx(-111.01)


@pytest.mark.parametrize("resolution", [0.1, 0.05, 0.01, 0.002])
@pytest.mark.parametrize("base", [40.0, -112.0, -180.0])
def test_grid_cell_starts_keep_last_cell_for_inexact_bounds(base, resolution):
    # Bounds on the 0.01 grid are not exact in binary; the rounding error in
    # ``maximum - minimum`` must not cost the final cell (or the only cell).
    for i in range(100):
        minimum = round(base + i * 0.01, 10)
        for k in range(1, 60):
            maximum = round(minimum + k * resolution, 10)
            assert len(_grid_cell_starts(minimum, maximum, resolution)) == k, (
                minimum,
                maximum,
            )


def test_grid_cell_starts_inexact_bound_examples():
    assert len(_grid_cell_starts(40.45, 40.93, 0.01)) == 48
    np.testing.assert_allclose(_grid_cell_starts(40.0, 40.01, 0.01), [40.0])


def test_grid_cell_starts_still_drop_partial_cell():
    # The tolerance covers float roundoff only, not a real shortfall.
    assert len(_grid_cell_starts(40.45, 40.93 - 1e-8, 0.01)) == 47
    with pytest.raises(ValueError, match="at least one complete cell"):
        _grid_cell_starts(40.0, 40.01 - 1e-8, 0.01)


def test_make_gauss_kernel_sigma_zero():
    k = _make_gauss_kernel((0.1, 0.1), sigma=0)
    assert k.shape == (1, 1)
    assert k[0, 0] == pytest.approx(1.0)


def test_aggregate_returns_dataframe():
    foot = _make_footprint(n_times=1)
    t0 = pd.Timestamp("2023-01-01 12:00")
    foot.loc[t0, 39.05, -113.95] = 1e-4

    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(target=_one_cell(-113.95, 39.05, 0.1), time_bins=bins)

    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0, 0] == pytest.approx(1e-4)


def test_aggregate_multiple_bins():
    foot = _make_footprint(n_times=3)
    t0 = pd.Timestamp("2023-01-01 12:00")
    t1 = t0 + pd.Timedelta(hours=1)
    t2 = t0 + pd.Timedelta(hours=2)

    foot.loc[t0, 39.05, -113.95] = 1e-4
    foot.loc[t1, 39.05, -113.95] = 2e-4
    foot.loc[t2, 39.05, -113.95] = 3e-4

    bins = pd.interval_range(start=t0, periods=3, freq="1h", closed="left")
    result = foot.stilt.aggregate(target=_one_cell(-113.95, 39.05, 0.1), time_bins=bins)

    assert list(result.columns) == list(bins.left)
    assert result.iloc[0, 0] == pytest.approx(1e-4)
    assert result.iloc[0, 1] == pytest.approx(2e-4)
    assert result.iloc[0, 2] == pytest.approx(3e-4)


@pytest.mark.parametrize("closed", ["right", "both", "neither"])
def test_aggregate_rejects_bins_not_closed_on_the_left(closed):
    """A right-closed bin would silently be summed as if left-closed (#58)."""
    foot = _make_footprint(n_times=2)
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=2, freq="1h", closed=closed)

    with pytest.raises(ValueError, match="closed on the left"):
        foot.stilt.aggregate(target=_one_cell(-113.95, 39.05, 0.1), time_bins=bins)


def test_netcdf_roundtrip_preserves_name(tmp_path):
    foot = _make_footprint(n_times=1)
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    path = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    foot.stilt.to_netcdf(path)

    loaded = read_footprint(path)
    assert loaded.stilt.name == "slv"
    assert loaded.stilt.grid.xres == pytest.approx(0.1)


def test_from_netcdf_forwards_chunks_to_xarray(tmp_path, monkeypatch):
    foot = _make_footprint(n_times=1)
    path = tmp_path / "chunked_foot.nc"
    foot.stilt.to_netcdf(path)
    seen_kwargs = {}
    real_open_dataset = xr.open_dataset

    def fake_open_dataset(path_arg, **kwargs):
        seen_kwargs.update(kwargs)
        return real_open_dataset(path_arg)

    monkeypatch.setattr("stilt.footprint.io.xr.open_dataset", fake_open_dataset)

    loaded = read_footprint(path, chunks={"time": 1})

    assert loaded.stilt.name == "slv"
    assert seen_kwargs["chunks"] == {"time": 1}


def test_netcdf_writes_cf_grid_mapping_and_coordinates(tmp_path):
    foot = _make_footprint(n_times=1)
    path = tmp_path / "cf_foot.nc"

    foot.stilt.to_netcdf(path)

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
        assert "stilt_receptor" in ds["foot"].attrs
        assert "stilt_footprint" in ds["foot"].attrs
        assert ds["receptor"].item() == "202301011200_-111.85_40.77_5"
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
    foot.stilt.to_netcdf(original)

    ds = xr.open_dataset(original)
    ds.load()
    ds.close()
    ds["foot"].attrs["stilt_name"] = "stored"
    ds.to_netcdf(renamed)

    loaded = read_footprint(renamed)
    assert loaded.stilt.name == "stored"


def test_netcdf_roundtrip_preserves_transforms(tmp_path):
    foot = _make_footprint(n_times=1)
    kernel = AveragingKernel(levels=[0.0, 1000.0], values=[0.1, 0.9], coordinate="xhgt")
    config = FootprintConfig(grid=foot.stilt.grid, transforms=[kernel])
    foot = _describe(foot, foot.stilt.receptor, config, "slv")
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    path = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    foot.stilt.to_netcdf(path)

    loaded = read_footprint(path)

    assert len(loaded.stilt.config.transforms) == 1
    transform = loaded.stilt.config.transforms[0]
    assert isinstance(transform, AveragingKernel)
    assert transform == kernel


def test_netcdf_with_unimportable_transform_still_loads(tmp_path):
    # A footprint written where a user transform class was importable must
    # still open on a machine without that package; the transform is kept as
    # its settings mapping.
    foot = _make_footprint(n_times=1)
    config = FootprintConfig(
        grid=foot.stilt.grid,
        transforms=[AveragingKernel(levels=[0.0, 1000.0], values=[0.1, 0.9])],
    )
    foot = _describe(foot, foot.stilt.receptor, config, "slv")
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    original = sim_dir / "202301011200_-111.85_40.77_5_slv_foot.nc"
    patched = sim_dir / "202301011200_-111.85_40.77_5_user_foot.nc"
    foot.stilt.to_netcdf(original)

    missing_kind = "no_such_pkg_for_stilt_tests.transforms.MyKernel"
    ds = xr.open_dataset(original)
    ds.load()
    ds.close()
    settings = json.loads(ds["foot"].attrs["stilt_footprint"])
    settings["transforms"] = [
        {"kind": missing_kind, "levels": [0.0, 1000.0], "values": [0.1, 0.9]}
    ]
    ds["foot"].attrs["stilt_footprint"] = json.dumps(settings)
    ds.to_netcdf(patched)

    recorded = {"kind": missing_kind, "levels": [0.0, 1000.0], "values": [0.1, 0.9]}
    loaded = read_footprint(patched)
    with pytest.warns(UserWarning, match="could not be imported"):
        assert loaded.stilt.config.transforms == [recorded]

    # The mapping survives another write/read unchanged. The accessor read the
    # settings once above, so writing does not warn again.
    rewritten = sim_dir / "202301011200_-111.85_40.77_5_again_foot.nc"
    loaded.stilt.to_netcdf(rewritten)
    again = read_footprint(rewritten)
    with pytest.warns(UserWarning):
        assert again.stilt.config.transforms == [recorded]


def test_netcdf_roundtrip_no_name(tmp_path):
    """Footprint with no name (unnamed) roundtrips as empty string."""
    foot = _make_footprint(n_times=1)
    foot.attrs["stilt_name"] = ""
    sim_dir = tmp_path / "202301011200_-111.85_40.77_5"
    sim_dir.mkdir()
    path = sim_dir / "202301011200_-111.85_40.77_5_foot.nc"
    foot.stilt.to_netcdf(path)
    loaded = read_footprint(path)
    assert loaded.stilt.name == ""


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
    foot = _describe(data, receptor, config, "slv")

    path = tmp_path / "timezone_aware_foot.nc"
    foot.stilt.to_netcdf(path)

    loaded = read_footprint(path)
    assert tuple(loaded.dims) == ("time", "lat", "lon")
    assert loaded.shape == (1, 2, 2)
    assert float(loaded.sum()) == pytest.approx(4.0)
    # Time coord and receptor.time must come back as naive UTC.
    assert loaded.stilt.receptor.time.tzinfo is None
    loaded_time = pd.Timestamp(loaded.time.values[0])
    assert loaded_time.tzinfo is None
    assert loaded_time == pd.Timestamp("2023-01-01 12:00:00")


def test_aggregate_zero_values_in_domain():
    """Grid coordinate within domain but with all-zero values returns zeros."""
    foot = _make_footprint(n_times=1)
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    # -113.95 and 39.05 are valid cell centers in the footprint
    result = foot.stilt.aggregate(target=_one_cell(-113.95, 39.05, 0.1), time_bins=bins)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# Footprint.aggregate() — area integration onto coarse target grids
# ---------------------------------------------------------------------------


def _foot_on_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    *,
    xres: float,
    yres: float,
    t0: pd.Timestamp | None = None,
) -> xr.DataArray:
    """Build a Footprint from explicit lon/lat centers and a value array."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[None]
    n_times = values.shape[0]
    if t0 is None:
        t0 = pd.Timestamp("2023-01-01 12:00")
    times = [t0 + pd.Timedelta(hours=i) for i in range(n_times)]
    receptor = PointReceptor(
        time=t0.to_pydatetime(),
        longitude=float(lons.mean()),
        latitude=float(lats.mean()),
        altitude=5.0,
    )
    grid = Grid(
        xmin=float(lons.min() - xres / 2),
        xmax=float(lons.max() + xres / 2),
        ymin=float(lats.min() - yres / 2),
        ymax=float(lats.max() + yres / 2),
        xres=xres,
        yres=yres,
    )
    data = xr.DataArray(
        values,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"units": "ppm (umol-1 m2 s)"},
    )
    return _describe(data, receptor, FootprintConfig(grid=grid), "")


def _block_centers(n_blocks: int, res: float, origin: float = 0.0) -> np.ndarray:
    """Cell centers for ``n_blocks`` cells of width ``res`` starting at origin."""
    return origin + (np.arange(n_blocks) + 0.5) * res


def _grid_over(xs, ys, res: float) -> Grid:
    """The regular grid whose cell centers span ``xs`` by ``ys`` at ``res``."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    return Grid(
        xmin=float(xs.min() - res / 2),
        xmax=float(xs.max() + res / 2),
        ymin=float(ys.min() - res / 2),
        ymax=float(ys.max() + res / 2),
        xres=res,
        yres=res,
    )


def _one_cell(x: float, y: float, res: float) -> Grid:
    """A one-cell grid centered on ``(x, y)``."""
    return _grid_over([x], [y], res)


def test_aggregate_conserves_integral():
    """Uniform fine footprint fully inside a coarse grid: each cell == v * Npix."""
    native_res, coarse_res = 0.01, 0.03  # 3x3 native pixels per coarse cell
    fine = _block_centers(6, native_res)  # centers 0.005 .. 0.055
    v = 2.0
    foot = _foot_on_grid(
        fine, fine, np.full((len(fine), len(fine)), v), xres=native_res, yres=native_res
    )
    coarse = _block_centers(2, coarse_res)  # centers 0.015, 0.045

    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(
        _grid_over(coarse, coarse, coarse_res), time_bins=bins
    )

    assert result.to_numpy() == pytest.approx(v * 9)  # 3x3 native pixels per cell
    assert result.to_numpy().sum() == pytest.approx(v * 36)  # full native integral


def test_aggregate_block_sum_exact():
    """Coarse cell equals the exact KxK block sum of its native sub-cells."""
    native_res, coarse_res, k = 0.01, 0.03, 3
    fine = _block_centers(6, native_res)
    vals = np.arange(36, dtype=float).reshape(len(fine), len(fine))
    foot = _foot_on_grid(fine, fine, vals, xres=native_res, yres=native_res)

    coarse = _block_centers(2, coarse_res)
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(
        _grid_over(coarse, coarse, coarse_res), time_bins=bins
    )

    half = coarse_res / 2
    for (cx, cy), value in zip(result.index, result.to_numpy().ravel(), strict=True):
        in_x = (fine >= cx - half) & (fine < cx + half)
        in_y = (fine >= cy - half) & (fine < cy + half)
        expected = vals[np.ix_(in_y, in_x)].sum()
        assert value == pytest.approx(expected)
        assert in_x.sum() == k and in_y.sum() == k  # KxK block


def test_aggregate_drops_out_of_domain():
    """Native pixels outside the target grid are dropped, not folded into edges."""
    native_res = 0.01
    fine = _block_centers(4, native_res)  # 0.005 .. 0.035
    foot = _foot_on_grid(
        fine, fine, np.ones((len(fine), len(fine))), xres=native_res, yres=native_res
    )
    # Single coarse cell covering only the lower-left 2x2 native pixels.
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(_one_cell(0.01, 0.01, 0.02), time_bins=bins)

    # 4 in-domain pixels of value 1; the other 12 exterior pixels are dropped.
    assert result.to_numpy().sum() == pytest.approx(4.0)


def test_aggregate_matched_resolution_identity():
    """A target equal to the native grid returns each native pixel value unchanged."""
    res = 0.1
    lons = np.array([-113.95, -113.85])
    lats = np.array([39.05, 39.15])
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])  # [lat, lon]
    foot = _foot_on_grid(lons, lats, vals, xres=res, yres=res)

    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(_grid_over(lons, lats, res), time_bins=bins)

    for (cx, cy), value in zip(result.index, result.to_numpy().ravel(), strict=True):
        j = int(np.argmin(np.abs(lons - cx)))
        i = int(np.argmin(np.abs(lats - cy)))
        assert value == pytest.approx(vals[i, j])


def test_aggregate_total_invariance_to_target_resolution():
    """Same footprint aggregated to native vs coarse grid has identical totals."""
    native_res, coarse_res = 0.01, 0.03
    fine = _block_centers(6, native_res)
    rng = np.random.default_rng(0)
    vals = rng.uniform(1e-6, 1e-3, size=(len(fine), len(fine)))
    foot = _foot_on_grid(fine, fine, vals, xres=native_res, yres=native_res)

    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")

    coarse = _block_centers(2, coarse_res)
    native_grid = _grid_over(fine, fine, native_res)
    total_native = foot.stilt.aggregate(native_grid, bins).to_numpy().sum()
    total_coarse = (
        foot.stilt.aggregate(_grid_over(coarse, coarse, coarse_res), bins)
        .to_numpy()
        .sum()
    )

    assert total_native == pytest.approx(vals.sum())
    assert total_coarse == pytest.approx(total_native)


def test_aggregate_time_binning():
    """Summing all time-bin columns equals footprint summed over time then space."""
    res = 0.1
    lons = np.array([-113.95, -113.85])
    lats = np.array([39.05, 39.15])
    vals = np.stack(
        [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
            np.array([[9.0, 10.0], [11.0, 12.0]]),
        ]
    )  # (time, lat, lon)
    foot = _foot_on_grid(lons, lats, vals, xres=res, yres=res)

    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=3, freq="1h", closed="left")
    result = foot.stilt.aggregate(_grid_over(lons, lats, res), time_bins=bins)

    assert result.to_numpy().sum() == pytest.approx(vals.sum())


def test_aggregate_misaligned_conserves_and_splits():
    """A misaligned coarse target that covers the domain conserves the sum by splitting."""
    native_res, coarse_res = 0.01, 0.05
    fine = _block_centers(10, native_res)  # native domain [0, 0.10]
    rng = np.random.default_rng(2)
    vals = rng.uniform(0, 1e-3, (len(fine), len(fine)))
    foot = _foot_on_grid(fine, fine, vals, xres=native_res, yres=native_res)

    # coarse cells whose edges (-0.025, 0.025, 0.075, 0.125) cut through native
    # cells yet fully cover the native domain [0, 0.10].
    coarse = np.array([0.0, 0.05, 0.10])
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")

    result = foot.stilt.aggregate(_grid_over(coarse, coarse, coarse_res), bins)
    # full coverage with split native cells -> all native mass retained
    assert result.to_numpy().sum() == pytest.approx(vals.sum())


def test_aggregate_splits_native_cell_by_area():
    """A target cell straddling native cells gets each by its exact overlap fraction."""
    res = 0.01
    lons = np.array([0.005, 0.015])
    lats = np.array([0.005, 0.015])
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])  # [lat, lon]
    foot = _foot_on_grid(lons, lats, vals, xres=res, yres=res)

    # one target cell centered at (0.0075, 0.0075): overlaps the lower/left native
    # cells by 0.75 and the upper/right by 0.25 on each axis.
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(_one_cell(0.0075, 0.0075, res), bins)

    fx = fy = np.array([0.75, 0.25])
    expected = float((fy[:, None] * fx[None, :] * vals).sum())  # 1.75
    assert float(result.iloc[0, 0]) == pytest.approx(expected)


def test_aggregate_finer_target_downscaling_conserves():
    """Aggregating to a FINER grid spreads each native cell and conserves the sum."""
    native_res, fine_res, k = 0.05, 0.01, 5
    coarse = _block_centers(3, native_res)
    vals = np.arange(9, dtype=float).reshape(3, 3) + 1.0
    foot = _foot_on_grid(coarse, coarse, vals, xres=native_res, yres=native_res)

    fine = _block_centers(15, fine_res)  # 3 native cells x 5 = 15 fine cells per axis
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")

    fine_grid = _grid_over(fine, fine, fine_res)
    result = foot.stilt.aggregate(fine_grid, bins)
    assert result.to_numpy().sum() == pytest.approx(vals.sum())
    # each native cell spreads uniformly over its k*k fine sub-cells
    expected_corner = vals[0, 0] / (k * k)
    assert result.to_numpy().max() == pytest.approx(vals.max() / (k * k))
    first_x, first_y = fine_grid.axes[0][0], fine_grid.axes[1][0]
    assert float(result.loc[(first_x, first_y)].iloc[0]) == pytest.approx(
        expected_corner
    )


def test_grid_to_xarray_longlat_centers():
    """Grid.to_xarray() yields CF lon/lat cell centers matching the footprint grid."""
    grid = Grid(xmin=-114.0, xmax=-113.8, ymin=39.0, ymax=39.2, xres=0.1, yres=0.1)
    ds = grid.to_xarray()

    np.testing.assert_allclose(ds["lon"].values, [-113.95, -113.85])
    np.testing.assert_allclose(ds["lat"].values, [39.05, 39.15])
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert "crs" in ds
    assert ds["lon"].attrs["standard_name"] == "longitude"
    assert ds["lat"].attrs["units"] == "degrees_north"


def test_aggregate_onto_own_grid_is_identity():
    """Aggregating a footprint onto its own grid is the identity."""
    foot = _make_footprint(n_times=1)
    t0 = pd.Timestamp("2023-01-01 12:00")
    foot.loc[t0, 39.05, -113.95] = 1e-4
    foot.loc[t0, 39.15, -113.85] = 3e-4

    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(foot.stilt.config.grid, bins)

    # identity round-trip: the two seeded native cells reappear unchanged.
    assert result.shape == (4, 1)
    nonzero = np.sort(result.to_numpy().ravel())
    assert result.to_numpy().sum() == pytest.approx(4e-4)
    np.testing.assert_allclose(nonzero[-2:], [1e-4, 3e-4], rtol=1e-9)
    np.testing.assert_allclose(nonzero[:2], [0.0, 0.0], atol=1e-18)


# ---------------------------------------------------------------------------
# calculate()
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
    foot = calculate(particles, receptor=point_receptor, config=_foot_config())
    assert foot is not None
    assert isinstance(foot, xr.DataArray)


def test_calculate_dims_are_time_lat_lon(point_receptor):
    particles = _particles_in_domain()
    foot = calculate(particles, receptor=point_receptor, config=_foot_config())
    assert foot is not None
    assert tuple(foot.dims) == ("time", "lat", "lon")


def test_calculate_raises_when_particles_outside_domain(
    point_receptor,
):
    """All particles outside the domain is an EmptyFootprint, not zeros."""
    particles = _particles_in_domain()
    particles["long"] = 0.0  # far outside [-114, -113]
    particles["lati"] = 0.0
    config = _foot_config()
    with pytest.raises(EmptyFootprint) as info:
        calculate(particles, receptor=point_receptor, config=config)
    assert info.value.reason == "outside_domain"


def test_calculate_raises_when_there_are_no_particles(point_receptor):
    particles = _particles_in_domain().iloc[0:0]
    with pytest.raises(EmptyFootprint) as info:
        calculate(particles, receptor=point_receptor, config=_foot_config())
    assert info.value.reason == "no_particles"


def test_calculate_assigns_name(point_receptor):
    particles = _particles_in_domain()
    foot = calculate(
        particles, receptor=point_receptor, config=_foot_config(), name="test"
    )
    assert foot is not None
    assert foot.stilt.name == "test"


def test_calculate_time_integrate_collapses_to_single_timestep(point_receptor):
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1),
        time_integrate=True,
    )
    particles = _particles_in_domain()
    foot = calculate(particles, receptor=point_receptor, config=config)
    assert foot is not None
    assert len(foot.time) == 1


def test_calculate_nonnegative_foot_values(point_receptor):
    """Footprint values should be non-negative."""
    particles = _particles_in_domain()
    foot = calculate(particles, receptor=point_receptor, config=_foot_config())
    assert foot is not None
    assert float(foot.values.min()) >= 0.0


def test_calculate_smooth_factor_zero(point_receptor):
    """smooth_factor=0 is equivalent to no smoothing (identity kernel)."""
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1),
        smooth_factor=0.0,
    )
    particles = _particles_in_domain()
    foot = calculate(particles, receptor=point_receptor, config=config)
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

    foot = calculate(particles, receptor=point_receptor, config=config)

    np.testing.assert_allclose(foot.lon.values, [-113.85, -113.55, -113.25])
    np.testing.assert_allclose(foot.lat.values, [39.2, 39.6])
    assert foot.shape == (2, 2, 3)


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

    foot = calculate(particles, receptor=point_receptor, config=config)

    assert foot.sizes["lon"] == 100
    assert foot.sizes["lat"] == 20
    assert np.isfinite(foot.values).all()
    assert float(foot.values.min()) >= 0.0


def test_calculate_grid_property(point_receptor):
    particles = _particles_in_domain()
    config = _foot_config()
    foot = calculate(particles, receptor=point_receptor, config=config)
    assert foot is not None
    assert foot.stilt.grid.xres == pytest.approx(0.1)


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
# These properties must hold from pure math regardless of STILT-R agreement.
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

    foot_1x = calculate(particles, receptor=point_receptor, config=config)

    particles_2x = particles.copy()
    particles_2x["foot"] = particles_2x["foot"] * 2.0
    foot_2x = calculate(particles_2x, receptor=point_receptor, config=config)

    np.testing.assert_allclose(
        foot_2x.values,
        foot_1x.values * 2.0,
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

    foot = calculate(particles, receptor=point_receptor, config=config)

    expected = float(particles["foot"].sum()) / n
    assert float(foot.values.sum()) == pytest.approx(expected, rel=1e-10)


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

    foot_0 = calculate(particles, receptor=point_receptor, config=config_0)
    foot_s = calculate(particles, receptor=point_receptor, config=config_s)

    total_0 = float(foot_0.values.sum())
    total_s = float(foot_s.values.sum())

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

    foot1 = calculate(particles, receptor=point_receptor, config=config)
    foot2 = calculate(particles, receptor=point_receptor, config=config)

    np.testing.assert_array_equal(
        foot1.values,
        foot2.values,
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

    foot_ti = calculate(
        particles,
        receptor=point_receptor,
        config=FootprintConfig(grid=grid, time_integrate=True),
    )
    foot_no = calculate(
        particles,
        receptor=point_receptor,
        config=FootprintConfig(grid=grid, time_integrate=False),
    )

    np.testing.assert_allclose(
        foot_ti.values.squeeze(),
        foot_no.sum("time").values,
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

    foot = calculate(particles, receptor=point_receptor, config=config)

    # total = sum(foot) / n_particles = (n * foot_val) / n = foot_val
    assert float(foot.values.sum()) == pytest.approx(foot_val, rel=1e-10)

    # Exactly one non-zero cell across all time layers
    nonzero_count = int((foot.values > 0).sum())
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

    foot = calculate(particles, receptor=point_receptor, config=config)

    # Verify the footprint cell values are exactly what the formula predicts.
    expected_fa = n_a * foot_a / n_total  # 1e-4
    expected_fb = n_b * foot_b / n_total  # 1.5e-4

    f_a = float(foot.sel(lon=-113.85, lat=39.05, method="nearest").sum())
    f_b = float(foot.sel(lon=-113.35, lat=39.55, method="nearest").sum())

    assert f_a == pytest.approx(expected_fa, rel=1e-10), (
        f"Cell A footprint: expected {expected_fa:.3e}, got {f_a:.3e}"
    )
    assert f_b == pytest.approx(expected_fb, rel=1e-10), (
        f"Cell B footprint: expected {expected_fb:.3e}, got {f_b:.3e}"
    )

    # Apply flux field (non-zero at the two cluster cells only).
    q_a, q_b = 5.0, 8.0
    flux = np.zeros_like(foot.values)

    lons = foot.lon.values
    lats = foot.lat.values
    lon_a = int(np.argmin(np.abs(lons - (-113.85))))
    lat_a = int(np.argmin(np.abs(lats - 39.05)))
    lon_b = int(np.argmin(np.abs(lons - (-113.35))))
    lat_b = int(np.argmin(np.abs(lats - 39.55)))

    flux[:, lat_a, lon_a] = q_a
    flux[:, lat_b, lon_b] = q_b

    # c = F_A * q_A + F_B * q_B
    c_computed = float((foot.values * flux).sum())
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
            crs="+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs",
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
            crs="+proj=nope-not-a-projection",
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


# ---------------------------------------------------------------------------
# Spatial geometries: Grid, Mesh, Zones
# ---------------------------------------------------------------------------


def test_aggregate_rejects_targets_that_are_not_geometries():
    """An xarray grid or a list of centres is not a target; build a Grid instead."""
    foot = _make_footprint(n_times=1)
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    with pytest.raises(TypeError, match="Grid, stilt.Mesh, or stilt.Zones"):
        foot.stilt.aggregate(foot.stilt.config.grid.to_xarray(), bins)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Grid, stilt.Mesh, or stilt.Zones"):
        foot.stilt.aggregate([(-113.95, 39.05)], bins)  # type: ignore[arg-type]


def test_aggregate_mesh_window_sums_exactly():
    """Each Mesh window sums the native cells inside it, independently."""
    native_res = 0.01
    fine = _block_centers(6, native_res)  # 0.005 .. 0.055
    vals = np.arange(36, dtype=float).reshape(len(fine), len(fine))
    foot = _foot_on_grid(fine, fine, vals, xres=native_res, yres=native_res)

    # Two overlapping 3x3 windows: overlap is allowed and counted in both.
    mesh = Mesh.from_windows([(0.015, 0.015), (0.025, 0.025)], 0.03, ids=["a", "b"])
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(mesh, bins)

    assert result.index.equals(mesh.index)
    assert result.loc["a"].iloc[0] == pytest.approx(vals[0:3, 0:3].sum())
    assert result.loc["b"].iloc[0] == pytest.approx(vals[1:4, 1:4].sum())


def test_aggregate_mesh_off_lattice_splits_by_area():
    """A polygon not aligned to the native lattice takes area fractions."""
    native_res = 0.01
    fine = _block_centers(4, native_res)  # 0.005, 0.015, 0.025, 0.035
    vals = np.ones((4, 4))
    foot = _foot_on_grid(fine, fine, vals, xres=native_res, yres=native_res)
    mesh = Mesh.from_windows([(0.015, 0.015)], 0.02)  # area = 4 native cells
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(mesh, bins)
    assert result.iloc[0, 0] == pytest.approx(4.0)


def test_aggregate_mesh_ids_index_and_time_bins():
    foot = _make_footprint(n_times=2)
    t0 = pd.Timestamp("2023-01-01 12:00")
    foot.loc[t0, 39.05, -113.95] = 1e-4
    foot.loc[t0 + pd.Timedelta(hours=1), 39.15, -113.85] = 3e-4

    mesh = Mesh.from_windows([(-113.95, 39.05), (-113.85, 39.15)], 0.1, ids=["a", "b"])
    bins = pd.interval_range(start=t0, periods=2, freq="1h", closed="left")
    result = foot.stilt.aggregate(mesh, bins)

    assert result.index.tolist() == ["a", "b"]
    assert result.index.name == "cell"
    assert result.shape == (2, 2)
    assert result.loc["a"].tolist() == pytest.approx([1e-4, 0.0])
    assert result.loc["b"].tolist() == pytest.approx([0.0, 3e-4])


def test_aggregate_zones_merge_grid_cells():
    native_res, coarse_res = 0.01, 0.03
    fine = _block_centers(6, native_res)
    vals = np.arange(36, dtype=float).reshape(len(fine), len(fine))
    foot = _foot_on_grid(fine, fine, vals, xres=native_res, yres=native_res)
    coarse = Grid(
        xmin=0.0, xmax=0.06, ymin=0.0, ymax=0.06, xres=coarse_res, yres=coarse_res
    )
    part = Zones.from_labels(coarse, ["W", "W", "E", "E"])  # x outer: W=x<0.03

    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(part, bins)
    assert result.index.tolist() == ["W", "E"]
    assert result.loc["W"].iloc[0] == pytest.approx(vals[:, :3].sum())
    assert result.loc["E"].iloc[0] == pytest.approx(vals[:, 3:].sum())
    assert result.to_numpy().sum() == pytest.approx(vals.sum())


def test_aggregate_warns_when_target_cells_under_resolved():
    foot = _make_footprint()  # native 0.1 deg
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    mesh = Mesh.from_windows([(-113.95, 39.05)], 0.05)  # half a native cell
    with pytest.warns(UserWarning, match="under-resolved"):
        foot.stilt.aggregate(mesh, bins)


def test_aggregate_grid_in_other_crs_is_reprojected():
    pytest.importorskip("pyproj")
    foot = _make_footprint()
    t0 = pd.Timestamp("2023-01-01 12:00")
    foot.loc[t0, 39.05, -113.95] = 1.0
    foot.loc[t0, 39.15, -113.85] = 1.0
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    # Coarse 20 km UTM cells covering the seeded footprint cells
    grid = Grid(
        xmin=-114.2,
        xmax=-113.6,
        ymin=38.8,
        ymax=39.4,
        xres=20000.0,
        yres=20000.0,
        crs="EPSG:32612",
    )
    result = foot.stilt.aggregate(grid, bins)
    assert result.to_numpy().sum() == pytest.approx(2.0, rel=1e-6)


def test_aggregate_mesh_empty_footprint_returns_zeros():
    foot = _make_footprint(n_times=1)
    foot = foot.isel(time=slice(0, 0))
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    mesh = Mesh.from_windows([(-113.95, 39.05)], 0.1)
    result = foot.stilt.aggregate(mesh, bins)
    assert result.shape == (1, 1)
    assert result.to_numpy().sum() == 0.0


# ---------------------------------------------------------------------------
# Geometry hash: recorded with geometry-derived footprints
# ---------------------------------------------------------------------------


def _windows_spec(shift: float = 0.0):
    return {
        "kind": "windows",
        "coords": [(-113.95 + shift, 39.05), (-113.85 + shift, 39.15)],
        "size": 0.25,  # >= 2 native cells so no under-resolution warning
        "ids": ["a", "b"],
    }


def _geometry_footprint() -> tuple[xr.DataArray, FootprintConfig, Mesh]:
    """A footprint whose grid was derived for the windows geometry, and that geometry."""
    base = _make_footprint()
    fc = FootprintConfig(grid=base.stilt.grid, geometry=_windows_spec())
    assert fc.geometry is not None
    mesh = Mesh.from_spec(fc.geometry)
    return _describe(base, base.stilt.receptor, fc, "geo", mesh.hash), fc, mesh


def test_a_footprint_records_its_geometry_and_hash():
    foot, fc, mesh = _geometry_footprint()
    assert foot.stilt.config == fc
    assert foot.stilt.geometry_hash == mesh.hash
    assert _make_footprint().stilt.geometry_hash is None


def test_netcdf_roundtrip_keeps_geometry_and_hash(tmp_path):
    foot, fc, mesh = _geometry_footprint()
    path = foot.stilt.to_netcdf(tmp_path / "geo_foot.nc")
    loaded = read_footprint(path)
    assert loaded.stilt.config.geometry == fc.geometry
    assert loaded.stilt.geometry_hash == mesh.hash
    # a grid-only footprint carries no geometry attrs at all
    plain = _make_footprint().stilt.to_netcdf(tmp_path / "plain_foot.nc")
    with xr.open_dataset(plain) as ds:
        settings = json.loads(ds["foot"].attrs["stilt_footprint"])
    assert settings["geometry"] is None and settings["geometry_hash"] is None
    assert read_footprint(plain).stilt.geometry_hash is None


def test_aggregate_warns_when_geometry_hash_differs():
    foot, _, same = _geometry_footprint()
    base = _make_footprint()
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        foot.stilt.aggregate(same, bins)  # identical geometry: no warning
        foot.stilt.aggregate(base.stilt.grid, bins)  # grids are never checked

    other = Mesh.from_windows([(-113.9, 39.1)], 0.25, ids=["c"])
    with pytest.warns(UserWarning, match="derived for geometry"):
        foot.stilt.aggregate(other, bins)
    with pytest.warns(UserWarning, match="derived for geometry"):
        foot.stilt.aggregate(Zones.from_labels(other, ["z"]), bins)


# ---------------------------------------------------------------------------
# A footprint is a DataArray (#107)
# ---------------------------------------------------------------------------


def test_receptor_coordinate_survives_arithmetic_and_reductions():
    """The receptor id is a coordinate, which xarray keeps where it drops attributes."""
    foot = _make_footprint(n_times=2)
    rid = "202301011200_-111.85_40.77_5"
    for result in (foot * 2, foot.sum("time"), foot.isel(time=0), foot + foot):
        assert result["receptor"].item() == rid


def test_accessor_says_what_is_missing_without_attributes():
    """Older xarray drops attributes in arithmetic; that gives a clear error, not a wrong answer."""
    foot = _make_footprint()
    bare = foot.copy()
    bare.attrs = {}
    with pytest.raises(ValueError, match="keep_attrs"):
        _ = bare.stilt.grid
    with xr.set_options(keep_attrs=True):
        assert (foot * 2).stilt.grid == foot.stilt.grid


def test_footprints_stack_along_the_receptor_coordinate():
    """xr.concat labels a stack of footprints by receptor with no extra work."""
    a = _make_footprint()
    other = PointReceptor(
        time=a.stilt.receptor.time, longitude=-112.0, latitude=40.0, altitude=5.0
    )
    b = _describe(a * 2, other, a.stilt.config, "slv")
    stack = xr.concat([a, b], dim="receptor")
    assert stack["receptor"].values.tolist() == [
        str(a.stilt.receptor.id),
        str(other.id),
    ]
    assert stack.sizes["receptor"] == 2


def test_calculate_returns_a_named_dataarray_with_its_receptor(point_receptor):
    particles = _particles_in_domain()
    foot = calculate(particles, point_receptor, _foot_config(), name="hrrr")
    assert isinstance(foot, xr.DataArray)
    assert foot.name == "foot"
    assert foot["receptor"].item() == str(point_receptor.id)
    assert foot.stilt.receptor == point_receptor
    assert foot.stilt.name == "hrrr"
    assert foot.attrs["units"] == "ppm m2 s umol-1"


def _first_hour_particles(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Backward particles that all stay within the first hour (layer -1)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": np.repeat([-20.0, -40.0, -59.0], n),
            "indx": np.tile(np.arange(1, n + 1), 3),
            "long": rng.uniform(-112.2, -111.7, 3 * n),
            "lati": rng.uniform(40.6, 40.9, 3 * n),
            "zagl": rng.uniform(5, 100, 3 * n),
            "foot": rng.uniform(0.0, 1e-3, 3 * n),
        }
    )


def test_one_layer_footprint_is_stamped_at_its_hour():
    """A backward footprint with only layer -1 is stamped an hour before the receptor, as in STILT-R."""
    receptor = PointReceptor(
        time="2023-01-01 12:00", longitude=-111.95, latitude=40.75, altitude=5.0
    )
    grid = Grid(xmin=-112.3, xmax=-111.6, ymin=40.5, ymax=41.0, xres=0.01, yres=0.01)

    foot = calculate(_first_hour_particles(), receptor, FootprintConfig(grid=grid))
    integrated = calculate(
        _first_hour_particles(),
        receptor,
        FootprintConfig(grid=grid, time_integrate=True),
    )

    assert list(pd.DatetimeIndex(foot["time"].values)) == [
        pd.Timestamp("2023-01-01 11:00")
    ]
    assert list(pd.DatetimeIndex(integrated["time"].values)) == [
        pd.Timestamp("2023-01-01 12:00")
    ]


def test_calculated_coordinates_match_the_grid_axes():
    """A calculated footprint has the coordinates a stored one is read back with."""
    receptor = PointReceptor(
        time="2023-01-01 12:00", longitude=-111.95, latitude=40.75, altitude=5.0
    )
    grid = Grid(xmin=-112.3, xmax=-111.6, ymin=40.5, ymax=41.0, xres=0.01, yres=0.01)

    foot = calculate(_first_hour_particles(), receptor, FootprintConfig(grid=grid))

    x, y = grid.axes
    np.testing.assert_array_equal(foot["lon"].values, x)
    np.testing.assert_array_equal(foot["lat"].values, y)
