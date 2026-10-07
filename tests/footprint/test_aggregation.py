"""Tests for aggregating footprints onto grids, meshes, and zones."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.footprint.config import FootprintConfig
from stilt.footprint.targets import Mesh, Zones
from stilt.receptors import PointReceptor
from stilt.spatial import Grid

from ..fixtures.footprints import as_footprint, geometry_footprint, make_footprint


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
    return as_footprint(data, receptor, FootprintConfig(grid=grid), "")


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


def test_aggregate_returns_dataframe():
    foot = make_footprint(n_times=1)
    t0 = pd.Timestamp("2023-01-01 12:00")
    foot.loc[t0, 39.05, -113.95] = 1e-4

    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    result = foot.stilt.aggregate(target=_one_cell(-113.95, 39.05, 0.1), time_bins=bins)

    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0, 0] == pytest.approx(1e-4)


def test_aggregate_multiple_bins():
    foot = make_footprint(n_times=3)
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
    foot = make_footprint(n_times=2)
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=2, freq="1h", closed=closed)

    with pytest.raises(ValueError, match="closed on the left"):
        foot.stilt.aggregate(target=_one_cell(-113.95, 39.05, 0.1), time_bins=bins)


def test_aggregate_zero_values_in_domain():
    """Grid coordinate within domain but with all-zero values returns zeros."""
    foot = make_footprint(n_times=1)
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    # -113.95 and 39.05 are valid cell centers in the footprint
    result = foot.stilt.aggregate(target=_one_cell(-113.95, 39.05, 0.1), time_bins=bins)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# Footprint.aggregate() — area integration onto coarse target grids

# ---------------------------------------------------------------------------


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


def test_aggregate_onto_own_grid_is_identity():
    """Aggregating a footprint onto its own grid is the identity."""
    foot = make_footprint(n_times=1)
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
# Spatial geometries: Grid, Mesh, Zones

# ---------------------------------------------------------------------------


def test_aggregate_rejects_targets_that_are_not_geometries():
    """An xarray grid or a list of centres is not a target; build a Grid instead."""
    foot = make_footprint(n_times=1)
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
    foot = make_footprint(n_times=2)
    t0 = pd.Timestamp("2023-01-01 12:00")
    foot.loc[t0, 39.05, -113.95] = 1e-4
    foot.loc[t0 + pd.Timedelta(hours=1), 39.15, -113.85] = 3e-4

    mesh = Mesh.from_windows([(-113.95, 39.05), (-113.85, 39.15)], 0.1, ids=["a", "b"])
    bins = pd.interval_range(start=t0, periods=2, freq="1h", closed="left")
    # The mesh's cells are the footprint's own size.
    with pytest.warns(UserWarning, match="under-resolved"):
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
    foot = make_footprint()  # native 0.1 deg
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    mesh = Mesh.from_windows([(-113.95, 39.05)], 0.05)  # half a native cell
    with pytest.warns(UserWarning, match="under-resolved"):
        foot.stilt.aggregate(mesh, bins)


def test_aggregate_grid_in_other_crs_is_reprojected():
    pytest.importorskip("pyproj")
    foot = make_footprint()
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


def test_aggregate_of_a_time_summed_footprint_raises():
    """Summed over time, a footprint's hours are gone; zeros would be wrong."""
    foot = make_footprint(n_times=2)
    total = foot.sum("time", keep_attrs=True)
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    mesh = Mesh.from_windows([(-113.95, 39.05)], 0.1)
    with pytest.raises(ValueError, match="no time dimension"):
        total.stilt.aggregate(mesh, bins)


def test_aggregate_mesh_empty_footprint_returns_zeros():
    foot = make_footprint(n_times=1)
    foot = foot.isel(time=slice(0, 0))
    t0 = pd.Timestamp("2023-01-01 12:00")
    bins = pd.interval_range(start=t0, periods=1, freq="1h", closed="left")
    mesh = Mesh.from_windows([(-113.95, 39.05)], 0.1)
    result = foot.stilt.aggregate(mesh, bins)
    assert result.shape == (1, 1)
    assert result.to_numpy().sum() == 0.0


def test_aggregate_warns_when_geometry_hash_differs():
    foot, _, same = geometry_footprint()
    base = make_footprint()
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
