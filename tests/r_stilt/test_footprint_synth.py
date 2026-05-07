"""
Synthetic footprint comparison tests: PYSTILT vs R-STILT on hand-crafted
particle DataFrames, bypassing HYSPLIT entirely.

Exercises code paths the seeded end-to-end tests cannot reliably target:

  test_single_particle_gaussian     One particle; verify Gaussian rasterization.
  test_all_outside_domain           All particles off-grid → zero footprint.
  test_boundary_particles           Particles at exact grid-edge fencepost.
  test_time_integrate               time_integrate=True collapses time axis.
  test_smooth_factor_scaling        smooth_factor=0.5 produces narrower Gaussian.
  test_hnf_dilution_active          plume < mixing-depth → foot corrected.
  test_hnf_dilution_inactive        plume >> mixing-depth → foot unchanged.
  test_early_time_multi_timestep    Multiple timesteps in the 100-min window.
  test_latitude_kernel_scaling      cos(lat) bandwidth scaling agrees between tools.

Each test:
  1. Constructs a minimal synthetic particle DataFrame.
  2. Runs PYSTILT's Python implementation.
  3. Writes particles to a tmp parquet and calls the matching R-STILT function
     via Rscript.
  4. Compares both results at rtol=1e-7.

Requires STILT_R_DIR and Rscript (fixtures in tests/conftest.py skip
automatically when either is absent).
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.config import FootprintConfig, Grid
from stilt.footprint import (
    Footprint,
    _build_buffered_grid,
    _compute_kernel_bandwidths,
    _filter_and_rasterize_particles,
    _grid_cell_starts,
    _interpolate_early_timesteps,
    _make_gauss_kernel,
    _project_particles_to_crs,
    _wrap_antimeridian_longitudes,
)
from stilt.receptors import PointReceptor
from stilt.trajectory import calc_plume_dilution

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_R_HELPERS = Path(__file__).parents[1] / "fixtures" / "r_helpers"

_RECEPTOR_TIME = dt.datetime(2015, 12, 10, 0, 0)
_RECEPTOR = PointReceptor(_RECEPTOR_TIME, -112.0, 40.5, 5.0)

_GRID = Grid(
    xmin=-113.0,
    xmax=-111.0,
    ymin=39.5,
    ymax=41.5,
    xres=0.01,
    yres=0.01,
)

_VEGHT = 0.5  # default STILT veght


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r_footprint(
    tmp_path: Path,
    rscript: str,
    r_stilt_dir: Path,
    particles: pd.DataFrame,
    grid: Grid = _GRID,
    smooth_factor: float = 1.0,
    time_integrate: bool = False,
) -> xr.Dataset:
    """Call R-STILT calc_footprint on *particles*, return xr.Dataset."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p_path = tmp_path / "particles.parquet"
    nc_path = tmp_path / "r_foot.nc"
    particles.to_parquet(p_path, index=False)

    result = subprocess.run(
        [
            rscript,
            str(_R_HELPERS / "calc_footprint.r"),
            str(p_path),
            str(nc_path),
            str(grid.xmin),
            str(grid.xmax),
            str(grid.ymin),
            str(grid.ymax),
            str(grid.xres),
            str(grid.yres),
            str(smooth_factor),
            str(time_integrate).upper(),
            str(r_stilt_dir),
            grid.projection,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"calc_footprint.r failed (exit {result.returncode}):\n{result.stderr}"
        )

    if not nc_path.exists():
        # calc_footprint returned NULL (all particles outside domain).
        # Return an all-zero Dataset so the caller can assert values == 0.
        lat = _grid_cell_starts(grid.ymin, grid.ymax, grid.yres)
        lon = _grid_cell_starts(grid.xmin, grid.xmax, grid.xres)
        n_lat = len(lat)
        n_lon = len(lon)
        return xr.Dataset(
            {
                "foot": (
                    ["time", "lat", "lon"],
                    np.zeros((1, n_lat, n_lon), dtype=np.float32),
                )
            },
            coords={"lat": lat, "lon": lon, "time": [0.0]},
        )

    ds = xr.open_dataset(nc_path)
    ds.load()
    ds.close()
    return ds


def _r_footprint_debug(
    tmp_path: Path,
    rscript: str,
    r_stilt_dir: Path,
    particles: pd.DataFrame,
    grid: Grid = _GRID,
    smooth_factor: float = 1.0,
    time_integrate: bool = False,
) -> dict[str, pd.DataFrame]:
    """Call the R debug helper and return calc_footprint intermediate tables."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p_path = tmp_path / "particles.parquet"
    out_dir = tmp_path / "debug"
    particles.to_parquet(p_path, index=False)

    result = subprocess.run(
        [
            rscript,
            str(_R_HELPERS / "calc_footprint_debug.r"),
            str(p_path),
            str(out_dir),
            str(grid.xmin),
            str(grid.xmax),
            str(grid.ymin),
            str(grid.ymax),
            str(grid.xres),
            str(grid.yres),
            str(smooth_factor),
            str(time_integrate).upper(),
            str(r_stilt_dir),
            grid.projection,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"calc_footprint_debug.r failed (exit {result.returncode}):\n"
            f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

    return {
        name: pd.read_parquet(out_dir / f"{name}.parquet")
        for name in [
            "p_after_interp",
            "p_with_rtime",
            "grid_x",
            "grid_y",
            "kernel",
            "raster",
            "metadata",
        ]
    }


def _py_footprint(
    tmp_path: Path,
    particles: pd.DataFrame,
    receptor: PointReceptor = _RECEPTOR,
    grid: Grid = _GRID,
    smooth_factor: float = 1.0,
    time_integrate: bool = False,
) -> xr.Dataset:
    """Call PYSTILT Footprint.calculate on *particles*, return xr.Dataset."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = FootprintConfig(
        grid=grid, smooth_factor=smooth_factor, time_integrate=time_integrate
    )
    foot = Footprint.calculate(particles, receptor=receptor, config=config, name="test")
    nc_path = tmp_path / "py_foot.nc"
    foot.to_netcdf(nc_path)
    ds = xr.open_dataset(nc_path)
    ds.load()
    ds.close()
    return ds


def _py_footprint_debug(
    particles: pd.DataFrame,
    grid: Grid = _GRID,
    smooth_factor: float = 1.0,
    time_integrate: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return Python-side calc_footprint intermediate tables matching the R helper."""
    projection = grid.projection
    xmin, xmax, xres = grid.xmin, grid.xmax, grid.xres
    ymin, ymax, yres = grid.ymin, grid.ymax, grid.yres
    is_longlat = "+proj=longlat" in projection

    p = particles.copy(deep=False)
    time_sign = int(np.sign(p["time"].median()))
    if is_longlat:
        p, xmin, xmax, _ = _wrap_antimeridian_longitudes(p, xmin=xmin, xmax=xmax)

    p_after_interp = _interpolate_early_timesteps(
        p, xres=xres, yres=yres, time_sign=time_sign
    )
    p_with_rtime = p_after_interp.copy()
    min_abs_time = (
        p_with_rtime["time"]
        .abs()
        .groupby(p_with_rtime["indx"], sort=False)
        .transform("min")
    )
    p_with_rtime["rtime"] = p_with_rtime["time"] - time_sign * min_abs_time

    if not is_longlat:
        p_with_rtime, xmin, xmax, ymin, ymax = _project_particles_to_crs(
            p_with_rtime,
            projection=projection,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )

    glong = _grid_cell_starts(xmin, xmax, xres)
    glati = _grid_cell_starts(ymin, ymax, yres)
    kernel_df, w = _compute_kernel_bandwidths(
        p_with_rtime, smooth_factor=smooth_factor, is_longlat=is_longlat
    )
    max_kernel = (
        _make_gauss_kernel((xres, yres), float(np.max(w)))
        if len(w) > 0
        else np.array([[1.0]])
    )
    buffered = _build_buffered_grid(
        xmin=xmin,
        ymin=ymin,
        xres=xres,
        yres=yres,
        n_lon=len(glong),
        n_lat=len(glati),
        max_kernel=max_kernel,
    )
    raster, _ = _filter_and_rasterize_particles(
        p_with_rtime,
        buffered=buffered,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        xres=xres,
        yres=yres,
        time_integrate=time_integrate,
    )

    kernel_out = kernel_df.copy()
    if "varsum" not in kernel_out.columns:
        kernel_out["varsum"] = 0.0
    kernel_out["lati"] = kernel_out["lat_mean"]
    kernel_out["w"] = w

    return {
        "p_after_interp": p_after_interp.reset_index(drop=True),
        "p_with_rtime": p_with_rtime.reset_index(drop=True),
        "grid_x": pd.DataFrame({"axis": "x", "value": glong}),
        "grid_y": pd.DataFrame({"axis": "y", "value": glati}),
        "kernel": kernel_out[["rtime", "varsum", "lati", "w"]].reset_index(drop=True),
        "raster": raster.reset_index(drop=True),
    }


def _assert_debug_tables_close(
    py: dict[str, pd.DataFrame],
    r: dict[str, pd.DataFrame],
    *,
    source_particles: pd.DataFrame,
    label: str,
) -> None:
    """Compare Python and R calc_footprint debug tables."""
    particle_cols = [
        "time",
        "indx",
        "long",
        "lati",
        "zagl",
        "foot",
        "mlht",
        "dens",
        "samt",
        "sigw",
        "tlgr",
    ]
    py_interp = py["p_after_interp"].sort_values(
        ["indx", "time"], ascending=[True, False]
    )
    r_interp = r["p_after_interp"].sort_values(
        ["indx", "time"], ascending=[True, False]
    )
    np.testing.assert_allclose(
        py_interp[particle_cols].to_numpy(dtype=float),
        r_interp[particle_cols].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-15,
        err_msg=f"[{label}] post-interpolation particle table differs",
    )

    rtime_cols = [*particle_cols, "rtime"]
    py_rtime = py["p_with_rtime"].sort_values(["indx", "time"], ascending=[True, False])
    r_rtime = r["p_with_rtime"].sort_values(["indx", "time"], ascending=[True, False])
    np.testing.assert_allclose(
        py_rtime[rtime_cols].to_numpy(dtype=float),
        r_rtime[rtime_cols].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-8,
        err_msg=f"[{label}] rtime particle table differs",
    )

    np.testing.assert_allclose(
        py["grid_x"]["value"].to_numpy(),
        r["grid_x"]["value"].to_numpy(),
        rtol=1e-12,
        atol=1e-8,
        err_msg=f"[{label}] x grid starts differ",
    )
    np.testing.assert_allclose(
        py["grid_y"]["value"].to_numpy(),
        r["grid_y"]["value"].to_numpy(),
        rtol=1e-12,
        atol=1e-8,
        err_msg=f"[{label}] y grid starts differ",
    )

    py_kernel = py["kernel"].sort_values("rtime").reset_index(drop=True)
    r_kernel = r["kernel"].sort_values("rtime").reset_index(drop=True)
    np.testing.assert_allclose(
        py_kernel[["rtime", "varsum", "lati", "w"]].to_numpy(dtype=float),
        r_kernel[["rtime", "varsum", "lati", "w"]].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-8,
        err_msg=f"[{label}] kernel bandwidth table differs",
    )

    py_raster = (
        py["raster"]
        .sort_values(["loi", "lai", "time", "rtime", "layer"])
        .reset_index(drop=True)
    )
    r_raster = r["raster"].copy()
    r_raster["loi"] = r_raster["loi"] - 1
    r_raster["lai"] = r_raster["lai"] - 1
    r_raster = r_raster.sort_values(
        ["loi", "lai", "time", "rtime", "layer"]
    ).reset_index(drop=True)
    raster_cols = ["loi", "lai", "time", "rtime", "foot", "layer"]
    np.testing.assert_allclose(
        py_raster[raster_cols].to_numpy(dtype=float),
        r_raster[raster_cols].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-15,
        err_msg=f"[{label}] pre-smoothing raster table differs",
    )

    assert int(r["metadata"]["np"].iloc[0]) == source_particles["indx"].nunique()


def _r_plume_dilution(
    tmp_path: Path,
    rscript: str,
    r_stilt_dir: Path,
    particles: pd.DataFrame,
    r_zagl: float,
    numpar: int = 1000,
    veght: float = _VEGHT,
) -> pd.DataFrame:
    """Call R-STILT calc_plume_dilution on *particles*, return DataFrame."""
    p_in = tmp_path / "particles_in.parquet"
    p_out = tmp_path / "particles_out.parquet"
    particles.to_parquet(p_in, index=False)

    result = subprocess.run(
        [
            rscript,
            str(_R_HELPERS / "calc_plume_dilution.r"),
            str(p_in),
            str(p_out),
            str(numpar),
            str(r_zagl),
            str(veght),
            str(r_stilt_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"calc_plume_dilution.r failed (exit {result.returncode}):\n{result.stderr}"
        )

    return pd.read_parquet(p_out)


def _particles(
    *,
    n: int = 1,
    times: list[float] | None = None,
    long: list[float] | None = None,
    lati: list[float] | None = None,
    foot: list[float] | None = None,
    # HNF columns — supply to test calc_plume_dilution
    mlht: list[float] | None = None,
    dens: list[float] | None = None,
    samt: list[float] | None = None,
    sigw: list[float] | None = None,
    tlgr: list[float] | None = None,
    zagl: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal synthetic particle DataFrame."""
    times = times or [-1.0] * n
    long = long or [-112.0] * n
    lati = lati or [40.5] * n
    foot = foot or [1e-4] * n
    zagl = zagl or [5.0] * n

    df = pd.DataFrame(
        {
            "time": [float(t) for t in times],
            "indx": [float(i + 1) for i in range(n)],
            "long": [float(v) for v in long],
            "lati": [float(v) for v in lati],
            "zagl": [float(v) for v in zagl],
            "foot": [float(v) for v in foot],
        }
    )

    # Add HNF columns when provided (needed for calc_plume_dilution)
    for col, vals in [
        ("mlht", mlht),
        ("dens", dens),
        ("samt", samt),
        ("sigw", sigw),
        ("tlgr", tlgr),
    ]:
        if vals is not None:
            df[col] = [float(v) for v in vals]

    return df


def _assert_footprint_close(
    py_ds: xr.Dataset, r_ds: xr.Dataset, *, label: str = ""
) -> None:
    """Compare PYSTILT and R-STILT footprint datasets."""
    prefix = f"[{label}] " if label else ""

    np.testing.assert_allclose(
        py_ds.lat.values,
        r_ds.lat.values,
        rtol=0,
        atol=1e-12,
        err_msg=f"{prefix}lat coordinates differ",
    )
    np.testing.assert_allclose(
        py_ds.lon.values,
        r_ds.lon.values,
        rtol=0,
        atol=1e-12,
        err_msg=f"{prefix}lon coordinates differ",
    )
    np.testing.assert_allclose(
        py_ds.foot.values.astype(np.float64),
        r_ds.foot.values.astype(np.float64),
        rtol=1e-7,
        atol=1e-15,
        err_msg=f"{prefix}footprint values differ",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_particle_gaussian(rscript, r_stilt_dir, tmp_path):
    """
    Cluster of particles at grid centre → Gaussian blob centred on (-112, 40.5).

    R-STILT's kernel bandwidth requires ≥2 particles so that var(long) is
    non-NA.  We use 100 particles drawn from a tight normal distribution;
    this is realistic (STILT always runs an ensemble) and exercises the full
    Gaussian rasterization path.  Both tools must agree on kernel shape,
    binning, and cell-area normalization.
    """
    rng = np.random.default_rng(42)
    n = 100
    p = _particles(
        n=n,
        long=rng.normal(-112.0, 0.05, n).tolist(),
        lati=rng.normal(40.5, 0.05, n).tolist(),
        foot=[1e-3] * n,
    )

    py_ds = _py_footprint(tmp_path / "py", p)
    r_ds = _r_footprint(tmp_path / "r", rscript, r_stilt_dir, p)

    _assert_footprint_close(py_ds, r_ds, label="single_particle_gaussian")


def test_all_outside_domain(rscript, r_stilt_dir, tmp_path):
    """
    All particles well outside the grid → footprint must be all zeros.

    Checks that neither tool crashes and both produce an all-zero array.
    """
    p = _particles(
        n=5,
        long=[-120.0, -121.0, -122.0, -123.0, -124.0],
        lati=[40.5] * 5,
        foot=[1e-4] * 5,
    )

    py_ds = _py_footprint(tmp_path / "py", p)
    r_ds = _r_footprint(tmp_path / "r", rscript, r_stilt_dir, p)

    assert np.all(py_ds.foot.values == 0), (
        "PYSTILT: expected zero footprint for out-of-domain particles"
    )
    assert np.all(r_ds.foot.values == 0), (
        "R-STILT: expected zero footprint for out-of-domain particles"
    )


def test_boundary_particles(rscript, r_stilt_dir, tmp_path):
    """
    Particles placed at the exact grid fencepost (xmin, xmax, ymin, ymax).

    Both tools must handle these the same way — the test catches any
    off-by-one disagreement in grid-edge binning.
    """
    p = _particles(
        n=4,
        long=[_GRID.xmin, _GRID.xmax, -112.0, -112.0],
        lati=[40.5, 40.5, _GRID.ymin, _GRID.ymax],
        foot=[1e-4] * 4,
    )

    py_ds = _py_footprint(tmp_path / "py", p)
    r_ds = _r_footprint(tmp_path / "r", rscript, r_stilt_dir, p)

    _assert_footprint_close(py_ds, r_ds, label="boundary_particles")


def test_irregular_grid_extent_matches_r_cell_centers(rscript, r_stilt_dir, tmp_path):
    """
    Domain spans that are not exact multiples of resolution keep complete cells.

    R-STILT uses lower-left cell starts from head(seq(xmn, xmx, by = xres), -1).
    PYSTILT should expose the matching cell centers without inventing a partial
    upper-bound cell.
    """
    grid = Grid(
        xmin=-114.0,
        xmax=-113.0,
        ymin=39.0,
        ymax=40.0,
        xres=0.3,
        yres=0.4,
    )
    p = _particles(
        n=4,
        long=[-113.85, -113.55, -113.25, -113.55],
        lati=[39.2, 39.6, 39.2, 39.6],
        foot=[1e-4, 2e-4, 3e-4, 4e-4],
    )

    py_ds = _py_footprint(tmp_path / "py", p, grid=grid, smooth_factor=0.0)
    r_ds = _r_footprint(
        tmp_path / "r", rscript, r_stilt_dir, p, grid=grid, smooth_factor=0.0
    )

    np.testing.assert_allclose(py_ds.lon.values, [-113.85, -113.55, -113.25])
    np.testing.assert_allclose(py_ds.lat.values, [39.2, 39.6])
    _assert_footprint_close(py_ds, r_ds, label="irregular_grid_extent")


def test_utm_projection_matches_r(rscript, r_stilt_dir, tmp_path):
    """
    Non-longlat grid: lon/lat particles are projected to UTM before rasterization.

    This catches projection-bound conversion, projected grid coordinates, kernel
    bandwidth scaling without the longlat cosine correction, and projected
    NetCDF dimensions.
    """
    try:
        from pyproj import Transformer
    except ImportError:
        pytest.skip("pyproj is required for the Python UTM projection path")

    r_proj4 = subprocess.run(
        [
            rscript,
            "-e",
            "if (!requireNamespace('proj4', quietly = TRUE)) quit(status = 1)",
        ],
        capture_output=True,
        text=True,
    )
    if r_proj4.returncode != 0:
        pytest.skip("R proj4 package is required for R-STILT non-longlat comparison")

    projection = "+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs"
    Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    grid = Grid(
        xmin=-112.3,
        xmax=-111.7,
        ymin=40.2,
        ymax=40.8,
        xres=1_000.0,
        yres=1_000.0,
        projection=projection,
    )
    rng = np.random.default_rng(21)
    n = 80
    p = pd.DataFrame(
        {
            "time": [0.0] * n + [-60.0] * n,
            "indx": [float(i + 1) for i in range(n)] * 2,
            "long": np.concatenate(
                [rng.normal(-112.0, 0.01, n), rng.normal(-112.02, 0.03, n)]
            ).tolist(),
            "lati": np.concatenate(
                [rng.normal(40.5, 0.01, n), rng.normal(40.49, 0.03, n)]
            ).tolist(),
            "zagl": [5.0] * (2 * n),
            "foot": [0.0] * n + [1e-3] * n,
        }
    )

    py_ds = _py_footprint(tmp_path / "py", p, grid=grid)
    r_ds = _r_footprint(tmp_path / "r", rscript, r_stilt_dir, p, grid=grid)

    assert "x" in py_ds.coords
    assert "y" in py_ds.coords
    py_compare = py_ds.rename({"x": "lon", "y": "lat"})
    r_compare = r_ds.rename({"x": "lon", "y": "lat"})
    _assert_footprint_close(py_compare, r_compare, label="utm_projection")


def test_time_integrate(rscript, r_stilt_dir, tmp_path):
    """
    Particles at three distinct time steps with time_integrate=True.

    The output footprint must have a single time layer whose values equal
    the sum of the per-time-step footprints.  Both tools should agree.
    """
    rng = np.random.default_rng(0)
    n_per_step = 20
    times = [-1.0, -2.0, -3.0]
    longs = rng.uniform(_GRID.xmin + 0.05, _GRID.xmax - 0.05, n_per_step * len(times))
    latis = rng.uniform(_GRID.ymin + 0.05, _GRID.ymax - 0.05, n_per_step * len(times))
    foot_vals = rng.exponential(1e-4, n_per_step * len(times))
    time_col = np.repeat(times, n_per_step).tolist()

    p = _particles(
        n=len(time_col),
        times=time_col,
        long=longs.tolist(),
        lati=latis.tolist(),
        foot=foot_vals.tolist(),
    )

    py_ds = _py_footprint(tmp_path / "py", p, time_integrate=True)
    r_ds = _r_footprint(tmp_path / "r", rscript, r_stilt_dir, p, time_integrate=True)

    assert py_ds.foot.values.shape[0] == 1, "PYSTILT: expected single time layer"
    assert r_ds.foot.values.shape[0] == 1, "R-STILT: expected single time layer"
    _assert_footprint_close(py_ds, r_ds, label="time_integrate")


def test_smooth_factor_scaling(rscript, r_stilt_dir, tmp_path):
    """
    smooth_factor=0.5 (half the default kernel) vs smooth_factor=1.0.

    At 0.5, the Gaussian is narrower: the peak cell must be higher and the
    spread smaller.  Both tools should agree on the 0.5-smoothed values.
    Uses the same 100-particle cluster as test_single_particle_gaussian so
    the kernel bandwidth is well-defined (non-zero variance).
    """
    # Two-step tracks (t=0 receptor + t=-360 min = 6 h back) so that
    # min(|time|)=0 per particle, giving rtime=-360 for the 6 h step.
    # Without t=0, rtime collapses to 0 for all steps → zero-width kernel
    # for every smooth_factor → both peaks identical.
    # foot=0 at t=0 is filtered before rasterization.
    rng = np.random.default_rng(42)
    n = 100
    longs = rng.normal(-112.0, 0.05, n).tolist()
    latis = rng.normal(40.5, 0.05, n).tolist()
    indx = [float(i + 1) for i in range(n)]
    p = pd.DataFrame(
        {
            "time": [0.0] * n + [-360.0] * n,
            "indx": indx * 2,
            "long": longs * 2,
            "lati": latis * 2,
            "zagl": [5.0] * (2 * n),
            "foot": [0.0] * n + [1e-3] * n,
        }
    )

    py_half = _py_footprint(tmp_path / "py_half", p, smooth_factor=0.5)
    r_half = _r_footprint(
        tmp_path / "r_half", rscript, r_stilt_dir, p, smooth_factor=0.5
    )

    py_full = _py_footprint(tmp_path / "py_full", p, smooth_factor=1.0)

    # Narrower kernel → higher peak
    assert py_half.foot.values.max() > py_full.foot.values.max(), (
        "smooth_factor=0.5 should produce a higher peak than smooth_factor=1.0"
    )

    _assert_footprint_close(py_half, r_half, label="smooth_factor_0.5")


def test_hnf_dilution_active(rscript, r_stilt_dir, tmp_path):
    """
    HNF correction active: plume << mixing depth → foot value is corrected.

    Single particle at t = -0.1 h with small sigw → small sigma → plume ≈ 12 m,
    well within pbl_mixing = veght * mlht = 0.5 * 2000 = 1000 m.
    The corrected foot differs from the raw value; both tools must agree.
    """
    r_zagl = 5.0
    p = _particles(
        times=[-0.1],
        foot=[1e-4],
        mlht=[2000.0],
        dens=[1.2],
        samt=[10.0],
        sigw=[0.01],
        tlgr=[50.0],
    )

    py_result = calc_plume_dilution(p.copy(), r_zagl=r_zagl, veght=_VEGHT)
    r_result = _r_plume_dilution(tmp_path, rscript, r_stilt_dir, p, r_zagl=r_zagl)

    # Correction should have changed foot
    assert not np.isclose(py_result["foot"].iloc[0], p["foot"].iloc[0], rtol=1e-3), (
        "HNF correction was expected to modify foot but it did not"
    )
    np.testing.assert_allclose(
        py_result["foot"].values,
        r_result["foot"].values,
        rtol=1e-7,
        err_msg="[hnf_active] corrected foot disagrees between PYSTILT and R-STILT",
    )
    np.testing.assert_allclose(
        py_result["foot_no_hnf_dilution"].values,
        r_result["foot_no_hnf_dilution"].values,
        rtol=1e-12,
        err_msg="[hnf_active] foot_no_hnf_dilution (raw foot) disagrees",
    )


def test_hnf_dilution_inactive(rscript, r_stilt_dir, tmp_path):
    """
    HNF correction inactive: plume >> mixing depth → foot value unchanged.

    Single particle at t = -6 h with large sigw → large sigma → plume ≈ 200 km,
    far above pbl_mixing = 0.5 * 100 = 50 m.
    foot_no_hnf_dilution must equal foot; both tools must agree.
    """
    r_zagl = 5.0
    p = _particles(
        times=[-6.0],
        foot=[1e-4],
        mlht=[100.0],
        dens=[1.2],
        samt=[500.0],
        sigw=[1.0],
        tlgr=[200.0],
    )

    py_result = calc_plume_dilution(p.copy(), r_zagl=r_zagl, veght=_VEGHT)
    r_result = _r_plume_dilution(tmp_path, rscript, r_stilt_dir, p, r_zagl=r_zagl)

    # Correction should NOT have changed foot
    np.testing.assert_allclose(
        py_result["foot"].values,
        p["foot"].values,
        rtol=1e-12,
        err_msg="[hnf_inactive] foot should be unchanged when plume >> mixing depth",
    )
    np.testing.assert_allclose(
        py_result["foot"].values,
        r_result["foot"].values,
        rtol=1e-7,
        err_msg="[hnf_inactive] foot disagrees between PYSTILT and R-STILT",
    )


def test_early_time_multi_timestep(rscript, r_stilt_dir, tmp_path):
    """
    Multiple timesteps within the 100-minute early-time window, slow movement.

    All existing synthetic tests use a single non-zero timestep per particle
    (anchor rows copy the receptor position, so only t=-1 has foot>0).  This
    means every particle has the same rtime, and the kernel is only computed
    once.  Here particles have genuine foot values at five timesteps spanning
    0–50 minutes, so the kernel bandwidth is computed at five distinct rtimes
    and the footprint accumulates over multiple independent raster layers.

    Particle movement is kept small (0.001° per step, far below xres=0.01°)
    so should_interpolate=FALSE and we avoid the sub-minute interpolation path.
    """
    rng = np.random.default_rng(7)
    n = 20
    times = [0.0, -1.0, -5.0, -20.0, -50.0]
    base_long = rng.normal(-112.0, 0.02, n)
    base_lati = rng.normal(40.5, 0.02, n)
    rows = []
    for i, t in enumerate(times):
        rows.append(
            pd.DataFrame(
                {
                    "time": [t] * n,
                    "indx": [float(j + 1) for j in range(n)],
                    # Slow drift: 0.001° per step << xres=0.01° → no sub-minute interp
                    "long": (base_long - i * 0.001).tolist(),
                    "lati": (base_lati + i * 0.001).tolist(),
                    "zagl": [5.0] * n,
                    "foot": [0.0 if t == 0.0 else 1e-3] * n,
                }
            )
        )
    p = pd.concat(rows, ignore_index=True)

    py_ds = _py_footprint(tmp_path / "py", p)
    r_ds = _r_footprint(tmp_path / "r", rscript, r_stilt_dir, p)

    assert py_ds.foot.values.sum() > 0, "footprint should be non-zero"
    _assert_footprint_close(py_ds, r_ds, label="early_time_multi_timestep")


def test_early_interpolation_with_hysplit_columns_matches_r_na_omit(
    rscript, r_stilt_dir, tmp_path
):
    """
    Fast early tracks with extra HYSPLIT columns follow R-STILT's na.omit path.

    R-STILT requests sub-minute interpolation here, but inserted rows have NA
    for non-interpolated HYSPLIT columns such as zagl/mlht/dens, so na.omit()
    removes those inserted rows.  PYSTILT must preserve that behavior for real
    trajectory tables even though minimal synthetic tables can still densify.
    """
    rng = np.random.default_rng(17)
    n = 20
    times = [0.0, -1.0, -5.0, -50.0, -120.0]
    base_long = rng.normal(-112.0, 0.02, n)
    base_lati = rng.normal(40.5, 0.02, n)
    rows = []
    for t in times:
        rows.append(
            pd.DataFrame(
                {
                    "time": [t] * n,
                    "indx": [float(j + 1) for j in range(n)],
                    "long": (base_long + abs(t) * 0.005).tolist(),
                    "lati": (base_lati - abs(t) * 0.001).tolist(),
                    "zagl": [5.0 + abs(t) * 0.01] * n,
                    "foot": [0.0 if t == 0.0 else 1e-3] * n,
                    "mlht": [1000.0] * n,
                    "dens": [1.2] * n,
                    "samt": [abs(t) * 60.0] * n,
                    "sigw": [0.5] * n,
                    "tlgr": [100.0] * n,
                }
            )
        )
    p = pd.concat(rows, ignore_index=True)

    py_ds = _py_footprint(tmp_path / "py", p)
    r_ds = _r_footprint(tmp_path / "r", rscript, r_stilt_dir, p)

    assert py_ds.foot.values.sum() > 0, "footprint should be non-zero"
    _assert_footprint_close(py_ds, r_ds, label="early_interp_hysplit_columns")


def test_calc_footprint_intermediate_tables_match_r(rscript, r_stilt_dir, tmp_path):
    """
    R-STILT parity at the main calc_footprint intermediate stages.

    Final footprint agreement can hide where a future drift starts.  This
    checks the post-interpolation particle table, rtime calculation, half-open
    grid starts, kernel bandwidth table, and pre-smoothing raster bins.
    """
    rng = np.random.default_rng(31)
    n = 20
    times = [0.0, -1.0, -5.0, -50.0, -120.0]
    base_long = rng.normal(-112.0, 0.02, n)
    base_lati = rng.normal(40.5, 0.02, n)
    rows = []
    for t in times:
        rows.append(
            pd.DataFrame(
                {
                    "time": [t] * n,
                    "indx": [float(j + 1) for j in range(n)],
                    "long": (base_long + abs(t) * 0.005).tolist(),
                    "lati": (base_lati - abs(t) * 0.001).tolist(),
                    "zagl": [5.0 + abs(t) * 0.01] * n,
                    "foot": [0.0 if t == 0.0 else 1e-3] * n,
                    "mlht": [1000.0] * n,
                    "dens": [1.2] * n,
                    "samt": [abs(t) * 60.0] * n,
                    "sigw": [0.5] * n,
                    "tlgr": [100.0] * n,
                }
            )
        )
    p = pd.concat(rows, ignore_index=True)

    py = _py_footprint_debug(p)
    r = _r_footprint_debug(tmp_path, rscript, r_stilt_dir, p)

    _assert_debug_tables_close(py, r, source_particles=p, label="longlat")


def test_calc_footprint_utm_intermediate_tables_match_r(rscript, r_stilt_dir, tmp_path):
    """
    R-STILT parity for projected calc_footprint intermediate stages.

    This extends the final UTM footprint comparison by checking that both tools
    project particle coordinates and grid limits the same way before computing
    grid starts, kernel bandwidths, and raster bins.
    """
    try:
        from pyproj import Transformer
    except ImportError:
        pytest.skip("pyproj is required for the Python UTM projection path")

    r_proj4 = subprocess.run(
        [
            rscript,
            "-e",
            "if (!requireNamespace('proj4', quietly = TRUE)) quit(status = 1)",
        ],
        capture_output=True,
        text=True,
    )
    if r_proj4.returncode != 0:
        pytest.skip("R proj4 package is required for R-STILT non-longlat comparison")

    projection = "+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs"
    Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    grid = Grid(
        xmin=-112.3,
        xmax=-111.7,
        ymin=40.2,
        ymax=40.8,
        xres=1_000.0,
        yres=1_000.0,
        projection=projection,
    )

    rng = np.random.default_rng(32)
    n = 40
    rows = []
    for t, foot in [(0.0, 0.0), (-60.0, 1e-3), (-180.0, 2e-3)]:
        rows.append(
            pd.DataFrame(
                {
                    "time": [t] * n,
                    "indx": [float(j + 1) for j in range(n)],
                    "long": rng.normal(-112.0 + abs(t) * 0.00005, 0.015, n).tolist(),
                    "lati": rng.normal(40.5 - abs(t) * 0.00002, 0.015, n).tolist(),
                    "zagl": [5.0 + abs(t) * 0.01] * n,
                    "foot": [foot] * n,
                    "mlht": [1000.0] * n,
                    "dens": [1.2] * n,
                    "samt": [abs(t) * 60.0] * n,
                    "sigw": [0.5] * n,
                    "tlgr": [100.0] * n,
                }
            )
        )
    p = pd.concat(rows, ignore_index=True)

    py = _py_footprint_debug(p, grid=grid)
    r = _r_footprint_debug(tmp_path, rscript, r_stilt_dir, p, grid=grid)

    _assert_debug_tables_close(py, r, source_particles=p, label="utm")


def test_latitude_kernel_scaling(rscript, r_stilt_dir, tmp_path):
    """
    Kernel bandwidth scales with cos(lat) for longlat projections.

    R-STILT computes grid_conv = cos(lat * π/180) and divides it into the
    bandwidth: w = smooth_factor * 0.06 * di * ti / grid_conv.  At lat=60°
    (cos≈0.50) the kernel is ~1.9× wider than at lat=20° (cos≈0.94).

    Both Python and R are run on identical particle clusters centred at each
    latitude; the test passes only if both agree at rtol=1e-7.  If either
    implementation omits or mismeasures the cos correction the footprint
    values will diverge.
    """
    rng = np.random.default_rng(99)
    n = 100
    spread_lon = rng.normal(0.0, 0.03, n)
    spread_lat = rng.normal(0.0, 0.03, n)

    for lat_center, lon_center in [(20.0, -112.0), (60.0, -112.0)]:
        grid = Grid(
            xmin=lon_center - 1.0,
            xmax=lon_center + 1.0,
            ymin=lat_center - 1.0,
            ymax=lat_center + 1.0,
            xres=0.01,
            yres=0.01,
        )
        p = _particles(
            n=n,
            long=(lon_center + spread_lon).tolist(),
            lati=(lat_center + spread_lat).tolist(),
            foot=[1e-3] * n,
        )
        tag = str(int(lat_center))
        py_ds = _py_footprint(tmp_path / f"py_{tag}", p, grid=grid)
        r_ds = _r_footprint(tmp_path / f"r_{tag}", rscript, r_stilt_dir, p, grid=grid)
        _assert_footprint_close(py_ds, r_ds, label=f"lat_{tag}")
