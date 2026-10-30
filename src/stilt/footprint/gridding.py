"""
Calculating a footprint from particles, as STILT-R's ``calc_footprint`` does.

Footprints match STILT-R at ``rtol=1e-7`` per cell, except on a projected
grid, where the smoothing (``_compute_kernel_bandwidths``) and the test for
the early-time interpolation (``_interpolate_early_timesteps``) differ on
purpose. Run the ``fidelity`` tests before merging any change to this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from stilt.exceptions import EmptyFootprint
from stilt.footprint.config import FootprintConfig
from stilt.particles import FOOTPRINT_COLUMNS, check_particles
from stilt.receptors import Receptor
from stilt.spatial import Grid, _grid_cell_starts, is_longlat
from stilt.transforms import apply_transforms

from .io import _footprint_array


def _make_gauss_kernel(rs: tuple[float, float], sigma: float) -> np.ndarray:
    """2D Gaussian smoothing kernel on a physical-unit grid, normalized to sum=1."""
    if sigma == 0:
        return np.array([[1.0]])
    d = 3 * sigma
    nx = 1 + 2 * int(np.floor(d / rs[0]))
    ny = 1 + 2 * int(np.floor(d / rs[1]))
    x = (np.arange(nx) - nx // 2) * rs[0]
    y = (np.arange(ny) - ny // 2) * rs[1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    m = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    w = m / m.sum()
    return np.where(np.isnan(w), 1.0, w)


#: The particle columns a footprint is made from. STILT-R's ``simulation_step``
#: passes ``calc_footprint`` only these (``footprint_varsiwant``), so the rows the
#: early-time interpolation inserts have no missing values and are kept.
_FOOTPRINT_INPUTS = ("particle", "age", "lon", "lat", "foot")


def _r_seq(start: float, stop: float, by: float) -> np.ndarray:
    """Return R's ``seq(start, stop, by = by)``: ``start + (0:n) * by``, capped at ``stop``."""
    n = int((stop - start) / by + 1e-10)
    return np.minimum(start + np.arange(n + 1) * by, stop)


def _interpolation_times(time_sign: int) -> np.ndarray:
    """
    Return STILT-R's interpolation times for the first 100 minutes, in minutes.

    They are built as R builds them, so they match STILT-R's to the last bit:
    R's ``seq(0, 10, by = 0.1)`` holds ``2.9000000000000004``, not ``2.9``.
    A position interpolated at a time one bit off can fall in the next grid
    cell, which moves that step's ``foot`` with it.
    """
    times = np.concatenate(
        [_r_seq(0, 10, 0.1), _r_seq(10.2, 20, 0.2), _r_seq(20.5, 100, 0.5)]
    )
    return times * time_sign


def _wrap_antimeridian_longitudes(
    p: pd.DataFrame, *, xmin: float, xmax: float
) -> tuple[pd.DataFrame, float, float, bool]:
    """
    Shift longitudes to 0 to 360 when the grid crosses the antimeridian.

    Only for longitude/latitude grids. Returns ``(p, xmin, xmax, wrapped)``.
    """
    xdist = ((180 - xmin) - (-180 - xmax)) % 360
    if xdist == 0:
        return p, -180.0, 180.0, False
    if (xmax < xmin) or (xmax > 180):
        p = p.copy()
        p["lon"] = ((p["lon"] % 360) + 360) % 360
        xmin = ((xmin % 360) + 360) % 360
        xmax = ((xmax % 360) + 360) % 360
        return p, xmin, xmax, True
    return p, xmin, xmax, False


def _interpolate_early_timesteps(
    p: pd.DataFrame,
    *,
    xres: float,
    yres: float,
    time_sign: int,
    crs: str = "+proj=longlat",
) -> pd.DataFrame:
    """
    Add interpolated positions in the first 100 minutes when particles jump grid cells.

    Near the receptor, particles can move more than a grid cell per output
    step. When the median step does, along either axis of the grid,
    positions and ``foot`` are linearly interpolated onto finer times (every
    0.1 min to 10 min, 0.2 min to 20 min, and 0.5 min to 100 min). ``foot``
    is then rescaled so each of those three windows keeps its total, as
    STILT-R does.

    *p* holds longitudes and latitudes in degrees. On a projected *crs* the
    steps are measured on the grid, from the particles' positions projected
    to it, so they compare with *xres* and *yres* in its units. STILT-R
    compares the steps in degrees with the cell size in the grid's units,
    which on a grid in meters never interpolates. The interpolation itself
    is done in degrees on any grid.
    """
    early = cast(pd.DataFrame, p[np.abs(p["age"]) < 100])
    if early.empty:
        return p

    # Match R: per-particle median(abs(diff(lon/lat))), then median across particles.
    # should_interpolate = median(dx) > xres OR median(dy) > yres
    sorted_early = early.sort_values(by=["particle", "age"])
    if not is_longlat(crs):
        # The steps along the grid's axes, in its units.
        from pyproj import Transformer

        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = tr.transform(
            sorted_early["lon"].to_numpy(), sorted_early["lat"].to_numpy()
        )
        sorted_early = sorted_early.assign(lon=x, lat=y)
    diffs = cast(
        pd.DataFrame,
        sorted_early.groupby("particle", sort=False)[["lon", "lat"]].diff().abs(),
    )
    per_particle_med = cast(
        pd.DataFrame, diffs.groupby(sorted_early["particle"]).median()
    )
    dx_values = per_particle_med["lon"].dropna().to_numpy()
    dy_values = per_particle_med["lat"].dropna().to_numpy()
    dx_med = float(np.median(dx_values)) if dx_values.size else np.nan
    dy_med = float(np.median(dy_values)) if dy_values.size else np.nan

    if (np.isnan(dx_med) or dx_med <= xres) and (np.isnan(dy_med) or dy_med <= yres):
        return p

    t_new = _interpolation_times(time_sign)

    # Store pre-interpolation foot sums per window for rescaling later.
    atime = np.abs(p["age"])
    foot_sums = [
        p.loc[atime <= 10, "foot"].sum(),
        p.loc[(atime > 10) & (atime <= 20), "foot"].sum(),
        p.loc[(atime > 20) & (atime <= 100), "foot"].sum(),
    ]

    p = _interpolate_particle_tracks(p, t_new=t_new)
    p["age"] = p["age"].round(1)

    # Rescale foot so total influence per window matches original.
    atime = np.abs(p["age"])
    masks = [
        atime <= 10,
        (atime > 10) & (atime <= 20),
        (atime > 20) & (atime <= 100),
    ]
    for mask, total in zip(masks, foot_sums, strict=False):
        s = p.loc[mask, "foot"].sum()
        if s > 0:
            p.loc[mask, "foot"] *= total / s

    return p


def _interpolate_particle_tracks(p: pd.DataFrame, *, t_new: np.ndarray) -> pd.DataFrame:
    """Interpolate lon/lat/foot for each particle on a dense time grid."""
    frames: list[pd.DataFrame] = []
    original_columns = list(p.columns)
    for number, group in p.groupby("particle", sort=False):
        source_time = group["age"].to_numpy(dtype=float)
        target_time = np.unique(np.concatenate([source_time, t_new]))
        order = np.argsort(source_time)
        sorted_time = source_time[order]
        unique_time, unique_idx = np.unique(sorted_time, return_index=True)

        # Mirror STILT-R's full_join(expand.grid(...), by = c("particle", "time")):
        # original rows keep every column, while inserted rows only receive the
        # interpolated lon/lat/foot values below.  A later dropna() therefore
        # matches R's na.omit() across all columns.
        expanded = pd.DataFrame(
            {
                "particle": np.full(len(target_time), number),
                "age": target_time,
            }
        )
        join_group = group.copy(deep=False)
        join_group["age"] = join_group["age"].astype(float)
        frame = expanded.merge(
            join_group, on=["particle", "age"], how="left", sort=False
        )
        for col in ["lon", "lat", "foot"]:
            values = group[col].to_numpy(dtype=float)[order][unique_idx]
            if len(unique_time) >= 2:
                # left/right=nan matches R's na_interp: no extrapolation
                # beyond the observed [min_time, max_time] range.
                frame[col] = np.interp(
                    target_time, unique_time, values, left=np.nan, right=np.nan
                )
            elif len(unique_time) == 1:
                frame[col] = np.full(len(target_time), values[0])
            else:
                frame[col] = np.full(len(target_time), np.nan)
        frames.append(frame.loc[:, original_columns])
    if not frames:
        return p.iloc[0:0].copy()
    return (
        pd.concat(frames, ignore_index=True)
        .dropna()
        .sort_values(["particle", "age"], ascending=[True, False], kind="stable")
        .reset_index(drop=True)
    )


def _project_particles_to_crs(
    p: pd.DataFrame,
    *,
    crs: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[pd.DataFrame, float, float, float, float]:
    """
    Project particle positions and the grid bounds to the grid's CRS.

    Grid bounds are given in degrees and projected along with the
    particles.
    """
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    p = p.copy()
    p["lon"], p["lat"] = tr.transform(p["lon"].values, p["lat"].values)
    corners_x, corners_y = tr.transform([xmin, xmax], [ymin, ymax])
    return (
        p,
        float(np.min(corners_x)),
        float(np.max(corners_x)),
        float(np.min(corners_y)),
        float(np.max(corners_y)),
    )


def _compute_kernel_bandwidths(
    p: pd.DataFrame, *, smooth_factor: float, crs: str = "+proj=longlat"
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Return ``(kernel_df, w)``, the Gaussian kernel width for each ``rtime``, in *crs* units.

    *p* holds the particles' longitudes and latitudes in degrees, before any
    projection. The width grows with the spread of the particles and the
    time since release, as in STILT-R, and is divided by ``cos(lat)`` so it
    covers the same distance at high latitudes::

        w = smooth_factor * 0.06 * varsum**0.25 * (|rtime| / 1440)**0.5 / cos(lat)

    This is in degrees of longitude. On a projected grid it is converted to
    the grid's units by the length of one degree of longitude at the
    particles' mean position (``_grid_units_per_degree_longitude``), so the
    kernel is as wide from east to west, on the ground, as on a
    longitude/latitude grid. STILT-R applies the formula to the projected
    coordinates instead, which on a grid in meters gives a kernel a few
    meters wide.
    """
    kernel_df = (
        p.groupby("rtime")
        .agg(
            long_var=("lon", "var"),
            lati_var=("lat", "var"),
            lat_mean=("lat", "mean"),
            lon_mean=("lon", "mean"),
        )
        .reset_index()
        .dropna()
    )
    if kernel_df.empty:
        # Single-particle / no-variance case: zero-width kernel per rtime.
        rtime_vals = np.sort(np.asarray(p["rtime"].dropna().unique()))
        kernel_df = pd.DataFrame(
            {
                "rtime": rtime_vals,
                "lat_mean": [float(p["lat"].to_numpy().mean())] * len(rtime_vals),
            }
        )
        return kernel_df, np.zeros(len(kernel_df), dtype=float)

    kernel_df["varsum"] = kernel_df["long_var"] + kernel_df["lati_var"]
    di = kernel_df["varsum"].to_numpy() ** 0.25
    ti = (np.abs(kernel_df["rtime"].to_numpy()) / 1440) ** 0.5
    grid_conv = np.cos(kernel_df["lat_mean"].to_numpy() * np.pi / 180)
    w = smooth_factor * 0.06 * di * ti / grid_conv
    if not is_longlat(crs):
        w = w * _grid_units_per_degree_longitude(
            crs,
            lon=kernel_df["lon_mean"].to_numpy(),
            lat=kernel_df["lat_mean"].to_numpy(),
        )
    return kernel_df, w


def _grid_units_per_degree_longitude(
    crs: str, *, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """
    Return the length of one degree of longitude at each ``(lon, lat)``, in *crs* units.

    It is measured with the projection itself: the distance between the
    points a hundredth of a degree east and west, divided by the
    difference in degrees. So it includes the projection's scale at that
    point and its units (meters, feet, kilometers), and does not depend on
    how the projection's axes are turned.
    """
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    step = 0.01
    x_west, y_west = tr.transform(lon - step, lat)
    x_east, y_east = tr.transform(lon + step, lat)
    return np.hypot(
        np.asarray(x_east) - np.asarray(x_west),
        np.asarray(y_east) - np.asarray(y_west),
    ) / (2 * step)


def _padded_axes(
    *,
    xmin: float,
    ymin: float,
    xres: float,
    yres: float,
    n_lon: int,
    n_lat: int,
    xbuf: int,
    ybuf: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the lower cell edges of the output grid padded by *xbuf* and *ybuf* cells per side.

    The padding, the size of the largest kernel, lets particles just outside
    the grid be smoothed into it.
    """
    glong_buf = xmin - xbuf * xres + np.arange(n_lon + 2 * xbuf) * xres
    glati_buf = ymin - ybuf * yres + np.arange(n_lat + 2 * ybuf) * yres
    return glong_buf, glati_buf


def _filter_and_rasterize_particles(
    p: pd.DataFrame,
    *,
    glong_buf: np.ndarray,
    glati_buf: np.ndarray,
    xbufh: int,
    ybufh: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xres: float,
    yres: float,
    time_integrate: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Sum ``foot`` by padded grid cell and time step for particles on the padded grid.

    Returns ``(p, layers)``. ``p`` has columns ``loi, lai, time, rtime,
    foot, layer``, and ``layers`` holds the sorted hour indices, or ``[0]``
    when ``time_integrate`` is set. Both are empty when no particle is on
    the padded grid.
    """
    filtered = cast(
        pd.DataFrame,
        p[
            (p["foot"] > 0)
            & (p["lon"] >= xmin - xbufh * xres)
            & (p["lon"] < xmax + xbufh * xres)
            & (p["lat"] >= ymin - ybufh * yres)
            & (p["lat"] < ymax + ybufh * yres)
        ].copy(),
    )

    if filtered.empty:
        return filtered, np.zeros(0, dtype=int)

    filtered["loi"] = (
        np.searchsorted(
            glong_buf,
            filtered["lon"].to_numpy(dtype=float),
            side="right",
        )
        - 1
    )
    filtered["lai"] = (
        np.searchsorted(
            glati_buf,
            filtered["lat"].to_numpy(dtype=float),
            side="right",
        )
        - 1
    )
    # Sum foot for particles in the same cell at the same time step.
    p = cast(
        pd.DataFrame,
        filtered.groupby(["loi", "lai", "age", "rtime"], as_index=False)["foot"].sum(),
    )

    # time_integrate=True collapses all steps into a single layer; otherwise
    # bin into hourly layers for time-resolved output.
    p["layer"] = 0 if time_integrate else np.floor(p["age"] / 60).astype(int)
    layers = np.sort(np.asarray(p["layer"].unique(), dtype=int))
    return p, layers


# Most (cell, kernel cell) pairs ``_stamp_kernel`` holds at once, about 16 to
# 50 bytes each. It keeps memory bounded when late-time kernels are wide.
_MAX_STAMP_PAIRS = 2**20


def _stamp_kernel(
    out: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    kernel: np.ndarray,
) -> None:
    """
    Add ``values[j] * kernel``, centered on cell ``(rows[j], cols[j])``, to *out* in place.

    This is STILT-R's ``permute.f90``: the kernel is added at each occupied
    cell, so the cost is cells x kernel cells. Kernel mass that falls outside
    *out* is dropped, as a convolution with zeros beyond the edge drops it.
    Cells are stamped in chunks of at most ``_MAX_STAMP_PAIRS`` (cell, kernel
    cell) pairs, so memory stays bounded however wide the kernel is.
    """
    n_rows, n_cols = out.shape
    hx, hy = kernel.shape[0] // 2, kernel.shape[1] // 2
    dx, dy = np.meshgrid(np.arange(-hx, hx + 1), np.arange(-hy, hy + 1), indexing="ij")
    dx, dy, weights = dx.ravel(), dy.ravel(), kernel.ravel()
    chunk = max(1, _MAX_STAMP_PAIRS // weights.size)
    for start in range(0, len(values), chunk):
        r = rows[start : start + chunk]
        c = cols[start : start + chunk]
        v = values[start : start + chunk]
        # The box these kernels cover, cut to the edges of out.
        r0, r1 = r.min() - hx, r.max() + hx + 1
        c0, c1 = c.min() - hy, c.max() + hy + 1
        crosses_edge = r0 < 0 or c0 < 0 or r1 > n_rows or c1 > n_cols
        r0, r1 = max(r0, 0), min(r1, n_rows)
        c0, c1 = max(c0, 0), min(c1, n_cols)
        width = c1 - c0
        # Flat index in the box of each (cell, kernel cell) pair, and its mass.
        idx = ((r - r0) * width + (c - c0))[:, None] + (dx * width + dy)
        mass = v[:, None] * weights
        if crosses_edge:
            rr = r[:, None] + dx
            cc = c[:, None] + dy
            inside = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
            idx, mass = idx[inside], mass[inside]
        out[r0:r1, c0:c1] += np.bincount(
            idx.ravel(), weights=mass.ravel(), minlength=(r1 - r0) * width
        ).reshape(r1 - r0, width)


def _accumulate_smoothed_footprint(
    p: pd.DataFrame,
    *,
    layers: np.ndarray,
    n_lon_buf: int,
    n_lat_buf: int,
    rtimes: np.ndarray,
    w: np.ndarray,
    rs: tuple[float, float],
) -> np.ndarray:
    """
    Grid each time step's ``foot`` on the padded grid and smooth it with a Gaussian kernel.

    Returns an array of shape ``(n_lon_buf, n_lat_buf, n_layers)``. Each
    step's particles are summed by cell, then the step's kernel is added at
    each occupied cell (``_stamp_kernel``), as STILT-R's ``permute.f90``
    does. This equals a convolution of the gridded step with the kernel, but
    costs cells x kernel cells instead of grid cells x kernel cells.
    """
    foot_arr = np.zeros((n_lon_buf, n_lat_buf, len(layers)), dtype=float)
    kernel_cache: dict[float, np.ndarray] = {}

    layer_index = {int(layer): i for i, layer in enumerate(layers)}
    for key, step in p.groupby(["layer", "rtime"], sort=False):
        layer, rtime_val = cast(tuple[Any, Any], key)
        i = layer_index[int(layer)]

        # Nearest-neighbor kernel bandwidth for this rtime.
        step_w_idx = int(np.argmin(np.abs(rtimes - rtime_val)))
        step_w = float(w[step_w_idx])
        if step_w not in kernel_cache:
            kernel_cache[step_w] = _make_gauss_kernel(rs, step_w)
        k = kernel_cache[step_w]

        loi_arr = step["loi"].to_numpy().astype(int)
        lai_arr = step["lai"].to_numpy().astype(int)
        foot_vals = step["foot"].to_numpy(dtype=float)
        valid = (
            (loi_arr >= 0)
            & (loi_arr < n_lon_buf)
            & (lai_arr >= 0)
            & (lai_arr < n_lat_buf)
        )
        # Sum the particles in each occupied cell, then stamp the kernel once per cell.
        cells, inverse = np.unique(
            loi_arr[valid] * n_lat_buf + lai_arr[valid], return_inverse=True
        )
        if len(cells) == 0:
            continue
        cell_foot = np.bincount(inverse, weights=foot_vals[valid])
        _stamp_kernel(
            foot_arr[:, :, i], cells // n_lat_buf, cells % n_lat_buf, cell_foot, k
        )

    return foot_arr


def calc_footprint(
    particles: pd.DataFrame,
    receptor: Receptor,
    grid: Grid,
    *,
    smooth_factor: float = 1.0,
    time_integrate: bool = False,
    transforms: Sequence[Any] = (),
    name: str = "",
    directory: str | Path | None = None,
    geometry_hash: str | None = None,
) -> xr.DataArray:
    """
    Calculate a footprint from particles, as STILT-R's ``calc_footprint`` does.

    The particle *transforms* are applied first, in order, so the footprint
    records exactly the transforms it was made with. Near the receptor,
    particle tracks are interpolated to finer times when particles cross
    more than a grid cell per step. Each particle's ``foot`` is added to the
    cell it is in, and each time step is smoothed with a Gaussian kernel
    that widens with the particles' spread and age. The sum is divided by
    the number of particles and binned by hour, unless *time_integrate* is
    set.

    On a projected grid the kernel's width is worked out in degrees, as on
    a longitude/latitude grid, and converted to the grid's units, so the
    footprint is smoothed as much. STILT-R applies its formula to the
    projected coordinates and barely smooths a footprint there. The step
    that decides the early-time interpolation is measured on the projected
    grid, in its units, so a grid of 100 m cells is interpolated as a 0.001°
    grid is. STILT-R measures it in degrees and never interpolates on a grid
    in meters.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table, such as :func:`stilt.run_trajectories` returns or
        ``sim.particles``, with columns ``particle``, ``age`` (minutes since
        release), ``lon``, ``lat``, and ``foot``.
    receptor : Receptor
        Receptor the particles were released from.
    grid : Grid
        Domain and resolution of the footprint. For a footprint given by a
        geometry, derive it first: ``stilt.Mesh.from_spec(geometry).to_grid()``.
    smooth_factor : float, default 1.0
        Factor on the width of the Gaussian smoothing kernel. 0 turns
        smoothing off.
    time_integrate : bool, default False
        Sum the footprint over time into one layer instead of hourly layers.
    transforms : sequence, optional
        Particle transforms applied before the footprint is calculated, as
        objects or as the mappings ``config.yaml`` takes
        (``{"kind": "pressure_weighting"}``).
    name : str, optional
        Name of the footprint, usually the variant name.
    directory : str or Path, optional
        Where a relative file name in a transform's settings starts, such as
        an averaging-kernel table: the project directory.
    geometry_hash : str, optional
        Hash of the geometry the grid was derived for
        (``sim.variant.geometry_hash``), recorded with the footprint so
        aggregating it onto another mesh warns.

    Returns
    -------
    xarray.DataArray
        The footprint, named ``foot``, with dimensions ``(time, lat, lon)``
        or ``(time, y, x)`` on a projected grid. Coordinates are cell
        centers, and ``time`` is the start of each hour. The ``.stilt``
        accessor has the methods that use it.

    Raises
    ------
    ImportError
        If a transform is a mapping whose ``kind`` cannot be imported.
    EmptyFootprint
        If no particle is over the grid, or the table has none.

    Examples
    --------
    >>> grid = stilt.Grid(  # doctest: +SKIP
    ...     xmin=-113, xmax=-110.5, ymin=40, ymax=42, xres=0.01, yres=0.01
    ... )
    >>> foot = stilt.calc_footprint(
    ...     particles, receptor, grid, smooth_factor=0.5
    ... )  # doctest: +SKIP
    """
    if grid is None:
        raise ValueError(
            "A footprint needs a grid. Give one, or derive it from a geometry "
            "with stilt.Mesh.from_spec(geometry).to_grid()."
        )
    grid = Grid.model_validate(grid)
    # Checks the settings, and builds the transforms given as mappings.
    config = FootprintConfig.model_validate(
        {
            "grid": grid,
            "smooth_factor": smooth_factor,
            "time_integrate": time_integrate,
            "transforms": list(transforms),
        }
    )
    if not particles.empty:
        check_particles(particles, need=FOOTPRINT_COLUMNS)
    if config.transforms:
        particles = apply_transforms(particles, config.transforms, receptor, directory)
    crs = grid.crs
    xmin, xmax, xres = grid.xmin, grid.xmax, grid.xres
    ymin, ymax, yres = grid.ymin, grid.ymax, grid.yres
    smooth_factor = config.smooth_factor
    time_integrate = config.time_integrate

    if particles.empty:
        raise EmptyFootprint()

    # Only the columns the footprint is made from, as STILT-R passes them.
    p = particles.loc[:, list(_FOOTPRINT_INPUTS)]
    n_particles = p["particle"].nunique()
    # time_sign: -1 for backward runs, +1 for forward.
    time_sign = int(np.sign(p["age"].median()))

    wrapped_longitude = False
    if grid.is_longlat:
        p, xmin, xmax, wrapped_longitude = _wrap_antimeridian_longitudes(
            p, xmin=xmin, xmax=xmax
        )

    p = _interpolate_early_timesteps(
        p, xres=xres, yres=yres, time_sign=time_sign, crs=crs
    )

    # rtime = time elapsed since each particle's first output step.
    # Used below to compute kernel bandwidth (particles spread more with time).
    min_abs_time = p["age"].abs().groupby(p["particle"], sort=False).transform("min")
    p["rtime"] = p["age"] - time_sign * min_abs_time

    # The kernel width comes from the spread in degrees, before projecting,
    # and is returned in the grid's units.
    kernel_df, w = _compute_kernel_bandwidths(p, smooth_factor=smooth_factor, crs=crs)

    if not grid.is_longlat:
        p, xmin, xmax, ymin, ymax = _project_particles_to_crs(
            p,
            crs=crs,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )

    # Output grid lower-left corners for half-open extents [min, max).
    glong = _grid_cell_starts(xmin, xmax, xres)
    glati = _grid_cell_starts(ymin, ymax, yres)
    n_lon = len(glong)
    n_lat = len(glati)
    rs = (xres, yres)

    max_kernel = (
        _make_gauss_kernel(rs, float(np.max(w))) if len(w) > 0 else np.array([[1.0]])
    )
    xbuf, ybuf = max_kernel.shape
    glong_buf, glati_buf = _padded_axes(
        xmin=xmin,
        ymin=ymin,
        xres=xres,
        yres=yres,
        n_lon=n_lon,
        n_lat=n_lat,
        xbuf=xbuf,
        ybuf=ybuf,
    )

    p, layers = _filter_and_rasterize_particles(
        p,
        glong_buf=glong_buf,
        glati_buf=glati_buf,
        xbufh=(xbuf - 1) // 2,
        ybufh=(ybuf - 1) // 2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        xres=xres,
        yres=yres,
        time_integrate=time_integrate,
    )

    if p.empty:
        raise EmptyFootprint()

    foot_arr = _accumulate_smoothed_footprint(
        p,
        layers=layers,
        n_lon_buf=len(glong_buf),
        n_lat_buf=len(glati_buf),
        rtimes=kernel_df["rtime"].to_numpy(),
        w=w,
        rs=rs,
    )

    # Trim the padding and normalize by the particle count.
    foot_arr = foot_arr[xbuf : xbuf + n_lon, ybuf : ybuf + n_lat, :] / n_particles

    values = foot_arr.transpose(2, 1, 0)  # (time, y, x)
    # Rounded as Grid.axes rounds them, so a footprint read back from its
    # file has the same coordinates as the one calculated.
    x_coords = np.round(glong + xres / 2, 10)
    y_coords = np.round(glati + yres / 2, 10)
    if wrapped_longitude:
        unwrapped = ((x_coords + 180.0) % 360.0) - 180.0
        order = np.argsort(unwrapped)
        x_coords = np.round(unwrapped[order], 10)
        values = values[:, :, order]
    return _footprint_array(
        values, layers, receptor, config, name, x_coords, y_coords, geometry_hash
    )
