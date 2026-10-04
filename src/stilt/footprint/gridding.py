"""
Calculating a footprint from particles, as STILT-R's ``calc_footprint`` does.

Footprints match STILT-R at ``rtol=1e-7`` per cell. Run the ``fidelity``
tests before merging any change to this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import convolve as _convolve

from stilt.exceptions import EmptyFootprint
from stilt.footprint.config import FootprintConfig
from stilt.receptors import Receptor
from stilt.spatial import _grid_cell_starts
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


def _interpolation_times(time_sign: int) -> np.ndarray:
    """Return STILT-R's interpolation times for the first 100 minutes, in minutes."""
    times = np.concatenate(
        [
            np.arange(0, 101, dtype=float) / 10,
            np.arange(102, 201, 2, dtype=float) / 10,
            np.arange(205, 1001, 5, dtype=float) / 10,
        ]
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
        p["long"] = ((p["long"] % 360) + 360) % 360
        xmin = ((xmin % 360) + 360) % 360
        xmax = ((xmax % 360) + 360) % 360
        return p, xmin, xmax, True
    return p, xmin, xmax, False


def _interpolate_early_timesteps(
    p: pd.DataFrame, *, xres: float, yres: float, time_sign: int
) -> pd.DataFrame:
    """
    Add interpolated positions in the first 100 minutes when particles jump grid cells.

    Near the receptor, particles can move more than a grid cell per output
    step. When the median step does, positions and ``foot`` are linearly
    interpolated onto finer times (every 0.1 min to 10 min, 0.2 min to 20
    min, and 0.5 min to 100 min). ``foot`` is then rescaled so each of those
    three windows keeps its total, as STILT-R does.
    """
    early = cast(pd.DataFrame, p[np.abs(p["time"]) < 100])
    if early.empty:
        return p

    # Match R: per-particle median(abs(diff(long/lati))), then median across particles.
    # should_interpolate = median(dx) > xres OR median(dy) > yres
    sorted_early = early.sort_values(by=["indx", "time"])
    diffs = cast(
        pd.DataFrame,
        sorted_early.groupby("indx", sort=False)[["long", "lati"]].diff().abs(),
    )
    per_particle_med = cast(pd.DataFrame, diffs.groupby(sorted_early["indx"]).median())
    dx_values = per_particle_med["long"].dropna().to_numpy()
    dy_values = per_particle_med["lati"].dropna().to_numpy()
    dx_med = float(np.median(dx_values)) if dx_values.size else np.nan
    dy_med = float(np.median(dy_values)) if dy_values.size else np.nan

    if (np.isnan(dx_med) or dx_med <= xres) and (np.isnan(dy_med) or dy_med <= yres):
        return p

    t_new = _interpolation_times(time_sign)

    # Store pre-interpolation foot sums per window for rescaling later.
    atime = np.abs(p["time"])
    foot_sums = [
        p.loc[atime <= 10, "foot"].sum(),
        p.loc[(atime > 10) & (atime <= 20), "foot"].sum(),
        p.loc[(atime > 20) & (atime <= 100), "foot"].sum(),
    ]

    p = _interpolate_particle_tracks(p, t_new=t_new)
    p["time"] = p["time"].round(1)

    # Rescale foot so total influence per window matches original.
    atime = np.abs(p["time"])
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
    """Interpolate long/lati/foot for each particle on a dense time grid."""
    frames: list[pd.DataFrame] = []
    original_columns = list(p.columns)
    for indx, group in p.groupby("indx", sort=False):
        source_time = group["time"].to_numpy(dtype=float)
        target_time = np.unique(np.concatenate([source_time, t_new]))
        order = np.argsort(source_time)
        sorted_time = source_time[order]
        unique_time, unique_idx = np.unique(sorted_time, return_index=True)

        # Mirror STILT-R's full_join(expand.grid(...), by = c("indx", "time")):
        # original rows keep every column, while inserted rows only receive the
        # interpolated long/lati/foot values below.  A later dropna() therefore
        # matches R's na.omit() across all columns.
        expanded = pd.DataFrame(
            {
                "indx": np.full(len(target_time), indx),
                "time": target_time,
            }
        )
        join_group = group.copy(deep=False)
        join_group["time"] = join_group["time"].astype(float)
        frame = expanded.merge(join_group, on=["indx", "time"], how="left", sort=False)
        for col in ["long", "lati", "foot"]:
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
        .sort_values(["indx", "time"], ascending=[True, False], kind="stable")
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
    p["long"], p["lati"] = tr.transform(p["long"].values, p["lati"].values)
    corners_x, corners_y = tr.transform([xmin, xmax], [ymin, ymax])
    return (
        p,
        float(np.min(corners_x)),
        float(np.max(corners_x)),
        float(np.min(corners_y)),
        float(np.max(corners_y)),
    )


def _compute_kernel_bandwidths(
    p: pd.DataFrame, *, smooth_factor: float, is_longlat: bool
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Return ``(kernel_df, w)``, the Gaussian kernel width for each ``rtime``.

    The width grows with the spread of the particles and the time since
    release, as in STILT-R. On a longitude/latitude grid it is divided by
    ``cos(lat)`` so it covers the same distance at high latitudes::

        w = smooth_factor * 0.06 * varsum**0.25 * (|rtime| / 1440)**0.5 / cos(lat)
    """
    kernel_df = (
        p.groupby("rtime")
        .agg(
            long_var=("long", "var"),
            lati_var=("lati", "var"),
            lat_mean=("lati", "mean"),
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
                "lat_mean": [float(p["lati"].to_numpy().mean())] * len(rtime_vals),
            }
        )
        return kernel_df, np.zeros(len(kernel_df), dtype=float)

    kernel_df["varsum"] = kernel_df["long_var"] + kernel_df["lati_var"]
    di = kernel_df["varsum"].to_numpy() ** 0.25
    ti = (np.abs(kernel_df["rtime"].to_numpy()) / 1440) ** 0.5
    grid_conv = (
        np.cos(kernel_df["lat_mean"].to_numpy() * np.pi / 180) if is_longlat else 1.0
    )
    w = smooth_factor * 0.06 * di * ti / grid_conv
    return kernel_df, w


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
            & (p["long"] >= xmin - xbufh * xres)
            & (p["long"] < xmax + xbufh * xres)
            & (p["lati"] >= ymin - ybufh * yres)
            & (p["lati"] < ymax + ybufh * yres)
        ].copy(),
    )

    if filtered.empty:
        return filtered, np.zeros(0, dtype=int)

    filtered["loi"] = (
        np.searchsorted(
            glong_buf,
            filtered["long"].to_numpy(dtype=float),
            side="right",
        )
        - 1
    )
    filtered["lai"] = (
        np.searchsorted(
            glati_buf,
            filtered["lati"].to_numpy(dtype=float),
            side="right",
        )
        - 1
    )
    # Sum foot for particles in the same cell at the same time step.
    p = cast(
        pd.DataFrame,
        filtered.groupby(["loi", "lai", "time", "rtime"], as_index=False)["foot"].sum(),
    )

    # time_integrate=True collapses all steps into a single layer; otherwise
    # bin into hourly layers for time-resolved output.
    p["layer"] = 0 if time_integrate else np.floor(p["time"] / 60).astype(int)
    layers = np.sort(np.asarray(p["layer"].unique(), dtype=int))
    return p, layers


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

    Returns an array of shape ``(n_lon_buf, n_lat_buf, n_layers)``. The
    convolution covers only the bounding box of nonzero cells, which gives
    the same result as the whole grid because the cells outside it are
    zero.
    """
    foot_arr = np.zeros((n_lon_buf, n_lat_buf, len(layers)), dtype=float)
    kernel_cache: dict[float, np.ndarray] = {}

    layer_index = {int(layer): i for i, layer in enumerate(layers)}
    for key, step in p.groupby(["layer", "rtime"], sort=False):
        layer, rtime_val = cast(tuple[Any, Any], key)
        i = layer_index[int(layer)]

        # Nearest-neighbour kernel bandwidth for this rtime.
        step_w_idx = int(np.argmin(np.abs(rtimes - rtime_val)))
        step_w = float(w[step_w_idx])
        if step_w not in kernel_cache:
            kernel_cache[step_w] = _make_gauss_kernel(rs, step_w)
        k = kernel_cache[step_w]

        loi_arr = step["loi"].values.astype(int)
        lai_arr = step["lai"].values.astype(int)
        foot_vals = step["foot"].to_numpy(dtype=float)
        valid = (
            (loi_arr >= 0)
            & (loi_arr < n_lon_buf)
            & (lai_arr >= 0)
            & (lai_arr < n_lat_buf)
        )
        lin_idx = loi_arr[valid] * n_lat_buf + lai_arr[valid]
        sparse = np.bincount(
            lin_idx,
            weights=foot_vals[valid],
            minlength=n_lon_buf * n_lat_buf,
        ).reshape(n_lon_buf, n_lat_buf)

        nz_r, nz_c = np.nonzero(sparse)
        if len(nz_r) == 0:
            continue
        kh_x = k.shape[0] // 2
        kh_y = k.shape[1] // 2
        r0 = max(0, nz_r.min() - kh_x)
        r1 = min(n_lon_buf, nz_r.max() + kh_x + 1)
        c0 = max(0, nz_c.min() - kh_y)
        c1 = min(n_lat_buf, nz_c.max() + kh_y + 1)
        foot_arr[r0:r1, c0:c1, i] += _convolve(
            sparse[r0:r1, c0:c1], k, mode="constant", cval=0.0
        )

    return foot_arr


def calculate(
    particles: pd.DataFrame,
    receptor: Receptor,
    config: FootprintConfig,
    name: str = "",
    directory: str | Path | None = None,
    geometry_hash: str | None = None,
) -> xr.DataArray:
    """
    Calculate a footprint from particles.

    The particle transforms in ``config.transforms`` are applied first, in
    order, so the footprint records exactly the transforms it was made
    with. The rest follows STILT-R's ``calc_footprint``. Near the receptor,
    particle tracks are interpolated to finer times when particles cross
    more than a grid cell per step. Each particle's ``foot`` is added to the
    cell it is in, and each time step is smoothed with a Gaussian kernel
    that widens with the particles' spread and age. The sum is divided by
    the number of particles and binned by hour, unless
    ``config.time_integrate`` is set.

    :meth:`stilt.Simulation.generate_footprint` calls this with a
    simulation's own particles and receptor.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table, such as ``sim.particles``, with columns ``indx``,
        ``time`` (minutes since release), ``long``, ``lati``, and ``foot``.
    receptor : Receptor
        Receptor the particles were released from.
    config : FootprintConfig
        Grid, smoothing, and particle transforms. For a footprint given by a
        geometry, derive the grid first: ``Mesh.from_spec(config.geometry).to_grid()``.
    name : str, optional
        Name of the footprint, usually the variant name.
    directory : str or Path, optional
        Where a relative file name in a transform's settings starts, such as
        an averaging-kernel table: the project directory.
    geometry_hash : str, optional
        Hash of the geometry the grid was derived for, recorded with the
        footprint so aggregating it onto another mesh warns.

    Returns
    -------
    xarray.DataArray
        The footprint, named ``foot``, with dimensions ``(time, lat, lon)``
        or ``(time, y, x)`` on a projected grid. Coordinates are cell
        centres, and ``time`` is the start of each hour. The ``.stilt``
        accessor has the methods that use it.

    Raises
    ------
    ImportError
        If a transform in ``config`` is a settings mapping that could not
        be imported.
    EmptyFootprint
        If no particle is over the grid. ``reason`` is ``"no_particles"``
        when the table is empty and ``"outside_domain"`` otherwise.
    """
    grid = config.grid
    if grid is None:
        raise ValueError(
            "The footprint config has no grid. Give one, or derive it from the "
            "geometry with stilt.Mesh.from_spec(config.geometry).to_grid()."
        )
    if config.transforms:
        unresolved = [t["kind"] for t in config.transforms if isinstance(t, dict)]
        if unresolved:
            raise ImportError(
                f"Transforms {unresolved} could not be rebuilt, so they "
                "cannot be applied."
            )
        particles = apply_transforms(particles, config.transforms, receptor, directory)
    crs = grid.crs
    xmin, xmax, xres = grid.xmin, grid.xmax, grid.xres
    ymin, ymax, yres = grid.ymin, grid.ymax, grid.yres
    is_longlat = grid.is_longlat
    smooth_factor = config.smooth_factor
    time_integrate = config.time_integrate

    if particles.empty:
        raise EmptyFootprint("no_particles")

    p = particles.copy(deep=False)
    n_particles = p["indx"].nunique()
    # time_sign: -1 for backward runs, +1 for forward.
    time_sign = int(np.sign(p["time"].median()))

    wrapped_longitude = False
    if is_longlat:
        p, xmin, xmax, wrapped_longitude = _wrap_antimeridian_longitudes(
            p, xmin=xmin, xmax=xmax
        )

    p = _interpolate_early_timesteps(p, xres=xres, yres=yres, time_sign=time_sign)

    # rtime = time elapsed since each particle's first output step.
    # Used below to compute kernel bandwidth (particles spread more with time).
    min_abs_time = p["time"].abs().groupby(p["indx"], sort=False).transform("min")
    p["rtime"] = p["time"] - time_sign * min_abs_time

    if not is_longlat:
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

    kernel_df, w = _compute_kernel_bandwidths(
        p, smooth_factor=smooth_factor, is_longlat=is_longlat
    )
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
        raise EmptyFootprint("outside_domain")

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
