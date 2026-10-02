"""Footprints of receptors, calculated from particles and applied to surface fluxes."""

from __future__ import annotations

import datetime as dt
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import convolve as _convolve

from stilt._atomic import atomic_path
from stilt.config import FootprintConfig, Grid
from stilt.config.spatial import (
    _cf_grid_mapping_attrs,
    _grid_cell_starts,
    cf_axis_attrs,
)
from stilt.exceptions import EmptyFootprint
from stilt.geometry import (
    Mesh,
    SpatialTarget,
    Zones,
    check_resolution,
    overlap_weights,
)
from stilt.receptors import Receptor
from stilt.transforms import (
    TransformContext,
    apply_transforms,
    dump_transform,
    load_transform,
    transform_kind,
)

if TYPE_CHECKING:
    from stilt.visualization import FootprintPlotAccessor


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


def _utc_index(values: Any) -> pd.DatetimeIndex:
    """Return *values* as a UTC DatetimeIndex."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def _naive_utc_timestamp(
    value: dt.datetime | pd.Timestamp | None,
) -> pd.Timestamp | None:
    """Return one optional timestamp in UTC without a timezone, or ``None`` for none or NaT."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if not isinstance(ts, pd.Timestamp):  # NaT
        return None
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def _build_footprint_array(
    *,
    foot_arr: np.ndarray,
    layers: np.ndarray,
    receptor: Receptor,
    is_longlat: bool,
    glong: np.ndarray,
    glati: np.ndarray,
    xres: float,
    yres: float,
    wrapped_longitude: bool,
) -> xr.DataArray:
    """Build one footprint DataArray from rasterized numpy output."""
    if len(layers) == 0:
        layers = np.array([0], dtype=int)
    if len(foot_arr.shape) != 3:
        raise ValueError("foot_arr must be 3D in (time, y, x) order.")
    # Layer k is the hour from k hours after the receptor time, and is stamped
    # there, as in STILT-R; a backward run's first hour is layer -1. A
    # time-integrated footprint is the one layer 0, at the receptor time.
    time_out = [receptor.time + pd.Timedelta(hours=int(layer)) for layer in layers]
    time_index = _utc_index(time_out).tz_localize(None)
    x_dim = "lon" if is_longlat else "x"
    y_dim = "lat" if is_longlat else "y"
    # Rounded as Grid.axes rounds them, so a footprint read back from its
    # file has the same coordinates as the one calculated.
    x_coords = np.round(glong + xres / 2, 10)
    y_coords = np.round(glati + yres / 2, 10)
    if wrapped_longitude:
        unwrapped = ((x_coords + 180.0) % 360.0) - 180.0
        order = np.argsort(unwrapped)
        x_coords = np.round(unwrapped[order], 10)
        foot_arr = foot_arr[:, :, order]
    return xr.DataArray(
        foot_arr,
        dims=["time", y_dim, x_dim],
        coords={"time": time_index, y_dim: y_coords, x_dim: x_coords},
        attrs={
            "units": "ppm m2 s umol-1"
        },  # surface influence function: ppm per (µmol m⁻² s⁻¹)
    )


def _with_cf_metadata(ds: xr.Dataset, *, grid: Grid) -> xr.Dataset:
    """Add CF coordinate and CRS attributes to a footprint dataset."""
    ds.attrs.setdefault("Conventions", "CF-1.8")
    ds["crs"] = xr.DataArray(0, attrs=_cf_grid_mapping_attrs(grid.projection))
    ds["foot"].attrs["grid_mapping"] = "crs"

    for dim in ("lon", "lat", "x", "y"):
        if dim in ds.coords:
            ds[dim].attrs.update(cf_axis_attrs(dim))
    if "time" in ds.coords:
        ds["time"].attrs.update({"standard_name": "time", "axis": "T"})
    return ds


@dataclass(frozen=True, slots=True)
class _BufferedGrid:
    """Output grid padded by the widest smoothing kernel on each side."""

    glong_buf: np.ndarray
    glati_buf: np.ndarray
    xbuf: int
    ybuf: int

    @property
    def n_lon_buf(self) -> int:
        """Number of longitude (x) cells, padding included."""
        return len(self.glong_buf)

    @property
    def n_lat_buf(self) -> int:
        """Number of latitude (y) cells, padding included."""
        return len(self.glati_buf)

    @property
    def xbufh(self) -> int:
        """Half-width of the widest kernel along x, in cells."""
        return (self.xbuf - 1) // 2

    @property
    def ybufh(self) -> int:
        """Half-width of the widest kernel along y, in cells."""
        return (self.ybuf - 1) // 2


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
    projection: str,
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

    tr = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
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


def _build_buffered_grid(
    *,
    xmin: float,
    ymin: float,
    xres: float,
    yres: float,
    n_lon: int,
    n_lat: int,
    max_kernel: np.ndarray,
) -> _BufferedGrid:
    """
    Pad the output grid by the size of the largest kernel on each side.

    The padding lets particles just outside the grid be smoothed into it.
    """
    xbuf = max_kernel.shape[0]
    ybuf = max_kernel.shape[1]
    return _BufferedGrid(
        glong_buf=xmin - xbuf * xres + np.arange(n_lon + 2 * xbuf) * xres,
        glati_buf=ymin - ybuf * yres + np.arange(n_lat + 2 * ybuf) * yres,
        xbuf=xbuf,
        ybuf=ybuf,
    )


def _filter_and_rasterize_particles(
    p: pd.DataFrame,
    *,
    buffered: _BufferedGrid,
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
    when ``time_integrate`` is set.
    """
    # Layer axis is derived from unfiltered particles so that empty
    # footprints still carry the right layer count downstream.
    layer_series = (
        pd.Series(0, index=p.index, dtype=int)
        if time_integrate
        else np.floor(p["time"] / 60).astype(int)
    )
    all_layers = np.sort(np.asarray(pd.Series(layer_series).unique(), dtype=int))
    if len(all_layers) == 0:
        all_layers = np.array([0], dtype=int)

    xbufh, ybufh = buffered.xbufh, buffered.ybufh
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
        return filtered, all_layers

    filtered["loi"] = (
        np.searchsorted(
            buffered.glong_buf,
            filtered["long"].to_numpy(dtype=float),
            side="right",
        )
        - 1
    )
    filtered["lai"] = (
        np.searchsorted(
            buffered.glati_buf,
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
    buffered: _BufferedGrid,
    kernel_df: pd.DataFrame,
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
    foot_arr = np.zeros(
        (buffered.n_lon_buf, buffered.n_lat_buf, len(layers)), dtype=float
    )
    rtimes_all = kernel_df["rtime"].values
    kernel_cache: dict[float, np.ndarray] = {}

    layer_index = {int(layer): i for i, layer in enumerate(layers)}
    for key, step in p.groupby(["layer", "rtime"], sort=False):
        layer, rtime_val = cast(tuple[Any, Any], key)
        i = layer_index[int(layer)]

        # Nearest-neighbour kernel bandwidth for this rtime.
        step_w_idx = int(np.argmin(np.abs(rtimes_all - rtime_val)))
        step_w = float(w[step_w_idx])
        if step_w not in kernel_cache:
            kernel_cache[step_w] = _make_gauss_kernel(rs, step_w)
        k = kernel_cache[step_w]

        loi_arr = step["loi"].values.astype(int)
        lai_arr = step["lai"].values.astype(int)
        foot_vals = step["foot"].to_numpy(dtype=float)
        valid = (
            (loi_arr >= 0)
            & (loi_arr < buffered.n_lon_buf)
            & (lai_arr >= 0)
            & (lai_arr < buffered.n_lat_buf)
        )
        lin_idx = loi_arr[valid] * buffered.n_lat_buf + lai_arr[valid]
        sparse = np.bincount(
            lin_idx,
            weights=foot_vals[valid],
            minlength=buffered.n_lon_buf * buffered.n_lat_buf,
        ).reshape(buffered.n_lon_buf, buffered.n_lat_buf)

        nz_r, nz_c = np.nonzero(sparse)
        if len(nz_r) == 0:
            continue
        kh_x = k.shape[0] // 2
        kh_y = k.shape[1] // 2
        r0 = max(0, nz_r.min() - kh_x)
        r1 = min(buffered.n_lon_buf, nz_r.max() + kh_x + 1)
        c0 = max(0, nz_c.min() - kh_y)
        c1 = min(buffered.n_lat_buf, nz_c.max() + kh_y + 1)
        foot_arr[r0:r1, c0:c1, i] += _convolve(
            sparse[r0:r1, c0:c1], k, mode="constant", cval=0.0
        )

    return foot_arr


def _record_transform(transform: Any) -> dict[str, Any]:
    """Return a transform as recorded in a footprint file, its ``kind`` alone if it has no settings to write."""
    try:
        return dump_transform(transform)
    except TypeError:
        return {"kind": transform_kind(transform)}


def _read_transform(spec: dict[str, Any], source: str) -> Any:
    """Return a recorded transform, or its mapping when it cannot be rebuilt here."""
    try:
        return load_transform(spec)
    except (ImportError, TypeError, ValueError) as exc:
        warnings.warn(
            f"{source}: transform {spec.get('kind')!r} could not be rebuilt "
            f"({exc}). It is kept as its settings and cannot be applied.",
            stacklevel=3,
        )
        return spec


#: Units of a footprint: ppm per (µmol m⁻² s⁻¹).
UNITS = "ppm m2 s umol-1"


def _settings_json(config: FootprintConfig) -> str:
    """Return footprint settings as JSON, with each transform recorded as far as it can be."""
    data = config.model_dump(mode="json", exclude={"transforms"})
    data["transforms"] = [_record_transform(t) for t in config.transforms]
    return json.dumps(data)


def _settings_from_json(text: str, source: str) -> FootprintConfig:
    """Return footprint settings from :func:`_settings_json`, keeping a transform that cannot be rebuilt as its mapping."""
    data = json.loads(text)
    specs = data.pop("transforms", [])
    config = FootprintConfig.model_validate(data)
    # model_copy skips validation, so a mapping stays a mapping.
    return config.model_copy(
        update={"transforms": [_read_transform(s, source) for s in specs]}
    )


def _describe(
    data: xr.DataArray, receptor: Receptor, config: FootprintConfig, name: str
) -> xr.DataArray:
    """
    Name a footprint array and attach what it belongs to.

    The receptor id is a scalar coordinate, which xarray keeps through
    arithmetic in every version. The full receptor and the settings are
    attributes, which the ``.stilt`` accessor reads.
    """
    return (
        data.rename("foot")
        .assign_coords(receptor=str(receptor.id))
        .assign_attrs(
            {
                "units": UNITS,
                "long_name": "footprint",
                "stilt_name": name,
                "stilt_receptor": json.dumps(receptor.to_dict()),
                "stilt_footprint": _settings_json(config),
            }
        )
    )


def calculate(
    particles: pd.DataFrame,
    receptor: Receptor,
    config: FootprintConfig,
    name: str = "",
    context: TransformContext | None = None,
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
        Grid, smoothing, and particle transforms. A grid given by
        ``geometry`` is derived first
        (:meth:`~stilt.config.FootprintConfig.resolve`).
    name : str, optional
        Name of the footprint, usually the variant name.
    context : TransformContext, optional
        Passed to every transform. Defaults to one holding ``receptor``
        and ``name``, with no project directory.

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
    config = config.resolve()
    grid = config.grid
    if grid is None:
        raise ValueError("A footprint needs settings with a grid.")
    if config.transforms:
        unresolved = [t["kind"] for t in config.transforms if isinstance(t, dict)]
        if unresolved:
            raise ImportError(
                f"Transforms {unresolved} could not be rebuilt, so they "
                "cannot be applied."
            )
        if context is None:
            context = TransformContext(receptor=receptor, variant=name)
        particles = apply_transforms(particles, config.transforms, context)
    projection = grid.projection
    xmin, xmax, xres = grid.xmin, grid.xmax, grid.xres
    ymin, ymax, yres = grid.ymin, grid.ymax, grid.yres
    is_longlat = "+proj=longlat" in projection
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
            projection=projection,
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
    buffered = _build_buffered_grid(
        xmin=xmin,
        ymin=ymin,
        xres=xres,
        yres=yres,
        n_lon=n_lon,
        n_lat=n_lat,
        max_kernel=max_kernel,
    )

    p, layers = _filter_and_rasterize_particles(
        p,
        buffered=buffered,
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
        buffered=buffered,
        kernel_df=kernel_df,
        w=w,
        rs=rs,
    )

    # Trim buffer and normalize by particle count.
    foot_arr = (
        foot_arr[
            buffered.xbuf : buffered.xbuf + n_lon,
            buffered.ybuf : buffered.ybuf + n_lat,
            :,
        ]
        / n_particles
    )

    if foot_arr.shape != (n_lon, n_lat, len(layers)):
        raise ValueError(
            f"foot_arr shape mismatch: expected ({n_lon}, {n_lat}, {len(layers)}), "
            f"got {foot_arr.shape}"
        )

    data = _build_footprint_array(
        foot_arr=foot_arr.transpose(2, 1, 0),
        layers=layers,
        receptor=receptor,
        is_longlat=is_longlat,
        glong=glong,
        glati=glati,
        xres=xres,
        yres=yres,
        wrapped_longitude=wrapped_longitude,
    )
    return _describe(data, receptor, config, name)


def _from_sparse_table(table: Any, config: FootprintConfig) -> xr.DataArray | None:
    """Return the dense footprint of a stored sparse table, or ``None`` when it is empty."""
    meta = table.schema.metadata or {}
    if meta.get(b"stilt:empty_reason", b""):
        return None
    receptor = Receptor.from_dict(json.loads(meta[b"stilt:receptor"]))
    hours = json.loads(meta[b"stilt:hours"])
    name = meta.get(b"stilt:name", b"").decode()
    grid = config.grid
    if grid is None:
        raise ValueError("A stored footprint needs settings with a grid.")

    x_axis, y_axis = grid.axes
    values = np.zeros((len(hours), len(y_axis), len(x_axis)), dtype=np.float64)
    if table.num_rows:
        layer = {h: i for i, h in enumerate(hours)}
        t = np.fromiter(
            (layer[h] for h in table["hour"].to_numpy()),
            dtype=np.intp,
            count=table.num_rows,
        )
        values[t, table["y"].to_numpy(), table["x"].to_numpy()] = table[
            "foot"
        ].to_numpy()

    receptor_time = pd.Timestamp(receptor.time)
    times = pd.DatetimeIndex(
        [receptor_time + pd.Timedelta(hours=int(h)) for h in hours]
    )
    x_dim, y_dim = ("lon", "lat") if grid.is_longlat else ("x", "y")
    data = xr.DataArray(
        values,
        dims=["time", y_dim, x_dim],
        coords={"time": times, y_dim: y_axis, x_dim: x_axis},
    )
    return _describe(data, receptor, config, name)


def read_footprint(
    path: str | Path,
    *,
    chunks: Any | None = None,
) -> xr.DataArray | None:
    """
    Read a footprint file, NetCDF or the Parquet files PYSTILT stores.

    Each file holds its receptor and settings, so it needs nothing else.

    Parameters
    ----------
    path : str or Path
        A ``.nc`` file written by ``foot.stilt.to_netcdf``, or a footprint
        file from an output directory.
    chunks : dict, int or "auto", optional
        For a NetCDF file, passed to :func:`xarray.open_dataset` to load the
        data lazily with dask.

    Returns
    -------
    xarray.DataArray or None
        The footprint, or ``None`` for an empty one (no particle reached the
        grid).

    Examples
    --------
    >>> foot = stilt.read_footprint(sim.footprint_path)
    >>> foot.stilt.receptor
    """
    import pyarrow.parquet as pq

    path = Path(path)
    if path.suffix == ".nc":
        if chunks is not None:
            foot = xr.open_dataset(path, chunks=chunks)["foot"]
        else:
            with xr.open_dataset(path) as ds:
                foot = ds["foot"].load()
        foot.attrs.pop("grid_mapping", None)
        return foot
    table = pq.read_table(path)
    stored = (table.schema.metadata or {}).get(b"stilt:footprint")
    if stored is None:
        raise ValueError(f"{path} does not record its footprint settings.")
    return _from_sparse_table(table, _settings_from_json(stored.decode(), path.name))


@xr.register_dataarray_accessor("stilt")
class FootprintAccessor:
    """
    PYSTILT methods on a footprint, as ``foot.stilt``.

    A footprint is an :class:`xarray.DataArray`, so sums, selections, and
    plots are plain xarray (``foot.sum("time")``,
    ``foot.sel(time=slice(a, b))``). The accessor adds what needs the
    receptor or the footprint settings, which ride in the array's
    attributes. Recent xarray keeps attributes through arithmetic. Older
    versions drop them (``foot * 2``), so with those, call these methods
    on the footprint as read, or use ``xr.set_options(keep_attrs=True)``.

    Examples
    --------
    >>> foot = sim.footprint
    >>> foot.stilt.receptor
    >>> foot.stilt.enhancement(flux).sum()
    >>> foot.stilt.aggregate(counties, time_bins=bins)
    """

    def __init__(self, foot: xr.DataArray) -> None:
        self._foot = foot

    def _attr(self, key: str) -> str:
        """Return a PYSTILT attribute, or raise when arithmetic has dropped it."""
        value = self._foot.attrs.get(key)
        if value is None:
            raise ValueError(
                f"This array has no {key!r} attribute. Older xarray versions "
                "drop attributes in arithmetic. Call this on the footprint as "
                "read, or use xr.set_options(keep_attrs=True)."
            )
        return value

    @property
    def receptor(self) -> Receptor:
        """The receptor the footprint belongs to."""
        return Receptor.from_dict(json.loads(self._attr("stilt_receptor")))

    @property
    def config(self) -> FootprintConfig:
        """The footprint settings: grid, smoothing, and particle transforms."""
        return _settings_from_json(self._attr("stilt_footprint"), "footprint")

    @property
    def grid(self) -> Grid:
        """The grid the footprint is on."""
        grid = self.config.grid
        if grid is None:  # calculate() never makes a footprint without one
            raise ValueError("The footprint settings have no grid.")
        return grid

    @property
    def name(self) -> str:
        """The footprint's name, usually the variant name."""
        return str(self._foot.attrs.get("stilt_name", ""))

    @property
    def plot(self) -> FootprintPlotAccessor:
        """Plotting methods, such as ``foot.stilt.plot.map()``."""
        from stilt.visualization import FootprintPlotAccessor

        return FootprintPlotAccessor(self._foot)

    def to_netcdf(self, path: str | Path) -> Path:
        """
        Write the footprint to a CF-1.8 NetCDF file.

        :func:`stilt.read_footprint` reads it back with its receptor and
        settings.

        Parameters
        ----------
        path : str or Path
            File to write.

        Returns
        -------
        Path
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset({"foot": self._foot})
        # NetCDF stores naive times; footprint times are UTC.
        ds = ds.assign_coords(time=_utc_index(ds["time"].values).tz_convert(None))
        ds = _with_cf_metadata(ds, grid=self.grid)
        ds.attrs["time_created"] = (
            dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat()
        )
        with atomic_path(path) as tmp:
            ds.to_netcdf(tmp, encoding={"foot": {"zlib": True, "complevel": 4}})
        return path

    def enhancement(self, flux: xr.DataArray) -> xr.DataArray:
        """
        Return the modelled enhancement at the receptor, footprint times flux summed over the grid.

        The flux is taken at each footprint cell centre from the nearest
        flux cell (:func:`stilt.flux.sample_flux`). Regrid a flux with much
        smaller cells than the footprint's before calling this.

        Parameters
        ----------
        flux : xarray.DataArray
            Surface flux on a ``lat``/``lon`` grid (``y``/``x`` for a
            projected footprint), in µmol m⁻² s⁻¹ for an enhancement in
            ppm. A flux with a ``time`` dimension is taken at each footprint
            time step.

        Returns
        -------
        xarray.DataArray
            Enhancement for each footprint time step. ``.sum()`` gives the
            total.
        """
        from stilt.flux import sample_flux

        foot = self._foot
        y_dim, x_dim = foot.dims[-2], foot.dims[-1]
        yy, xx = np.meshgrid(
            foot[y_dim].to_numpy(), foot[x_dim].to_numpy(), indexing="ij"
        )
        shape = yy.shape
        if "time" in flux.dims:
            layers = [
                sample_flux(flux, xx.ravel(), yy.ravel(), np.full(xx.size, t)).reshape(
                    shape
                )
                for t in foot["time"].to_numpy()
            ]
            sampled = np.stack(layers)
        else:
            sampled = sample_flux(flux, xx.ravel(), yy.ravel()).reshape(shape)[None]
        values = (foot.to_numpy() * sampled).sum(axis=(1, 2))
        return xr.DataArray(
            values,
            dims=["time"],
            coords={"time": foot["time"]},
            name="enhancement",
        )

    def aggregate(
        self,
        target: SpatialTarget,
        time_bins: pd.IntervalIndex,
    ) -> pd.DataFrame:
        """
        Sum the footprint onto the cells of another grid or set of polygons, per time bin.

        A footprint value belongs to its whole cell, so it is summed when
        cells are combined. Each footprint cell is split among the target
        cells it overlaps, in proportion to the overlapping area. For a
        target whose cells are whole blocks of footprint cells, this is a
        plain block sum. Footprint values outside the target are dropped.
        Time steps are summed within each of ``time_bins``.

        Parameters
        ----------
        target : SpatialTarget
            Cells to sum onto:

            - :class:`~stilt.config.Grid`, every cell of a regular grid, in
              ``Grid.index`` order.
            - :class:`~stilt.Mesh`, any polygons (a shapefile, H3 hexagons,
              nested grids). Rows are indexed by cell id.
            - :class:`~stilt.Zones`, groups of the cells of a grid or mesh.
              Use it to keep only some cells of a grid.

            A target in another coordinate system is projected onto the
            footprint grid.
        time_bins : pandas.IntervalIndex
            Time intervals to sum over, such as the time steps of a flux
            inventory. They must be closed on the left
            (``closed="left"``): each bin holds the footprint hours that
            start in it.

        Returns
        -------
        pandas.DataFrame
            One row per target cell and one column per time bin, labelled
            by the bin's left edge. Rows are indexed by ``(x, y)`` for grid
            targets and by cell id for meshes and zones. Cells and bins the
            footprint does not reach are 0.

        Raises
        ------
        ValueError
            If ``time_bins`` is not closed on the left.
        """
        if time_bins.closed != "left":
            raise ValueError(
                f"time_bins must be closed on the left, not {time_bins.closed!r}. "
                "A footprint time is the start of its hour, so each bin takes "
                "the hours that start in it. Build the bins with "
                "closed='left', for example "
                "pd.interval_range(start, end, freq='1h', closed='left')."
            )
        if not isinstance(target, (Grid, Mesh, Zones)):
            raise TypeError(
                "aggregate target must be a stilt.Grid, stilt.Mesh, or stilt.Zones, "
                f"not {type(target).__name__}."
            )
        foot = self._foot
        is_latlon = "lon" in foot.dims and "lat" in foot.dims
        x_dim = "lon" if is_latlon else "x"
        y_dim = "lat" if is_latlon else "y"
        config = self.config
        self._check_geometry_hash(target, config)
        return self._aggregate_geometry(target, time_bins, x_dim, y_dim, config)

    def _check_geometry_hash(self, target: object, config: FootprintConfig) -> None:
        """Warn when the target mesh differs from the one the footprint grid was chosen for."""
        expected = config.geometry_hash
        if not expected:
            return
        mesh = target.base if isinstance(target, Zones) else target
        if isinstance(mesh, Mesh) and mesh.hash != expected:
            warnings.warn(
                f"Footprint {self.name!r} was derived for geometry {expected} but is "
                f"being aggregated onto geometry {mesh.hash}; the geometry source may "
                "have changed since the footprint was computed, so the raster "
                "resolution and extent may no longer suit it.",
                stacklevel=3,
            )

    def _aggregate_geometry(
        self,
        target: Grid | Mesh | Zones,
        time_bins: pd.IntervalIndex,
        x_dim: str,
        y_dim: str,
        config: FootprintConfig,
    ) -> pd.DataFrame:
        """
        Sum onto a geometry with its cached overlap weights.

        The weights ``W`` (``n_cells × n_native``) hold the fraction of each
        footprint cell inside each target cell, so each time bin is
        ``W @ F.ravel()``.
        """
        foot = self._foot
        columns = _utc_index(time_bins.left).tz_localize(None)
        result = pd.DataFrame(0.0, index=target.index, columns=columns)

        ntime = int(foot.sizes.get("time", 0))
        if foot.size == 0 or ntime == 0:
            return result

        grid = config.grid
        if grid is None:
            raise ValueError("The footprint settings have no grid.")
        px = np.asarray(foot[x_dim].values, dtype=float)
        py = np.asarray(foot[y_dim].values, dtype=float)
        xres, yres = grid.xres, grid.yres
        crs = grid.projection
        check_resolution(target, xres, yres, crs)
        weights = overlap_weights(target, px, py, xres, yres, crs)

        data_arr = foot.transpose("time", y_dim, x_dim).to_numpy()
        native_times = _utc_index(foot["time"].values).tz_localize(None)
        for interval, left_edge in zip(time_bins, columns, strict=False):
            left = _naive_utc_timestamp(interval.left)
            right = _naive_utc_timestamp(interval.right)
            if left is None or right is None:
                raise ValueError(
                    f"Could not convert interval bounds to UTC timestamps: {interval}"
                )
            in_bin = np.asarray((native_times >= left) & (native_times < right))
            if not in_bin.any():
                continue
            f_bin = data_arr[in_bin].sum(axis=0).ravel()  # (Ny * Nx,)
            result[left_edge] = weights @ f_bin
        return result
