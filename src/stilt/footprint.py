"""Footprint computation and serialization for STILT runs."""

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import convolve as _convolve
from typing_extensions import Self

from stilt.config import FootprintConfig, Grid
from stilt.receptor import Receptor

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


def _calc_digits(res: float) -> int:
    """Decimal digits needed to round coordinates at a given resolution."""
    if res <= 0:
        raise ValueError("Resolution must be positive")
    if res < 1:
        return int(math.ceil(math.log10(1 / res))) + 1
    return int(-math.log10(res))  # 0 for res >= 1


def _interpolation_times(time_sign: int) -> np.ndarray:
    """Exact R-STILT early-time interpolation schedule in minutes."""
    times = np.concatenate(
        [
            np.arange(0, 101, dtype=float) / 10,
            np.arange(102, 201, 2, dtype=float) / 10,
            np.arange(205, 1001, 5, dtype=float) / 10,
        ]
    )
    return times * time_sign


class Footprint:
    """STILT footprint container with grid metadata and data array."""

    def __init__(
        self,
        receptor: Receptor,
        config: FootprintConfig,
        data: xr.DataArray,
        name: str = "",
    ):
        self.receptor = receptor
        self.config = config
        self.data = data
        self.name = name
        self._plot: FootprintPlotAccessor | None = None

    @property
    def plot(self) -> "FootprintPlotAccessor":
        """Plotting namespace (e.g. ``foot.plot.map()``)."""
        if self._plot is None:
            from stilt.visualization import FootprintPlotAccessor

            self._plot = FootprintPlotAccessor(self)
        return self._plot

    @property
    def grid(self) -> Grid:
        """Convenience accessor for the footprint grid metadata from the config."""
        return self.config.grid

    @property
    def time_range(self) -> tuple[dt.datetime, dt.datetime]:
        """Get time range of footprint data."""
        times = sorted(self.data.time.values)
        start = pd.Timestamp(times[0]).to_pydatetime()
        stop = pd.Timestamp(times[-1]) + pd.Timedelta(hours=1)
        return start, stop.to_pydatetime()  # type: ignore[return-value]

    @classmethod
    def from_netcdf(cls, path: str | Path, **kwargs) -> Self:
        """Create a footprint from a netCDF file.

        Parameters
        ----------
        path : str or Path
            NetCDF footprint file path.
        **kwargs
            Passed to :func:`xarray.open_dataset`.

        Returns
        -------
        Footprint
            Reconstructed footprint with config parsed from global attributes.
        """
        path = Path(path).resolve()

        # Read dataset
        ds = xr.open_dataset(path, **kwargs)

        # Build receptor
        receptor = Receptor.from_dict(json.loads(ds.attrs.pop("receptor")))

        # Build config
        foot_config = FootprintConfig(
            grid=Grid(
                xmin=ds.attrs["xmin"],
                xmax=ds.attrs["xmax"],
                ymin=ds.attrs["ymin"],
                ymax=ds.attrs["ymax"],
                xres=ds.attrs["xres"],
                yres=ds.attrs["yres"],
                projection=ds.attrs.get("projection", "+proj=longlat"),
            ),
            smooth_factor=ds.attrs.get("smooth_factor", 1.0),
            time_integrate=ds.attrs.get("time_integrate", False),
            transforms=json.loads(ds.attrs.get("transforms", "[]")),
        )

        name = ds.attrs.pop("name", "")

        if "time_created" in ds.attrs:
            ds.attrs["time_created"] = dt.datetime.fromisoformat(
                ds.attrs["time_created"]
            )

        return cls(
            receptor=receptor,
            config=foot_config,
            data=ds.foot,
            name=name,
        )

    @classmethod
    def calculate(
        cls,
        particles: pd.DataFrame,
        receptor: Receptor,
        config: FootprintConfig,
        name: str = "",
    ) -> Self | None:
        """
        Calculate footprint from particle trajectories.

        Parameters
        ----------
        particles : pd.DataFrame
            Particle data from ``Simulation.execute()``. Must include:
            long, lati, indx, foot, time.
        config : FootprintConfig
            Grid and smoothing parameters.
        receptor : Receptor
            Receptor metadata for the returned Footprint.
        name : str, optional
            Name for the footprint.

        Returns
        -------
        Footprint or None
            Footprint object when at least one particle contributes inside the
            buffered domain, otherwise ``None``.
        """
        p = particles.copy()

        # Unpack config and derive convenience flags
        grid = config.grid
        projection = grid.projection
        xmin, xmax, xres = grid.xmin, grid.xmax, grid.xres
        ymin, ymax, yres = grid.ymin, grid.ymax, grid.yres
        is_longlat = "+proj=longlat" in projection
        smooth_factor = config.smooth_factor
        time_integrate = config.time_integrate

        n_particles = p["indx"].nunique()
        # time_sign: -1 for backward runs (negative time), +1 for forward
        time_sign = int(np.sign(p["time"].median()))

        # --- Antimeridian handling ---
        # If the domain crosses the dateline, wrap longitudes to 0-360 so
        # coordinate arithmetic stays consistent.
        if is_longlat:
            xdist = ((180 - xmin) - (-180 - xmax)) % 360
            if xdist == 0:
                xmin, xmax = -180.0, 180.0
            elif (xmax < xmin) or (xmax > 180):
                p["long"] = ((p["long"] % 360) + 360) % 360
                xmin = ((xmin % 360) + 360) % 360
                xmax = ((xmax % 360) + 360) % 360

        # --- Sub-minute interpolation for first 100 min ---
        # Near the receptor, particles move quickly relative to the grid.
        # If the median inter-particle step exceeds one grid cell, insert
        # sub-minute time points (0.0-10.0 by 0.1 min, 10.2-20.0 by 0.2,
        # 20.5-100.0 by 0.5) and linearly interpolate positions and foot.
        # Foot values are rescaled after interpolation to preserve the
        # total influence in each time window.
        early = p[np.abs(p["time"]) < 100]
        bp = early.groupby("indx")
        dx_med = (
            bp["long"]
            .apply(
                lambda s: (
                    float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else np.nan
                )
            )
            .median()
        )
        dy_med = (
            bp["lati"]
            .apply(
                lambda s: (
                    float(np.abs(np.diff(s.values)).mean()) if len(s) > 1 else np.nan
                )
            )
            .median()
        )

        if (not np.isnan(dx_med) and dx_med > xres) or (
            not np.isnan(dy_med) and dy_med > yres
        ):
            t_new = _interpolation_times(time_sign)

            # Store pre-interpolation foot sums per window for rescaling later
            atime = np.abs(p["time"])
            foot_sums = [
                p.loc[atime <= 10, "foot"].sum(),
                p.loc[(atime > 10) & (atime <= 20), "foot"].sum(),
                p.loc[(atime > 20) & (atime <= 100), "foot"].sum(),
            ]

            # Outer-join each particle track with the dense time grid, then
            # interpolate long/lati/foot at the new time points
            p["time"] = p["time"].astype(float)
            grid = pd.MultiIndex.from_product(
                [p["indx"].unique(), t_new], names=["indx", "time"]
            ).to_frame(index=False)
            p = p.merge(grid, on=["indx", "time"], how="outer").sort_values(
                ["indx", "time"], ascending=[True, False]
            )

            def _interp_cols(g):
                """Interpolate long/lati/foot columns onto a dense time grid."""
                t = g["time"].values.astype(float)
                out = {}
                for col in ["long", "lati", "foot"]:
                    y = g[col].values.astype(float)
                    valid = ~np.isnan(y)
                    out[col] = (
                        np.interp(t, t[valid], y[valid]) if valid.sum() >= 2 else y
                    )
                return pd.DataFrame(out, index=g.index)

            p[["long", "lati", "foot"]] = p.groupby("indx", group_keys=False).apply(
                _interp_cols
            )

            p = p.dropna(subset=["long", "lati", "foot"]).copy()
            p["time"] = p["time"].round(1)

            # Rescale foot so total influence per window matches original
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

        # --- Relative time per particle ---
        # rtime = time elapsed since each particle's first output step.
        # Used to compute kernel bandwidth (particles spread more with time).
        p["rtime"] = p.groupby("indx")["time"].transform(
            lambda s: s - time_sign * np.abs(s).min()
        )

        # --- Project coordinates if not longlat ---
        # Transform particle positions and domain bounds into the output CRS.
        if not is_longlat:
            try:
                from pyproj import Transformer
            except ImportError as e:
                raise ImportError(
                    "pyproj is required for non-longlat projections"
                ) from e
            tr = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
            p["long"], p["lati"] = tr.transform(p["long"].values, p["lati"].values)
            corners = tr.transform([xmin, xmax], [ymin, ymax])
            xmin, xmax = float(np.min(corners[0])), float(np.max(corners[0]))
            ymin, ymax = float(np.min(corners[1])), float(np.max(corners[1]))

        # --- Output grid cell lower-left corners ---
        n_lon = max(1, int(round((xmax - xmin) / xres)))
        n_lat = max(1, int(round((ymax - ymin) / yres)))
        glong = xmin + np.arange(n_lon) * xres
        glati = ymin + np.arange(n_lat) * yres

        # --- Kernel bandwidths per rtime ---
        # Bandwidth w scales with particle spread (di) and elapsed time (ti),
        # corrected for grid convergence at high latitudes (grid_conv).
        # Formula: w = smooth_factor * 0.06 * varsum^(1/4) * (|rtime|/1440)^(1/2) / cos(lat)
        rs = (xres, yres)
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
            # Single-particle/no-variance case: use zero-width kernel per rtime,
            # equivalent to no smoothing while still allowing downstream indexing.
            rtime_vals = np.sort(np.asarray(p["rtime"].dropna().unique()))
            kernel_df = pd.DataFrame(
                {
                    "rtime": rtime_vals,
                    "lat_mean": [float(p["lati"].to_numpy().mean())] * len(rtime_vals),
                }
            )
            w = np.zeros(len(kernel_df), dtype=float)
        else:
            kernel_df["varsum"] = kernel_df["long_var"] + kernel_df["lati_var"]
            di = kernel_df["varsum"].to_numpy() ** 0.25
            ti = (np.abs(kernel_df["rtime"].to_numpy()) / 1440) ** 0.5
            grid_conv = (
                np.cos(kernel_df["lat_mean"].to_numpy() * np.pi / 180)
                if is_longlat
                else 1.0
            )
            w = smooth_factor * 0.06 * di * ti / grid_conv

        # --- Buffer grid sized to the largest kernel ---
        # Extend the output grid by the largest kernel half-width on each side
        # so that particles near the domain edge are smoothed correctly.
        max_k = (
            _make_gauss_kernel(rs, float(np.max(w)))
            if len(w) > 0
            else np.array([[1.0]])
        )
        xbuf = max_k.shape[0]
        ybuf = max_k.shape[1]
        xbufh = (xbuf - 1) // 2
        ybufh = (ybuf - 1) // 2

        n_lon_buf = n_lon + 2 * xbuf
        n_lat_buf = n_lat + 2 * ybuf
        glong_buf = xmin - xbuf * xres + np.arange(n_lon_buf) * xres
        glati_buf = ymin - ybuf * yres + np.arange(n_lat_buf) * yres

        # --- Filter to particles with positive foot inside the buffered domain ---
        p = p[
            (p["foot"] > 0)
            & (p["long"] >= xmin - xbufh * xres)
            & (p["long"] < xmax + xbufh * xres)
            & (p["lati"] >= ymin - ybufh * yres)
            & (p["lati"] < ymax + ybufh * yres)
        ]
        if p.empty:
            return None

        # --- Assign particles to buffered grid cells and aggregate ---
        p = p.copy()
        assert isinstance(p, pd.DataFrame)
        p["loi"] = np.searchsorted(glong_buf, p["long"].to_numpy(), side="right") - 1
        p["lai"] = np.searchsorted(glati_buf, p["lati"].to_numpy(), side="right") - 1
        # Sum foot for particles in the same cell at the same time step
        p = p.groupby(["loi", "lai", "time", "rtime"], as_index=False)["foot"].sum()
        assert isinstance(p, pd.DataFrame)

        # --- Assign each time step to an output layer ---
        # time_integrate=True collapses all steps into a single layer;
        # otherwise bin into hourly layers for time-resolved output.
        interval_mins = 60
        p["layer"] = (
            0 if time_integrate else np.floor(p["time"] / interval_mins).astype(int)
        )
        layers = np.sort(np.asarray(p["layer"].unique()))
        n_layers = len(layers)

        # --- Accumulate footprint per layer ---
        foot_arr = np.zeros((n_lon_buf, n_lat_buf, n_layers))
        rtimes_all = kernel_df["rtime"].values
        kernel_cache: dict[float, np.ndarray] = {}

        for i, layer in enumerate(layers):
            layer_p = p.loc[p["layer"] == layer]
            for rtime_val in layer_p["rtime"].unique():
                step = layer_p[layer_p["rtime"] == rtime_val]

                # Look up the kernel bandwidth (w) for this rtime - nearest
                # neighbor in the precomputed kernel_df rtime grid.
                step_w_idx = int(np.argmin(np.abs(rtimes_all - rtime_val)))
                step_w = float(w[step_w_idx])
                # Build Gaussian kernel at this bandwidth; cache by exact sigma
                # value since the same rtime can appear across multiple layers.
                if step_w not in kernel_cache:
                    kernel_cache[step_w] = _make_gauss_kernel(rs, step_w)
                k = kernel_cache[step_w]

                # Scatter particle foot values onto the buffered grid using
                # bincount (faster than np.add.at for this use case).
                loi_arr = step["loi"].values.astype(int)
                lai_arr = step["lai"].values.astype(int)
                foot_vals = step["foot"].values
                valid = (
                    (loi_arr >= 0)
                    & (loi_arr < n_lon_buf)
                    & (lai_arr >= 0)
                    & (lai_arr < n_lat_buf)
                )
                lin_idx = loi_arr[valid] * n_lat_buf + lai_arr[valid]
                sparse = np.bincount(
                    lin_idx, weights=foot_vals[valid], minlength=n_lon_buf * n_lat_buf
                ).reshape(n_lon_buf, n_lat_buf)

                # Subgrid convolution: find the bounding box of nonzero cells
                # and convolve only that region, padded by the kernel half-width.
                # This is mathematically equivalent to convolving the full
                # buffered grid because all cells outside the bounding box are
                # zero and mode='constant' zero-pads beyond array edges - but
                # it's much faster when particles occupy a small fraction of the
                # domain (typical for fine-resolution grids like 0.01°).
                nz_r, nz_c = np.nonzero(sparse)
                if len(nz_r) == 0:
                    continue  # no particles in domain this timestep
                kh_x, kh_y = (
                    k.shape[0] // 2,
                    k.shape[1] // 2,
                )  # kernel half-width in cells
                r0 = max(
                    0, nz_r.min() - kh_x
                )  # left edge of bounding box, extended by kernel half-width
                r1 = min(n_lon_buf, nz_r.max() + kh_x + 1)  # right edge of bounding box
                c0 = max(0, nz_c.min() - kh_y)  # top edge of bounding box
                c1 = min(
                    n_lat_buf, nz_c.max() + kh_y + 1
                )  # bottom edge of bounding box
                foot_arr[r0:r1, c0:c1, i] += (
                    _convolve(  # convolve the sparse grid within the bounding box
                        sparse[r0:r1, c0:c1], k, mode="constant", cval=0.0
                    )
                )

        # --- Trim buffer and normalize ---
        foot_arr = foot_arr[xbuf : xbuf + n_lon, ybuf : ybuf + n_lat, :] / n_particles

        # --- Build time coordinate ---
        if time_integrate:
            time_out = [receptor.time]
        else:
            time_out = [
                receptor.time + pd.Timedelta(seconds=int(layer) * 3600)
                for layer in layers
            ]
        time_index = pd.DatetimeIndex(pd.to_datetime(time_out, utc=True)).tz_convert(
            None
        )

        # --- Build xarray DataArray ---
        x_dim = "lon" if is_longlat else "x"
        y_dim = "lat" if is_longlat else "y"
        x_coords = glong + xres / 2  # cell centers
        y_coords = glati + yres / 2

        # Transpose from (lon, lat, time) to (time, lat, lon) - CF convention.
        foot_arr = foot_arr.transpose(2, 1, 0)

        data = xr.DataArray(
            foot_arr,
            dims=["time", y_dim, x_dim],
            coords={x_dim: x_coords, y_dim: y_coords, "time": time_index},
            attrs={"units": "ppm (umol-1 m2 s)"},
        )

        return cls(
            receptor=receptor,
            config=config,
            data=data,
            name=name,
        )

    def to_netcdf(self, path: str | Path) -> Path:
        """Write footprint to a netCDF file with CF-convention attributes.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Returns
        -------
        Path
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        grid = self.config.grid

        ds = xr.Dataset({"foot": self.data})
        if "time" in ds.coords:
            ds = ds.assign_coords(
                time=pd.DatetimeIndex(
                    pd.to_datetime(ds["time"].values, utc=True)
                ).tz_convert(None)
            )
        ds.attrs.update(
            {
                "name": self.name,
                "receptor": json.dumps(self.receptor.to_dict()),
                "projection": grid.projection,
                "xmin": grid.xmin,
                "xmax": grid.xmax,
                "ymin": grid.ymin,
                "ymax": grid.ymax,
                "xres": grid.xres,
                "yres": grid.yres,
                "smooth_factor": self.config.smooth_factor,
                "time_integrate": int(self.config.time_integrate),
                "transforms": json.dumps(
                    [
                        transform.model_dump(mode="json")
                        for transform in self.config.transforms
                    ]
                ),
                "time_created": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            ds.to_netcdf(tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return path

    def integrate_over_time(
        self, start: dt.datetime | None = None, end: dt.datetime | None = None
    ) -> xr.DataArray:
        """Integrate this footprint over an optional time range.

        Parameters
        ----------
        start : datetime, optional
            Inclusive start bound.
        end : datetime, optional
            Inclusive end bound.

        Returns
        -------
        xr.DataArray
            Time-summed footprint.
        """
        return self.data.sel(time=slice(start, end)).sum("time")

    def aggregate(
        self,
        coords: list[tuple[float, float]],
        time_bins: pd.IntervalIndex,
    ) -> pd.DataFrame:
        """Sample footprint at coordinates and integrate over time bins.

        Parameters
        ----------
        coords : list[tuple[float, float]]
            (x, y) pairs to sample. Use (lon, lat) order for geographic
            footprints.
        time_bins : pd.IntervalIndex
            Flux time intervals to sum over.

        Returns
        -------
        pd.DataFrame
            Indexed by (x, y) with one column per time bin (labeled by bin
            left edge). Missing coord/bin combinations are 0.
        """
        is_latlon = "lon" in self.data.dims and "lat" in self.data.dims
        x_dim = "lon" if is_latlon else "x"
        y_dim = "lat" if is_latlon else "y"

        # Convert to pandas and round to avoid floating-point mismatch
        foot = self.data.to_series().reset_index()
        xdigits = _calc_digits(self.config.grid.xres)
        ydigits = _calc_digits(self.config.grid.yres)
        foot[x_dim] = foot[x_dim].round(xdigits)
        foot[y_dim] = foot[y_dim].round(ydigits)
        foot = foot.set_index([x_dim, y_dim, "time"])

        # Round requested coords to match footprint grid
        raw = pd.MultiIndex.from_tuples(coords, names=[x_dim, y_dim])
        coord_index = pd.MultiIndex.from_arrays(
            [
                raw.get_level_values(0).round(xdigits),
                raw.get_level_values(1).round(ydigits),
            ],
            names=[x_dim, y_dim],
        )

        # Filter to requested coordinates
        filtered = (
            foot.reset_index(level="time")
            .loc[coord_index]
            .set_index("time", append=True)
        )

        if filtered.empty:
            cols = time_bins.left.rename("time")
            return pd.DataFrame(0.0, index=coord_index, columns=cols)

        # Bin time axis and sum within each bin
        tmp = filtered.reset_index()
        time_col = tmp["time"]
        if hasattr(time_bins, "dtype") and hasattr(time_bins.dtype, "subtype"):
            target_dtype = time_bins.dtype.subtype  # type: ignore[union-attr]
            if time_col.dtype != target_dtype:
                time_col = time_col.astype(target_dtype)
        tmp["time"] = pd.cut(time_col, bins=time_bins, include_lowest=True, right=False)
        tmp["time"] = tmp["time"].cat.rename_categories(lambda x: x.left)

        return (
            tmp.groupby([x_dim, y_dim, "time"], observed=True)
            .sum()
            .unstack("time")
            .droplevel(0, axis=1)
            .reindex(columns=time_bins.left, fill_value=0.0)
        )
