"""
Plotting for footprints and simulations: STILT's plots, which draw the footprint.

``foot.stilt.plot`` (:class:`FootprintPlotAccessor`) and ``sim.plot``
(:class:`SimulationPlotAccessor`). The plots of particles, releases and
projects are in :mod:`stilt.visualization`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from stilt.spatial import horizontal_dims
from stilt.visualization import (
    _bounds_extent,
    _data_crs,
    _draw_bounds_box,
    _draw_met_box,
    _extent,
    _legend,
    _make_ax,
    _release_points,
)

if TYPE_CHECKING:
    import xarray as xr
    from cartopy.io.img_tiles import GoogleTiles

    from stilt.spatial import Bounds
    from stilt.stilt_simulation import Simulation


def _cell_lonlat(foot: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the longitude and latitude of each footprint cell center, as 2-D arrays.

    A footprint on a projected grid (``x`` and ``y``) is converted with its
    grid's projection, from ``foot.stilt.grid``.
    """
    y_dim, x_dim = horizontal_dims(foot)
    if x_dim == "lon":
        return np.meshgrid(foot["lon"].values, foot["lat"].values)
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs(foot.stilt.grid.crs, "EPSG:4326", always_xy=True)
    x, y = np.meshgrid(foot["x"].values, foot["y"].values)
    lon, lat = to_lonlat.transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def _log10_safe(vals: np.ndarray) -> np.ndarray:
    """Return log10 of ``vals``, with NaN where a value is zero or negative."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vals > 0, np.log10(vals), np.nan)


class FootprintPlotAccessor:
    """Plotting methods of a footprint, as ``foot.stilt.plot``."""

    def __init__(self, foot: xr.DataArray) -> None:
        self._foot = foot

    def map(
        self,
        time=None,
        log: bool = True,
        ax: Axes | None = None,
        cmap: str = "cool",
        show_grid: bool = False,
        show_release: bool = True,
        met_bounds: Bounds | None = None,
        tiler: GoogleTiles | None = None,
        tiler_zoom: int = 8,
        **kwargs,
    ) -> Axes:
        """
        Map the footprint.

        Parameters
        ----------
        time : datetime-like, optional
            Plot the time step nearest this time. By default all time steps
            are summed.
        log : bool, default True
            Plot log10 of the footprint. Cells of zero are left blank.
        ax : Axes, optional
            Axes to plot on. By default a new map is made.
        cmap : str, default "cool"
            Colormap name.
        show_grid : bool, default False
            Outline the footprint grid with a dashed line.
        show_release : bool, default True
            Mark the release the footprint was made for.
        met_bounds : Bounds, optional
            Outline the meteorology domain with a dotted blue line.
        tiler : cartopy.io.img_tiles.GoogleTiles, optional
            Map tiles drawn as a background on a new cartopy map.
        tiler_zoom : int, default 8
            Zoom level of the map tiles.
        **kwargs
            Forwarded to :func:`matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        Axes
        """
        foot = self._foot
        if time is not None:
            data = foot.sel(time=time, method="nearest")
        else:
            data = foot.sum("time")

        LON, LAT = _cell_lonlat(foot)
        vals = data.values.astype(float)

        if log:
            vals = _log10_safe(vals)
            cbar_label = "log₁₀(footprint)"
        else:
            cbar_label = "footprint"

        extent = _extent(LON, LAT, pad=0.05)
        fig, ax = _make_ax(ax, extent=extent, tiler=tiler, tiler_zoom=tiler_zoom)

        mesh = ax.pcolormesh(
            LON,
            LAT,
            vals,
            cmap=cmap,
            shading="auto",
            **{**_data_crs(ax), **kwargs},
        )
        fig.colorbar(mesh, ax=ax, label=cbar_label, shrink=0.7, pad=0.02)

        if show_grid:
            _draw_bounds_box(ax, foot.stilt.grid, label="Domain")
        if met_bounds is not None:
            _draw_met_box(ax, met_bounds)

        if show_release and "stilt_release" in foot.attrs:
            foot.stilt.release.plot.map(ax=ax)  # also makes the legend
        elif show_grid or met_bounds is not None:
            _legend(ax)
        title = "Footprint" if time is None else f"Footprint — {pd.Timestamp(time)}"
        ax.set(xlabel="Longitude", ylabel="Latitude", title=title)
        return ax

    def facet(
        self,
        ncols: int = 3,
        log: bool = True,
        cmap: str = "cool",
        figsize: tuple[float, float] | None = None,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        """
        Plot each time step of the footprint in its own panel, with one colorbar.

        This is xarray's faceted plot, in the footprint's own coordinates. Each
        panel is titled with its time.

        Parameters
        ----------
        ncols : int, default 3
            Number of panel columns.
        log : bool, default True
            Plot log10 of the footprint. Cells of zero are left blank.
        cmap : str, default "cool"
            Colormap name.
        figsize : tuple of float, optional
            Figure size. By default the figure is sized by its panels, each
            2.2 inches high and 1.25 times as wide as high (xarray's ``size``
            and ``aspect``, which can be passed instead).
        **kwargs
            Forwarded to :meth:`xarray.DataArray.plot`. ``cbar_kwargs`` is
            merged into the slim colorbar's settings.

        Returns
        -------
        fig : Figure
        axes : ndarray of Axes
        """
        foot = self._foot
        data = foot
        if log:
            blank_zeros = foot.where(foot > 0)
            data = blank_zeros.copy(data=np.log10(blank_zeros.values))
        y_dim, x_dim = horizontal_dims(foot)
        if figsize is None:
            kwargs.setdefault("size", 2.2)
            kwargs.setdefault("aspect", 1.25)
        else:
            kwargs["figsize"] = figsize
        cbar_kwargs = {
            "label": "log₁₀(footprint)" if log else "footprint",
            "fraction": 0.08,
            "aspect": 30,
            "shrink": 0.6,
            "pad": 0.02,
            **kwargs.pop("cbar_kwargs", {}),
        }
        facets = data.plot(
            x=x_dim,
            y=y_dim,
            col="time",
            col_wrap=ncols,
            cmap=cmap,
            cbar_kwargs=cbar_kwargs,
            **kwargs,
        )
        # xarray titles a panel "time = 2021-01-15T01:00:00", and a midnight one with no hour.
        for ax, time in zip(facets.axs.flat, foot["time"].values, strict=False):
            ax.set_title(pd.Timestamp(time).strftime("%Y-%m-%d %H:%M"), fontsize=9)
        return facets.fig, facets.axs


class SimulationPlotAccessor:
    """Plotting methods of :class:`stilt.Simulation`, as ``sim.plot``."""

    def __init__(self, sim: Simulation) -> None:
        self._sim = sim

    def map(
        self,
        show_particles: bool = True,
        show_release: bool = True,
        log: bool = True,
        foot_cmap: str = "YlOrRd",
        particles_cmap: str = "viridis_r",
        particles_color_by: str = "age",
        particles_s: float = 1.0,
        particles_alpha: float = 0.3,
        show_grid: bool = True,
        met_bounds: Bounds | None = None,
        ax: Axes | None = None,
        tiler: GoogleTiles | None = None,
        tiler_zoom: int = 8,
    ) -> Axes:
        """
        Map the footprint, the particles, and the release together.

        The footprint is drawn first, then the particles, then the release.
        A layer the simulation has no output for is skipped.

        Parameters
        ----------
        show_particles : bool, default True
            Draw the particle positions.
        show_release : bool, default True
            Mark the release.
        log : bool, default True
            Plot log10 of the footprint.
        foot_cmap : str, default "YlOrRd"
            Colormap of the footprint.
        particles_cmap : str, default "viridis_r"
            Colormap of the particles.
        particles_color_by : {"age", "zagl", "foot"}, default "age"
            Particle variable to color by.
        particles_s : float, default 1.0
            Particle marker size.
        particles_alpha : float, default 0.3
            Particle marker opacity.
        show_grid : bool, default True
            Outline the footprint grid with a dashed line.
        met_bounds : Bounds, optional
            Outline the meteorology domain with a dotted blue line.
        ax : Axes, optional
            Axes to plot on. By default a new map is made.
        tiler : cartopy.io.img_tiles.GoogleTiles, optional
            Map tiles drawn as a background on a new cartopy map.
        tiler_zoom : int, default 8
            Zoom level of the map tiles.

        Returns
        -------
        Axes
        """
        sim = self._sim
        foot = sim.footprint if sim.has_footprint else None
        particles = sim.particles if sim.has_particles else None

        # Map extent: the footprint grid, else the particles, else the release
        if foot is not None:
            extent = _bounds_extent(foot.stilt.grid, pad=0.1)
        elif particles is not None:
            extent = _extent(particles["lon"], particles["lat"], pad=0.5)
        else:
            lons, lats, _ = _release_points(sim.release)
            extent = _extent(lons, lats, pad=2.0)

        _, ax = _make_ax(ax, extent=extent, tiler=tiler, tiler_zoom=tiler_zoom)

        if foot is not None:
            # The release is marked once, below, whatever the footprint records.
            foot.stilt.plot.map(
                ax=ax,
                log=log,
                cmap=foot_cmap,
                show_grid=show_grid,
                show_release=False,
            )

        if show_particles and particles is not None:
            # pandas-stubs read the registered .stilt accessor as a column
            # pyrefly: ignore[missing-attribute]
            particles.stilt.plot.map(
                ax=ax,
                color_by=particles_color_by,
                cmap=particles_cmap,
                s=particles_s,
                alpha=particles_alpha,
            )

        if show_release:
            sim.release.plot.map(ax=ax, color="red")

        if met_bounds is not None:
            _draw_met_box(ax, met_bounds)
            _legend(ax)

        ax.set_title(f"Simulation {sim}")
        return ax


__all__ = ["FootprintPlotAccessor", "SimulationPlotAccessor"]
