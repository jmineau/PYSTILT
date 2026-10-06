"""Plotting for particles, footprints, receptors, simulations, and projects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import cartopy  # type: ignore[import-untyped]
    import xarray as xr

    from stilt.project import Project
    from stilt.receptors import Receptor
    from stilt.simulation import Simulation
    from stilt.spatial import Bounds

from stilt.receptors import ColumnReceptor, MultiPointReceptor, Receptor
from stilt.spatial import horizontal_dims

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_ax(
    ax: Axes | None = None,
    extent: tuple[float, float, float, float] | None = None,
    tiler: cartopy.io.img_tiles.GoogleTiles | None = None,
    tiler_zoom: int = 8,
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, making a cartopy map when cartopy is installed."""
    if ax is not None:
        return ax.get_figure(), ax  # type: ignore[return-value]
    try:
        import cartopy.crs as ccrs  # type: ignore[import-untyped]
        import cartopy.feature as cfeature  # type: ignore[import-untyped]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="0.4")  # type: ignore[attr-defined]
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)  # type: ignore[attr-defined]
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore[attr-defined]

            if tiler is not None:
                ax.add_image(tiler, tiler_zoom)  # type: ignore[attr-defined]
    except ImportError:
        fig, ax = plt.subplots()
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
    return fig, ax


def _cell_lonlat(foot: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the longitude and latitude of each footprint cell centre, as 2-D arrays.

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


def _extent(lons, lats, pad: float) -> tuple[float, float, float, float]:
    """Return the ``(west, east, south, north)`` box around *lons* and *lats*, widened by *pad* degrees."""
    lons, lats = np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    return (
        float(lons.min()) - pad,
        float(lons.max()) + pad,
        float(lats.min()) - pad,
        float(lats.max()) + pad,
    )


def _bounds_extent(bounds: Bounds, pad: float) -> tuple[float, float, float, float]:
    """Return the ``(west, east, south, north)`` box of *bounds*, widened by *pad* degrees."""
    return _extent([bounds.xmin, bounds.xmax], [bounds.ymin, bounds.ymax], pad)


def _receptor_points(receptor: Receptor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the longitudes, latitudes, and heights of a receptor's release points."""
    lats, lons, alts = (
        np.array(v, dtype=float) for v in zip(*receptor.coords(), strict=True)
    )
    return lons, lats, alts


def _log10_safe(vals: np.ndarray) -> np.ndarray:
    """Return log10 of ``vals``, with NaN where a value is zero or negative."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vals > 0, np.log10(vals), np.nan)


def _draw_bounds_box(
    ax: Axes,
    bounds: Bounds,
    label: str = "",
    edgecolor: str = "black",
    linestyle: str = "--",
    linewidth: float = 1.5,
) -> None:
    """Draw the outline of ``bounds`` on ``ax``."""
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (bounds.xmin, bounds.ymin),
        bounds.xmax - bounds.xmin,
        bounds.ymax - bounds.ymin,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor="none",
        linestyle=linestyle,
        label=label or None,
    )
    ax.add_patch(rect)


def _draw_met_box(ax: Axes, bounds: Bounds) -> None:
    """Draw the outline of a met's crop box on ``ax``."""
    _draw_bounds_box(
        ax,
        bounds,
        label="Met domain",
        edgecolor="steelblue",
        linestyle=":",
        linewidth=1.2,
    )


# ---------------------------------------------------------------------------
# Particles plot accessor
# ---------------------------------------------------------------------------


class ParticlesPlotAccessor:
    """Plotting methods of a particle table, as ``particles.stilt.plot``."""

    def __init__(self, particles: pd.DataFrame) -> None:
        self._particles = particles

    def map(
        self,
        color_by: str = "time",
        receptor: Receptor | None = None,
        ax: Axes | None = None,
        cmap: str = "viridis_r",
        s: float = 1.0,
        alpha: float = 0.3,
        tiler: cartopy.io.img_tiles.GoogleTiles | None = None,
        tiler_zoom: int = 8,
        **kwargs,
    ) -> Axes:
        """
        Map every particle position, colored by a particle variable.

        Parameters
        ----------
        color_by : {"time", "zagl", "foot"}, default "time"
            Particle variable to color by.
        receptor : Receptor, optional
            Mark this receptor on the map.
        ax : Axes, optional
            Axes to plot on. By default a new map is made, with cartopy when
            it is installed.
        cmap : str, default "viridis_r"
            Colormap name.
        s : float, default 1.0
            Marker size.
        alpha : float, default 0.3
            Marker opacity.
        tiler : cartopy.io.img_tiles.GoogleTiles, optional
            Map tiles drawn as a background on a new cartopy map.
        tiler_zoom : int, default 8
            Zoom level of the map tiles.
        **kwargs
            Forwarded to :func:`matplotlib.axes.Axes.scatter`.

        Returns
        -------
        Axes
        """
        p = self._particles
        lons: np.ndarray = p["long"].to_numpy(dtype=float)
        lats: np.ndarray = p["lati"].to_numpy(dtype=float)

        _color_labels = {
            "time": "Time (min)",
            "zagl": "Altitude AGL (m)",
            "foot": "Footprint influence",
        }
        if color_by not in _color_labels:
            raise ValueError(
                f"color_by must be one of {list(_color_labels)!r}, got {color_by!r}"
            )
        c: np.ndarray = p[color_by].to_numpy(dtype=float)
        cbar_label = _color_labels[color_by]

        extent = _extent(lons, lats, pad=0.5)
        fig, ax = _make_ax(ax, extent=extent, tiler=tiler, tiler_zoom=tiler_zoom)

        sc = ax.scatter(lons, lats, c=c, cmap=cmap, s=s, alpha=alpha, **kwargs)
        fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.7, pad=0.02)

        if receptor is not None:
            receptor.plot.map(ax=ax)
        ax.set(xlabel="Longitude", ylabel="Latitude", title="Particles")
        return ax


# ---------------------------------------------------------------------------
# Footprint plot accessor
# ---------------------------------------------------------------------------


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
        met_bounds: Bounds | None = None,
        tiler: cartopy.io.img_tiles.GoogleTiles | None = None,
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

        mesh = ax.pcolormesh(LON, LAT, vals, cmap=cmap, shading="auto", **kwargs)
        fig.colorbar(mesh, ax=ax, label=cbar_label, shrink=0.7, pad=0.02)

        if show_grid:
            _draw_bounds_box(ax, foot.stilt.grid, label="Domain")
        if met_bounds is not None:
            _draw_met_box(ax, met_bounds)

        if "stilt_receptor" in foot.attrs:
            foot.stilt.receptor.plot.map(ax=ax)
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

        This is xarray's faceted plot, in the footprint's own coordinates.

        Parameters
        ----------
        ncols : int, default 3
            Number of panel columns.
        log : bool, default True
            Plot log10 of the footprint. Cells of zero are left blank.
        cmap : str, default "cool"
            Colormap name.
        figsize : tuple of float, optional
            Figure size. Defaults to ``(ncols * 4, nrows * 3)``.
        **kwargs
            Forwarded to :meth:`xarray.DataArray.plot`.

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
        nrows = -(-foot.sizes["time"] // ncols)
        facets = data.plot(
            x=x_dim,
            y=y_dim,
            col="time",
            col_wrap=ncols,
            cmap=cmap,
            figsize=figsize or (ncols * 4, nrows * 3),
            cbar_kwargs={"label": "log₁₀(footprint)" if log else "footprint"},
            **kwargs,
        )
        facets.fig.suptitle("Footprint by time step")
        return facets.fig, facets.axs


# ---------------------------------------------------------------------------
# Receptor plot accessor
# ---------------------------------------------------------------------------


class ReceptorPlotAccessor:
    """Plotting methods of :class:`stilt.Receptor`, as ``receptor.plot``."""

    def __init__(self, receptor: Receptor) -> None:
        self._receptor = receptor

    def map(
        self,
        ax: Axes | None = None,
        domain: Bounds | None = None,
        met_bounds: Bounds | None = None,
        color: str = "red",
        tiler: cartopy.io.img_tiles.GoogleTiles | None = None,
        tiler_zoom: int = 8,
        **kwargs,
    ) -> Axes:
        """
        Map the receptor's location.

        A multipoint receptor's points are colored by height.

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. By default a new map is made, with cartopy when
            it is installed.
        domain : Bounds, optional
            Outline a domain, such as a footprint :class:`~stilt.Grid`, with
            a dashed line, and fit the map to it.
        met_bounds : Bounds, optional
            Outline the meteorology domain with a dotted blue line.
        color : str, default "red"
            Marker color of a point or column receptor.
        tiler : cartopy.io.img_tiles.GoogleTiles, optional
            Map tiles drawn as a background on a new cartopy map.
        tiler_zoom : int, default 8
            Zoom level of the map tiles.
        **kwargs
            Passed to :meth:`matplotlib.axes.Axes.scatter`.

        Returns
        -------
        Axes
        """
        standalone = ax is None

        r = self._receptor
        lons, lats, alts = _receptor_points(r)

        if domain is not None:
            extent = _bounds_extent(domain, pad=0.5)
        else:
            pad = max(
                1.0,
                (lons.max() - lons.min()) * 0.3 + 0.5,
                (lats.max() - lats.min()) * 0.3 + 0.5,
            )
            extent = _extent(lons, lats, pad=pad)

        fig, ax = _make_ax(ax, extent=extent, tiler=tiler, tiler_zoom=tiler_zoom)

        if isinstance(r, MultiPointReceptor):
            sc = ax.scatter(
                lons,
                lats,
                c=alts,
                cmap="plasma",
                s=80,
                zorder=5,
                label="Receptors",
                **kwargs,
            )
            fig.colorbar(sc, ax=ax, label="Height AGL (m)", shrink=0.7, pad=0.02)
        else:  # a point, or a column at one location
            ax.scatter(
                [lons[0]],
                [lats[0]],
                marker="*",
                s=200,
                color=color,
                zorder=5,
                label="Receptor",
                **kwargs,
            )
            if isinstance(r, ColumnReceptor):
                ax.annotate(
                    f"{r.bottom:.0f}–{r.top:.0f} m AGL",
                    xy=(lons[0], lats[0]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

        if domain is not None:
            _draw_bounds_box(ax, domain, label="Domain")
        if met_bounds is not None:
            _draw_met_box(ax, met_bounds)

        ax.legend(fontsize=8)
        if standalone:
            ax.set(xlabel="Longitude", ylabel="Latitude", title="Receptor")
        return ax


# ---------------------------------------------------------------------------
# Simulation plot accessor
# ---------------------------------------------------------------------------


class SimulationPlotAccessor:
    """Plotting methods of :class:`stilt.Simulation`, as ``sim.plot``."""

    def __init__(self, sim: Simulation) -> None:
        self._sim = sim

    def map(
        self,
        show_particles: bool = True,
        show_receptor: bool = True,
        log: bool = True,
        foot_cmap: str = "YlOrRd",
        particles_cmap: str = "viridis_r",
        particles_color_by: str = "time",
        particles_s: float = 1.0,
        particles_alpha: float = 0.3,
        show_grid: bool = True,
        met_bounds: Bounds | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """
        Map the footprint, the particles, and the receptor together.

        The footprint is drawn first, then the particles, then the receptor.
        A layer the simulation has no output for is skipped.

        Parameters
        ----------
        show_particles : bool, default True
            Draw the particle positions.
        show_receptor : bool, default True
            Mark the receptor.
        log : bool, default True
            Plot log10 of the footprint.
        foot_cmap : str, default "YlOrRd"
            Colormap of the footprint.
        particles_cmap : str, default "viridis_r"
            Colormap of the particles.
        particles_color_by : {"time", "zagl", "foot"}, default "time"
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

        Returns
        -------
        Axes
        """
        sim = self._sim
        foot = sim.footprint if sim.has_footprint else None
        particles = sim.particles if sim.has_particles else None

        # Map extent: the footprint grid, else the particles, else the receptor
        if foot is not None:
            extent = _bounds_extent(foot.stilt.grid, pad=0.1)
        elif particles is not None:
            extent = _extent(particles["long"], particles["lati"], pad=0.5)
        else:
            lons, lats, _ = _receptor_points(sim.receptor)
            extent = _extent(lons, lats, pad=2.0)

        _, ax = _make_ax(ax, extent=extent)

        if foot is not None:
            foot.stilt.plot.map(ax=ax, log=log, cmap=foot_cmap, show_grid=show_grid)

        if show_particles and particles is not None:
            particles.stilt.plot.map(
                ax=ax,
                color_by=particles_color_by,
                cmap=particles_cmap,
                s=particles_s,
                alpha=particles_alpha,
            )

        if show_receptor:
            sim.receptor.plot.map(ax=ax, color="red")

        if met_bounds is not None:
            _draw_met_box(ax, met_bounds)
            ax.legend(fontsize=8)

        ax.set_title(f"Simulation {sim}")
        return ax


# ---------------------------------------------------------------------------
# Project plot accessor
# ---------------------------------------------------------------------------


class ProjectPlotAccessor:
    """Plotting methods of :class:`stilt.Project`, as ``project.plot``."""

    def __init__(self, project: Project):
        self._project = project

    def availability(self, ax: Axes | None = None, **kwargs) -> Axes:
        """
        Plot the project's receptors by location and time.

        Each receptor is a one-hour bar at its time, in the row of its
        location. Whether its simulations have run is not shown.

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. By default a new figure is made.
        **kwargs
            Passed to :meth:`matplotlib.axes.Axes.barh`.

        Returns
        -------
        Axes
        """
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None

        receptors = self._project.receptors
        if receptors.empty:
            return ax

        for location, time in zip(
            receptors["location"], receptors["time"], strict=True
        ):
            ax.barh(  # type: ignore[arg-type]
                y=location,
                width=pd.Timedelta(hours=1),  # type: ignore[arg-type]
                left=time,  # type: ignore[arg-type]
                height=0.6,
                align="center",
                edgecolor="black",
                alpha=0.6,
                **kwargs,
            )

        ax.figure.autofmt_xdate()
        ax.set(title="Simulation Availability", xlabel="Time", ylabel="Location ID")
        return ax
