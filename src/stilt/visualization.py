"""Plotting for trajectories, footprints, receptors, simulations, and models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import cartopy  # type: ignore[import-untyped]

    from stilt.config import Bounds
    from stilt.footprint import Footprint
    from stilt.model import Model
    from stilt.simulation import Simulation
    from stilt.trajectory import Trajectories

from stilt.receptors import ColumnReceptor, MultiPointReceptor, Receptor

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


# ---------------------------------------------------------------------------
# Trajectories plot accessor
# ---------------------------------------------------------------------------


class TrajectoriesPlotAccessor:
    """Plotting methods of :class:`stilt.Trajectories`, as ``traj.plot``."""

    def __init__(self, traj: Trajectories) -> None:
        self._traj = traj

    def map(
        self,
        color_by: str = "time",
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
        p = self._traj.data
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

        pad = 0.5
        extent = (
            lons.min() - pad,
            lons.max() + pad,
            lats.min() - pad,
            lats.max() + pad,
        )
        fig, ax = _make_ax(ax, extent=extent, tiler=tiler, tiler_zoom=tiler_zoom)

        sc = ax.scatter(lons, lats, c=c, cmap=cmap, s=s, alpha=alpha, **kwargs)
        fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.7, pad=0.02)

        self._traj.receptor.plot.map(ax=ax)
        ax.set(xlabel="Longitude", ylabel="Latitude", title="Particle Trajectories")
        return ax


# ---------------------------------------------------------------------------
# Footprint plot accessor
# ---------------------------------------------------------------------------


class FootprintPlotAccessor:
    """Plotting methods of :class:`stilt.Footprint`, as ``foot.plot``."""

    def __init__(self, foot: Footprint) -> None:
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
            data = foot.data.sel(time=time, method="nearest")
        else:
            data = foot.data.sum("time")

        lons = data.lon.values
        lats = data.lat.values
        vals = data.values.astype(float)

        if log:
            vals = _log10_safe(vals)
            cbar_label = "log₁₀(footprint)"
        else:
            cbar_label = "footprint"

        pad = 0.05
        extent = (
            lons.min() - pad,
            lons.max() + pad,
            lats.min() - pad,
            lats.max() + pad,
        )
        fig, ax = _make_ax(ax, extent=extent, tiler=tiler, tiler_zoom=tiler_zoom)

        LON, LAT = np.meshgrid(lons, lats)
        mesh = ax.pcolormesh(LON, LAT, vals, cmap=cmap, shading="auto", **kwargs)
        fig.colorbar(mesh, ax=ax, label=cbar_label, shrink=0.7, pad=0.02)

        if show_grid:
            _draw_bounds_box(ax, foot.grid, label="Domain")
        if met_bounds is not None:
            _draw_bounds_box(
                ax,
                met_bounds,
                label="Met domain",
                edgecolor="steelblue",
                linestyle=":",
                linewidth=1.2,
            )

        foot.receptor.plot.map(ax=ax)
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
        Map each time step of the footprint in its own panel, with one colorbar.

        Parameters
        ----------
        ncols : int, default 3
            Number of panel columns.
        log : bool, default True
            Plot log10 of the footprint.
        cmap : str, default "cool"
            Colormap name.
        figsize : tuple of float, optional
            Figure size. Defaults to ``(ncols * 4, nrows * 3)``.
        **kwargs
            Forwarded to :func:`matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        fig : Figure
        axes : ndarray of Axes
        """
        foot = self._foot
        times = foot.data.time.values
        n = len(times)
        nrows = (n + ncols - 1) // ncols

        if figsize is None:
            figsize = (ncols * 4, nrows * 3)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        axes_flat: np.ndarray = np.array(axes).flatten()

        lons = foot.data.lon.values
        lats = foot.data.lat.values
        LON, LAT = np.meshgrid(lons, lats)

        all_vals = foot.data.values.astype(float)
        if log:
            all_vals_plot = _log10_safe(all_vals)
            cbar_label = "log₁₀(footprint)"
        else:
            all_vals_plot = all_vals
            cbar_label = "footprint"

        vmin = float(np.nanmin(all_vals_plot))
        vmax = float(np.nanmax(all_vals_plot))

        mesh = None
        for i, (t, panel_ax) in enumerate(zip(times, axes_flat, strict=False)):
            vals = all_vals_plot[i]
            mesh = panel_ax.pcolormesh(
                LON,
                LAT,
                vals,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                **kwargs,
            )
            panel_time_raw = pd.Timestamp(t)
            title = (
                "NaT"
                if not isinstance(panel_time_raw, pd.Timestamp)
                else panel_time_raw.strftime("%Y-%m-%d %H:%M")
            )
            panel_ax.set_title(title, fontsize=9)
            panel_ax.tick_params(labelsize=7)

        for panel_ax in axes_flat[n:]:
            panel_ax.set_visible(False)

        if mesh is not None:
            fig.colorbar(mesh, ax=axes_flat[:n], label=cbar_label, shrink=0.6)
        fig.suptitle("Footprint by Time Step")
        return fig, axes


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
        coords = [(lat, lon, alt) for lat, lon, alt in r]
        lats = np.array([c[0] for c in coords])
        lons = np.array([c[1] for c in coords])
        alts = np.array([c[2] for c in coords])

        if domain is not None:
            pad = 0.5
            extent = (
                domain.xmin - pad,
                domain.xmax + pad,
                domain.ymin - pad,
                domain.ymax + pad,
            )
        else:
            pad = max(
                1.0,
                (lons.max() - lons.min()) * 0.3 + 0.5,
                (lats.max() - lats.min()) * 0.3 + 0.5,
            )
            extent = (
                lons.min() - pad,
                lons.max() + pad,
                lats.min() - pad,
                lats.max() + pad,
            )

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
        elif isinstance(r, ColumnReceptor):
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
            ax.annotate(
                f"{r.bottom:.0f}–{r.top:.0f} m AGL",
                xy=(lons[0], lats[0]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
            )
        else:
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

        if domain is not None:
            _draw_bounds_box(ax, domain, label="Domain")
        if met_bounds is not None:
            _draw_bounds_box(
                ax,
                met_bounds,
                label="Met domain",
                edgecolor="steelblue",
                linestyle=":",
                linewidth=1.2,
            )

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
        show_traj: bool = True,
        show_receptor: bool = True,
        log: bool = True,
        foot_cmap: str = "YlOrRd",
        traj_cmap: str = "viridis_r",
        traj_color_by: str = "time",
        traj_s: float = 1.0,
        traj_alpha: float = 0.3,
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
        show_traj : bool, default True
            Draw the particle positions.
        show_receptor : bool, default True
            Mark the receptor.
        log : bool, default True
            Plot log10 of the footprint.
        foot_cmap : str, default "YlOrRd"
            Colormap of the footprint.
        traj_cmap : str, default "viridis_r"
            Colormap of the particles.
        traj_color_by : {"time", "zagl", "foot"}, default "time"
            Particle variable to color by.
        traj_s : float, default 1.0
            Particle marker size.
        traj_alpha : float, default 0.3
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
        traj = sim.trajectories if sim.has_trajectory else None

        # Determine map extent: prefer footprint grid, fall back to traj bounds
        if foot is not None:
            g = foot.grid
            pad = 0.1
            extent: tuple[float, float, float, float] = (
                g.xmin - pad,
                g.xmax + pad,
                g.ymin - pad,
                g.ymax + pad,
            )
        elif traj is not None:
            p = traj.data
            pad = 0.5
            extent = (
                p["long"].min() - pad,
                p["long"].max() + pad,
                p["lati"].min() - pad,
                p["lati"].max() + pad,
            )
        else:
            r = sim.receptor
            pad = 2.0
            _lats = np.array([lat for lat, lon, alt in r])
            _lons = np.array([lon for lat, lon, alt in r])
            extent = (
                _lons.min() - pad,
                _lons.max() + pad,
                _lats.min() - pad,
                _lats.max() + pad,
            )

        _, ax = _make_ax(ax, extent=extent)

        if foot is not None:
            foot.plot.map(ax=ax, log=log, cmap=foot_cmap, show_grid=show_grid)

        if show_traj and traj is not None:
            traj.plot.map(
                ax=ax,
                color_by=traj_color_by,
                cmap=traj_cmap,
                s=traj_s,
                alpha=traj_alpha,
            )

        if show_receptor:
            sim.receptor.plot.map(ax=ax, color="red")

        if met_bounds is not None:
            _draw_bounds_box(
                ax,
                met_bounds,
                label="Met domain",
                edgecolor="steelblue",
                linestyle=":",
                linewidth=1.2,
            )
            ax.legend(fontsize=8)

        ax.set_title(f"Simulation {sim.id}")
        return ax


# ---------------------------------------------------------------------------
# Model plot accessor
# ---------------------------------------------------------------------------


class ModelPlotAccessor:
    """Plotting methods of :class:`stilt.Model`, as ``model.plot``."""

    def __init__(self, model: Model):
        self._model = model

    def availability(self, ax: Axes | None = None, **kwargs) -> Axes:
        """
        Plot the model's receptors by location and time.

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

        receptors = list(self._model.receptors)
        if not receptors:
            return ax

        for receptor in receptors:
            ax.barh(  # type: ignore[arg-type]
                y=receptor.location_id,
                width=pd.Timedelta(hours=1),  # type: ignore[arg-type]
                left=receptor.time,  # type: ignore[arg-type]
                height=0.6,
                align="center",
                edgecolor="black",
                alpha=0.6,
                **kwargs,
            )

        ax.figure.autofmt_xdate()
        ax.set(title="Simulation Availability", xlabel="Time", ylabel="Location ID")
        return ax
