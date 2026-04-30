"""Visualization helpers and accessors for STILT objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from stilt.simulation import SimID

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
    """Return (fig, ax), using a cartopy GeoAxes when available."""
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
    """log10 of vals, with zeros/negatives mapped to NaN."""
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
    """Draw a bounding-box rectangle on *ax*."""
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
    """Plot namespace for :class:`stilt.trajectory.Trajectories`."""

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
        Scatter all particle positions colored by a trajectory variable.

        Parameters
        ----------
        color_by : {'time', 'zagl', 'foot'}
            Variable to use for the color mapping.
        ax : Axes, optional
            Existing axes to plot into.  If *None* and cartopy is installed,
            a ``PlateCarree`` GeoAxes is created automatically.
        cmap : str
            Colormap name.
        s : float
            Marker size (passed to ``scatter``).
        alpha : float
            Marker transparency.
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
    """Plot namespace for :class:`stilt.footprint.Footprint`."""

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
        2-D map of the footprint, optionally log-scaled.

        Parameters
        ----------
        time : scalar, optional
            Select a single time step (passed to ``xr.DataArray.sel`` with
            ``method='nearest'``).  If *None*, all time steps are summed.
        log : bool
            Apply a log₁₀ transform to values before plotting.
        ax : Axes, optional
            Existing axes to plot into.
        cmap : str
            Colormap name.
        show_grid : bool
            Overlay the footprint domain bounding box as a dashed rectangle.
        met_bounds : Bounds, optional
            Draw the meteorology domain as a dotted blue bounding box.
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
        One subplot per time step, with a shared colorbar.

        Parameters
        ----------
        ncols : int
            Number of columns in the subplot grid.
        log : bool
            Apply a log₁₀ transform to values before plotting.
        cmap : str
            Colormap name.
        figsize : (float, float), optional
            Figure size.  Defaults to ``(ncols * 4, nrows * 3)``.
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
    """Plot namespace for :class:`stilt.receptor.Receptor`."""

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
        Plot receptor location(s) on a map.

        Parameters
        ----------
        ax : Axes, optional
            Existing axes to plot into.  If *None*, a new figure is created
            (with a cartopy GeoAxes if cartopy is installed).
        domain : Bounds, optional
            Draw a footprint/model domain as a dashed black bounding box and
            size the map extent to encompass it.  Accepts any :class:`Bounds`
            subclass (including :class:`Grid`).
        met_bounds : Bounds, optional
            Draw the meteorology domain as a dotted blue bounding box.
        color : str
            Marker color for point/column receptors.
        **kwargs
            Forwarded to the scatter call.

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
    """Plot namespace for :class:`stilt.simulation.Simulation`."""

    def __init__(self, sim: Simulation) -> None:
        self._sim = sim

    def map(
        self,
        foot_name: str = "",
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
        Composite map stacking footprint → trajectories → receptor.

        Layers are rendered in order from bottom to top.  Any layer whose
        data is unavailable (e.g. footprint not yet computed) is silently
        skipped.

        Parameters
        ----------
        foot_name : str
            Name of the footprint to display (default ``""``).
        show_traj : bool
            Overlay particle trajectory scatter if trajectories exist.
        show_receptor : bool
            Mark the receptor location on top.
        log : bool
            Apply a log₁₀ transform to footprint values.
        foot_cmap : str
            Colormap for the footprint layer.
        traj_cmap : str
            Colormap for the trajectory layer.
        traj_color_by : {'time', 'zagl', 'foot'}
            Variable used to color trajectory particles.
        traj_s : float
            Particle marker size.
        traj_alpha : float
            Particle marker transparency.
        show_grid : bool
            Overlay the footprint domain bounding box (dashed black rectangle).
            Only applied when a footprint is available.
        met_bounds : Bounds, optional
            Draw the meteorology domain as a dotted blue bounding box.
        ax : Axes, optional
            Existing axes to plot into.

        Returns
        -------
        Axes
        """
        sim = self._sim
        foot = sim.get_footprint(foot_name)
        traj = sim.trajectories

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
    """Plot namespace for :class:`stilt.model.Model`."""

    def __init__(self, model: Model):
        self._model = model

    def availability(self, ax: Axes | None = None, **kwargs) -> Axes:
        """
        Plot simulation availability by location and time.

        Parameters
        ----------
        ax : Axes, optional
            Existing matplotlib axes to draw on. If ``None``, a new figure
            and axes are created.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`~matplotlib.axes.Axes.barh`.

        Returns
        -------
        Axes
            The axes containing the availability plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        assert ax is not None

        sim_ids = self._model.index.sim_ids()
        if not sim_ids:
            return ax

        for sim_id in sim_ids:
            sid = SimID(sim_id)
            ax.barh(  # type: ignore[arg-type]
                y=sid.location,
                width=pd.Timedelta(hours=1),  # type: ignore[arg-type]
                left=sid.time,  # type: ignore[arg-type]
                height=0.6,
                align="center",
                edgecolor="black",
                alpha=0.6,
                **kwargs,
            )

        fig.autofmt_xdate()
        ax.set(title="Simulation Availability", xlabel="Time", ylabel="Location ID")
        return ax
