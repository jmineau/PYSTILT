"""
Plotting for particles, receptors, and projects.

The plots of a footprint and of a simulation, which draw the footprint, are
STILT's: :mod:`stilt.stilt_visualization`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from cartopy.io.img_tiles import GoogleTiles

    from stilt.project import Project
    from stilt.receptors import Receptor
    from stilt.spatial import Bounds

from stilt.receptors import ColumnReceptor, MultiPointReceptor, Receptor

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_ax(
    ax: Axes | None = None,
    extent: tuple[float, float, float, float] | None = None,
    tiler: GoogleTiles | None = None,
    tiler_zoom: int = 8,
) -> tuple[Figure, Axes]:
    """Return ``(fig, ax)``, making a cartopy map when cartopy is installed."""
    if ax is not None:
        # pyrefly: ignore[bad-return]
        return ax.get_figure(), ax
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        # Constrained layout makes room for each colorbar and its label.
        fig = plt.figure(layout="constrained")
        center = _central_longitude(extent)
        projection = ccrs.PlateCarree(central_longitude=center)
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        # pyrefly: ignore[missing-attribute]
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="0.4")
        # pyrefly: ignore[missing-attribute]
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        if extent is not None:
            # The extent is given in the axes' own coordinates, which count
            # degrees east of the center, so a map across 180 stays narrow.
            west, east, south, north = extent
            # pyrefly: ignore[missing-attribute]
            ax.set_extent((west - center, east - center, south, north), crs=projection)

            if tiler is not None:
                # pyrefly: ignore[bad-argument-type, bad-argument-count]
                ax.add_image(tiler, tiler_zoom)
    except ImportError:
        fig, ax = plt.subplots()
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
    return fig, ax


def _central_longitude(extent: tuple[float, float, float, float] | None) -> float:
    """
    Return the longitude a new map is centered on.

    This is 0 unless the extent runs past 180 or -180, as a footprint grid
    from 170 to 190 does. Then it is the extent's middle, so the map shows
    that box and not the whole globe.
    """
    if extent is None:
        return 0.0
    west, east = extent[0], extent[1]
    if west < -180 or east > 180:
        return (west + east) / 2
    return 0.0


def _data_crs(ax: Axes) -> dict:
    """
    Return the ``transform`` keyword that draws longitude and latitude on *ax*.

    A cartopy map is centered on the longitude it was made with, so data in
    degrees must say so. Plain matplotlib axes need no keyword.
    """
    if not hasattr(ax, "projection"):
        return {}
    import cartopy.crs as ccrs

    return {"transform": ccrs.PlateCarree()}


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
        **_data_crs(ax),
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


def _legend(ax: Axes) -> None:
    """Add a legend of the labeled artists in the upper right corner (the best spot is slow to search for)."""
    ax.legend(fontsize=8, loc="upper right")


# ---------------------------------------------------------------------------
# Particles plot accessor
# ---------------------------------------------------------------------------


class ParticlesPlotAccessor:
    """Plotting methods of a particle table, as ``particles.stilt.plot``."""

    def __init__(self, particles: pd.DataFrame) -> None:
        self._particles = particles

    def map(
        self,
        color_by: str = "age",
        receptor: Receptor | None = None,
        ax: Axes | None = None,
        cmap: str = "viridis_r",
        s: float = 1.0,
        alpha: float = 0.3,
        tiler: GoogleTiles | None = None,
        tiler_zoom: int = 8,
        **kwargs,
    ) -> Axes:
        """
        Map every particle position, colored by a particle variable.

        Parameters
        ----------
        color_by : {"age", "zagl", "foot"}, default "age"
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
        lons: np.ndarray = p["lon"].to_numpy(dtype=float)
        lats: np.ndarray = p["lat"].to_numpy(dtype=float)

        _color_labels = {
            "age": "Minutes since release",
            "zagl": "Altitude AGL (m)",
            "foot": "Footprint influence",
        }
        if color_by not in _color_labels:
            raise ValueError(
                f"color_by must be one of {list(_color_labels)!r}, got {color_by!r}"
            )
        if color_by == "foot" and "foot" not in p.columns:
            raise ValueError(
                "The particles have no 'foot' column to color by: the transport "
                "model wrote none. stilt.footprint.footprint_particles adds "
                "the one the footprint uses."
            )
        c: np.ndarray = p[color_by].to_numpy(dtype=float)
        cbar_label = _color_labels[color_by]

        extent = _extent(lons, lats, pad=0.5)
        fig, ax = _make_ax(ax, extent=extent, tiler=tiler, tiler_zoom=tiler_zoom)

        sc = ax.scatter(
            lons,
            lats,
            c=c,
            cmap=cmap,
            s=s,
            alpha=alpha,
            **{**_data_crs(ax), **kwargs},
        )
        fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.7, pad=0.02)

        if receptor is not None:
            receptor.plot.map(ax=ax)
        ax.set(xlabel="Longitude", ylabel="Latitude", title="Particles")
        return ax


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
        tiler: GoogleTiles | None = None,
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
                **{**_data_crs(ax), **kwargs},
            )
            fig.colorbar(
                sc,
                ax=ax,
                label=f"Height {r.altitude_ref.upper()} (m)",
                shrink=0.7,
                pad=0.02,
            )
        else:  # a point, or a column at one location
            ax.scatter(
                [lons[0]],
                [lats[0]],
                marker="*",
                s=200,
                color=color,
                zorder=5,
                label="Receptor",
                **{**_data_crs(ax), **kwargs},
            )
            if isinstance(r, ColumnReceptor):
                ax.annotate(
                    f"{r.bottom:.0f}–{r.top:.0f} m {r.altitude_ref.upper()}",
                    xy=(lons[0], lats[0]),
                    xycoords=_data_crs(ax).get("transform", "data"),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

        if domain is not None:
            _draw_bounds_box(ax, domain, label="Domain")
        if met_bounds is not None:
            _draw_met_box(ax, met_bounds)

        _legend(ax)
        if standalone:
            ax.set(xlabel="Longitude", ylabel="Latitude", title="Receptor")
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
        Plot how many receptors each location has on each day.

        Each location is a row, and each day with receptors there a one-day
        bar shaded by their number, so the plot stays quick for projects of
        tens of thousands of receptors. Whether their simulations have run
        is not shown; ``project.status()`` says that.

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. By default a new figure is made.
        **kwargs
            Passed to :meth:`matplotlib.axes.Axes.broken_barh`, except
            ``cmap``, the colormap of the counts (``viridis``).

        Returns
        -------
        Axes
        """
        import matplotlib.dates as mdates
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        if ax is None:
            # Constrained layout fits location labels of any length.
            _, ax = plt.subplots(layout="constrained")
        assert ax is not None

        receptors = self._project.receptors
        if receptors.empty:
            return ax

        cmap = plt.get_cmap(kwargs.pop("cmap", "viridis"))
        days = pd.to_datetime(receptors["time"]).dt.floor("D")
        counts = receptors.assign(day=days).groupby(["location", "day"]).size()
        norm = Normalize(vmin=1, vmax=max(int(counts.to_numpy().max()), 2))
        locations = list(dict.fromkeys(receptors["location"]))
        for row, location in enumerate(locations):
            per_day = counts.loc[location]
            starts = mdates.date2num(per_day.index.to_pydatetime())
            ax.broken_barh(
                [(float(start), 1.0) for start in starts],
                (row - 0.3, 0.6),
                facecolors=[cmap(norm(int(n))) for n in per_day.to_numpy()],
                **kwargs,
            )
        ax.set_yticks(range(len(locations)), labels=locations)
        ax.xaxis_date()
        ax.figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Receptors per day"
        )
        if ax.figure.get_layout_engine() is None:
            ax.figure.autofmt_xdate()  # rotates the dates and makes room for them
        else:  # the layout makes the room; autofmt_xdate would warn on older matplotlib
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment("right")
        ax.set(title="Receptors", xlabel="Day", ylabel="Location")
        return ax
