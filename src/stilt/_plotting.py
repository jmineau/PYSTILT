"""
Map helpers for footprint and simulation plots: the axes, the extent, and the boxes drawn on them.

Copies of the helpers :mod:`runtraj.visualization` keeps private, so that PYSTILT
imports only runtraj's public names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from runtraj.releases import Release

if TYPE_CHECKING:
    from cartopy.io.img_tiles import GoogleTiles
    from runtraj import Bounds


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

    This is 0 unless the extent runs past 180 or -180, as a grid
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


def _release_points(release: Release) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the longitudes, latitudes, and heights of a release's release points."""
    lons, lats, alts = (
        np.array(v, dtype=float) for v in zip(*release.coords(), strict=True)
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
