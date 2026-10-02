"""Bounding boxes and footprint grids."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

VerticalReference = Literal["agl", "msl"]


def kmsl_from_vertical_reference(reference: VerticalReference) -> int:
    """Return HYSPLIT's ``KMSL`` value for a vertical reference: 0 for AGL, 1 for MSL."""
    return 0 if reference == "agl" else 1


def _grid_cell_starts(minimum: float, maximum: float, resolution: float) -> np.ndarray:
    """
    Return the lower edges of the whole cells between two grid bounds.

    Cells start at ``minimum`` and step by ``resolution`` while the whole cell
    fits inside ``[minimum, maximum]``, so ``maximum`` is the outer edge of
    the last cell.

    Decimal bounds are not exact in binary, so ``maximum - minimum`` carries a
    rounding error that grows with the bounds (``40.93 - 40.45`` is
    ``0.4799999999999969``). The tolerance is sized to that error so the last
    intended cell is kept (48 cells here at 0.01, as STILT-R's ``seq()``
    gives) and a partial cell is still dropped.
    """
    if resolution <= 0:
        raise ValueError("Grid resolution must be positive.")
    quotient = (maximum - minimum) / resolution
    scale = max(abs(minimum), abs(maximum), abs(maximum - minimum))
    tol = 16 * np.finfo(float).eps * scale / resolution
    n_cells = int(np.floor(quotient + tol))
    if n_cells < 1:
        raise ValueError("Grid extent must contain at least one complete cell.")
    return minimum + np.arange(n_cells, dtype=float) * resolution


def _cf_grid_mapping_attrs(projection: str) -> dict[str, object]:
    """Return CF-style grid-mapping attributes for a PROJ string."""
    attrs: dict[str, object] = {"proj4_params": projection}
    from pyproj import CRS

    crs = CRS.from_user_input(projection)
    attrs.update(crs.to_cf())
    wkt = crs.to_wkt()
    attrs["spatial_ref"] = wkt
    attrs["crs_wkt"] = wkt
    return {
        key: value
        for key, value in attrs.items()
        if isinstance(value, str | int | float | np.number)
    }


def cf_axis_attrs(dim: str) -> dict[str, str]:
    """
    Return the CF attributes for one coordinate axis: ``lon``, ``lat``, ``x`` or ``y``.

    Grids and footprints both use this, so their axes carry the same
    attributes.
    """
    return {
        "lon": {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
        "lat": {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
        "x": {
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "units": "m",
            "axis": "X",
        },
        "y": {
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection",
            "units": "m",
            "axis": "Y",
        },
    }[dim]


class Bounds(BaseModel):
    """Longitude/latitude bounding box, in degrees."""

    model_config = ConfigDict(frozen=True)

    xmin: float = Field(..., description="Western edge, in degrees longitude.")
    xmax: float = Field(..., description="Eastern edge, in degrees longitude.")
    ymin: float = Field(..., description="Southern edge, in degrees latitude.")
    ymax: float = Field(..., description="Northern edge, in degrees latitude.")


class Grid(Bounds):
    """
    Footprint grid: longitude/latitude bounds, cell size, and projection.

    The bounds are always longitude/latitude. With a projected ``projection``,
    the grid covers the bounds' extent in that projection and ``xres`` and
    ``yres`` are in its units.
    """

    model_config = ConfigDict(frozen=True)

    xres: float = Field(
        ...,
        description="Cell width in projection units (degrees for longlat, meters for UTM).",
    )
    yres: float = Field(
        ...,
        description="Cell height in projection units (degrees for longlat, meters for UTM).",
    )
    projection: str = Field(
        "+proj=longlat",
        description=(
            "Projection of the footprint grid, as a PROJ string. Particles and "
            "bounds are projected to it before gridding."
        ),
    )

    @property
    def resolution(self) -> str:
        """Cell size as text, such as ``'0.01x0.01'``."""
        return f"{self.xres}x{self.yres}"

    @property
    def is_longlat(self) -> bool:
        """Whether the grid is in longitude/latitude degrees."""
        return "+proj=longlat" in self.projection

    @property
    def min_cell_width(self) -> float:
        """Smaller of ``xres`` and ``yres``, as other geometries report it."""
        return float(min(self.xres, self.yres))

    @classmethod
    def from_geometry(
        cls,
        geometry,
        *,
        cells_per_target: float = 4.0,
        projection: str | None = None,
        max_cells: int = 50_000_000,
    ) -> Grid:
        """
        Return a grid fine enough to resolve the cells of a geometry.

        The bounds are the geometry's extent, rounded outward to whole cells.
        The cell size is the smallest geometry cell width divided by
        ``cells_per_target``, rounded down to one significant figure.

        Parameters
        ----------
        geometry : Mesh, Zones, or Grid
            Geometry to resolve. Any object with ``bounds``,
            ``min_cell_width``, ``crs`` or ``projection``, and ``is_longlat``.
        cells_per_target : float, default 4
            Grid cells across the smallest geometry cell.
        projection : str, optional
            Projection of the grid. Defaults to the geometry's CRS; the
            geometry is reprojected when they differ.
        max_cells : int, default 50_000_000
            Warn when the grid would have more cells than this.

        Returns
        -------
        Grid
            The derived grid, with longitude/latitude bounds.
        """
        import math
        import warnings

        crs = getattr(geometry, "crs", None) or getattr(geometry, "projection", None)
        if not isinstance(crs, str):
            raise TypeError("geometry must expose a 'crs' or 'projection' string.")
        if projection is not None and projection != crs:
            from stilt.geometry import Mesh, Zones

            geometry = geometry.base if isinstance(geometry, Zones) else geometry
            if isinstance(geometry, Grid):
                geometry = Mesh.from_grid(geometry)
            geometry = geometry.to_crs(projection)
            crs = projection
        projection = crs

        width = float(geometry.min_cell_width) / float(cells_per_target)
        if width <= 0:
            raise ValueError("Geometry cells must have positive width.")
        exp = math.floor(math.log10(width))
        res = math.floor(width / 10**exp) * 10**exp  # round down, 1 sig fig
        res = float(f"{res:.1g}")

        xmin, ymin, xmax, ymax = geometry.bounds
        xmin, ymin = math.floor(xmin / res) * res, math.floor(ymin / res) * res
        xmax, ymax = math.ceil(xmax / res) * res, math.ceil(ymax / res) * res
        if xmax <= xmin:
            xmax = xmin + res
        if ymax <= ymin:
            ymax = ymin + res

        n_cells = ((xmax - xmin) / res) * ((ymax - ymin) / res)
        if n_cells > max_cells:
            warnings.warn(
                f"Derived grid has ~{n_cells:.3g} cells at resolution {res:g}; "
                "consider a coarser cells_per_target or a smaller domain.",
                stacklevel=2,
            )

        if not geometry.is_longlat:
            # Bounds are always lon/lat: back-transform the snapped envelope.
            # Pad by one cell first; ``axes`` re-projects the lon/lat corners
            # and takes their extremes, which can shave an edge cell off a
            # rotated projection otherwise.
            xmin, xmax, ymin, ymax = xmin - res, xmax + res, ymin - res, ymax + res
            from pyproj import Transformer

            tr = Transformer.from_crs(projection, "EPSG:4326", always_xy=True)
            xs, ys = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
            xmin, xmax = float(min(xs)), float(max(xs))
            ymin, ymax = float(min(ys)), float(max(ys))

        return cls(
            xmin=float(xmin),
            xmax=float(xmax),
            ymin=float(ymin),
            ymax=float(ymax),
            xres=res,
            yres=res,
            projection=projection,
        )

    @classmethod
    def from_geometries(cls, geometries, **kwargs) -> Grid:
        """
        Return one grid that resolves several geometries.

        The grid covers all their extents at the finest cell size any of them
        needs. Keyword arguments are passed to :meth:`from_geometry`.
        """
        grids = [cls.from_geometry(g, **kwargs) for g in geometries]
        res = min(min(g.xres, g.yres) for g in grids)
        projection = grids[0].projection
        return cls(
            xmin=min(g.xmin for g in grids),
            xmax=max(g.xmax for g in grids),
            ymin=min(g.ymin for g in grids),
            ymax=max(g.ymax for g in grids),
            xres=res,
            yres=res,
            projection=projection,
        )

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cell-center coordinates ``(x, y)`` in projection units, ascending.

        These are the footprint's coordinates on this grid. They are rounded
        to 10 decimals (``40.35`` rather than ``40.349999999999994``) so they
        match labels built elsewhere from the same bounds and resolution.
        Projected grids need ``pyproj``.
        """
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        if not self.is_longlat:
            from pyproj import Transformer

            tr = Transformer.from_crs("EPSG:4326", self.projection, always_xy=True)
            corners_x, corners_y = tr.transform([xmin, xmax], [ymin, ymax])
            xmin, xmax = float(np.min(corners_x)), float(np.max(corners_x))
            ymin, ymax = float(np.min(corners_y)), float(np.max(corners_y))

        x_centers = _grid_cell_starts(xmin, xmax, self.xres) + self.xres / 2
        y_centers = _grid_cell_starts(ymin, ymax, self.yres) + self.yres / 2
        return np.round(x_centers, 10), np.round(y_centers, 10)

    @property
    def cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Every cell center as flat ``(x, y)`` arrays, with ``x`` varying slowest."""
        x, y = self.axes
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return xx.ravel(), yy.ravel()

    @property
    def index(self) -> pd.MultiIndex:
        """
        Index of every cell, in the same order as ``cells``.

        The levels are named ``lon`` and ``lat`` for a longitude/latitude grid
        and ``x`` and ``y`` for a projected one, as in the footprint.
        """
        import pandas as pd

        x, y = self.axes
        names = ["lon", "lat"] if self.is_longlat else ["x", "y"]
        return pd.MultiIndex.from_product([x, y], names=names)

    def to_xarray(self) -> xr.Dataset:
        """
        Return the grid as a CF-1.8 dataset of cell centers.

        The dataset has ``lon`` and ``lat`` coordinates (``x`` and ``y`` when
        projected) matching the footprint, and a ``crs`` grid-mapping
        variable. Pass it to ``foot.stilt.aggregate`` as a target, or
        to other tools that read CF grids. Projected grids need ``pyproj``.
        """
        import xarray as xr

        is_longlat = self.is_longlat
        x_centers, y_centers = self.axes
        x_dim, y_dim = ("lon", "lat") if is_longlat else ("x", "y")

        ds = xr.Dataset(coords={x_dim: x_centers, y_dim: y_centers})
        ds.attrs["Conventions"] = "CF-1.8"
        ds["crs"] = xr.DataArray(0, attrs=_cf_grid_mapping_attrs(self.projection))
        ds[x_dim].attrs.update(cf_axis_attrs(x_dim))
        ds[y_dim].attrs.update(cf_axis_attrs(y_dim))
        return ds


__all__ = ["Bounds", "Grid", "cf_axis_attrs"]
