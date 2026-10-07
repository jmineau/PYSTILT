"""
Rasters and coordinate systems: bounding boxes, grids, and CRS helpers.

:class:`Bounds` is a longitude/latitude box, such as the crop box of a met.
:class:`Grid` is a rectilinear raster: the grid a footprint is computed on,
and the grid a flux field is put on to be compared with it. This module
does not need shapely, so any part of PYSTILT can use it. Polygons
(``Mesh``, ``Zones``), which do, are in :mod:`stilt.footprint.targets`.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import xarray as xr

# ---------------------------------------------------------------------------
# Bounds and grids
# ---------------------------------------------------------------------------


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


def _with_cf_grid(ds: xr.Dataset, crs: str) -> xr.Dataset:
    """Add the CF-1.8 grid mapping for *crs* and the axis attributes to *ds*, and return it."""
    import xarray as xr

    ds.attrs.setdefault("Conventions", "CF-1.8")
    ds["crs"] = xr.DataArray(0, attrs=_cf_grid_mapping_attrs(crs))
    for dim in ("lon", "lat", "x", "y"):
        if dim in ds.coords:
            ds[dim].attrs.update(cf_axis_attrs(dim))
    return ds


_HORIZONTAL_DIMS = (("lat", "lon"), ("y", "x"))


def horizontal_dims(data: xr.DataArray) -> tuple[str, str]:
    """
    Return the names of the horizontal dimensions, ``(y_dim, x_dim)``.

    Raises
    ------
    ValueError
        If *data* has neither ``lat``/``lon`` nor ``y``/``x`` dimensions.
    """
    for y_dim, x_dim in _HORIZONTAL_DIMS:
        if y_dim in data.dims and x_dim in data.dims:
            return y_dim, x_dim
    raise ValueError(
        f"Expected 'lat'/'lon' or 'y'/'x' dimensions; got {tuple(data.dims)}."
    )


class Bounds(BaseModel):
    """Longitude/latitude bounding box, in degrees."""

    model_config = ConfigDict(frozen=True)

    xmin: float = Field(..., description="Western edge, in degrees longitude.")
    xmax: float = Field(..., description="Eastern edge, in degrees longitude.")
    ymin: float = Field(..., description="Southern edge, in degrees latitude.")
    ymax: float = Field(..., description="Northern edge, in degrees latitude.")


class Grid(Bounds):
    """
    Footprint grid: longitude/latitude bounds, cell size, and CRS.

    The bounds are always longitude/latitude. With a projected ``crs``,
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
    crs: str = Field(
        "+proj=longlat",
        validation_alias=AliasChoices("crs", "projection"),
        description=(
            "Coordinate reference system of the footprint grid, such as a PROJ "
            "string or ``EPSG:32612``. Particles and bounds are projected to it "
            "before gridding. Also read under STILT-R's name, ``projection``."
        ),
    )

    @property
    def is_longlat(self) -> bool:
        """Whether the grid is in longitude/latitude degrees."""
        return is_longlat(self.crs)

    @property
    def dims(self) -> tuple[str, str]:
        """The names of a footprint's horizontal dimensions on this grid: ``("lat", "lon")`` or ``("y", "x")``."""
        return ("lat", "lon") if self.is_longlat else ("y", "x")

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

            tr = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
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
        return pd.MultiIndex.from_product([pd.Index(x), pd.Index(y)], names=names)

    def to_xarray(self) -> xr.Dataset:
        """
        Return the grid as a CF-1.8 dataset of cell centers.

        The dataset has ``lon`` and ``lat`` coordinates (``x`` and ``y`` when
        projected) matching the footprint, and a ``crs`` grid-mapping
        variable. Use it to regrid a flux field onto the footprint grid, or
        with other tools that read CF grids. To sum a footprint onto this
        grid, pass the grid itself to ``foot.stilt.aggregate``. Projected
        grids need ``pyproj``.
        """
        import xarray as xr

        x_centers, y_centers = self.axes
        y_dim, x_dim = self.dims
        ds = xr.Dataset(coords={x_dim: x_centers, y_dim: y_centers})
        return _with_cf_grid(ds, self.crs)


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def is_longlat(crs: str) -> bool:
    """
    Return whether ``crs`` is geographic (longitude/latitude degrees).

    This is the one test PYSTILT uses, for grids, meshes, and plots alike.
    PROJ strings are read by eye; other forms, such as ``EPSG:4326``, by
    pyproj.
    """
    if "+proj=longlat" in crs:
        return True
    if crs.upper() in {"EPSG:4326", "WGS84", "OGC:CRS84"}:
        return True
    if crs.startswith("+"):
        return False
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(crs).is_geographic)
    except Exception:  # not a CRS pyproj can read
        return False


def same_crs(a: str, b: str) -> bool:
    """Return whether two CRS strings describe the same CRS, however they are spelled."""
    if a == b:
        return True
    if is_longlat(a) and is_longlat(b):
        return True
    try:
        from pyproj import CRS

        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:  # not a CRS pyproj can read
        return False


def haversine_km(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    """Return the great-circle distance in km, broadcasting over the inputs."""
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = (
        np.sin(d_lat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


__all__ = [
    "Bounds",
    "Grid",
    "cf_axis_attrs",
    "haversine_km",
    "horizontal_dims",
    "is_longlat",
    "same_crs",
]
