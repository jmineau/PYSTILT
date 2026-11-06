"""
Rasters and coordinate systems: bounding boxes, grids, and CRS helpers.

:class:`Bounds` is a longitude/latitude box, such as the crop box of a met.
:class:`Grid` is a rectilinear raster: the grid a footprint is computed on,
and the grid a flux field is put on to be compared with it. This module
does not need shapely, so any part of PYSTILT can use it. Polygons
(``Mesh``, ``Zones``), which do, are in :mod:`stilt.footprint.targets`.
"""

from __future__ import annotations

import difflib
import functools
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Bounds and grids
# ---------------------------------------------------------------------------


def _grid_cell_count(minimum: float, maximum: float, resolution: float) -> int:
    """
    Return the number of whole cells between two grid bounds.

    Cells start at ``minimum`` and step by ``resolution`` while the whole cell
    fits inside ``[minimum, maximum]``, so ``maximum`` is the outer edge of
    the last cell.

    Decimal bounds are not exact in binary, so ``maximum - minimum`` carries a
    rounding error that grows with the bounds (``40.93 - 40.45`` is
    ``0.4799999999999969``). The tolerance is sized to that error so the last
    intended cell is kept (48 cells here at 0.01, as STILT-R's ``seq()``
    gives) and a partial cell is still dropped.

    Raises
    ------
    ValueError
        If *resolution* is not positive or no whole cell fits.
    """
    if resolution <= 0:
        raise ValueError("Grid resolution must be positive.")
    quotient = (maximum - minimum) / resolution
    scale = max(abs(minimum), abs(maximum), abs(maximum - minimum))
    tol = 16 * np.finfo(float).eps * scale / resolution
    n_cells = int(np.floor(quotient + tol))
    if n_cells < 1:
        raise ValueError("Grid extent must contain at least one complete cell.")
    return n_cells


def _grid_cell_starts(minimum: float, maximum: float, resolution: float) -> np.ndarray:
    """Return the lower edges of the whole cells between two grid bounds (:func:`_grid_cell_count`)."""
    n_cells = _grid_cell_count(minimum, maximum, resolution)
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


def _east_edge(xmin: float, xmax: float) -> float:
    """Return a grid's east edge in degrees, a turn later when *xmax* is written west of *xmin*."""
    return xmax + 360.0 if xmax < xmin else xmax


class Bounds(BaseModel):
    """Longitude/latitude bounding box, in degrees."""

    model_config = ConfigDict(frozen=True)

    xmin: float = Field(..., description="Western edge, in degrees longitude.")
    xmax: float = Field(..., description="Eastern edge, in degrees longitude.")
    ymin: float = Field(..., description="Southern edge, in degrees latitude.")
    ymax: float = Field(..., description="Northern edge, in degrees latitude.")

    def contains(self, longitude: ArrayLike, latitude: ArrayLike) -> np.ndarray:
        """
        Return whether each point is in the box, counting longitudes modulo 360.

        A longitude is in the box when, counted east from ``xmin``, it is
        within the box's width, so a box may cross the dateline (``xmin=170``,
        ``xmax=-170``) or reach past 180 (``xmax=190``), and ``-175`` and
        ``185`` are the same meridian. A box 360 degrees wide or more holds
        every longitude. Points on an edge are inside; the longitude edges
        allow ``1e-9`` degree for rounding, since a sum such as
        ``184.7 + 175.3`` need not give exactly 360. A point with a
        missing coordinate is not in the box.

        Parameters
        ----------
        longitude, latitude : array_like of float
            Degrees. Scalars or arrays, broadcast against each other.

        Returns
        -------
        numpy.ndarray of bool
            One value per point (a 0-d array for scalars).

        Examples
        --------
        >>> import stilt
        >>> utah = stilt.Bounds(xmin=-114, xmax=-109, ymin=37, ymax=42)
        >>> bool(utah.contains(-111.9, 40.8))
        True
        >>> bool(utah.contains(-111.9, 45.0))
        False

        A box may cross the dateline, and ``-175`` and ``185`` are one meridian:

        >>> pacific = stilt.Bounds(xmin=170, xmax=-170, ymin=-10, ymax=10)
        >>> pacific.contains([175, -175, 185, 0], 0).tolist()
        [True, True, True, False]
        """
        lon = np.asarray(longitude, dtype=float)
        lat = np.asarray(latitude, dtype=float)
        in_latitude = (lat >= self.ymin) & (lat <= self.ymax)
        if self.xmax - self.xmin >= 360.0:
            in_longitude = ~np.isnan(lon)
        else:
            tolerance = 1e-9
            width = (self.xmax - self.xmin) % 360.0
            from_west = (lon - self.xmin) % 360.0
            in_longitude = (from_west <= width + tolerance) | (
                from_west >= 360.0 - tolerance
            )
        return np.asarray(in_longitude & in_latitude)


class Grid(Bounds):
    """
    Footprint grid: longitude/latitude bounds, cell size, and CRS.

    The bounds are always longitude/latitude. With a projected ``crs``,
    the grid covers the bounds' extent in that projection and ``xres`` and
    ``yres`` are in its units. Longitudes count modulo 360, so a grid
    across 180 degrees is written ``xmin=170, xmax=190`` or ``xmin=170,
    xmax=-170``. A footprint on either has longitudes 170 to 190. A grid
    spans at most 360 degrees of longitude.
    """

    model_config = ConfigDict(frozen=True)

    xres: float = Field(
        ...,
        gt=0,
        description="Cell width in projection units (degrees for longlat, meters for UTM).",
    )
    yres: float = Field(
        ...,
        gt=0,
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

    @model_validator(mode="before")
    @classmethod
    def _no_unknown_keys(cls, data: object) -> object:
        """Refuse a key that is no grid setting, naming the nearest one, so ``csr`` is not dropped."""
        if isinstance(data, dict):
            known = [*cls.model_fields, "projection"]
            for key in data:
                if key not in known:
                    close = difflib.get_close_matches(str(key), known, n=1)
                    hint = f" Did you mean {close[0]!r}?" if close else ""
                    raise ValueError(f"{key!r} is not a grid setting.{hint}")
        return data

    @model_validator(mode="after")
    def _at_most_one_turn(self) -> Self:
        """Refuse a grid wider than 360 degrees of longitude: it would hold some longitudes twice."""
        if not -360.0 < self.xmax - self.xmin <= 360.0:
            raise ValueError(
                f"A grid spans at most 360 degrees of longitude; xmin={self.xmin:g} "
                f"to xmax={self.xmax:g} does not. A grid "
                "across 180 degrees can be written either way, such as xmin=170, "
                "xmax=190 or xmin=170, xmax=-170."
            )
        return self

    @model_validator(mode="after")
    def _latitudes_and_cells(self) -> Self:
        """Refuse latitudes off the globe and, on a longitude/latitude grid, a box that holds no whole cell."""
        if not -90.0 <= self.ymin < self.ymax <= 90.0:
            raise ValueError(
                f"A grid needs -90 <= ymin < ymax <= 90 degrees latitude; got "
                f"ymin={self.ymin:g}, ymax={self.ymax:g}."
            )
        if self.xmax == self.xmin:
            raise ValueError(
                f"A grid needs xmin and xmax to differ; both are {self.xmin:g}."
            )
        if self.is_longlat:
            # The grid's own message, with the axis it is about.
            for axis, low, high, res in (
                ("x", self.xmin, _east_edge(self.xmin, self.xmax), self.xres),
                ("y", self.ymin, self.ymax, self.yres),
            ):
                try:
                    _grid_cell_count(low, high, res)
                except ValueError as error:
                    raise ValueError(
                        f"{error} The {axis} axis runs {low:g} to {high:g} "
                        f"with {axis}res={res:g}."
                    ) from None
        return self

    @property
    def is_longlat(self) -> bool:
        """Whether the grid is in longitude/latitude degrees."""
        return is_longlat(self.crs)

    @property
    def dims(self) -> tuple[str, str]:
        """The names of a footprint's horizontal dimensions on this grid, ``("lat", "lon")`` or ``("y", "x")``."""
        return ("lat", "lon") if self.is_longlat else ("y", "x")

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cell-center coordinates ``(x, y)`` in projection units, ascending.

        These are the footprint's coordinates on this grid. On a
        longitude/latitude grid they run east from ``xmin``, past 180 if the
        grid does: a grid from 170 to -170 has longitudes 170 to 190. They are rounded
        to 10 decimals (``40.35`` rather than ``40.349999999999994``) so they
        match labels built elsewhere from the same bounds and resolution.
        Projected grids need ``pyproj``.
        """
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        if self.is_longlat:
            xmax = _east_edge(xmin, xmax)
        else:
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
