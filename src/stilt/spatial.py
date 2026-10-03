"""
Where things are: bounding boxes, footprint grids, and the geometries footprints are aggregated onto.

Footprints are always computed on a rectilinear grid (:class:`Grid`), as in
STILT-R. An inversion's state vector may use another geometry: a coarser or
shifted grid, hexagons, polygons from a shapefile, nested grids, or cells
merged into larger regions. A footprint value is a per-cell sensitivity
that adds up over area, so moving it onto another geometry is a sparse
matrix product. The matrix entry ``W[cell, native]`` is the fraction of a
native grid cell inside a target cell. It depends only on the grid and the
geometry, so it is built once and cached.

Geometries
----------
:class:`Grid`
    A rectilinear grid, also the grid footprints are computed on.
:class:`Mesh`
    Polygons with ids and a CRS: shapefiles, nested grids, H3 hexagons
    (:meth:`Mesh.from_h3`), and windows around point sources
    (:meth:`Mesh.from_windows`).
:class:`Zones`
    Labels that merge the cells of a ``Grid`` or ``Mesh`` into larger
    regions.

Each has ``index`` (the cells in result order), ``bounds``, ``crs`` or
``projection``, ``is_longlat``, ``min_cell_width``, and ``hash``.
:func:`overlap_weights` builds the weight matrix for any of them.
"""

from __future__ import annotations

import functools
import hashlib
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import shapely
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from scipy import sparse
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    import xarray as xr

# ---------------------------------------------------------------------------
# Bounds and grids
# ---------------------------------------------------------------------------

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
    def min_cell_width(self) -> float:
        """Smaller of ``xres`` and ``yres``, as other geometries report it."""
        return float(min(self.xres, self.yres))

    @classmethod
    def from_geometry(
        cls,
        geometry,
        *,
        cells_per_target: float = 4.0,
        crs: str | None = None,
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
            ``min_cell_width``, ``crs``, and ``is_longlat``.
        cells_per_target : float, default 4
            Grid cells across the smallest geometry cell.
        crs : str, optional
            CRS of the grid. Defaults to the geometry's; the geometry is
            reprojected when they differ.
        max_cells : int, default 50_000_000
            Warn when the grid would have more cells than this.

        Returns
        -------
        Grid
            The derived grid, with longitude/latitude bounds.
        """
        import math
        import warnings

        if crs is not None and crs != geometry.crs:
            geometry = geometry.base if isinstance(geometry, Zones) else geometry
            if isinstance(geometry, Grid):
                geometry = Mesh.from_grid(geometry)
            geometry = geometry.to_crs(crs)
        crs = geometry.crs

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

            tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
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
            crs=crs,
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
        return pd.MultiIndex.from_product([x, y], names=names)

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
        ds.attrs["Conventions"] = "CF-1.8"
        ds["crs"] = xr.DataArray(0, attrs=_cf_grid_mapping_attrs(self.crs))
        ds[x_dim].attrs.update(cf_axis_attrs(x_dim))
        ds[y_dim].attrs.update(cf_axis_attrs(y_dim))
        return ds


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


def _transform_geometries(
    geoms: Sequence[BaseGeometry], src: str, dst: str
) -> tuple[BaseGeometry, ...]:
    """Return shapely geometries reprojected from ``src`` to ``dst``. Requires pyproj."""
    if same_crs(src, dst):
        return tuple(geoms)
    from pyproj import Transformer

    tr = Transformer.from_crs(src, dst, always_xy=True)

    def _fn(coords: np.ndarray) -> np.ndarray:
        """Return one array of coordinates reprojected."""
        x, y = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack((x, y))

    return tuple(shapely.transform(g, _fn) for g in geoms)


def _grid_boxes(x_centers: np.ndarray, y_centers: np.ndarray, xres: float, yres: float):
    """Return a shapely box for every grid cell, with ``x`` varying fastest."""
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")  # (ny, nx)
    return shapely.box(
        (xx - xres / 2).ravel(),
        (yy - yres / 2).ravel(),
        (xx + xres / 2).ravel(),
        (yy + yres / 2).ravel(),
    )


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


class Mesh(BaseModel):
    """
    Polygon cells with ids, the most general geometry.

    Parameters
    ----------
    ids : sequence of str
        Unique label for each cell. These become the ``index``.
    geometries : sequence of shapely Polygon or MultiPolygon
        One polygon per id, in ``crs`` coordinates.
    crs : str, default "+proj=longlat"
        PROJ string or ``"EPSG:xxxx"`` code of the polygon coordinates.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    ids: tuple[str, ...] = Field(..., description="Unique cell labels.")
    geometries: tuple[Any, ...] = Field(
        ..., description="Shapely polygon per cell, in ``crs`` coordinates."
    )
    crs: str = Field(
        "+proj=longlat", description="Coordinate reference system of the polygons."
    )

    @model_validator(mode="after")
    def _check(self) -> Mesh:
        """Require unique ids, one non-empty polygon per id."""
        if len(self.ids) == 0:
            raise ValueError("Mesh requires at least one cell.")
        if len(self.ids) != len(self.geometries):
            raise ValueError("ids and geometries must have the same length.")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("ids must be unique.")
        for g in self.geometries:
            if not isinstance(g, BaseGeometry) or g.is_empty:
                raise ValueError("geometries must be non-empty shapely geometries.")
            if shapely.get_type_id(g) not in (3, 6):  # Polygon, MultiPolygon
                raise ValueError("Mesh geometries must be Polygon or MultiPolygon.")
        return self

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_geodataframe(cls, gdf, ids: str | Sequence[str] | None = None) -> Mesh:
        """
        Return a mesh of the polygons in a GeoDataFrame's geometry column.

        ``ids`` is the name of a column holding the cell labels, or the
        labels themselves. Defaults to the GeoDataFrame's index.
        """
        if ids is None:
            labels = [str(i) for i in gdf.index]
        elif isinstance(ids, str):
            labels = [str(v) for v in gdf[ids]]
        else:
            labels = [str(v) for v in ids]
        crs = gdf.crs.to_string() if gdf.crs is not None else "+proj=longlat"
        return cls(ids=tuple(labels), geometries=tuple(gdf.geometry.tolist()), crs=crs)

    @classmethod
    def from_file(cls, path, ids: str | None = None, **read_kwargs) -> Mesh:
        """Return a mesh read from a vector file such as a shapefile, with geopandas."""
        import geopandas as gpd

        return cls.from_geodataframe(gpd.read_file(path, **read_kwargs), ids=ids)

    @classmethod
    def from_windows(
        cls,
        coords: Sequence[tuple[float, float]],
        size: float | tuple[float, float],
        *,
        ids: Sequence[str] | None = None,
        crs: str = "+proj=longlat",
    ) -> Mesh:
        """
        Return rectangular windows centered on points, such as known point sources.

        ``size`` is the window width, or ``(width, height)``, in ``crs``
        units. Without ``ids``, each window is labeled ``"x,y"``.
        """
        arr = np.asarray(list(coords), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("coords must be a sequence of (x, y) pairs.")
        if isinstance(size, (int, float)):
            w = h = float(size)
        else:
            w, h = float(size[0]), float(size[1])
        if w <= 0 or h <= 0:
            raise ValueError("window size must be positive.")
        boxes = shapely.box(
            arr[:, 0] - w / 2, arr[:, 1] - h / 2, arr[:, 0] + w / 2, arr[:, 1] + h / 2
        )
        if ids is None:
            labels = tuple(f"{x:g},{y:g}" for x, y in arr)
        else:
            labels = tuple(str(i) for i in ids)
        return cls(ids=labels, geometries=tuple(boxes.tolist()), crs=crs)

    @classmethod
    def from_grid(cls, grid: Grid) -> Mesh:
        """Return every cell of a grid as a polygon, in ``grid.index`` order."""
        x, y = grid.cells
        boxes = shapely.box(
            x - grid.xres / 2, y - grid.yres / 2, x + grid.xres / 2, y + grid.yres / 2
        )
        labels = tuple(f"{xi:g},{yi:g}" for xi, yi in zip(x, y, strict=True))
        return cls(ids=labels, geometries=tuple(boxes.tolist()), crs=grid.crs)

    @classmethod
    def from_h3(cls, resolution: int, bounds) -> Mesh:
        """
        Return the H3 hexagons of one resolution inside ``bounds``. Requires ``h3``.

        ``bounds`` is a :class:`~stilt.config.Bounds` or ``Grid``, or an
        ``(xmin, ymin, xmax, ymax)`` tuple in degrees. The cell ids are the H3
        cell strings.
        """
        try:
            import h3  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("Mesh.from_h3 requires the 'h3' package.") from exc

        if hasattr(bounds, "xmin"):
            xmin, ymin, xmax, ymax = bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax
        else:
            xmin, ymin, xmax, ymax = (float(v) for v in bounds)
        ring = [(ymin, xmin), (ymin, xmax), (ymax, xmax), (ymax, xmin)]  # (lat, lng)
        poly = h3.LatLngPoly(ring)
        cells = sorted(h3.h3shape_to_cells(poly, resolution))
        if not cells:
            raise ValueError("No H3 cells found inside bounds at this resolution.")
        polys = [
            shapely.Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)])
            for c in cells
        ]
        return cls(ids=tuple(cells), geometries=tuple(polys), crs="+proj=longlat")

    # -- properties ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def index(self) -> pd.Index:
        """The cell ids, as an index named ``"cell"``."""
        return pd.Index(list(self.ids), name="cell")

    @property
    def is_longlat(self) -> bool:
        """Whether the polygon coordinates are longitude/latitude degrees."""
        return is_longlat(self.crs)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` extent of all cells."""
        b = shapely.total_bounds(np.asarray(self.geometries, dtype=object))
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    @property
    def min_cell_width(self) -> float:
        """Shortest side of any cell's bounding box, in ``crs`` units."""
        b = shapely.bounds(np.asarray(self.geometries, dtype=object))
        widths = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        return float(widths.min())

    @property
    def hash(self) -> str:
        """First 10 characters of a SHA-256 hash of the ids, polygons, and CRS."""
        h = hashlib.sha256()
        h.update(self.crs.encode())
        for i, g in zip(self.ids, self.geometries, strict=True):
            h.update(i.encode())
            h.update(shapely.to_wkb(g))
        return h.hexdigest()[:10]

    def to_crs(self, crs: str) -> Mesh:
        """Return the mesh reprojected to ``crs``. Requires pyproj."""
        if same_crs(self.crs, crs):
            return self
        return Mesh(
            ids=self.ids,
            geometries=_transform_geometries(self.geometries, self.crs, crs),
            crs=crs,
        )

    def __repr__(self) -> str:
        return f"Mesh(n_cells={len(self)}, crs={self.crs!r})"

    __str__ = __repr__


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------


class Zones(BaseModel):
    """
    Regions made by merging the cells of a ``Grid`` or ``Mesh``.

    Parameters
    ----------
    base : Grid or Mesh
        Geometry whose cells are merged.
    labels : sequence of str
        One label per cell of ``base``, in ``base.index`` order. Cells with
        the same label form one region. The ``index`` lists the labels in
        order of first appearance.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    base: Grid | Mesh
    labels: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Zones:
        """Require one label per base cell."""
        n = len(self.base.index)
        if len(self.labels) != n:
            raise ValueError(
                f"labels must have one entry per base cell ({n}), got {len(self.labels)}."
            )
        return self

    @classmethod
    def from_labels(cls, base: Grid | Mesh, labels: Sequence[Any]) -> Zones:
        """Return zones from labels of any type, converted to strings."""
        return cls(base=base, labels=tuple(str(v) for v in labels))

    def __len__(self) -> int:
        return len(self.index)

    @property
    def index(self) -> pd.Index:
        """Unique labels in order of first appearance, as an index named ``"cell"``."""
        return pd.Index(pd.unique(np.asarray(self.labels, dtype=object)), name="cell")

    @property
    def crs(self) -> str:
        """CRS of the base geometry."""
        return self.base.crs

    @property
    def is_longlat(self) -> bool:
        """Whether the base geometry is in longitude/latitude degrees."""
        return self.base.is_longlat

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` extent of the base geometry."""
        if isinstance(self.base, Grid):
            b = self.base
            return b.xmin, b.ymin, b.xmax, b.ymax
        return self.base.bounds

    @property
    def min_cell_width(self) -> float:
        """Smallest cell width of the base geometry, since no region is smaller."""
        if isinstance(self.base, Grid):
            return float(min(self.base.xres, self.base.yres))
        return self.base.min_cell_width

    @property
    def membership(self) -> sparse.csr_matrix:
        """Sparse ``(n_regions, n_base_cells)`` matrix, 1 where a base cell is in a region."""
        codes = self.index.get_indexer(list(self.labels))
        n_base = len(self.labels)
        return sparse.csr_matrix(
            (np.ones(n_base), (codes, np.arange(n_base))),
            shape=(len(self.index), n_base),
        )

    @property
    def hash(self) -> str:
        """First 10 characters of a SHA-256 hash of the base geometry and labels."""
        h = hashlib.sha256()
        h.update(_geometry_key(self.base).encode())
        h.update("\x1f".join(self.labels).encode())
        return h.hexdigest()[:10]

    def __repr__(self) -> str:
        return f"Zones(n_cells={len(self)}, base={self.base!r})"

    __str__ = __repr__


# ---------------------------------------------------------------------------
# Overlap weights
# ---------------------------------------------------------------------------

Geometry = Grid | Mesh | Zones
"""A geometry a footprint can be aggregated onto."""


def _geometry_key(geometry: Geometry) -> str:
    """Return a cache key identifying the geometry."""
    if isinstance(geometry, Grid):
        return "grid:" + geometry.model_dump_json()
    return f"{type(geometry).__name__.lower()}:{geometry.hash}"


def _raster_key(
    x: np.ndarray, y: np.ndarray, xres: float, yres: float, crs: str
) -> str:
    """Return a cache key identifying a raster's axes and projection."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=float).tobytes())
    h.update(np.ascontiguousarray(y, dtype=float).tobytes())
    h.update(f"{xres!r}|{yres!r}|{crs}".encode())
    return h.hexdigest()[:16]


_weight_cache: dict[tuple[str, str], sparse.csr_matrix] = {}


def _overlap_1d(src_edges: np.ndarray, dst_edges: np.ndarray) -> sparse.csr_matrix:
    """Return the fraction of each source cell inside each destination cell, along one axis."""
    s_lo, s_hi = src_edges[:-1], src_edges[1:]
    d_lo, d_hi = dst_edges[:-1], dst_edges[1:]
    lo = np.maximum(s_lo[None, :], d_lo[:, None])
    hi = np.minimum(s_hi[None, :], d_hi[:, None])
    frac = np.clip(hi - lo, 0.0, None) / (s_hi - s_lo)[None, :]
    return sparse.csr_matrix(frac)


def _edges(centers: np.ndarray, res: float) -> np.ndarray:
    """Return cell edges from cell centers and a resolution."""
    c = np.asarray(centers, dtype=float)
    return np.concatenate(([c[0] - res / 2], c + res / 2))


def _grid_weights(
    grid: Grid, x: np.ndarray, y: np.ndarray, xres: float, yres: float
) -> sparse.csr_matrix:
    """Return the weights of a grid in the raster's CRS, with rows in ``grid.index`` order."""
    tx, ty = grid.axes
    px = _overlap_1d(_edges(x, xres), _edges(tx, grid.xres))  # (Tx, Nx)
    py = _overlap_1d(_edges(y, yres), _edges(ty, grid.yres))  # (Ty, Ny)
    w = sparse.kron(py, px).tocsr()  # rows (ty, tx); cols (iy, ix)
    n_tx, n_ty = len(tx), len(ty)
    # reorder rows from (ty outer, tx inner) to (tx outer, ty inner)
    order = (np.arange(n_tx)[:, None] + n_tx * np.arange(n_ty)[None, :]).ravel()
    return w[order]


def _exactextract_available() -> bool:
    """Return whether the optional exactextract backend is importable."""
    try:
        import exactextract  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


def _mesh_weights_exactextract(
    mesh: Mesh, x: np.ndarray, y: np.ndarray, xres: float, yres: float
) -> sparse.csr_matrix:
    """
    Return polygon weights computed with ``exactextract``.

    Gives the same result as :func:`_mesh_weights`, about 100 times faster
    on large rasters. ``exactextract`` numbers cells from the top row, so its
    ``cell_id`` is renumbered to start from the bottom row.
    """
    from exactextract import exact_extract  # pyright: ignore[reportMissingImports]
    from exactextract.raster import (  # pyright: ignore[reportMissingImports]
        NumPyRasterSource,
    )

    ny, nx = len(y), len(x)
    raster = NumPyRasterSource(
        np.zeros((ny, nx), dtype=np.float32),
        xmin=float(x[0] - xres / 2),
        ymin=float(y[0] - yres / 2),
        xmax=float(x[-1] + xres / 2),
        ymax=float(y[-1] + yres / 2),
    )
    features = [
        {
            "type": "Feature",
            "id": i,
            "properties": {"fid": i},
            "geometry": shapely.geometry.mapping(g),
        }
        for i, g in enumerate(mesh.geometries)
    ]
    out = exact_extract(
        raster, features, ["cell_id", "coverage"], include_cols=["fid"], output="pandas"
    )
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for fid, cell_id, coverage in zip(
        out["fid"], out["cell_id"], out["coverage"], strict=True
    ):
        cell_id = np.asarray(cell_id, dtype=np.int64)
        coverage = np.asarray(coverage, dtype=float)
        if cell_id.size == 0:
            continue
        row_top, col = np.divmod(cell_id, nx)
        native = (ny - 1 - row_top) * nx + col
        keep = coverage > 0
        rows.append(np.full(int(keep.sum()), int(fid), dtype=np.int64))
        cols.append(native[keep])
        vals.append(coverage[keep])
    if not rows:
        return sparse.csr_matrix((len(mesh), ny * nx))
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(mesh), ny * nx),
    )


def _mesh_weights(
    mesh: Mesh, x: np.ndarray, y: np.ndarray, xres: float, yres: float
) -> sparse.csr_matrix:
    """Return polygon weights from exact area intersections with shapely."""
    boxes = _grid_boxes(x, y, xres, yres)
    tree = shapely.STRtree(boxes)
    polys = np.asarray(mesh.geometries, dtype=object)
    p_idx, b_idx = tree.query(polys, predicate="intersects")
    if len(p_idx) == 0:
        return sparse.csr_matrix((len(mesh), len(boxes)))
    inter = shapely.area(shapely.intersection(polys[p_idx], boxes[b_idx]))
    frac = inter / shapely.area(boxes[b_idx])
    keep = frac > 0
    return sparse.csr_matrix(
        (frac[keep], (p_idx[keep], b_idx[keep])), shape=(len(mesh), len(boxes))
    )


def _polygon_weights(
    mesh: Mesh, x: np.ndarray, y: np.ndarray, xres: float, yres: float
) -> sparse.csr_matrix:
    """Return a mesh's weights, with exactextract when it is installed."""
    if _exactextract_available():
        return _mesh_weights_exactextract(mesh, x, y, xres, yres)
    return _mesh_weights(mesh, x, y, xres, yres)


def overlap_weights(
    geometry: Geometry,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    xres: float,
    yres: float,
    crs: str,
) -> sparse.csr_matrix:
    """
    Return the fraction of each raster cell inside each geometry cell.

    Polygon overlaps (a ``Mesh``, or a ``Grid`` in another CRS) use
    `exactextract <https://github.com/isciences/exactextract>`_ when it is
    installed, which is about 100 times faster on large rasters, and shapely
    otherwise. Both give the same fractions. Results are cached, so repeated
    aggregations onto the same geometry are one matrix product each.

    Parameters
    ----------
    geometry : Grid, Mesh, or Zones
        Target geometry. It is reprojected to ``crs`` when needed.
    x_centers, y_centers : numpy.ndarray
        Cell-center coordinates of the raster, in ``crs`` units.
    xres, yres : float
        Cell size of the raster, in ``crs`` units.
    crs : str
        CRS of the raster.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n_cells, ny * nx)``, with rows in ``geometry.index`` order
        and raster cells ordered with ``x`` varying fastest (the order of
        ``data.transpose("time", y, x).reshape(nt, -1)``).
    """
    x = np.asarray(x_centers, dtype=float)
    y = np.asarray(y_centers, dtype=float)
    key = (_raster_key(x, y, xres, yres, crs), _geometry_key(geometry))
    cached = _weight_cache.get(key)
    if cached is not None:
        return cached

    if isinstance(geometry, Zones):
        base_w = overlap_weights(geometry.base, x, y, xres, yres, crs)
        w = sparse.csr_matrix(geometry.membership @ base_w)
    elif isinstance(geometry, Grid):
        if same_crs(geometry.crs, crs):
            w = _grid_weights(geometry, x, y, xres, yres)
        else:
            w = _polygon_weights(Mesh.from_grid(geometry).to_crs(crs), x, y, xres, yres)
    else:
        w = _polygon_weights(geometry.to_crs(crs), x, y, xres, yres)

    _weight_cache[key] = w
    return w


def check_resolution(geometry: Geometry, xres: float, yres: float, crs: str) -> None:
    """
    Warn when the raster is too coarse to resolve the smallest target cell.

    The warning is about the error of rasterizing polygon boundaries. A grid
    in the raster's own CRS is exact at any resolution, so it never warns,
    and neither does a geometry in a different CRS, whose units differ.
    """
    width = geometry.min_cell_width
    if isinstance(geometry, Grid) and same_crs(geometry.crs, crs):
        return  # exact per-axis overlap; no rasterization error to warn about
    if not same_crs(geometry.crs, crs):
        return  # units differ; skip the heuristic rather than mislead
    if width < 2.0 * max(xres, yres):
        warnings.warn(
            f"Smallest target cell ({width:g}) spans fewer than two native raster "
            f"cells ({xres:g} x {yres:g}); the aggregate is under-resolved. "
            "Regenerate the footprint on a finer grid (sim.generate_footprint with "
            "Grid.from_geometry) for boundary accuracy.",
            stacklevel=3,
        )


__all__ = [
    "Bounds",
    "cf_axis_attrs",
    "check_resolution",
    "Geometry",
    "Grid",
    "horizontal_dims",
    "is_longlat",
    "kmsl_from_vertical_reference",
    "Mesh",
    "overlap_weights",
    "same_crs",
    "VerticalReference",
    "Zones",
]
