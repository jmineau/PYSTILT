"""
Spatial geometries: the state geometry footprints are aggregated onto.

Footprints are always *computed* on a rectilinear raster
(:class:`~stilt.config.Grid`), which preserves the STILT kernel and R-STILT
parity.  An inversion's state vector may live on any other geometry: a
coarser or shifted grid, hexagons, polygons from a shapefile, nested grids,
or cells of an existing geometry merged into super-cells.  Because a
footprint is an extensive per-cell sensitivity, moving it onto another
geometry is a linear operation: a sparse *overlap-weight matrix* whose entry
``W[cell, native]`` is the fraction of a native raster cell lying inside a
target cell.  That matrix depends only on the raster and the geometry, so it
is built once and cached, and every footprint is then a single matmul.

Geometries
----------
- :class:`~stilt.config.Grid` — rectilinear; also the native raster.  Fast
  per-axis overlap path.
- :class:`Mesh` — arbitrary polygons with ids and a CRS.  Covers shapefiles,
  nested multi-resolution grids, H3 hexagons (:meth:`Mesh.from_h3`) and
  point-source windows (:meth:`Mesh.from_windows`).
- :class:`Zones` — labels over a ``Grid`` or ``Mesh`` that merge its
  cells into super-cells.

All three expose ``index`` (the state index, in result order), ``bounds``,
``crs``/``is_longlat``, ``min_cell_width`` and ``hash``.
:func:`overlap_weights` builds the cached weight matrix for any of them.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pydantic import BaseModel, ConfigDict, model_validator
from scipy import sparse
from shapely.geometry.base import BaseGeometry

from stilt.config.fields import cfg_field
from stilt.config.spatial import Grid

# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------


def is_longlat_crs(crs: str) -> bool:
    """``True`` when ``crs`` is geographic (lon/lat degrees)."""
    if "+proj=longlat" in crs:
        return True
    if crs.upper() in {"EPSG:4326", "WGS84", "OGC:CRS84"}:
        return True
    if crs.startswith("+"):
        return False
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(crs).is_geographic)
    except Exception:  # pragma: no cover - pyproj optional
        return False


def same_crs(a: str, b: str) -> bool:
    """Compare two CRS descriptions, tolerating PROJ/EPSG spelling differences."""
    if a == b:
        return True
    if is_longlat_crs(a) and is_longlat_crs(b):
        return True
    try:
        from pyproj import CRS

        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:  # pragma: no cover - pyproj optional
        return False


def _transform_geometries(
    geoms: Sequence[BaseGeometry], src: str, dst: str
) -> tuple[BaseGeometry, ...]:
    """Reproject shapely geometries from ``src`` to ``dst`` (requires pyproj)."""
    if same_crs(src, dst):
        return tuple(geoms)
    from pyproj import Transformer

    tr = Transformer.from_crs(src, dst, always_xy=True)

    def _fn(coords: np.ndarray) -> np.ndarray:
        x, y = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack((x, y))

    return tuple(shapely.transform(g, _fn) for g in geoms)


def _grid_boxes(x_centers: np.ndarray, y_centers: np.ndarray, xres: float, yres: float):
    """Shapely boxes for every raster cell, flattened y-outer / x-inner."""
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
    Arbitrary polygon cells with ids: the general state geometry.

    Parameters
    ----------
    ids : sequence of str
        Unique label per cell; becomes the state index (``index``).
    geometries : sequence of shapely Polygon / MultiPolygon
        One polygon per id, in ``crs`` coordinates.
    crs : str
        PROJ string or ``"EPSG:xxxx"`` of the polygon coordinates.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    ids: tuple[str, ...] = cfg_field(..., description="Unique cell labels.")
    geometries: tuple[Any, ...] = cfg_field(
        ..., description="Shapely polygon per cell, in ``crs`` coordinates."
    )
    crs: str = cfg_field("+proj=longlat", description="CRS of the polygons.")

    @model_validator(mode="after")
    def _check(self) -> Mesh:
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
        Build from a GeoDataFrame (polygons in its active geometry column).

        ``ids`` names a column to use as cell labels, or supplies labels
        directly; by default the DataFrame index is used.
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
        """Build from a shapefile / GeoPackage / GeoJSON via geopandas."""
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
        Rectangular windows centred on points, e.g. named point sources.

        ``size`` is the window width, or ``(width, height)``, in ``crs`` units.
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
        """Every cell of a rectilinear grid as a polygon, in ``grid.index`` order."""
        x, y = grid.cells
        boxes = shapely.box(
            x - grid.xres / 2, y - grid.yres / 2, x + grid.xres / 2, y + grid.yres / 2
        )
        labels = tuple(f"{xi:g},{yi:g}" for xi, yi in zip(x, y, strict=True))
        return cls(ids=labels, geometries=tuple(boxes.tolist()), crs=grid.projection)

    @classmethod
    def from_h3(cls, resolution: int, bounds) -> Mesh:
        """
        H3 hexagons of ``resolution`` covering ``bounds`` (requires ``h3``).

        ``bounds`` is a :class:`~stilt.config.Bounds`/``Grid`` or an
        ``(xmin, ymin, xmax, ymax)`` tuple in lon/lat degrees.  Cell ids are
        the H3 cell strings; coordinates are lon/lat.
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
        """State index: the cell ids, named ``"cell"``."""
        return pd.Index(list(self.ids), name="cell")

    @property
    def is_longlat(self) -> bool:
        """``True`` when polygon coordinates are lon/lat degrees."""
        return is_longlat_crs(self.crs)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` envelope of all cells."""
        b = shapely.total_bounds(np.asarray(self.geometries, dtype=object))
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    @property
    def min_cell_width(self) -> float:
        """Smallest cell envelope side, in ``crs`` units (sets raster resolution)."""
        b = shapely.bounds(np.asarray(self.geometries, dtype=object))
        widths = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        return float(widths.min())

    @property
    def hash(self) -> str:
        """10-char SHA-256 of ids, WKB geometries and CRS."""
        h = hashlib.sha256()
        h.update(self.crs.encode())
        for i, g in zip(self.ids, self.geometries, strict=True):
            h.update(i.encode())
            h.update(shapely.to_wkb(g))
        return h.hexdigest()[:10]

    def to_crs(self, crs: str) -> Mesh:
        """Return this mesh reprojected to ``crs`` (requires pyproj)."""
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
    Super-cells: labels that merge the cells of a ``Grid`` or ``Mesh``.

    Parameters
    ----------
    base : Grid | Mesh
        The geometry being merged.
    labels : sequence of str
        One label per base cell, in ``base.index`` order.  Cells sharing a
        label form one super-cell; the state index lists labels in order of
        first appearance.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    base: Grid | Mesh
    labels: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Zones:
        n = len(self.base.index)
        if len(self.labels) != n:
            raise ValueError(
                f"labels must have one entry per base cell ({n}), got {len(self.labels)}."
            )
        return self

    @classmethod
    def from_labels(cls, base: Grid | Mesh, labels: Sequence[Any]) -> Zones:
        """Build from any label sequence (values are stringified)."""
        return cls(base=base, labels=tuple(str(v) for v in labels))

    def __len__(self) -> int:
        return len(self.index)

    @property
    def index(self) -> pd.Index:
        """Unique labels in order of first appearance, named ``"cell"``."""
        return pd.Index(pd.unique(np.asarray(self.labels, dtype=object)), name="cell")

    @property
    def crs(self) -> str:
        """CRS of the base geometry."""
        return self.base.projection if isinstance(self.base, Grid) else self.base.crs

    @property
    def is_longlat(self) -> bool:
        """``True`` when the base geometry is lon/lat."""
        return self.base.is_longlat

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Envelope of the base geometry."""
        if isinstance(self.base, Grid):
            b = self.base
            return b.xmin, b.ymin, b.xmax, b.ymax
        return self.base.bounds

    @property
    def min_cell_width(self) -> float:
        """Smallest base cell width (super-cells are never smaller)."""
        if isinstance(self.base, Grid):
            return float(min(self.base.xres, self.base.yres))
        return self.base.min_cell_width

    @property
    def membership(self) -> sparse.csr_matrix:
        """``(n_groups, n_base)`` indicator matrix mapping base cells to labels."""
        codes = self.index.get_indexer(list(self.labels))
        n_base = len(self.labels)
        return sparse.csr_matrix(
            (np.ones(n_base), (codes, np.arange(n_base))),
            shape=(len(self.index), n_base),
        )

    @property
    def hash(self) -> str:
        """10-char SHA-256 of the base geometry and the labels."""
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
"""A state geometry a footprint can be aggregated onto."""

SpatialTarget = Geometry | xr.DataArray | xr.Dataset | list[tuple[float, float]]
"""Every form :meth:`stilt.Footprint.aggregate` accepts as a target."""


def _geometry_key(geometry: Geometry) -> str:
    if isinstance(geometry, Grid):
        return "grid:" + geometry.model_dump_json()
    return f"{type(geometry).__name__.lower()}:{geometry.hash}"


def _raster_key(
    x: np.ndarray, y: np.ndarray, xres: float, yres: float, crs: str
) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=float).tobytes())
    h.update(np.ascontiguousarray(y, dtype=float).tobytes())
    h.update(f"{xres!r}|{yres!r}|{crs}".encode())
    return h.hexdigest()[:16]


_weight_cache: dict[tuple[str, str, str], sparse.csr_matrix] = {}


def _overlap_1d(src_edges: np.ndarray, dst_edges: np.ndarray) -> sparse.csr_matrix:
    """Per-axis fraction of each source cell inside each destination cell."""
    s_lo, s_hi = src_edges[:-1], src_edges[1:]
    d_lo, d_hi = dst_edges[:-1], dst_edges[1:]
    lo = np.maximum(s_lo[None, :], d_lo[:, None])
    hi = np.minimum(s_hi[None, :], d_hi[:, None])
    frac = np.clip(hi - lo, 0.0, None) / (s_hi - s_lo)[None, :]
    return sparse.csr_matrix(frac)


def _edges(centers: np.ndarray, res: float) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    return np.concatenate(([c[0] - res / 2], c + res / 2))


def _grid_weights(
    grid: Grid, x: np.ndarray, y: np.ndarray, xres: float, yres: float
) -> sparse.csr_matrix:
    """Grid-to-raster weights: rows in ``grid.index`` order (x outer, y inner)."""
    tx, ty = grid.axes
    px = _overlap_1d(_edges(x, xres), _edges(tx, grid.xres))  # (Tx, Nx)
    py = _overlap_1d(_edges(y, yres), _edges(ty, grid.yres))  # (Ty, Ny)
    w = sparse.kron(py, px).tocsr()  # rows (ty, tx); cols (iy, ix)
    n_tx, n_ty = len(tx), len(ty)
    # reorder rows from (ty outer, tx inner) to (tx outer, ty inner)
    order = (np.arange(n_tx)[:, None] + n_tx * np.arange(n_ty)[None, :]).ravel()
    return w[order]


def _exactextract_available() -> bool:
    try:
        import exactextract  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


def _mesh_weights_exactextract(
    mesh: Mesh, x: np.ndarray, y: np.ndarray, xres: float, yres: float
) -> sparse.csr_matrix:
    """
    Polygon-to-raster weights via ``exactextract`` (C++ exact area fractions).

    Same result as :func:`_mesh_weights` but ~100x faster on large rasters.
    ``exactextract`` numbers raster cells row-major from the *top* row, so
    its ``cell_id`` is remapped to the y-ascending flatten used here.
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
    """Polygon-to-raster weights by exact area intersection (shapely)."""
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


Backend = Literal["auto", "shapely", "exactextract"]


def _polygon_weights(
    mesh: Mesh, x: np.ndarray, y: np.ndarray, xres: float, yres: float, backend: str
) -> sparse.csr_matrix:
    if backend == "auto":
        backend = "exactextract" if _exactextract_available() else "shapely"
    if backend == "exactextract":
        return _mesh_weights_exactextract(mesh, x, y, xres, yres)
    if backend == "shapely":
        return _mesh_weights(mesh, x, y, xres, yres)
    raise ValueError(f"Unknown overlap backend {backend!r}.")


def overlap_weights(
    geometry: Geometry,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    xres: float,
    yres: float,
    crs: str,
    *,
    backend: Backend = "auto",
) -> sparse.csr_matrix:
    """
    Sparse ``(n_cells, ny * nx)`` fraction of each raster cell inside each cell.

    The raster is described by its cell-centre axes, resolution and CRS; its
    cells are flattened y-outer / x-inner (the order of
    ``data.transpose("time", y, x).reshape(nt, -1)``).  ``geometry`` is
    reprojected to the raster CRS when needed.  Results are cached on the
    (raster, geometry, backend) triple so repeated aggregations are a single
    matmul.

    Polygon overlaps (``Mesh``, and ``Grid`` in another CRS) use
    `exactextract <https://github.com/isciences/exactextract>`_ when it is
    installed (``backend="auto"``, roughly 100x faster on large rasters) and
    fall back to shapely otherwise.  It is not a dependency; ``pip install
    exactextract`` enables it.  Both backends give the same fractions.
    """
    x = np.asarray(x_centers, dtype=float)
    y = np.asarray(y_centers, dtype=float)
    key = (_raster_key(x, y, xres, yres, crs), _geometry_key(geometry), backend)
    cached = _weight_cache.get(key)
    if cached is not None:
        return cached

    if isinstance(geometry, Zones):
        base_w = overlap_weights(geometry.base, x, y, xres, yres, crs, backend=backend)
        w = sparse.csr_matrix(geometry.membership @ base_w)
    elif isinstance(geometry, Grid):
        if same_crs(geometry.projection, crs):
            w = _grid_weights(geometry, x, y, xres, yres)
        else:
            w = _polygon_weights(
                Mesh.from_grid(geometry).to_crs(crs), x, y, xres, yres, backend
            )
    else:
        w = _polygon_weights(geometry.to_crs(crs), x, y, xres, yres, backend)

    _weight_cache[key] = w
    return w


def check_resolution(geometry: Geometry, xres: float, yres: float, crs: str) -> None:
    """Warn when the raster is too coarse to resolve the smallest target cell."""
    width = geometry.min_cell_width
    if not same_crs(
        geometry.crs if not isinstance(geometry, Grid) else geometry.projection, crs
    ):
        return  # units differ; skip the heuristic rather than mislead
    if width < 2.0 * max(xres, yres):
        warnings.warn(
            f"Smallest target cell ({width:g}) spans fewer than two native raster "
            f"cells ({xres:g} x {yres:g}); the aggregate is under-resolved. "
            "Regenerate the footprint on a finer grid (Trajectories.footprint with "
            "Grid.from_geometry) for boundary accuracy.",
            stacklevel=3,
        )


__all__ = [
    "Backend",
    "Geometry",
    "Mesh",
    "Zones",
    "SpatialTarget",
    "check_resolution",
    "is_longlat_crs",
    "overlap_weights",
    "same_crs",
]
