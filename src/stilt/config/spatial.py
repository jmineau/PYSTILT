"""Spatial config models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xarray as xr

VerticalReference = Literal["agl", "msl"]


def validate_vertical_reference(reference: str) -> VerticalReference:
    """Return a normalized vertical reference or raise for invalid input."""
    normalized = reference.lower()
    if normalized not in {"agl", "msl"}:
        raise ValueError(
            f"Vertical reference must be 'agl' or 'msl'. Got {reference!r}."
        )
    return cast(VerticalReference, normalized)


def kmsl_from_vertical_reference(reference: VerticalReference) -> int:
    """Map a vertical reference onto the HYSPLIT ``KMSL`` control value."""
    return 0 if reference == "agl" else 1


class Bounds(BaseModel):
    """Immutable geographic bounding box (always lon/lat degrees)."""

    model_config = ConfigDict(frozen=True)

    xmin: float = Field(..., description="Western longitude (degrees).")
    xmax: float = Field(..., description="Eastern longitude (degrees).")
    ymin: float = Field(..., description="Southern latitude (degrees).")
    ymax: float = Field(..., description="Northern latitude (degrees).")


class Grid(Bounds):
    """Footprint grid: lon/lat bounds, optional output projection, cell resolution."""

    model_config = ConfigDict(frozen=True)

    xres: float = Field(
        ...,
        description="Cell width in projection units (degrees for longlat, metres for UTM, etc.).",
    )
    yres: float = Field(
        ...,
        description="Cell height in projection units (degrees for longlat, metres for UTM, etc.).",
    )
    projection: str = Field(
        "+proj=longlat",
        description=(
            "Output CRS as a PROJ string.  Bounds are always lon/lat; "
            "particles and bounds are projected to this CRS before gridding."
        ),
    )

    @property
    def resolution(self) -> str:
        """Human-readable cell resolution string, e.g. ``'0.01x0.01'``."""
        return f"{self.xres}x{self.yres}"

    @property
    def is_longlat(self) -> bool:
        """``True`` when the output grid is geographic (lon/lat degrees)."""
        return "+proj=longlat" in self.projection

    @property
    def min_cell_width(self) -> float:
        """Smaller of ``xres`` and ``yres`` (parity with other geometries)."""
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
        Derive the native raster that resolves a non-rectilinear geometry.

        Bounds are the geometry's envelope snapped outward to whole cells.
        Resolution is the smallest target cell width divided by
        ``cells_per_target``, rounded *down* to one significant figure so
        file names and hashes stay readable.  ``projection`` defaults to the
        geometry's CRS.  Warns when the raster would exceed ``max_cells``.

        Parameters
        ----------
        geometry : Mesh | Zones | Grid
            Target geometry (any object with ``bounds``, ``min_cell_width``,
            ``crs``/``projection`` and ``is_longlat``).
        cells_per_target : float, default 4
            Native cells across the smallest target cell.
        projection : str, optional
            Output CRS; the geometry is reprojected when it differs.  Bounds
            are always stored in lon/lat.
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
        """One raster that resolves several geometries: union envelope, finest cell."""
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
        1-D cell-centre coordinates ``(x, y)`` in projection units, ascending.

        These are the coordinates of the footprint's native raster for this
        grid, rounded to 10 decimals to strip floating-point noise
        (``40.35`` rather than ``40.349999999999994``) so that ``index``
        matches coordinate labels built elsewhere from the same bounds and
        resolution exactly.  ``pyproj`` is required only for non-longlat
        projections.
        """
        import numpy as np

        from stilt.footprint import _grid_cell_starts

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
        """Every cell centre as flat ``(x, y)`` arrays, x outer / y inner."""
        import numpy as np

        x, y = self.axes
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return xx.ravel(), yy.ravel()

    @property
    def index(self) -> pd.MultiIndex:
        """
        State index over all cells, in the same order as ``cells``.

        Named ``("lon", "lat")`` for geographic grids and ``("x", "y")`` for
        projected ones, matching the coordinate names of the footprint raster.
        """
        import pandas as pd

        x, y = self.axes
        names = ["lon", "lat"] if self.is_longlat else ["x", "y"]
        return pd.MultiIndex.from_product([x, y], names=names)

    def to_xarray(self) -> xr.Dataset:
        """
        Return this grid as a CF-style xarray Dataset of cell centers.

        The result carries 1-D ``lon``/``lat`` (or projected ``x``/``y``)
        coordinates matching the footprint's native output grid, plus a ``crs``
        grid-mapping variable.  This is the interchange form of the grid: pass it
        as the target of :meth:`stilt.Footprint.aggregate`, or hand it to other
        tools via the shared xarray/CF grid convention.

        ``pyproj`` is required only for non-longlat projections.
        """
        import xarray as xr

        from stilt.footprint import _cf_grid_mapping_attrs

        is_longlat = self.is_longlat
        x_centers, y_centers = self.axes
        x_dim, y_dim = ("lon", "lat") if is_longlat else ("x", "y")

        ds = xr.Dataset(coords={x_dim: x_centers, y_dim: y_centers})
        ds.attrs["Conventions"] = "CF-1.8"
        ds["crs"] = xr.DataArray(0, attrs=_cf_grid_mapping_attrs(self.projection))
        if is_longlat:
            ds[x_dim].attrs.update(
                standard_name="longitude",
                long_name="longitude",
                units="degrees_east",
                axis="X",
            )
            ds[y_dim].attrs.update(
                standard_name="latitude",
                long_name="latitude",
                units="degrees_north",
                axis="Y",
            )
        else:
            ds[x_dim].attrs.update(
                standard_name="projection_x_coordinate",
                long_name="x coordinate of projection",
                units="m",
                axis="X",
            )
            ds[y_dim].attrs.update(
                standard_name="projection_y_coordinate",
                long_name="y coordinate of projection",
                units="m",
                axis="Y",
            )
        return ds


__all__ = ["Bounds", "Grid"]
