"""Spatial config models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict

from .fields import cfg_field

if TYPE_CHECKING:
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

    xmin: float = cfg_field(..., description="Western longitude (degrees).")
    xmax: float = cfg_field(..., description="Eastern longitude (degrees).")
    ymin: float = cfg_field(..., description="Southern latitude (degrees).")
    ymax: float = cfg_field(..., description="Northern latitude (degrees).")


class Grid(Bounds):
    """Footprint grid: lon/lat bounds, optional output projection, cell resolution."""

    model_config = ConfigDict(frozen=True)

    xres: float = cfg_field(
        ...,
        description="Cell width in projection units (degrees for longlat, metres for UTM, etc.).",
    )
    yres: float = cfg_field(
        ...,
        description="Cell height in projection units (degrees for longlat, metres for UTM, etc.).",
    )
    projection: str = cfg_field(
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
        import numpy as np
        import xarray as xr

        from stilt.footprint import _cf_grid_mapping_attrs, _grid_cell_starts

        is_longlat = "+proj=longlat" in self.projection
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        if not is_longlat:
            from pyproj import Transformer

            tr = Transformer.from_crs("EPSG:4326", self.projection, always_xy=True)
            corners_x, corners_y = tr.transform([xmin, xmax], [ymin, ymax])
            xmin, xmax = float(np.min(corners_x)), float(np.max(corners_x))
            ymin, ymax = float(np.min(corners_y)), float(np.max(corners_y))

        x_centers = _grid_cell_starts(xmin, xmax, self.xres) + self.xres / 2
        y_centers = _grid_cell_starts(ymin, ymax, self.yres) + self.yres / 2
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
