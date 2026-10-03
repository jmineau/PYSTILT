"""
Where things are: a longitude/latitude box and the CRS helpers.

:class:`Bounds` is a longitude/latitude box, such as the crop box of a met.
:func:`is_longlat` and :func:`same_crs` are the CRS tests every part of
PYSTILT uses, and :func:`horizontal_dims` names the horizontal dimensions of
a gridded field. The footprint grid is :class:`stilt.footprint.grid.Grid`.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import xarray as xr


class Bounds(BaseModel):
    """Longitude/latitude bounding box, in degrees."""

    model_config = ConfigDict(frozen=True)

    xmin: float = Field(..., description="Western edge, in degrees longitude.")
    xmax: float = Field(..., description="Eastern edge, in degrees longitude.")
    ymin: float = Field(..., description="Southern edge, in degrees latitude.")
    ymax: float = Field(..., description="Northern edge, in degrees latitude.")


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


__all__ = [
    "Bounds",
    "horizontal_dims",
    "is_longlat",
    "same_crs",
]
