"""
Sampling gridded fields, such as a surface flux or a mole fraction, at points.

A field is an :class:`xarray.DataArray` on a regular ``lat``/``lon`` grid
(``y``/``x`` for a projected footprint grid), with an optional vertical
dimension and an optional ``time`` dimension. Each point takes the value of
the cell it falls in. ``foot.stilt.enhancement`` and
``particles.stilt.enhancement`` sample a surface flux this way, and
:func:`stilt.particles.background` a mole-fraction field. PYSTILT does
not convert units. A flux in µmol m⁻² s⁻¹ times a footprint in ppm per
(µmol m⁻² s⁻¹) gives ppm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from stilt.spatial import horizontal_dims


def nearest_cell(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Return the index of the cell nearest each value, or ``-1`` outside the cells.

    Each cell reaches halfway to its neighbors. The end cells reach the
    same distance beyond their centers. With a single cell, every finite
    value is inside it.

    Parameters
    ----------
    coords : numpy.ndarray
        Cell centers along one axis, ascending or descending.
    values : numpy.ndarray
        Positions to look up.

    Returns
    -------
    numpy.ndarray
        Index into *coords* for each value.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("Field coordinates must be a non-empty 1-D array.")
    ascending = coords[0] <= coords[-1]
    c = coords if ascending else coords[::-1]
    if c.size == 1:
        inside = np.isfinite(values)
        return np.where(inside, 0, -1)
    mids = (c[:-1] + c[1:]) / 2.0
    lo = c[0] - (c[1] - c[0]) / 2.0
    hi = c[-1] + (c[-1] - c[-2]) / 2.0
    idx = np.searchsorted(mids, values)
    inside = (values >= lo) & (values <= hi)
    idx = np.where(inside, idx, -1)
    if not ascending:
        idx = np.where(idx >= 0, c.size - 1 - idx, -1)
    return idx


def vertical_dim(field: xr.DataArray) -> str | None:
    """
    Return the name of a field's vertical dimension.

    That is the one dimension that is neither horizontal nor ``time``.
    Returns ``None`` for a field without one, such as a column mean or a
    surface field, and raises if there is more than one candidate.
    """
    y_dim, x_dim = horizontal_dims(field)
    extra = [str(d) for d in field.dims if d not in (y_dim, x_dim, "time")]
    if len(extra) > 1:
        raise ValueError(
            f"Expected at most one vertical dimension besides {y_dim!r}/{x_dim!r} "
            f"and 'time'; got {extra}."
        )
    return extra[0] if extra else None


def sample_field(
    field: xr.DataArray,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike | None = None,
    times: ArrayLike | None = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Return the field's value in the cell nearest each point.

    A point outside the field horizontally, or in a missing cell, gives
    ``fill_value``. Longitudes are wrapped to the field's convention (-180
    to 180 or 0 to 360).

    Parameters
    ----------
    field : xarray.DataArray
        Field with horizontal dimensions (``lat`` and ``lon``, or ``y`` and
        ``x``), and optionally a vertical dimension and ``time``.
    x, y : array-like
        Point coordinates: longitude and latitude for a ``lat``/``lon``
        field.
    z : array-like, optional
        Vertical coordinate of each point, in the units of the field's
        vertical dimension. Required when the field has one. Matched to the
        nearest level, so a point above the top level takes the top level.
    times : array-like, optional
        Time of each point. Required when the field has a ``time``
        dimension. Matched to the nearest time, so times outside the
        field's span take the first or last step.
    fill_value : float, default NaN
        Value for points outside the field and for missing cells. Use 0 for
        a surface flux, where no data means no emission. ``NaN`` suits a
        mole fraction, which is unknown there.

    Returns
    -------
    numpy.ndarray
        One value per point.
    """
    y_dim, x_dim = horizontal_dims(field)
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same length.")
    lons = field[x_dim].to_numpy()
    if x_dim == "lon":
        xs = xs % 360.0 if lons.max() > 180.0 else ((xs + 180.0) % 360.0) - 180.0
    ix = nearest_cell(lons, xs)
    iy = nearest_cell(field[y_dim].to_numpy(), ys)
    inside = (ix >= 0) & (iy >= 0)

    indexers: dict[str, xr.DataArray] = {
        x_dim: xr.DataArray(np.where(inside, ix, 0), dims="points"),
        y_dim: xr.DataArray(np.where(inside, iy, 0), dims="points"),
    }
    zdim = vertical_dim(field)
    if zdim is not None:
        if z is None:
            raise ValueError(f"field has a {zdim!r} dimension; pass z.")
        zs = np.asarray(z, dtype=float).ravel()
        if zs.shape != xs.shape:
            raise ValueError("z must have the same length as x and y.")
        indexers[zdim] = xr.DataArray(_nearest_level(field[zdim], zs), dims="points")
    if "time" in field.dims:
        if times is None:
            raise ValueError("field has a time dimension; pass times.")
        stamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(times).ravel()))
        if len(stamps) != len(xs):
            raise ValueError("times must have the same length as x and y.")
        it = field.indexes["time"].get_indexer(stamps, method="nearest")
        indexers["time"] = xr.DataArray(it, dims="points")
    sampled = field.isel(indexers).to_numpy().astype(float)
    return np.where(inside & ~np.isnan(sampled), sampled, fill_value)


def _nearest_level(levels: xr.DataArray, z: np.ndarray) -> np.ndarray:
    """Return the index of the level nearest each ``z``."""
    coords = levels.to_numpy().astype(float)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("The vertical coordinate must be a non-empty 1-D array.")
    return np.abs(coords[None, :] - z[:, None]).argmin(axis=1)


__all__ = [
    "nearest_cell",
    "sample_field",
    "vertical_dim",
]
