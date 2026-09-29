"""
Sampling a surface flux field along particles and under footprints.

A flux field is an :class:`xarray.DataArray` on a regular ``lat``/``lon``
grid (``y``/``x`` for a projected footprint grid), with an optional ``time``
dimension. Each point takes the value of the flux cell it falls in, and a
point outside the field gets zero. PYSTILT does not convert units. A flux
in µmol m⁻² s⁻¹ times a footprint in ppm per (µmol m⁻² s⁻¹) gives ppm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

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


def nearest_cell(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Return the index of the cell nearest each value, or ``-1`` outside the cells.

    Each cell reaches halfway to its neighbours. The end cells reach the
    same distance beyond their centres. With a single cell, every finite
    value is inside it.

    Parameters
    ----------
    coords : numpy.ndarray
        Cell centres along one axis, ascending or descending.
    values : numpy.ndarray
        Positions to look up.

    Returns
    -------
    numpy.ndarray
        Index into *coords* for each value.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("Flux coordinates must be a non-empty 1-D array.")
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


def sample_flux(
    flux: xr.DataArray,
    x: ArrayLike,
    y: ArrayLike,
    times: ArrayLike | None = None,
) -> np.ndarray:
    """
    Return the flux at each point, or zero outside the field.

    Parameters
    ----------
    flux : xarray.DataArray
        Flux on a ``lat``/``lon`` or ``y``/``x`` grid, with an optional
        ``time`` dimension. NaN values count as zero.
    x, y : array-like
        Point coordinates, longitude and latitude for a ``lat``/``lon``
        field.
    times : array-like, optional
        Time of each point. Required when *flux* has a ``time`` dimension.
        Each point takes the nearest time step, so times outside the
        field's span take the first or last one.

    Returns
    -------
    numpy.ndarray
        Flux at each point, in the flux's units.

    Raises
    ------
    ValueError
        If the lengths of *x*, *y*, and *times* differ, or *flux* has a
        ``time`` dimension and *times* is not given.
    """
    y_dim, x_dim = horizontal_dims(flux)
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same length.")
    ix = nearest_cell(flux[x_dim].to_numpy(), xs)
    iy = nearest_cell(flux[y_dim].to_numpy(), ys)
    inside = (ix >= 0) & (iy >= 0)

    indexers: dict[str, xr.DataArray] = {
        x_dim: xr.DataArray(np.where(inside, ix, 0), dims="points"),
        y_dim: xr.DataArray(np.where(inside, iy, 0), dims="points"),
    }
    if "time" in flux.dims:
        if times is None:
            raise ValueError("flux has a time dimension; pass times.")
        stamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(times).ravel()))
        if len(stamps) != len(xs):
            raise ValueError("times must have the same length as x and y.")
        it = flux.indexes["time"].get_indexer(stamps, method="nearest")
        indexers["time"] = xr.DataArray(it, dims="points")
    elif times is not None and np.size(times) != len(xs):
        raise ValueError("times must have the same length as x and y.")
    sampled = flux.isel(indexers).to_numpy().astype(float)
    return np.where(inside, np.nan_to_num(sampled), 0.0)


def particle_enhancement(particles: pd.DataFrame, flux: xr.DataArray) -> pd.Series:
    """
    Return each particle's enhancement, ``foot`` times flux summed along its trajectory.

    The mean over particles, after any weighting, is the modelled
    enhancement at the receptor. Unlike :meth:`stilt.Footprint.enhancement`,
    the flux is taken at each particle position, with no gridding or
    smoothing.

    Parameters
    ----------
    particles : pandas.DataFrame
        Particle table with ``indx``, ``long``, ``lati``, and ``foot``
        columns, and ``datetime`` when the flux varies in time.
    flux : xarray.DataArray
        Surface flux field. See :func:`sample_flux`.

    Returns
    -------
    pandas.Series
        Enhancement indexed by ``indx``, in the flux's units times the
        footprint's. A particle that never crosses the flux field gets 0.

    Raises
    ------
    ValueError
        If the flux varies in time and the particles have no ``datetime``
        column.
    """
    times = (
        particles["datetime"].to_numpy() if "datetime" in particles.columns else None
    )
    if "time" in flux.dims and times is None:
        raise ValueError(
            "flux varies in time but the particles have no 'datetime' column."
        )
    sampled = sample_flux(
        flux, particles["long"].to_numpy(), particles["lati"].to_numpy(), times
    )
    contribution = particles["foot"].to_numpy(dtype=float) * sampled
    indx = particles["indx"].to_numpy()
    unique, inverse = np.unique(indx, return_inverse=True)
    sums = np.bincount(inverse, weights=contribution, minlength=unique.size)
    return pd.Series(sums, index=pd.Index(unique, name="indx"), name="enhancement")


__all__ = ["horizontal_dims", "nearest_cell", "particle_enhancement", "sample_flux"]
