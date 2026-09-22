"""
Background mole fraction at a receptor, from a field sampled at trajectory endpoints.

A back-trajectory ends where the receptor's air came from. Sampling a
mole-fraction field (a global model such as CarbonTracker or CAMS, or an
observed curtain) at every particle's endpoint and averaging over the
particles gives the background: what the receptor would see with no fluxes
inside the domain. Adding the modelled enhancement gives the modelled mole
fraction. X-STILT does the same per particle in ``endpts.trajfoot``, and
CT-STILT is the same idea with CarbonTracker.

The average is weighted the way the footprint is. The particle transforms
that weight the enhancement (averaging kernel, pressure weighting, lifetime
decay) weight the background too, so the two add. Readers stay outside
PYSTILT: the field comes in as an :class:`xarray.DataArray`, or already
sampled as one value per particle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from stilt.flux import horizontal_dims, nearest_cell
from stilt.trajectory import endpoint_rows
from stilt.transforms import TransformContext, apply_transforms


@dataclass(frozen=True)
class Background:
    """
    Result of :func:`background`.

    ``value`` is the background at the receptor, weighted like the footprint:
    ``Σ weights × per_particle``. ``per_particle`` is the field at each
    particle's endpoint, indexed by ``indx``; ``NaN`` where the endpoint
    lies outside the field. ``weights`` is each particle's share, indexed
    the same way: ``1 / N`` each without transforms, so they sum to one; with
    pressure weighting they sum to the fraction of the column's air mass the
    particles cover. Particles without a value are left out of ``value`` and
    the others carry their weight, as if they had the same mean.
    """

    value: float
    per_particle: pd.Series
    weights: pd.Series


def vertical_dim(field: xr.DataArray) -> str | None:
    """
    The field's vertical dimension: the one that is not horizontal or ``time``.

    ``None`` for a field with no vertical dimension (a column mean or a surface
    field). More than one candidate is an error.
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
) -> np.ndarray:
    """
    Field value at the cell nearest each ``(x, y[, z][, time])`` point.

    ``x`` / ``y`` are longitudes / latitudes for a ``lat`` / ``lon`` field.
    ``z`` is required when the field has a vertical dimension (see
    :func:`vertical_dim`) and is in that coordinate's units; it is matched to
    the nearest level and held at the ends, so a point above the top level
    takes the top level. ``times`` is required when the field has a ``time``
    dimension and is matched the same way. Horizontally, a point outside the
    field's cells is ``NaN``: a missing mole fraction is not zero, unlike a
    missing flux in :func:`stilt.flux.sample_flux`. Longitudes are wrapped
    into the field's convention (``-180..180`` or ``0..360``).
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
    return np.where(inside, sampled, np.nan)


def _nearest_level(levels: xr.DataArray, z: np.ndarray) -> np.ndarray:
    """Index of the level nearest each ``z``, in either ordering, clamped to the ends."""
    coords = levels.to_numpy().astype(float)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("The vertical coordinate must be a non-empty 1-D array.")
    return np.abs(coords[None, :] - z[:, None]).argmin(axis=1)


def particle_background(particles: pd.DataFrame, field: xr.DataArray) -> pd.Series:
    """
    The field at each particle's endpoint, indexed by ``indx``.

    The endpoint is the row farthest in time from release
    (:func:`stilt.trajectory.endpoint_rows`). The field's vertical dimension,
    if any, must be named after the particle column it is matched against:
    ``pres`` for pressure in hPa or ``zagl`` for height above ground in
    metres (rename it with ``field.rename(level="pres")``), or a column you
    add, such as height above sea level from ``zagl + zsfc``. A time-varying
    field is sampled at the endpoint's ``datetime``.
    """
    ends = endpoint_rows(particles)
    zdim = vertical_dim(field)
    z = None
    if zdim is not None:
        if zdim not in ends.columns:
            raise ValueError(
                f"The field's vertical dimension {zdim!r} is not a particle column. "
                "Name it after the column to match it against ('pres' or 'zagl'), "
                "or add that column to the particles."
            )
        z = ends[zdim].to_numpy(dtype=float)
    times = None
    if "time" in field.dims:
        if "datetime" not in ends.columns:
            raise ValueError(
                "field varies in time but the particles have no 'datetime' column."
            )
        times = ends["datetime"].to_numpy()
    values = sample_field(
        field, ends["long"].to_numpy(), ends["lati"].to_numpy(), z=z, times=times
    )
    return pd.Series(
        values, index=pd.Index(ends["indx"].to_numpy(), name="indx"), name="background"
    )


def endpoint_weights(
    particles: pd.DataFrame,
    transforms: Sequence[Any] = (),
    context: TransformContext | None = None,
) -> pd.Series:
    """
    Each particle's weight at its endpoint after the transforms, indexed by ``indx``.

    A transform is a multiplicative factor on ``foot``, so applying the
    transforms to a table whose ``foot`` is one everywhere leaves that factor
    behind: the averaging kernel and pressure weight of the particle, and the
    lifetime decay at its endpoint age. Without transforms every weight is
    one. Divided by the particle count, these are the weights
    :meth:`stilt.Footprint.calculate` gives the particles.
    """
    transforms = list(transforms)
    if transforms:
        if context is None:
            context = default_context()
        particles = apply_transforms(particles.assign(foot=1.0), transforms, context)
        ends = endpoint_rows(particles)
        weights = ends["foot"].to_numpy(dtype=float)
    else:
        ends = endpoint_rows(particles)
        weights = np.ones(len(ends))
    return pd.Series(
        weights, index=pd.Index(ends["indx"].to_numpy(), name="indx"), name="weight"
    )


def fill_missing(per_particle: pd.Series, weights: pd.Series) -> pd.Series:
    """
    Replace ``NaN`` per-particle values with the weighted mean of the others.

    A particle whose endpoint lies outside the field then neither adds to nor
    dilutes the background; the result is ``NaN`` everywhere when no particle
    has a value.
    """
    values = per_particle.reindex(weights.index).to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    ok = np.isfinite(values)
    if ok.all():
        return pd.Series(values, index=weights.index, name=per_particle.name)
    total = w[ok].sum()
    mean = (w[ok] * values[ok]).sum() / total if total > 0 else np.nan
    return pd.Series(
        np.where(ok, values, mean), index=weights.index, name=per_particle.name
    )


def background(
    particles: pd.DataFrame,
    field: xr.DataArray | pd.Series,
    *,
    transforms: Sequence[Any] = (),
    context: TransformContext | None = None,
) -> Background:
    """
    Background mole fraction at the receptor, from the field at the trajectory endpoints.

    Parameters
    ----------
    particles
        The simulation's particle table (``sim.trajectories.data``).
    field
        The background field (see :func:`particle_background` for its
        layout), or one value per particle that you sampled yourself, as a
        Series indexed by ``indx``: for example lair's
        ``CarbonTracker.sample`` on ``sim.trajectories.endpoints()``.
    transforms, context
        The footprint's particle transforms and the context to apply them
        with (``config.transforms`` and ``sim.transform_context(name)``), so
        the background is weighted the way the footprint is and adds to its
        enhancement. For a tower receptor there is nothing to pass.

    Notes
    -----
    Without transforms the value is the plain mean over particles. With
    pressure weighting the weights sum to the fraction of the column's air
    mass the particles cover, ``(p_sfc - p_top) / p_sfc``, the same fraction
    the enhancement covers; the part of the column above the receptor top is
    still yours to add from the same field.
    """
    if isinstance(field, pd.Series):
        per_particle = field.rename("background")
    else:
        per_particle = particle_background(particles, field)
    weights = endpoint_weights(particles, transforms, context)
    weights = weights / len(weights)
    filled = fill_missing(per_particle, weights)
    value = float((weights * filled).sum()) if np.isfinite(filled).any() else np.nan
    return Background(
        value=value, per_particle=per_particle.reindex(weights.index), weights=weights
    )


def default_context() -> TransformContext:
    """
    A placeholder context for transforms that do not read it.

    Transforms that do (an averaging-kernel ``table``) need the real one from
    ``sim.transform_context(name)``.
    """
    from stilt.receptors import PointReceptor

    return TransformContext(receptor=PointReceptor("2000-01-01", 0.0, 0.0, 0.0))


__all__ = [
    "Background",
    "background",
    "endpoint_weights",
    "fill_missing",
    "particle_background",
    "sample_field",
    "vertical_dim",
]
