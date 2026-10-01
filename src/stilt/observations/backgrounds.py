"""
Background mole fraction at a receptor, from a field sampled where the particles end.

A back-trajectory ends where the receptor's air came from. Sampling a
mole-fraction field, such as CarbonTracker or CAMS, at every particle's
endpoint and averaging over the particles gives the background: what the
receptor would see with no fluxes inside the domain. Adding the modeled
enhancement gives the modeled mole fraction. X-STILT does the same in
``endpts.trajfoot``, and CT-STILT does it with CarbonTracker.

The average is weighted the way the footprint is, so the particle
transforms (averaging kernel, pressure weighting, lifetime decay) apply to
the background too, and the background and enhancement add. The field is
passed in as an :class:`xarray.DataArray`, or already sampled as one value
per particle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from stilt.flux import sample_field, vertical_dim
from stilt.transforms import TransformContext, apply_transforms


@dataclass(frozen=True)
class Background:
    """
    Background at a receptor, returned by :func:`background`.

    Attributes
    ----------
    value : float
        Background at the receptor, weighted like the footprint:
        ``Σ weights × per_particle``. Particles with no value count as the
        weighted mean of the others.
    per_particle : pandas.Series
        Field value at each particle's endpoint, indexed by ``indx``. ``NaN``
        where the endpoint is outside the field.
    weights : pandas.Series
        Each particle's weight, indexed by ``indx``. Without transforms each
        is ``1 / N`` and they sum to one. With pressure weighting they sum to
        the fraction of the atmosphere's mass inside the column.
    """

    value: float
    per_particle: pd.Series
    weights: pd.Series


def particle_background(particles: pd.DataFrame, field: xr.DataArray) -> pd.Series:
    """
    Return the field at each particle's endpoint, indexed by ``indx``.

    The endpoint is the row farthest in time from release
    (``particles.stilt.endpoints()``). A vertical dimension must be
    named after the particle column it is matched against: ``pres`` for
    pressure in hPa, ``zagl`` for height above ground in meters, or a column
    you add, such as height above sea level from ``zagl + zsfc``. Rename it
    with, for example, ``field.rename(level="pres")``. A field with a
    ``time`` dimension is sampled at the endpoint's ``datetime``.
    """
    ends = particles.stilt.endpoints()
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
    Return each particle's transform weight at its endpoint, indexed by ``indx``.

    Transforms multiply ``foot``, so applying them to particles whose
    ``foot`` is 1 leaves each particle's weight: its averaging kernel and
    pressure weight, and the lifetime decay at its endpoint age. Without
    transforms every weight is 1. Divided by the particle count, these are
    the weights :func:`stilt.footprint.calculate` gives the particles.
    """
    transforms = list(transforms)
    if transforms:
        if context is None:
            context = default_context()
        particles = apply_transforms(particles.assign(foot=1.0), transforms, context)
        ends = particles.stilt.endpoints()
        weights = ends["foot"].to_numpy(dtype=float)
    else:
        ends = particles.stilt.endpoints()
        weights = np.ones(len(ends))
    return pd.Series(
        weights, index=pd.Index(ends["indx"].to_numpy(), name="indx"), name="weight"
    )


def fill_missing(per_particle: pd.Series, weights: pd.Series) -> pd.Series:
    """
    Replace missing per-particle values with the weighted mean of the others.

    A particle whose endpoint is outside the field then does not change the
    background. The result is ``NaN`` everywhere when no particle has a
    value.
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
    Return the background mole fraction at a receptor.

    The background is the field at each particle's endpoint, averaged over
    the particles with the footprint's weights.

    Parameters
    ----------
    particles : pandas.DataFrame
        The simulation's particle table (``sim.particles``).
    field : xarray.DataArray or pandas.Series
        The background field (see :func:`particle_background` for its
        layout), or one value per particle that you sampled yourself, as a
        Series indexed by ``indx``. For example, lair's
        ``CarbonTracker.sample`` on ``sim.particles.stilt.endpoints()``.
    transforms : sequence, optional
        The footprint's particle transforms (``sim.variant.footprint.transforms``), so
        the background is weighted like the footprint and adds to its
        enhancement. A tower receptor has none.
    context : TransformContext, optional
        Context to apply the transforms with
        (``project.transform_context(sim)``). Required by an averaging kernel read
        from a table.

    Returns
    -------
    Background

    Notes
    -----
    Without transforms the value is the mean over particles. With pressure
    weighting the weights sum to the fraction of the atmosphere's mass inside
    the column, ``(p_sfc - p_top) / p_sfc``, the same fraction the
    enhancement covers. Add the part of the column above its top from the
    same field.
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
    Return a placeholder context for transforms that do not read it.

    An averaging kernel read from a ``table`` needs the real context from
    ``project.transform_context(sim)``.
    """
    import datetime as dt

    from stilt.receptors import PointReceptor

    return TransformContext(
        receptor=PointReceptor(
            time=dt.datetime(2000, 1, 1), longitude=0.0, latitude=0.0, altitude=0.0
        )
    )


__all__ = [
    "Background",
    "background",
    "endpoint_weights",
    "fill_missing",
    "particle_background",
]
