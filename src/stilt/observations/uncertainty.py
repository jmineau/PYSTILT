"""
Transport error on the modelled enhancement, from wind-perturbed trajectories.

The method is Lin and Gerbig (2005, GRL, doi:10.1029/2004GL021127). A
simulation run with wind-error settings (``siguverr`` and friends in
:class:`~stilt.config.STILTParams`) writes a second particle table whose
transport carries an extra random wind component with the statistics of the
meteorology's errors. Each particle's enhancement is its ``foot × flux``
summed along its trajectory. The perturbed particles sample more of the flux
field, so the variance of that enhancement across the ensemble is larger,
and the increase is the transport-error variance (their equation 4):

    var_transport = var(enhancement | perturbed) − var(enhancement | unperturbed)

For a column receptor the difference is taken per release level and the
levels are combined with the column weighting and a vertical error
correlation, following X-STILT (Wu et al., 2018, GMD). The difference of two
sample variances is noisy and can be negative; it is returned signed, with
an estimate of its own noise, so a batch of receptors can be aggregated with
a median and a single value can be judged against its noise floor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from stilt.flux import particle_enhancement
from stilt.observations.backgrounds import (
    default_context,
    endpoint_weights,
    fill_missing,
    particle_background,
)
from stilt.transforms import TransformContext, apply_transforms, release_coordinate

#: X-STILT's empirical mean vertical correlation length of transport errors, m.
DEFAULT_LENGTH_SCALE = 356.0


@dataclass(frozen=True)
class TransportError:
    """
    Result of :func:`transport_error`.

    All values are in the flux's units times the footprint's (ppm for a flux
    in µmol m⁻² s⁻¹).

    ``variance`` is the signed transport-error variance of the modelled
    enhancement: the extra ensemble variance the wind perturbation produced.
    A negative value is sampling noise. ``noise`` is the standard deviation
    of ``variance`` expected with no perturbation at all, estimated from
    random halves of the unperturbed particles; ``variance`` is resolved
    only when it is several times ``noise``. ``sd`` is ``sqrt(variance)``,
    or ``0`` when the variance is negative.

    ``enhancement`` and ``enhancement_perturbed`` are the modelled
    enhancement from the unperturbed and the perturbed particles. ``levels``
    has one row per release level: ``height`` (m, mean release height),
    ``n`` particles, ``weight`` (its share of the column), ``mean`` / ``var``
    of the per-particle enhancement without (``_orig``) and with (``_err``)
    the perturbation, ``dvar`` their difference, and ``sd_trans`` the signed
    square root of ``dvar`` (or X-STILT's regression-scaled value with
    ``regression=True``).

    With a ``background`` field the per-particle values are the modelled
    mole fraction, enhancement plus background at the particle's endpoint,
    so ``enhancement`` and ``enhancement_perturbed`` are then modelled mole
    fractions and ``variance`` includes the background's response to the
    wind errors. ``background`` is the weighted background from the
    unperturbed particles (``0`` when no field was given), so
    ``enhancement - background`` is the enhancement alone.
    """

    variance: float
    noise: float
    enhancement: float
    enhancement_perturbed: float
    levels: pd.DataFrame
    length_scale: float | None
    background: float = 0.0

    @property
    def sd(self) -> float:
        """Transport-error standard deviation; ``0`` when ``variance`` is negative."""
        return float(np.sqrt(max(self.variance, 0.0)))


def _level_bins(
    heights: pd.Series, levels: int | Sequence[float]
) -> tuple[pd.Series, pd.Series]:
    """Assign each particle a level label; return ``(label, level_height)``."""
    if not isinstance(levels, int):
        edges = np.asarray(levels, dtype=float)
        label = pd.cut(heights, edges, labels=False, include_lowest=True)
    else:
        if levels < 1:
            raise ValueError("levels must be >= 1.")
        unique = np.unique(heights.to_numpy())
        if unique.size <= levels:
            label = pd.Series(
                np.searchsorted(unique, heights.to_numpy()), index=heights.index
            )
        else:
            label = pd.cut(heights, levels, labels=False, include_lowest=True)
    lab = np.asarray(label, dtype=float)
    hgt = heights.to_numpy(dtype=float)
    uniq = np.unique(lab[np.isfinite(lab)])
    level_height = pd.Series([float(hgt[lab == u].mean()) for u in uniq], index=uniq)
    return pd.Series(lab, index=heights.index), level_height


def _level_stats(values: np.ndarray, percentile: float) -> tuple[float, float]:
    """Mean of all values, and population variance after dropping those above ``percentile``."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.nan, np.nan
    mean = float(v.mean())
    if percentile < 1.0:
        v = v[v <= np.quantile(v, percentile)]
    if v.size < 2:
        return mean, np.nan
    return mean, float(v.var(ddof=0))


def _scale_dvar(levels: pd.DataFrame) -> np.ndarray:
    """
    X-STILT's regression-scaled transport-error sd per level (Wu et al., 2018).

    Over the levels where ``var_err − var_orig`` is positive, ``var_err`` is
    regressed on ``var_orig`` (weighted by ``1 / var_err``), the fitted line
    is evaluated at every level, and its excess over ``var_orig`` is the
    scaled transport-error variance. With fewer than two positive levels the
    raw difference is used, clipped at zero.

    Selecting the positive levels biases the slope above one, so under pure
    sampling noise this reports a positive error at every level. It is kept
    for reproducing X-STILT results and is off by default.
    """
    var_orig = levels["var_orig"].to_numpy(dtype=float)
    var_err = levels["var_err"].to_numpy(dtype=float)
    dvar = var_err - var_orig
    positive = np.isfinite(dvar) & (dvar > 0) & (var_err > 0)
    if positive.sum() >= 2 and np.ptp(var_orig[positive]) > 0:
        slope, intercept = np.polyfit(
            var_orig[positive], var_err[positive], 1, w=1.0 / np.sqrt(var_err[positive])
        )
        scaled = slope * var_orig + intercept - var_orig
    else:
        scaled = dvar
    return np.sqrt(np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, None))


def _signed_sqrt(values: np.ndarray) -> np.ndarray:
    """Take the square root while keeping the sign of a signed variance."""
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    return np.sign(v) * np.sqrt(np.abs(v))


def _combine(
    levels: pd.DataFrame, length_scale: float | None, *, regression: bool
) -> float:
    """
    Column variance: ``Σ_i w_i² v_i + Σ_{i≠j} w_i w_j s_i s_j corr_ij``.

    ``s`` is the per-level ``sd_trans``. On the diagonal ``v`` is the signed
    variance difference itself (so a level whose spread fell contributes
    negatively), or ``s²`` for X-STILT's regression-scaled values, which are
    never negative. The cross terms use the signed square roots so correlated
    levels of like sign add and unlike sign cancel.
    """
    w = levels["weight"].to_numpy(dtype=float)
    s = levels["sd_trans"].to_numpy(dtype=float)
    h = levels["height"].to_numpy(dtype=float)
    if regression:
        diag = s**2
    else:
        diag = np.nan_to_num(levels["dvar"].to_numpy(dtype=float), nan=0.0)
    if length_scale is None:
        corr = np.eye(len(h))
    else:
        corr = np.exp(-np.abs(h[:, None] - h[None, :]) / length_scale)
    prod = w[:, None] * w[None, :] * s[:, None] * s[None, :] * corr
    np.fill_diagonal(prod, w**2 * diag)
    return float(prod.sum())


def _level_table(
    x_orig: pd.Series,
    x_err: pd.Series,
    label_orig: pd.Series,
    label_err: pd.Series,
    level_height: pd.Series,
    *,
    percentile: float,
    regression: bool,
) -> pd.DataFrame:
    """Build the per-release-level table of means, variances and weights."""
    rows = []
    n_total = len(x_orig)
    for lvl, height in level_height.items():
        idx_o = label_orig.index.to_numpy()[(label_orig == lvl).to_numpy()]
        idx_e = label_err.index.to_numpy()[(label_err == lvl).to_numpy()]
        mean_o, var_o = _level_stats(x_orig.reindex(idx_o).to_numpy(), percentile)
        mean_e, var_e = _level_stats(x_err.reindex(idx_e).to_numpy(), percentile)
        rows.append(
            {
                "height": float(height),
                "n": int(len(idx_o)),
                "weight": len(idx_o) / n_total,
                "mean_orig": mean_o,
                "mean_err": mean_e,
                "var_orig": var_o,
                "var_err": var_e,
            }
        )
    table = pd.DataFrame(rows)
    table["dvar"] = table["var_err"] - table["var_orig"]
    table["sd_trans"] = (
        _scale_dvar(table) if regression else _signed_sqrt(table["dvar"].to_numpy())
    )
    return table


def _noise(
    x_orig: pd.Series,
    label_orig: pd.Series,
    level_height: pd.Series,
    *,
    splits: int,
    percentile: float,
    regression: bool,
    length_scale: float | None,
) -> float:
    """
    Standard deviation of the estimator under no perturbation.

    The unperturbed particles are split into two random halves within each
    level and treated as the main and perturbed tables; the spread of that
    estimate over ``splits`` random splits, scaled from half to full
    ensembles by ``1/sqrt(2)``, is the noise of ``variance``.
    """
    if splits < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    indx = x_orig.index.to_numpy()
    lab = label_orig.reindex(indx).to_numpy()
    estimates = []
    for _ in range(splits):
        half = np.zeros(len(indx), dtype=bool)
        for lvl in np.unique(lab[np.isfinite(lab)]):
            members = np.flatnonzero(lab == lvl)
            chosen = rng.permutation(members)[: len(members) // 2]
            half[chosen] = True
        a, b = x_orig.iloc[half], x_orig.iloc[~half]
        table = _level_table(
            a,
            b,
            label_orig.reindex(a.index),
            label_orig.reindex(b.index),
            level_height,
            percentile=percentile,
            regression=regression,
        )
        table["weight"] = table["n"] / table["n"].sum()
        estimates.append(_combine(table, length_scale, regression=regression))
    return float(np.std(estimates, ddof=1) / np.sqrt(2.0))


def transport_error(
    particles: pd.DataFrame,
    error_particles: pd.DataFrame,
    flux: xr.DataArray,
    *,
    transforms: Sequence[Any] = (),
    context: TransformContext | None = None,
    levels: int | Sequence[float] = 20,
    length_scale: float | None = DEFAULT_LENGTH_SCALE,
    percentile: float = 1.0,
    regression: bool = False,
    noise_splits: int = 16,
    background: xr.DataArray | None = None,
) -> TransportError:
    """
    Transport-error variance of the modelled enhancement (Lin and Gerbig, 2005).

    Parameters
    ----------
    particles, error_particles
        The simulation's main and error-trajectory particle tables
        (``sim.trajectories.data`` and ``sim.error_trajectories.data``).
    flux
        Surface flux field (see :mod:`stilt.flux`).
    transforms, context
        The footprint's particle transforms and the context to apply them
        with (``config.transforms`` and ``sim.transform_context(name)``), so
        the error is weighted the way the footprint is (averaging kernel,
        pressure weighting, lifetime decay). Applied to both tables.
    levels
        Release-height levels to compute statistics on: a number of
        equal-width bins between the lowest and highest release height, or
        explicit bin edges. Particles with at most ``levels`` distinct release
        heights (a multipoint receptor) use those heights directly. A point
        receptor is one level.
    length_scale
        Vertical e-folding length of the error correlation between levels,
        in metres (X-STILT's 356 m); ``None`` treats levels as uncorrelated.
        Irrelevant for a point receptor.
    percentile
        Per level, drop particles above this quantile of the enhancement
        before taking the variance. ``1.0`` keeps every particle (Lin and
        Gerbig); X-STILT uses ``0.99`` to tame a few particles that cross a
        point source. Means always use every particle.
    regression
        Use X-STILT's regression scaling of the per-level variance
        differences (see :func:`_scale_dvar`) instead of the signed
        differences. Biased upward under sampling noise; for reproducing
        X-STILT results.
    noise_splits
        Random half-splits of the unperturbed particles used to estimate
        ``noise``; ``0`` skips it.
    background
        A background field to sample at each particle's endpoint (see
        :func:`~stilt.observations.background`). Wind errors move the
        endpoints as well as the surface contact, so with a field the
        statistics are of the modelled mole fraction, enhancement plus
        background per particle, as X-STILT computes them.

    Notes
    -----
    Per level ``l`` with ``n_l`` of ``N`` particles, ``dvar_l`` is the change
    in the variance of the per-particle enhancement under the perturbation,
    ``s_l = sign(dvar_l) sqrt(|dvar_l|)``, and the column variance is
    ``Σ_ij w_i w_j s_i s_j exp(-|h_i - h_j| / L)`` with ``w_l = n_l / N`` and
    ``h`` the level heights; for one level it is ``dvar`` itself. Because
    the transforms are applied to the particles first, the level statistics
    are already in column-weighted units and the weights are the particle
    counts.

    The signal is the extra spread the perturbation adds, so it is small
    when the wind error decorrelates quickly (HYSPLIT decorrelates it with
    the distance a particle travels as well as with time) and when turbulent
    dispersion already spreads the particles widely, as in a convective
    afternoon. Judge a single value against ``noise``; over many receptors,
    aggregate the signed ``variance`` with a median rather than clipping
    each one at zero.
    """
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1].")
    if length_scale is not None and length_scale <= 0:
        raise ValueError("length_scale must be > 0 or None.")
    if noise_splits < 0:
        raise ValueError("noise_splits must be >= 0.")
    if context is None:
        context = default_context()
    transforms = list(transforms)

    tables, backgrounds = [], []
    for table, is_error in ((particles, False), (error_particles, True)):
        ctx = TransformContext(
            receptor=context.receptor,
            footprint_name=context.footprint_name,
            is_error=is_error,
            store=context.store,
        )
        if background is not None:
            # each particle's background, weighted like its enhancement
            weights = endpoint_weights(table, transforms, ctx)
            sampled = fill_missing(particle_background(table, background), weights)
            backgrounds.append(weights * sampled)
        if transforms:
            table = apply_transforms(table, transforms, ctx)
        tables.append(table)
    main, err = tables

    x_orig = particle_enhancement(main, flux)
    x_err = particle_enhancement(err, flux)
    background_value = 0.0
    if backgrounds:
        b_orig, b_err = backgrounds
        x_orig = x_orig + b_orig.reindex(x_orig.index)
        x_err = x_err + b_err.reindex(x_err.index)
        background_value = float(b_orig.mean())
    h_orig = _release_heights(main)
    h_err = _release_heights(err)

    label_orig, level_height = _level_bins(h_orig, levels)
    edges = _edges_from_levels(h_orig, level_height, levels)
    label_err = pd.Series(
        pd.cut(h_err, edges, labels=False, include_lowest=True), index=h_err.index
    ).astype(float)

    table = _level_table(
        x_orig,
        x_err,
        label_orig,
        label_err,
        level_height,
        percentile=percentile,
        regression=regression,
    )
    w = table["weight"].to_numpy()
    return TransportError(
        variance=_combine(table, length_scale, regression=regression),
        noise=_noise(
            x_orig,
            label_orig,
            level_height,
            splits=noise_splits,
            percentile=percentile,
            regression=regression,
            length_scale=length_scale,
        ),
        enhancement=float(np.nansum(w * table["mean_orig"].to_numpy())),
        enhancement_perturbed=float(np.nansum(w * table["mean_err"].to_numpy())),
        levels=table,
        length_scale=length_scale,
        background=background_value,
    )


def _release_heights(particles: pd.DataFrame) -> pd.Series:
    """Release height per particle (``xhgt``), or zeros for a single-level receptor."""
    if "xhgt" in particles.columns:
        return release_coordinate(particles, "xhgt")
    indx = np.unique(particles["indx"].to_numpy())
    return pd.Series(0.0, index=indx)


def _edges_from_levels(
    heights: pd.Series, level_height: pd.Series, levels: int | Sequence[float]
) -> np.ndarray:
    """Bin edges that reproduce the main table's levels, for the error table."""
    if not isinstance(levels, int):
        return np.asarray(levels, dtype=float)
    centres = level_height.to_numpy(dtype=float)
    if centres.size == 1:
        return np.array([-np.inf, np.inf])
    unique = np.unique(heights.to_numpy())
    if unique.size <= levels:
        mids = (unique[:-1] + unique[1:]) / 2.0
        return np.concatenate(([-np.inf], mids, [np.inf]))
    lo, hi = float(heights.min()), float(heights.max())
    edges = np.linspace(lo, hi, levels + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


__all__ = ["DEFAULT_LENGTH_SCALE", "TransportError", "transport_error"]
