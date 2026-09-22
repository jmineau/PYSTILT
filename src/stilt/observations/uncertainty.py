"""
Transport error on the modelled enhancement, from wind-perturbed trajectories.

This is X-STILT's column transport-error method (Wu et al., 2018, GMD,
``cal.trajfoot.stat`` / ``scale.dvar`` / ``cal.var.cov`` / ``cal.trans.err``).
A simulation run with wind-error settings (``siguverr`` and friends in
:class:`~stilt.config.STILTParams`) writes a second particle table whose
transport carries an extra random wind component. Each particle's enhancement
is its ``foot × flux`` summed along its trajectory; the spread of that value
across the ensemble is larger with the perturbation, and the increase in
variance is the transport error. Column receptors get a value per release
level, combined with the column weighting and a vertical error correlation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from stilt.flux import particle_enhancement
from stilt.transforms import TransformContext, apply_transforms, release_coordinate

#: X-STILT's empirical mean vertical correlation length of transport errors, m.
DEFAULT_LENGTH_SCALE = 356.0


@dataclass(frozen=True)
class TransportError:
    """
    Result of :func:`transport_error`.

    ``sd`` is the transport-error standard deviation of the modelled
    enhancement, in the flux's units times the footprint's (ppm for a flux
    in µmol m⁻² s⁻¹). ``enhancement`` is the modelled enhancement itself from
    the unperturbed particles. ``levels`` has one row per release level:
    ``height`` (m, mean release height), ``n`` particles, ``weight`` (its share
    of the column), ``mean`` / ``var`` of the per-particle enhancement without
    (``_orig``) and with (``_err``) the perturbation, ``dvar`` their raw
    difference, and ``sd_trans`` the transport error at that level after the
    regression scaling.
    """

    sd: float
    enhancement: float
    levels: pd.DataFrame
    length_scale: float | None


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
    Transport-error sd per level from the variance difference, Wu et al. (2018).

    The per-level ``var_err − var_orig`` is noisy and can go negative. Over the
    levels where it is positive, ``var_err`` is regressed on ``var_orig``
    (weighted by ``1 / var_err``), the fitted line is evaluated at every level,
    and its excess over ``var_orig`` is the scaled transport-error variance.
    With fewer than two positive levels the raw difference is used, clipped
    at zero.
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


def transport_error(
    particles: pd.DataFrame,
    error_particles: pd.DataFrame,
    flux: xr.DataArray,
    *,
    transforms: Sequence[Any] = (),
    context: TransformContext | None = None,
    levels: int | Sequence[float] = 20,
    length_scale: float | None = DEFAULT_LENGTH_SCALE,
    percentile: float = 0.99,
) -> TransportError:
    """
    Transport-error standard deviation of the modelled enhancement.

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
        in metres; ``None`` treats levels as uncorrelated.
    percentile
        Per level, values above this quantile are dropped before the
        variance is taken (X-STILT removes the top 1%). Means use every
        particle.

    Notes
    -----
    Per level ``l`` with ``n_l`` of ``N`` particles, the transport error
    ``sd_l`` comes from the increase in the variance of the per-particle
    enhancement under the perturbation (see :func:`_scale_dvar`), and the
    column value is ``sqrt(Σ_ij w_i w_j sd_i sd_j exp(-|h_i - h_j| / L))``
    with ``w_l = n_l / N`` and ``h`` the level heights. Because the transforms
    are applied to the particles first, the level statistics are already in
    column-weighted units and the weights are the particle counts.
    """
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1].")
    if context is None:
        context = _default_context()
    transforms = list(transforms)

    tables = []
    for table, is_error in ((particles, False), (error_particles, True)):
        if transforms:
            ctx = TransformContext(
                receptor=context.receptor,
                footprint_name=context.footprint_name,
                is_error=is_error,
                store=context.store,
            )
            table = apply_transforms(table, transforms, ctx)
        tables.append(table)
    main, err = tables

    x_orig = particle_enhancement(main, flux)
    x_err = particle_enhancement(err, flux)
    h_orig = _release_heights(main)
    h_err = _release_heights(err)

    label_orig, level_height = _level_bins(h_orig, levels)
    edges = _edges_from_levels(h_orig, level_height, levels)
    label_err = pd.Series(
        pd.cut(h_err, edges, labels=False, include_lowest=True), index=h_err.index
    ).astype(float)

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
    table["sd_trans"] = _scale_dvar(table)

    w = table["weight"].to_numpy()
    sd = table["sd_trans"].to_numpy()
    h = table["height"].to_numpy()
    if length_scale is None:
        corr = np.eye(len(h))
    else:
        if length_scale <= 0:
            raise ValueError("length_scale must be > 0 or None.")
        corr = np.exp(-np.abs(h[:, None] - h[None, :]) / length_scale)
    total_var = float(
        (w[:, None] * w[None, :] * sd[:, None] * sd[None, :] * corr).sum()
    )
    enhancement = float(np.nansum(w * table["mean_orig"].to_numpy()))
    return TransportError(
        sd=float(np.sqrt(total_var)),
        enhancement=enhancement,
        levels=table,
        length_scale=length_scale,
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


def _default_context() -> TransformContext:
    """
    A placeholder context for transforms that do not read it.

    Transforms that do (an averaging-kernel ``table``) need the real one from
    ``sim.transform_context(name)``.
    """
    from stilt.receptors import PointReceptor

    return TransformContext(receptor=PointReceptor("2000-01-01", 0.0, 0.0, 0.0))


__all__ = ["DEFAULT_LENGTH_SCALE", "TransportError", "transport_error"]
