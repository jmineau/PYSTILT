"""
Transport error of the modeled enhancement, from wind-perturbed trajectories.

The method is that of Lin and Gerbig (2005, GRL, doi:10.1029/2004GL021127).
A variant with wind-error settings (``siguverr``, ``tluverr``,
``zcoruverr``, ``horcoruverr``) runs the particles again with an extra
random wind that has the statistics of the meteorology's errors. Each
particle's enhancement is its ``foot × flux`` summed along its trajectory.
The perturbed particles sample more of the flux field, so the enhancement
varies more across them. The increase is the transport-error variance
(their equation 4)::

    var_transport = var(enhancement | perturbed) − var(enhancement | unperturbed)

For a column receptor the difference is taken per release level, and the
levels are combined with the column weighting and a vertical error
correlation, as in X-STILT (Wu et al., 2018, GMD). The difference of two
sample variances is noisy and can be negative. It is returned with its sign
and with an estimate of its noise, so many receptors can be combined with a
median and a single value can be compared with its noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from stilt.observations import backgrounds
from stilt.transforms import apply_transforms, release_coordinate

if TYPE_CHECKING:
    from stilt.receptors import Receptor

#: X-STILT's empirical mean vertical correlation length of transport errors, m.
DEFAULT_LENGTH_SCALE = 356.0


@dataclass(frozen=True)
class TransportError:
    """
    Transport error of a modeled enhancement, returned by :func:`transport_error`.

    Values are in the flux's units times the footprint's, which is ppm for a
    flux in µmol m⁻² s⁻¹, and squared for variances. With a ``background``
    field, the per-particle values are modeled mole fractions (enhancement
    plus background at the endpoint), so ``enhancement`` and
    ``enhancement_perturbed`` are mole fractions and ``variance`` includes
    the background's response to the wind errors.

    Attributes
    ----------
    variance : float
        Transport-error variance, the extra spread the wind perturbation
        added. Kept with its sign, since a negative value is sampling noise.
    noise : float
        Standard deviation of ``variance`` expected with no perturbation,
        estimated from random halves of the unperturbed particles.
        ``variance`` is resolved only when it is several times ``noise``.
    enhancement : float
        Modeled enhancement from the unperturbed particles.
    enhancement_perturbed : float
        Modeled enhancement from the perturbed particles, averaged over the
        realizations.
    levels : pandas.DataFrame
        One row per release level, with columns ``height`` (mean release
        height, m), ``n`` (particles), ``weight`` (share of the particles),
        ``mean_orig``, ``var_orig``, ``mean_err``, and ``var_err`` (mean and
        variance of the per-particle enhancement without and with the
        perturbation), ``dvar`` (``var_err - var_orig``), and ``sd_trans``
        (signed square root of ``dvar``).
    length_scale : float or None
        Vertical correlation length used to combine the levels, in m.
    background : float
        Weighted background from the unperturbed particles, or 0 without a
        ``background`` field. ``enhancement - background`` is the
        enhancement alone.
    realizations : int
        Number of error realizations in the estimate.
    """

    variance: float
    noise: float
    enhancement: float
    enhancement_perturbed: float
    levels: pd.DataFrame
    length_scale: float | None
    background: float = 0.0
    realizations: int = 1

    @property
    def sd(self) -> float:
        """Transport-error standard deviation, or 0 when ``variance`` is negative."""
        return float(np.sqrt(max(self.variance, 0.0)))


def _level_edges(heights: pd.Series, levels: int | Sequence[float]) -> np.ndarray:
    """
    Return the release-height bin edges of the levels.

    Given edges are used as they are. With a number of levels, the edges
    split the height range evenly, or fall halfway between the distinct
    heights when there are no more of them than ``levels``. The outer edges
    are then open, so the perturbed particles fall in the same levels.
    """
    if not isinstance(levels, int):
        return np.asarray(levels, dtype=float)
    if levels < 1:
        raise ValueError("levels must be >= 1.")
    unique = np.unique(heights.to_numpy(dtype=float))
    if unique.size <= levels:
        edges = np.concatenate(([-np.inf], (unique[:-1] + unique[1:]) / 2.0, [np.inf]))
    else:
        edges = np.linspace(unique[0], unique[-1], levels + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _level_labels(heights: pd.Series, edges: np.ndarray) -> pd.Series:
    """Return each particle's level number, NaN outside the edges."""
    label = pd.cut(heights, edges, labels=False, include_lowest=True)
    return pd.Series(np.asarray(label, dtype=float), index=heights.index)


def _level_stats(x: pd.Series, labels: pd.Series, percentile: float) -> pd.DataFrame:
    """
    Return each level's mean and variance of ``x``, indexed by level.

    The mean is over all the level's finite values and the variance over
    those at or below ``percentile``. A level with fewer than two such
    values has NaN variance.
    """

    def var(values: pd.Series) -> float:
        v = _finite(values)
        if v.size and percentile < 1.0:
            v = v[v <= np.quantile(v, percentile)]
        return float(v.var(ddof=0)) if v.size >= 2 else np.nan

    groups = x.reindex(labels.index).groupby(labels)
    return pd.DataFrame({"mean": groups.apply(_mean), "var": groups.apply(var)})


def _finite(values: pd.Series) -> np.ndarray:
    """Return the finite values as an array."""
    v = values.to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _mean(values: pd.Series) -> float:
    """Return numpy's mean of the finite values, or NaN when there are none."""
    v = _finite(values)
    return float(v.mean()) if v.size else np.nan


def _signed_sqrt(values: np.ndarray) -> np.ndarray:
    """Return ``sign(v) * sqrt(|v|)``, with NaN as 0."""
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    return np.sign(v) * np.sqrt(np.abs(v))


def _combine(levels: pd.DataFrame, length_scale: float | None) -> float:
    """
    Return the column variance ``Σ_i w_i² v_i + Σ_{i≠j} w_i w_j s_i s_j corr_ij``.

    ``s`` is the per-level ``sd_trans``. On the diagonal, ``v`` is the
    signed variance difference, so a level whose spread fell counts
    negatively. The cross terms use the signed square roots, so correlated levels of the
    same sign add and levels of opposite sign cancel.
    """
    w = levels["weight"].to_numpy(dtype=float)
    s = levels["sd_trans"].to_numpy(dtype=float)
    h = levels["height"].to_numpy(dtype=float)
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
    x_errs: Sequence[pd.Series],
    label_orig: pd.Series,
    label_errs: Sequence[pd.Series],
    level_height: pd.Series,
    *,
    percentile: float,
) -> pd.DataFrame:
    """
    Return the table of means, variances, and weights per release level.

    The perturbed mean and variance of each level are averaged over the
    error realizations (one ``x_errs`` and ``label_errs`` pair each) before
    ``dvar`` is taken.
    """
    orig = _level_stats(x_orig, label_orig, percentile).reindex(level_height.index)
    errs = [
        _level_stats(x, label, percentile).reindex(level_height.index)
        for x, label in zip(x_errs, label_errs, strict=True)
    ]
    n = label_orig.value_counts().reindex(level_height.index, fill_value=0)
    table = pd.DataFrame(
        {
            "height": level_height.to_numpy(dtype=float),
            "n": n.to_numpy(dtype=int),
            "weight": n.to_numpy() / len(x_orig),
            "mean_orig": orig["mean"].to_numpy(),
            "mean_err": _nanmean([e["mean"].to_numpy(dtype=float) for e in errs]),
            "var_orig": orig["var"].to_numpy(),
            "var_err": _nanmean([e["var"].to_numpy(dtype=float) for e in errs]),
        }
    )
    table["dvar"] = table["var_err"] - table["var_orig"]
    table["sd_trans"] = _signed_sqrt(table["dvar"].to_numpy())
    return table


def _nanmean(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Return the mean across *columns* per row, ignoring NaN; NaN where all are."""
    arr = np.column_stack(columns)
    ok = np.isfinite(arr)
    count = ok.sum(axis=1)
    total = np.where(ok, arr, 0.0).sum(axis=1)
    return np.divide(total, count, out=np.full(len(arr), np.nan), where=count > 0)


def _noise(
    x_orig: pd.Series,
    label_orig: pd.Series,
    level_height: pd.Series,
    *,
    splits: int,
    percentile: float,
    length_scale: float | None,
) -> float:
    """
    Return the standard deviation of ``variance`` with no perturbation.

    The unperturbed particles of each level are split into two random halves,
    treated as the unperturbed and perturbed tables. The spread of the
    estimate over ``splits`` random splits, divided by ``sqrt(2)`` to go from
    half to full ensembles, is the noise. Returns NaN for fewer than two
    splits.
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
            [b],
            label_orig.reindex(a.index),
            [label_orig.reindex(b.index)],
            level_height,
            percentile=percentile,
        )
        table["weight"] = table["n"] / table["n"].sum()
        estimates.append(_combine(table, length_scale))
    return float(np.std(estimates, ddof=1) / np.sqrt(2.0))


def transport_error(
    particles: pd.DataFrame,
    error_particles: pd.DataFrame | Sequence[pd.DataFrame],
    flux: xr.DataArray,
    *,
    transforms: Sequence[Any] = (),
    receptor: Receptor | None = None,
    directory: str | Path | None = None,
    levels: int | Sequence[float] = 20,
    length_scale: float | None = DEFAULT_LENGTH_SCALE,
    percentile: float = 1.0,
    noise_splits: int = 16,
    background: xr.DataArray | None = None,
) -> TransportError:
    """
    Return the transport-error variance of a modeled enhancement (Lin and Gerbig, 2005).

    Parameters
    ----------
    particles : pandas.DataFrame
        Unperturbed particle table of a receptor (``sim.particles``).
    error_particles : pandas.DataFrame or sequence of pandas.DataFrame
        Particle table of a wind-error variant of the same receptor, or a
        list of them for a variant with ``realizations: N``, such as
        ``[rows for _, rows in project.particles(ensemble).groupby("variant")]``.
    flux : xarray.DataArray
        Surface flux field (see ``particles.stilt.enhancement``).
    transforms : sequence, optional
        The footprint's particle transforms (``sim.variant.footprint.transforms``),
        applied to both tables so the error is weighted like the footprint.
    receptor : Receptor, optional
        The receptor (``sim.receptor``), for the transforms.
    directory : str or Path, optional
        Where a kernel table's relative path starts (``project.directory``).
    levels : int or sequence of float, default 20
        Release-height levels to compute statistics on: a number of
        equal-width bins between the lowest and highest release height, or
        bin edges in meters. When the particles have no more distinct
        release heights than ``levels`` (a multipoint receptor), each height
        is a level. A point receptor is one level.
    length_scale : float or None, default 356.0
        Vertical e-folding length of the error correlation between levels,
        in meters (X-STILT's value). ``None`` treats the levels as
        uncorrelated. It has no effect for a point receptor.
    percentile : float, default 1.0
        In each level, drop particles above this quantile of the enhancement
        before taking the variance. 1.0 keeps every particle, as Lin and
        Gerbig do. X-STILT uses 0.99 to limit the few particles that cross a
        point source. Means always use every particle.
    noise_splits : int, default 16
        Number of random half-splits of the unperturbed particles used to
        estimate ``noise``. Fewer than 2 skips it and gives NaN.
    background : xarray.DataArray, optional
        Background field sampled at each particle's endpoint (see
        :func:`~stilt.observations.background`). Wind errors move the
        endpoints as well as the surface contact, so with a field the
        statistics are of the modeled mole fraction per particle, as X-STILT
        computes them.

    Returns
    -------
    TransportError

    Notes
    -----
    Per level ``l`` with ``n_l`` of ``N`` particles, ``dvar_l`` is the change
    in the variance of the per-particle enhancement under the perturbation,
    ``s_l = sign(dvar_l) sqrt(|dvar_l|)``, and the column variance is
    ``Σ_ij w_i w_j s_i s_j exp(-|h_i - h_j| / L)`` with ``w_l = n_l / N`` and
    ``h`` the level heights. For one level it is ``dvar`` itself. The
    transforms are applied to the particles first, so the level statistics
    are already column-weighted and the level weights are particle counts.

    With several error realizations, each level's ``mean_err`` and
    ``var_err`` are averaged over them before ``dvar`` is formed, so the
    perturbed side's sampling noise falls as ``1/sqrt(N)``. The unperturbed
    side is the same particles in every realization, so its noise does not
    fall: with ``N`` realizations the null spread of ``variance`` is
    ``sqrt((1 + 1/N) / 2)`` times the single-realization ``noise``, which
    tends to ``1/sqrt(2)``. ``noise`` includes that factor. More
    realizations therefore give at most a ``sqrt(2)`` tighter estimate and
    cannot resolve a case that one realization leaves unresolved.

    The signal is the extra spread the perturbation adds. It is small when
    the wind error decorrelates quickly (HYSPLIT decorrelates it with the
    distance a particle travels as well as with time), and when turbulence
    already spreads the particles widely, as on a convective afternoon.
    Compare a single value with ``noise``. Over many receptors, combine the
    signed ``variance`` values with a median instead of clipping each at
    zero.
    """
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1].")
    if length_scale is not None and length_scale <= 0:
        raise ValueError("length_scale must be > 0 or None.")
    if noise_splits < 0:
        raise ValueError("noise_splits must be >= 0.")
    transforms = list(transforms)
    error_tables = (
        [error_particles]
        if isinstance(error_particles, pd.DataFrame)
        else list(error_particles)
    )
    if not error_tables:
        raise ValueError("error_particles must hold at least one realization.")

    def _prepare(
        table: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, float]:
        """Return the modeled value and release height per particle, and the background."""
        weighted = (
            apply_transforms(table, transforms, receptor, directory)
            if transforms
            else table
        )
        x = weighted.stilt.enhancement(flux)
        if background is None:
            return x, _release_heights(weighted), 0.0
        bg = backgrounds.background(
            table,
            background,
            transforms=transforms,
            receptor=receptor,
            directory=directory,
        )
        # Each particle's background, weighted like its enhancement. A
        # particle's enhancement is N times its share of the footprint's.
        filled = backgrounds._fill_missing(bg.per_particle, bg.weights)
        x = x + (len(bg.weights) * bg.weights * filled).reindex(x.index)
        return x, _release_heights(weighted), bg.value

    x_orig, h_orig, background_value = _prepare(particles)

    edges = _level_edges(h_orig, levels)
    label_orig = _level_labels(h_orig, edges)
    level_height = h_orig.groupby(label_orig).apply(_mean)

    x_errs, label_errs = [], []
    for err in error_tables:
        x_err, h_err, _ = _prepare(err)
        x_errs.append(x_err)
        label_errs.append(_level_labels(h_err, edges))

    table = _level_table(
        x_orig,
        x_errs,
        label_orig,
        label_errs,
        level_height,
        percentile=percentile,
    )
    n_real = len(error_tables)
    # The main particles are shared by every realization, so only the
    # perturbed side's noise averages down; see the Notes.
    noise_factor = float(np.sqrt((1.0 + 1.0 / n_real) / 2.0))
    w = table["weight"].to_numpy()
    return TransportError(
        variance=_combine(table, length_scale),
        noise=noise_factor
        * _noise(
            x_orig,
            label_orig,
            level_height,
            splits=noise_splits,
            percentile=percentile,
            length_scale=length_scale,
        ),
        enhancement=float(np.nansum(w * table["mean_orig"].to_numpy())),
        enhancement_perturbed=float(np.nansum(w * table["mean_err"].to_numpy())),
        levels=table,
        length_scale=length_scale,
        background=background_value,
        realizations=n_real,
    )


def _release_heights(particles: pd.DataFrame) -> pd.Series:
    """Return each particle's release height (``xhgt``), or zeros without ``xhgt``."""
    if "xhgt" in particles.columns:
        return release_coordinate(particles, "xhgt")
    indx = np.unique(particles["indx"].to_numpy())
    return pd.Series(0.0, index=indx)


__all__ = ["DEFAULT_LENGTH_SCALE", "TransportError", "transport_error"]
