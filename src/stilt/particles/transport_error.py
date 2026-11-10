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
levels are added with the column weighting and X-STILT's vertical error
correlation (Wu et al., 2018, GMD). The difference of two sample variances
is noisy and can be negative. It is returned with its sign and with its
standard error, so many receptors can be combined with a median and a
single value can be compared with its noise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from stilt.particles.background import _background, _fill_missing
from stilt.receptors import Receptor
from stilt.transforms import apply_transforms, release_coordinate

#: X-STILT's empirical mean vertical correlation length of transport errors, m.
DEFAULT_LENGTH_SCALE = 356.0

#: The results :func:`_transport_errors` keeps for each receptor, in column order.
_RESULT_COLUMNS = (
    "enhancement",
    "enhancement_perturbed",
    "background",
    "variance",
    "noise",
    "realizations",
)


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
        Standard error of ``variance``, from the particles of both runs.
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
        perturbation), ``dvar`` (``var_err - var_orig``), ``noise`` (standard
        error of ``dvar``), and ``factor`` (what ``dvar`` is multiplied by in
        the column, so ``variance`` is ``(factor * dvar).sum()``).
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
    label = pd.cut(heights, edges.tolist(), labels=False, include_lowest=True)
    return pd.Series(np.asarray(label, dtype=float), index=heights.index)


def _level_stats(x: pd.Series, labels: pd.Series, percentile: float) -> pd.DataFrame:
    """
    Return each level's mean and variance of ``x``, indexed by level.

    The mean is over all the level's finite values and the variance over
    those at or below ``percentile``. ``var_of_var`` is the sampling variance
    of that variance, ``(m4 - var²) / n`` for ``n`` values with fourth
    central moment ``m4``. A level with fewer than two such values has NaN
    for both.
    """
    rows = {}
    for level, values in x.reindex(labels.index).groupby(labels):
        v = _finite(values)
        mean = float(v.mean()) if v.size else np.nan
        if v.size and percentile < 1.0:
            v = v[v <= np.quantile(v, percentile)]
        if v.size < 2:
            rows[level] = (mean, np.nan, np.nan)
            continue
        var = float(v.var(ddof=0))
        m4 = float(np.mean((v - v.mean()) ** 4))
        # m4 is never below var²; rounding can put it there when they are equal.
        rows[level] = (mean, var, max(m4 - var**2, 0.0) / v.size)
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["mean", "var", "var_of_var"]
    )


def _finite(values: pd.Series) -> np.ndarray:
    """Return the finite values as an array."""
    v = values.to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _mean(values: pd.Series) -> float:
    """Return numpy's mean of the finite values, or NaN when there are none."""
    v = _finite(values)
    return float(v.mean()) if v.size else np.nan


def _correlated_counts(heights: np.ndarray, length_scale: float) -> np.ndarray:
    """
    Return ``Σ_q exp(-|h_p - h_q| / length_scale)`` for each particle ``p``, itself included.

    With the heights sorted, the sum over the particles below ``p`` is a
    running sum of ``exp(h_q / L)`` times ``exp(-h_p / L)``, and likewise
    above. The running sums are kept as logarithms (``logaddexp``), so a tall
    column with a short length scale does not overflow.
    """
    order = np.argsort(heights)
    z = heights[order] / length_scale
    below = np.exp(np.logaddexp.accumulate(z) - z)  # particles at or below p
    above = np.exp(np.logaddexp.accumulate(-z[::-1])[::-1] + z)  # at or above p
    counts = np.empty_like(z)
    counts[order] = below + above - 1.0  # p itself is in both
    return counts


def _level_factors(
    heights: pd.Series, labels: pd.Series, length_scale: float | None
) -> pd.Series:
    """
    Return each level's factor in the column variance, indexed by level.

    The column variance is ``Σ_i factor_i dvar_i``. Two particles' transport
    errors correlate as ``exp(-|h_p - h_q| / length_scale)``, with ``h``
    their release heights, and a level's factor is the sum of that
    correlation over its particles and every particle, divided by ``N²``.
    That is ``w_i Σ_j w_j R_ij``, with ``w`` the levels' shares of the
    particles and ``R_ij`` the mean correlation between the particles of two
    levels, so a pair of levels counts with the mean of their two ``dvar``.
    The correlation is between particles, not between the levels' mean
    heights, so an error that is the same at every height gives the same
    column variance whatever the number of levels. With no ``length_scale``
    the levels are independent and the factor is ``w_i²``.
    """
    in_level = labels.notna()
    lab = labels[in_level]
    if length_scale is None:
        return (lab.value_counts().sort_index() / len(labels)) ** 2
    h = heights.reindex(lab.index).to_numpy(dtype=float)
    counts = pd.Series(_correlated_counts(h, length_scale), index=lab.index)
    return counts.groupby(lab).sum() / len(labels) ** 2


def _level_table(
    x_orig: pd.Series,
    x_errs: Sequence[pd.Series],
    h_orig: pd.Series,
    label_orig: pd.Series,
    label_errs: Sequence[pd.Series],
    *,
    percentile: float,
    length_scale: float | None,
) -> pd.DataFrame:
    """
    Return the table of means, variances, and weights per release level.

    The perturbed mean and variance of each level are averaged over the
    error realizations (one ``x_errs`` and ``label_errs`` pair each) before
    ``dvar`` is taken. ``noise`` is the standard error of ``dvar``: the
    sampling variance of the unperturbed variance plus that of the mean of
    the realizations' variances, which are independent runs.
    """
    level_height = h_orig.groupby(label_orig).apply(_mean)
    levels = level_height.index
    orig = _level_stats(x_orig, label_orig, percentile).reindex(levels)
    errs = [
        _level_stats(x, label, percentile).reindex(levels)
        for x, label in zip(x_errs, label_errs, strict=True)
    ]
    n = label_orig.value_counts().reindex(levels, fill_value=0)
    var_errs = np.column_stack([e["var"].to_numpy(dtype=float) for e in errs])
    var_of_var_errs = np.column_stack(
        [e["var_of_var"].to_numpy(dtype=float) for e in errs]
    )
    # A realization with too few particles in a level is left out of its mean.
    used = np.isfinite(var_errs).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_of_var_err = np.nansum(var_of_var_errs, axis=1) / used**2
    table = pd.DataFrame(
        {
            "height": level_height.to_numpy(dtype=float),
            "n": n.to_numpy(dtype=int),
            "weight": n.to_numpy() / len(x_orig),
            "mean_orig": orig["mean"].to_numpy(),
            "mean_err": _nanmean([e["mean"].to_numpy(dtype=float) for e in errs]),
            "var_orig": orig["var"].to_numpy(),
            "var_err": _nanmean(list(var_errs.T)),
        }
    )
    table["dvar"] = table["var_err"] - table["var_orig"]
    table["noise"] = np.sqrt(orig["var_of_var"].to_numpy() + var_of_var_err)
    table["factor"] = (
        _level_factors(h_orig, label_orig, length_scale).reindex(levels).to_numpy()
    )
    return table


def _nanmean(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Return the mean across *columns* per row, ignoring NaN; NaN where all are."""
    arr = np.column_stack(columns)
    ok = np.isfinite(arr)
    count = ok.sum(axis=1)
    total = np.where(ok, arr, 0.0).sum(axis=1)
    return np.divide(total, count, out=np.full(len(arr), np.nan), where=count > 0)


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
    background: xr.DataArray | None = None,
) -> TransportError:
    """
    Return the transport-error variance of a modeled enhancement (Lin and Gerbig, 2005).

    Parameters
    ----------
    particles : pandas.DataFrame
        Unperturbed particle table of a receptor (``sim.particles``).
    error_particles : pandas.DataFrame or sequence of pandas.DataFrame
        Particle table of a wind-error variant of the same receptor, or one
        table per realization for a variant with ``realizations: N`` (see
        Examples). A table is one simulation's particles: one receptor and
        one realization.
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
        independent. It has no effect for a point receptor.
    percentile : float, default 1.0
        In each level, drop particles above this quantile of the enhancement
        before taking the variance. 1.0 keeps every particle, as Lin and
        Gerbig do. X-STILT uses 0.99 to limit the few particles that cross a
        point source. Means always use every particle.
    background : xarray.DataArray, optional
        Background field sampled at each particle's endpoint (see
        :func:`background`). Wind errors move the
        endpoints as well as the surface contact, so with a field the
        statistics are of the modeled mole fraction per particle, as X-STILT
        computes them.

    Returns
    -------
    TransportError

    Notes
    -----
    Per level ``l`` with ``n_l`` of ``N`` particles, ``dvar_l`` is the change
    in the variance of the per-particle enhancement under the perturbation.
    Two particles' transport errors correlate as ``exp(-|h_p - h_q| / L)``,
    with ``h`` their release heights, and the column variance is
    ``Σ_l factor_l dvar_l``. ``factor_l`` is the sum of that correlation over
    the particles of level ``l`` and every particle, divided by ``N²``, which
    is ``w_l Σ_m w_m R_lm`` with ``w_l = n_l / N`` and ``R_lm`` the mean
    correlation between the particles of two levels. A pair of levels
    therefore counts with the mean of their two ``dvar``. For one level the
    variance is ``dvar`` itself, and with ``length_scale=None`` it is
    ``Σ_l w_l² dvar_l``. The transforms are applied to the particles first,
    so the level statistics are already column-weighted and the level
    weights are particle counts.

    The sum is linear in ``dvar``, so the sampling noise in each level's
    ``dvar`` averages out over many receptors. Where the error changes
    sharply with height within a correlation length, the mean of two levels'
    ``dvar`` is above the product of their standard deviations, and the
    column variance is high by up to about 20 %.

    ``noise`` is the standard error of ``variance``. A variance of ``n``
    values with fourth central moment ``m4`` has the sampling variance
    ``(m4 - var²) / n``. Each level's ``noise`` adds that of the unperturbed
    run and of the perturbed run, and the column's is
    ``sqrt(Σ_l (factor_l noise_l)²)``.

    With several error realizations, each level's ``mean_err`` and
    ``var_err`` are averaged over them before ``dvar`` is formed, so the
    perturbed side's sampling variance falls as ``1/N``. The unperturbed
    side is the same particles in every realization, so its part does not
    fall. More realizations therefore tighten the estimate by about
    ``sqrt(2)`` at most and cannot resolve a case that one realization
    leaves far from resolved.

    The signal is the extra spread the perturbation adds. It is small when
    the wind error decorrelates quickly (HYSPLIT decorrelates it with the
    distance a particle travels as well as with time), and when turbulence
    already spreads the particles widely, as on a convective afternoon.
    Compare a single value with ``noise``. Over many receptors, combine the
    signed ``variance`` values with a median instead of clipping each at
    zero.

    Examples
    --------
    For the variant ``hrrr-err`` with ``realizations: 4``, select one
    receptor's simulations and group their particles by realization. Grouping
    by ``variant`` gives one table, which counts as one realization and gives
    a wrong variance. Without the receptor, each realization's table holds
    every receptor's particles.

    >>> from stilt.particles import transport_error  # doctest: +SKIP
    >>> sim = project.simulation(rid, "hrrr")  # doctest: +SKIP
    >>> sims = project.simulations  # doctest: +SKIP
    >>> ensemble = sims[
    ...     (sims.variant == "hrrr-err") & (sims.receptor == rid)
    ... ]  # doctest: +SKIP
    >>> by_realization = project.particles(ensemble).groupby(
    ...     "realization"
    ... )  # doctest: +SKIP
    >>> error_particles = [rows for _, rows in by_realization]  # doctest: +SKIP
    >>> result = transport_error(sim.particles, error_particles, flux)  # doctest: +SKIP
    >>> result.realizations  # doctest: +SKIP
    4

    ``sim.transport_error(realizations, flux)`` builds this list from the
    realizations' simulations. It also passes ``transforms``, ``receptor``,
    and ``directory``, which a variant with footprint transforms needs and
    this call leaves out.
    """
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1].")
    if length_scale is not None and length_scale <= 0:
        raise ValueError("length_scale must be > 0 or None.")
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
        bg = _background(
            table,
            background,
            transforms=transforms,
            receptor=receptor,
            directory=directory,
        )
        # Each particle's background, weighted like its enhancement. A
        # particle's enhancement is N times its share of the footprint's.
        filled = _fill_missing(bg.per_particle, bg.weights)
        x = x + (len(bg.weights) * bg.weights * filled).reindex(x.index)
        return x, _release_heights(weighted), bg.value

    x_orig, h_orig, background_value = _prepare(particles)

    edges = _level_edges(h_orig, levels)
    label_orig = _level_labels(h_orig, edges)

    x_errs, label_errs = [], []
    for err in error_tables:
        x_err, h_err, _ = _prepare(err)
        x_errs.append(x_err)
        label_errs.append(_level_labels(h_err, edges))

    table = _level_table(
        x_orig,
        x_errs,
        h_orig,
        label_orig,
        label_errs,
        percentile=percentile,
        length_scale=length_scale,
    )
    w = table["weight"].to_numpy()
    factor = table["factor"].to_numpy()
    # A level with too few particles for a variance (NaN) adds nothing.
    return TransportError(
        variance=float(np.nansum(factor * table["dvar"].to_numpy())),
        noise=float(np.sqrt(np.nansum((factor * table["noise"].to_numpy()) ** 2))),
        enhancement=float(np.nansum(w * table["mean_orig"].to_numpy())),
        enhancement_perturbed=float(np.nansum(w * table["mean_err"].to_numpy())),
        levels=table,
        length_scale=length_scale,
        background=background_value,
        realizations=len(error_tables),
    )


def _transport_errors(
    read: Callable[[str], tuple[pd.DataFrame, list[pd.DataFrame]]],
    receptors: Sequence[Receptor],
    flux: xr.DataArray,
    *,
    transforms: Sequence[Any] = (),
    directory: str | Path | None = None,
    workers: int = 1,
    **options: Any,
) -> pd.DataFrame:
    """
    Return the transport error of many receptors, one receptor at a time.

    ``read(receptor_id)`` returns that receptor's unperturbed particle table
    and one table per error realization, as :func:`transport_error` takes
    them; every receptor must have both. Each receptor is read and estimated
    in one of *workers* threads, which holds only that receptor's tables.
    *transforms* are the unperturbed variant's footprint transforms, as
    ``sim.transport_error`` passes them, and *options* go to
    :func:`transport_error` unchanged.

    Returns one row per receptor, indexed by receptor id in *receptors*
    order, with the columns in :data:`_RESULT_COLUMNS`.
    """

    def one(receptor: Receptor) -> tuple[float, float, float, float, float, int]:
        particles, error_particles = read(receptor.id)
        result = transport_error(
            particles,
            error_particles,
            flux,
            transforms=transforms,
            receptor=receptor,
            directory=directory,
            **options,
        )
        return (
            result.enhancement,
            result.enhancement_perturbed,
            result.background,
            result.variance,
            result.noise,
            result.realizations,
        )

    if workers <= 1 or len(receptors) <= 1:
        rows = [one(r) for r in receptors]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, receptors))
    return pd.DataFrame(
        rows,
        index=pd.Index([r.id for r in receptors], name="receptor"),
        columns=pd.Index(_RESULT_COLUMNS),
    ).astype({"realizations": "int64"})


def _release_heights(particles: pd.DataFrame) -> pd.Series:
    """Return each particle's release height (``release_height``), or zeros without ``release_height``."""
    if "release_height" in particles.columns:
        return release_coordinate(particles, "release_height")
    indx = np.unique(particles["particle"].to_numpy())
    return pd.Series(0.0, index=indx)
