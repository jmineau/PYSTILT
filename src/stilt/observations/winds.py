"""
Variograms of wind errors, for transport-error runs (Lin and Gerbig, 2005, section 2.1).

HYSPLIT's wind-error perturbation needs four numbers: the standard deviation
of the analysis wind error and its correlation scales in time, height and
horizontal distance (``siguverr``, ``tluverr``, ``zcoruverr`` and
``horcoruverr`` in the configuration). Lin and Gerbig derive them from the
differences between analysed and observed winds: the standard deviation
directly, and each scale by fitting the exponential variogram

    γ(h) = σ² (1 − exp(−h / l))

to the mean squared difference of the error between pairs of points
separated by ``h`` in that coordinate. :func:`variogram` builds the
empirical variogram from an error array, its separation coordinate and a
grouping key that says which points may pair; :func:`fit_variogram` fits
the model. The errors themselves are the analysed wind minus the observed
wind at each observation; arlmet samples the analysis at the observation
points::

    met = arlmet.sample_points(files, points, ["UWND", "VWND"], earth_relative=True)
    u_err = met["UWND"] - observed_u

The Wind Error Statistics guide has the recipe from there to the four
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from .selection import _haversine_km


def _bin(
    sums: tuple[np.ndarray, np.ndarray, np.ndarray],
    edges: np.ndarray,
    lag: np.ndarray,
    sq: np.ndarray,
) -> None:
    """Add pairs to the running per-bin sums of squared difference, lag and count."""
    keep = (lag > 0) & (lag < edges[-1])
    lag, sq = lag[keep], sq[keep]
    sums[0][:] += np.histogram(lag, edges, weights=sq)[0]
    sums[1][:] += np.histogram(lag, edges, weights=lag)[0]
    sums[2][:] += np.histogram(lag, edges)[0]


def _pairs_1d(sums, edges: np.ndarray, errors: np.ndarray, coord: np.ndarray) -> None:
    """All pairs within one group along a 1-D coordinate, closer than the last edge."""
    order = np.argsort(coord, kind="stable")
    e, c = errors[order], coord[order]
    n = len(c)
    for d in range(1, n):
        i = np.arange(n - d)
        h = c[i + d] - c[i]
        if h.min() >= edges[-1]:
            break  # coordinates are sorted, so wider bands are farther still
        _bin(sums, edges, h, (e[i + d] - e[i]) ** 2)


def _pairs_geo(
    sums, edges: np.ndarray, errors: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> None:
    """All pairs within one group, separated by great-circle distance in km."""
    if len(errors) < 2:
        return
    i, j = np.triu_indices(len(errors), 1)
    h = _haversine_km(lon[i], lat[i], lon[j], lat[j])
    _bin(sums, edges, h, (errors[i] - errors[j]) ** 2)


def variogram(
    errors: ArrayLike,
    lag: ArrayLike,
    *,
    group: ArrayLike | None = None,
    bins: ArrayLike,
) -> pd.DataFrame:
    """
    Empirical semivariogram of ``errors`` over a separation coordinate.

    For every pair of points in the same group, the squared difference of
    their errors is binned by their separation; the variogram in a bin is
    half the mean squared difference. Bins are half-open, ``[a, b)``. Pairs
    at zero separation are skipped, as are pairs at or beyond the last edge.

    Parameters
    ----------
    errors
        One error value per point (a wind component's analysis minus
        observation, say).
    lag
        The coordinate the separation is measured in, one value per point:
        minutes, metres, or an ``(n, 2)`` array of longitude and latitude in
        degrees, in which case the separation is the great-circle distance
        in kilometres.
    group
        A label per point; only points sharing a label are paired. For the
        vertical variogram of radiosonde errors that is the launch, for the
        time variogram of a station's errors the station, for the horizontal
        variogram of a network the observation time. ``None`` pairs
        everything with everything, which is fine for a few thousand points
        and not for a few hundred thousand.
    bins
        Separation bin edges, in the units of ``lag``.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty bin: ``lag`` (the mean separation of the pairs
        in the bin), ``gamma`` (the semivariogram, in the units of ``errors``
        squared) and ``n`` (the number of pairs).
    """
    e = np.asarray(errors, dtype=float)
    coord = np.asarray(lag, dtype=float)
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bins must be increasing edges with at least two values.")
    if coord.ndim == 1:
        geo = False
    elif coord.ndim == 2 and coord.shape[1] == 2:
        geo = True
    else:
        raise ValueError("lag must be 1-D, or (n, 2) longitude/latitude.")
    if len(coord) != e.size:
        raise ValueError("errors and lag must have the same length.")

    valid = np.isfinite(e) & np.isfinite(coord).reshape(e.size, -1).all(axis=1)
    if group is None:
        labels = np.zeros(e.size, dtype=np.intp)
    else:
        labels = pd.factorize(pd.Series(np.asarray(group, dtype=object)))[0]
        if len(labels) != e.size:
            raise ValueError("group must have one label per point.")
        valid &= labels >= 0

    sums = tuple(np.zeros(edges.size - 1) for _ in range(3))
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(labels[idx], kind="stable")]
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    for members in np.split(order, boundaries):
        if members.size < 2:
            continue
        if geo:
            _pairs_geo(sums, edges, e[members], coord[members, 0], coord[members, 1])
        else:
            _pairs_1d(sums, edges, e[members], coord[members])

    sq, lag_sum, n = sums
    keep = n > 0
    return pd.DataFrame(
        {
            "lag": lag_sum[keep] / n[keep],
            "gamma": 0.5 * sq[keep] / n[keep],
            "n": n[keep].astype(int),
        }
    )


@dataclass(frozen=True)
class VariogramFit:
    """
    Exponential variogram ``σ² (1 − exp(−h / l))``.

    ``sigma`` is the error standard deviation (the square root of the sill)
    and ``length`` the e-folding correlation scale, in the units of the
    separation it was fitted over. Call it with separations to evaluate the
    model.
    """

    sigma: float
    length: float

    def __call__(self, lag: ArrayLike) -> np.ndarray:
        """Evaluate the model at ``lag``."""
        h = np.asarray(lag, dtype=float)
        return self.sigma**2 * (1.0 - np.exp(-h / self.length))


def fit_variogram(
    lag: ArrayLike, gamma: ArrayLike, *, sigma: float | None = None
) -> VariogramFit:
    """
    Fit the exponential variogram to an empirical one.

    Parameters
    ----------
    lag, gamma
        The empirical variogram, as :func:`variogram` returns it.
    sigma
        The error standard deviation. Given, the sill is fixed at ``sigma²``
        and only the correlation scale is fitted, which is what Lin and
        Gerbig's definition of the variogram implies and what you want when
        the sample standard deviation is known. ``None`` fits both.

    Returns
    -------
    VariogramFit
    """
    h = np.asarray(lag, dtype=float)
    g = np.asarray(gamma, dtype=float)
    keep = np.isfinite(h) & np.isfinite(g) & (h > 0)
    h, g = h[keep], g[keep]
    if sigma is not None:
        if h.size < 1:
            raise ValueError("fit_variogram needs at least one finite point.")
        (length,), _ = curve_fit(
            lambda x, ell: VariogramFit(sigma, ell)(x),
            h,
            g,
            p0=[float(np.median(h))],
            bounds=(1e-9, np.inf),
        )
        return VariogramFit(float(sigma), float(length))
    if h.size < 2:
        raise ValueError(
            "fit_variogram needs at least two finite points to fit sigma too."
        )
    (length, sig), _ = curve_fit(
        lambda x, ell, s: VariogramFit(s, ell)(x),
        h,
        g,
        p0=[float(np.median(h)), float(np.sqrt(max(g.max(), 1e-12)))],
        bounds=(1e-9, np.inf),
    )
    return VariogramFit(float(sig), float(length))


__all__ = ["VariogramFit", "fit_variogram", "variogram"]
