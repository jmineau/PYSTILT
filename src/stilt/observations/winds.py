"""
Wind-error statistics for transport-error runs (Lin and Gerbig, 2005, section 2.1).

HYSPLIT's wind-error perturbation needs four numbers: the standard deviation
of the analysis wind error and its correlation scales in time, height and
horizontal distance (``siguverr``, ``tluverr``, ``zcoruverr`` and
``horcoruverr`` in the configuration). Lin and Gerbig derive them from the
differences between analysed and observed winds: the standard deviation
directly, and each scale by fitting the exponential variogram

    γ(h) = σ² (1 − exp(−h / l))

to the mean squared difference of the error between pairs of points
separated by ``h`` in that coordinate.

:func:`variogram` builds the empirical variogram from an error array, its
separation coordinate and a grouping key that says which points may pair.
:func:`fit_variogram` fits the model. :func:`wind_error_scales` runs the
recipe on a table of upper-air errors and, optionally, a table of
surface-station errors, and returns the four parameters. Producing the
errors is arlmet's job::

    arlmet.sample_points(files, points, ["UWND", "VWND"], earth_relative=True)
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from .selection import _haversine_km

#: Pairs accumulated before they are binned; bounds memory for large groups.
_FLUSH_PAIRS = 2_000_000


# -- empirical variogram ---------------------------------------------------------


class _Accumulator:
    """Binned sums of squared differences, lags and pair counts."""

    def __init__(self, edges: np.ndarray):
        self.edges = edges
        self.max_lag = float(edges[-1])
        self.sq = np.zeros(len(edges) - 1)
        self.lag = np.zeros(len(edges) - 1)
        self.n = np.zeros(len(edges) - 1)
        self._lags: list[np.ndarray] = []
        self._sqs: list[np.ndarray] = []
        self._pending = 0

    def add(self, lag: np.ndarray, sq: np.ndarray) -> None:
        if lag.size == 0:
            return
        self._lags.append(lag)
        self._sqs.append(sq)
        self._pending += lag.size
        if self._pending >= _FLUSH_PAIRS:
            self.flush()

    def flush(self) -> None:
        if not self._lags:
            return
        lag = np.concatenate(self._lags)
        sq = np.concatenate(self._sqs)
        self.sq += np.histogram(lag, self.edges, weights=sq)[0]
        self.lag += np.histogram(lag, self.edges, weights=lag)[0]
        self.n += np.histogram(lag, self.edges)[0]
        self._lags, self._sqs, self._pending = [], [], 0

    def table(self) -> pd.DataFrame:
        self.flush()
        keep = self.n > 0
        return pd.DataFrame(
            {
                "lag": self.lag[keep] / self.n[keep],
                "gamma": 0.5 * self.sq[keep] / self.n[keep],
                "n": self.n[keep].astype(int),
            }
        )


def _pairs_1d(acc: _Accumulator, errors: np.ndarray, coord: np.ndarray) -> None:
    """All pairs within one group along a 1-D coordinate, no farther apart than the last bin."""
    order = np.argsort(coord, kind="stable")
    e = errors[order]
    c = coord[order]
    n = len(c)
    for d in range(1, n):
        i = np.arange(n - d)
        h = c[i + d] - c[i]
        if h.min() > acc.max_lag:
            break  # coordinates are sorted, so wider bands are farther still
        keep = (h > 0) & (h < acc.max_lag)
        acc.add(h[keep], (e[i + d] - e[i])[keep] ** 2)


def _pairs_geo(
    acc: _Accumulator, errors: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> None:
    """All pairs within one group, separated by great-circle distance in km."""
    m = len(errors)
    if m < 2:
        return
    i, j = np.triu_indices(m, 1)
    h = _haversine_km(lon[i], lat[i], lon[j], lat[j])
    keep = (h > 0) & (h < acc.max_lag)
    acc.add(h[keep], (errors[i] - errors[j])[keep] ** 2)


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

    acc = _Accumulator(edges)
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(labels[idx], kind="stable")]
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    for members in np.split(order, boundaries):
        if members.size < 2:
            continue
        if geo:
            _pairs_geo(acc, e[members], coord[members, 0], coord[members, 1])
        else:
            _pairs_1d(acc, e[members], coord[members])
    return acc.table()


# -- model fit -------------------------------------------------------------------


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
        p0 = [float(np.median(h))]
        (length,), _ = curve_fit(
            lambda x, ell: VariogramFit(sigma, ell)(x),
            h,
            g,
            p0=p0,
            bounds=(1e-9, np.inf),
        )
        return VariogramFit(float(sigma), float(length))
    if h.size < 2:
        raise ValueError(
            "fit_variogram needs at least two finite points to fit sigma too."
        )
    p0 = [float(np.median(h)), float(np.sqrt(max(g.max(), 1e-12)))]
    (length, sig), _ = curve_fit(
        lambda x, ell, s: VariogramFit(s, ell)(x), h, g, p0=p0, bounds=(1e-9, np.inf)
    )
    return VariogramFit(float(sig), float(length))


# -- the recipe ------------------------------------------------------------------


def _minutes(times: ArrayLike | pd.Series) -> np.ndarray:
    """Times as minutes since 2000-01-01 UTC."""
    t = pd.to_datetime(pd.Series(times))
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return (t - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy() / 60.0


def _stamps(times: ArrayLike | pd.Series) -> np.ndarray:
    """Times as strings, one label per distinct time."""
    return pd.to_datetime(pd.Series(times)).astype(str).to_numpy()


def _require(table: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}.")


def _site(table: pd.DataFrame) -> np.ndarray:
    """One label per site: ``site``, or the rounded position, or a constant."""
    if "site" in table.columns:
        return table["site"].to_numpy().astype(str)
    if "lon" in table.columns:
        lon = np.round(table["lon"].to_numpy(dtype=float), 4)
        lat = np.round(table["lat"].to_numpy(dtype=float), 4)
        return np.char.add(np.char.add(lon.astype(str), "|"), lat.astype(str))
    return np.full(len(table), "", dtype=object)


def _join(*parts: np.ndarray) -> np.ndarray:
    """Element-wise ``"a|b|..."`` labels from string arrays."""
    out = parts[0].astype(str)
    for part in parts[1:]:
        out = np.char.add(np.char.add(out, "|"), part.astype(str))
    return out


@dataclass(frozen=True)
class WindErrorScales:
    """
    Result of :func:`wind_error_scales`.

    ``siguverr`` (m/s), ``tluverr`` (min), ``zcoruverr`` (m) and
    ``horcoruverr`` (km) are the wind-error settings for the configuration,
    each the mean of the u and v values; ``horcoruverr`` is ``None`` without
    surface-station errors. ``fits`` has one row per component and
    coordinate with the fitted ``sigma`` and ``length``, the ``bias`` (mean
    error) of the data it was fitted to, the number of pairs and which
    ``source`` table it came from; ``variograms`` holds the empirical
    variogram behind each row, keyed the same way, for plotting against the
    fit.
    """

    siguverr: float
    tluverr: float
    zcoruverr: float
    horcoruverr: float | None
    fits: pd.DataFrame
    variograms: dict[tuple[str, str], pd.DataFrame]

    def to_dict(self) -> dict[str, float]:
        """The configuration parameters, omitting ``horcoruverr`` when it is ``None``."""
        out = {
            "siguverr": self.siguverr,
            "tluverr": self.tluverr,
            "zcoruverr": self.zcoruverr,
        }
        if self.horcoruverr is not None:
            out["horcoruverr"] = self.horcoruverr
        return out


def wind_error_scales(
    upper: pd.DataFrame,
    surface: pd.DataFrame | None = None,
    *,
    height_range: tuple[float, float] = (0.0, 3000.0),
    height_bins: ArrayLike | None = None,
    time_bins: ArrayLike | None = None,
    distance_bins: ArrayLike | None = None,
    time_from: Literal["surface", "upper"] | None = None,
) -> WindErrorScales:
    """
    HYSPLIT's wind-error parameters from analysis-minus-observation winds.

    Each parameter comes from the data that resolves it. The error standard
    deviation and its vertical correlation come from the upper-air profiles
    within ``height_range``, the vertical one from pairs of levels in the
    same launch. The horizontal correlation comes from pairs of surface
    stations at the same time. The time correlation comes from pairs of
    times at the same station when surface errors are given, because
    radiosondes twelve hours apart cannot resolve a scale of a few hours,
    and otherwise from pairs of launches at the same height. Each variogram
    is fitted with the sill fixed at the variance of the errors it was built
    from, and u and v are treated separately and averaged at the end.

    Parameters
    ----------
    upper
        Upper-air errors, one row per sounding level: ``time``, ``height``
        (metres above ground), ``u_err``, ``v_err`` and optionally ``site``.
    surface
        Surface-station errors, one row per station and time: ``time``,
        ``lon``, ``lat``, ``u_err``, ``v_err`` and optionally ``site``.
    height_range
        The layer of the upper-air errors to use, in metres above ground.
    height_bins, time_bins, distance_bins
        Separation bin edges for the vertical (m), time (min) and
        horizontal (km) variograms. The defaults are 100 m steps across
        ``height_range``, hourly steps out to ten days, and 1 km steps out
        to 50 km. The height bins also define the "same height" groups when
        the time scale comes from the upper-air data.
    time_from
        Which table the time scale comes from; the default is ``surface``
        when given, else ``upper``.

    Returns
    -------
    WindErrorScales
    """
    _require(upper, ("time", "height", "u_err", "v_err"), "upper")
    lo, hi = height_range
    heights_all = upper["height"].to_numpy(dtype=float)
    in_layer = (heights_all >= lo) & (heights_all <= hi)
    layer = upper.loc[in_layer].reset_index(drop=True)
    if layer.empty:
        raise ValueError(f"No upper-air rows within height_range {height_range}.")
    if surface is not None:
        _require(surface, ("time", "lon", "lat", "u_err", "v_err"), "surface")
        surface = surface.reset_index(drop=True)
    if time_from is None:
        time_from = "surface" if surface is not None else "upper"
    if time_from == "surface" and surface is None:
        raise ValueError("time_from='surface' needs a surface table.")

    z_edges = (
        np.arange(0.0, hi - lo + 100.0, 100.0)
        if height_bins is None
        else np.asarray(height_bins, dtype=float)
    )
    t_edges = (
        np.arange(0.0, 10 * 1440.0 + 60.0, 60.0)
        if time_bins is None
        else np.asarray(time_bins, dtype=float)
    )
    x_edges = (
        np.arange(0.0, 51.0, 1.0)
        if distance_bins is None
        else np.asarray(distance_bins, dtype=float)
    )

    rows: list[dict[str, Hashable]] = []
    curves: dict[tuple[str, str], pd.DataFrame] = {}

    def record(
        component: str,
        coordinate: str,
        source: str,
        errors: np.ndarray,
        table: pd.DataFrame,
        fit: VariogramFit,
    ) -> None:
        rows.append(
            {
                "component": component,
                "coordinate": coordinate,
                "source": source,
                "sigma": fit.sigma,
                "length": fit.length,
                "bias": float(np.nanmean(errors)),
                "n_pairs": int(table["n"].to_numpy().sum()),
            }
        )
        curves[(component, coordinate)] = table

    heights = layer["height"].to_numpy(dtype=float)
    upper_site = _site(layer)
    launch = _join(upper_site, _stamps(layer["time"]))
    upper_minutes = _minutes(layer["time"])
    height_bin = np.digitize(heights, lo + z_edges)
    same_height = _join(upper_site, height_bin)

    for component in ("u", "v"):
        e_up = layer[f"{component}_err"].to_numpy(dtype=float)
        sigma_up = float(np.nanstd(e_up, ddof=1))
        z_table = variogram(e_up, heights, group=launch, bins=z_edges)
        z_fit = fit_variogram(z_table["lag"], z_table["gamma"], sigma=sigma_up)
        record(component, "height", "upper", e_up, z_table, z_fit)

        if surface is not None:
            e_sf = surface[f"{component}_err"].to_numpy(dtype=float)
            sigma_sf = float(np.nanstd(e_sf, ddof=1))
            lonlat = surface[["lon", "lat"]].to_numpy(dtype=float)
            same_time = _stamps(surface["time"])
            x_table = variogram(e_sf, lonlat, group=same_time, bins=x_edges)
            x_fit = fit_variogram(x_table["lag"], x_table["gamma"], sigma=sigma_sf)
            record(component, "distance", "surface", e_sf, x_table, x_fit)
            if time_from == "surface":
                t_table = variogram(
                    e_sf, _minutes(surface["time"]), group=_site(surface), bins=t_edges
                )
                t_fit = fit_variogram(t_table["lag"], t_table["gamma"], sigma=sigma_sf)
                record(component, "time", "surface", e_sf, t_table, t_fit)
        if time_from == "upper":
            t_table = variogram(e_up, upper_minutes, group=same_height, bins=t_edges)
            t_fit = fit_variogram(t_table["lag"], t_table["gamma"], sigma=sigma_up)
            record(component, "time", "upper", e_up, t_table, t_fit)

    fits = pd.DataFrame(rows).set_index(["component", "coordinate"])

    def mean_of(coordinate: str, column: str) -> float:
        values = fits.xs(coordinate, level="coordinate")[column].to_numpy(dtype=float)
        return float(values.mean())

    return WindErrorScales(
        siguverr=mean_of("height", "sigma"),
        tluverr=mean_of("time", "length"),
        zcoruverr=mean_of("height", "length"),
        horcoruverr=mean_of("distance", "length") if surface is not None else None,
        fits=fits,
        variograms=curves,
    )


__all__ = [
    "VariogramFit",
    "WindErrorScales",
    "fit_variogram",
    "variogram",
    "wind_error_scales",
]
