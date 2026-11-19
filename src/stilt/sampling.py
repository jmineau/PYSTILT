"""
Sampling ARL met files at particle positions.

:func:`sample_met` gives the particle columns a transport model can write
from the met (``mlht``, ``dens``, ``pres``, ``zsfc``), for a model that
writes only positions, and :func:`calc_pbl_height` finds the mixed-layer
height from the met's profile, as HYSPLIT does by default. A gridded field
such as a surface flux or a mole fraction is read at points with
:func:`runtraj.sampling.sample_field`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import pandas as pd

#: The ARL variables each particle column is computed from, by :func:`sample_met`.
#: ``pres`` is the ``PRES`` field, not arlmet's virtual ``"pressure"``, which
#: is NaN at a height on sigma files (jmineau/arl-met#65).
_MET_VARIABLES: dict[str, tuple[str, ...]] = {
    "mlht": ("PBLH",),
    "dens": ("PRES", "TEMP", "SPHU"),
    "pres": ("PRES",),
    "zsfc": ("SHGT",),
}

#: Gas constant of dry air, J kg⁻¹ K⁻¹, the value HYSPLIT uses.
_R_DRY = 287.04

#: The top of the domain of a HYSPLIT run that does not set one, in meters
#: (``HysplitConfig.z_top``, which a test keeps this equal to: this module does
#: not import HYSPLIT's package).
_DEFAULT_Z_TOP = 25000.0


def sample_met(
    particles: pd.DataFrame,
    files: Iterable[str | os.PathLike[str]],
    columns: Iterable[str] = ("mlht", "dens", "pres", "zsfc"),
    *,
    z_top: float | None = _DEFAULT_Z_TOP,
) -> pd.DataFrame:
    """
    Return the particles with met columns sampled from ARL files at each particle's position.

    A transport model that writes only positions can get the columns
    HYSPLIT writes from the met this way. Each row is sampled at its
    ``lon``, ``lat``, ``zagl``, and ``time``: bilinear across the met's
    grid and linear in height between the internal levels of a HYSPLIT run
    with the top *z_top* (:func:`arlmet.sample_points`), then linear in time
    between the two met times around the particle's, as HYSPLIT does. Each column comes
    from these ARL variables:

    - ``mlht`` is ``PBLH``, the met model's boundary-layer height, in m
      above ground.
    - ``dens`` is 100 ``PRES`` / (R ``TEMP`` (1 + 0.61 ``SPHU``)), the
      density of moist air in kg m⁻³, with R = 287.04 J kg⁻¹ K⁻¹.
    - ``pres`` is ``PRES``, the pressure, in hPa.
    - ``zsfc`` is ``SHGT``, the terrain height, in m above sea level.

    ``pres`` and ``dens`` follow what HYSPLIT reads, to the differences
    :doc:`/development/validation` gives. ``mlht`` is the met's: HYSPLIT's is the
    met's ``PBLH`` only with ``kmixd: 0``, and then it is at least ``kmix0``;
    the default, ``kmixd: 3``, finds the mixed layer from the profile, as
    :func:`calc_pbl_height` does.

    Parameters
    ----------
    particles : pandas.DataFrame
        A particle table with ``lon``, ``lat``, ``zagl``, and ``time``.
        ``sim.particles`` has ``time``. For a table a transport model
        returned, add it: the release time plus ``age`` minutes.
    files : iterable of path-like
        ARL met files that cover the particles' times, such as the files the
        run read (``met_files`` of :class:`runtraj.models.ModelRun` or of
        :func:`runtraj.particles.particles_metadata`).
    columns : iterable of str, default ("mlht", "dens", "pres", "zsfc")
        The columns to add.
    z_top : float or None, default 25000
        The top of the run's domain in meters (``z_top`` of
        :class:`runtraj.models.hysplit.HysplitConfig`), since HYSPLIT's
        internal levels depend on it. A stored run records it:
        ``particles_metadata(path).settings["z_top"]``. ``None`` reads the
        met's own levels, for a transport model that is not HYSPLIT; a row
        below their lowest or above their top is then NaN.

    Returns
    -------
    pandas.DataFrame
        A copy of *particles* with the columns added. A row off the met's
        grid is NaN.

    Raises
    ------
    ValueError
        If a column is not one of the four or is in *particles* already
        (drop it first), *particles* has no ``time`` column, a row is outside
        the files' times, or *z_top* is not positive.
    KeyError
        If the files lack a variable a column needs.

    Examples
    --------
    >>> meta = runtraj.particles.particles_metadata(
    ...     sim.particles_path
    ... )  # doctest: +SKIP
    >>> hysplit = sim.particles  # doctest: +SKIP
    >>> sampled = stilt.sampling.sample_met(  # doctest: +SKIP
    ...     hysplit.drop(columns="pres"),
    ...     meta.met_files,
    ...     columns=["pres"],
    ...     z_top=meta.settings["z_top"],
    ... )
    >>> (sampled["pres"] - hysplit["pres"]).describe()  # doctest: +SKIP
    """
    import arlmet

    columns = list(dict.fromkeys(columns))
    unknown = [c for c in columns if c not in _MET_VARIABLES]
    if unknown:
        raise ValueError(
            f"sample_met cannot sample {unknown}; it samples {list(_MET_VARIABLES)}."
        )
    present = [c for c in columns if c in particles.columns]
    if present:
        raise ValueError(
            f"The particles already have {present}. Drop them first to sample them."
        )
    if "time" not in particles.columns:
        raise ValueError(
            "sample_met needs a 'time' column: the release time plus 'age' minutes."
        )
    variables = list(dict.fromkeys(v for c in columns for v in _MET_VARIABLES[c]))

    handles = [arlmet.File(path) for path in files]
    try:
        met_times = pd.DatetimeIndex(sorted({t for h in handles for t in h.times}))
        times = pd.DatetimeIndex(particles["time"])
        before, after, weight = _bracket(met_times, times)
        # Each row is sampled at the met time before it and, unless it is on
        # a met time, at the one after. One call samples both.
        later = weight > 0
        position = particles[["lon", "lat", "zagl"]].to_numpy(dtype=float)
        points = pd.DataFrame(
            np.concatenate([position, position[later]]), columns=["lon", "lat", "z"]
        )
        points["time"] = np.concatenate([before, after[later]])
        sampled = arlmet.sample_points(
            handles, points, variables, z_kind="agl", model_top=z_top
        )
    finally:
        for handle in handles:
            handle.close()

    n = len(particles)
    w = weight[later]
    met: dict[str, np.ndarray] = {}
    for name in variables:
        values = sampled[name].to_numpy(dtype=float)
        at_rows = values[:n].copy()
        at_rows[later] = (1.0 - w) * at_rows[later] + w * values[n:]
        met[name] = at_rows

    out = particles.copy()
    for column in columns:
        if column == "dens":
            virtual_temperature = met["TEMP"] * (1.0 + 0.61 * met["SPHU"])
            out[column] = 100.0 * met["PRES"] / (_R_DRY * virtual_temperature)
        else:
            out[column] = met[_MET_VARIABLES[column][0]]
    return out


def _bracket(
    met_times: pd.DatetimeIndex, times: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the met times before and after each time, and the weight of the one after.

    A time on a met time has that time for both, and weight 0.
    """
    if len(met_times) == 0:
        raise ValueError("The met files hold no time steps.")
    after = met_times.searchsorted(times, side="left")
    outside = (after == len(met_times)) | np.asarray(times < met_times[0])
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} particle rows are outside the met files' times "
            f"({met_times[0]} to {met_times[-1]}): {times[outside].min()} "
            f"to {times[outside].max()}."
        )
    t1 = met_times[after]
    exact = np.asarray(t1 == times)
    t0 = met_times[np.where(exact, after, after - 1)]
    elapsed = (times - t0).to_numpy()
    span = np.where(exact, np.timedelta64(1, "s"), (t1 - t0).to_numpy())
    weight = np.where(exact, 0.0, elapsed / span)
    return t0.to_numpy(), t1.to_numpy(), weight.astype(float)


#: HYSPLIT's constants for the mixed layer (``stbcon.f``): gravity in m s⁻²,
#: von Kármán's constant, and the heat capacity of air in J kg⁻¹ K⁻¹.
_HYSPLIT_GRAV = 9.80616
_VON_KARMAN = 0.4
_CP = 1005.0
#: The top of HYSPLIT's surface layer, in m above ground (``metlvl.f``).
_SURFACE_LAYER_TOP = 75.0
#: The bulk Richardson number at the top of the mixed layer (``stbanl.f``).
_CRITICAL_RICHARDSON = 0.25


def calc_pbl_height(
    particles: pd.DataFrame,
    files: Iterable[str | os.PathLike[str]],
    *,
    z_top: float = _DEFAULT_Z_TOP,
) -> np.ndarray:
    """
    Return the mixed-layer height at each particle, found from the met's profile as HYSPLIT's ``kmixd: 3`` finds it.

    This is HYSPLIT's default way to find the mixed layer, ported from its
    source (``stbanl.f``, Holtslag and Boville, 1993). Going up the column,
    the mixed layer ends where the bulk Richardson number between the
    lowest level and the level reached passes 0.25::

        Ri = g · (θ(z) − θ₁ − θₑ) · (z − z₁) / ((θ₁ + θₑ) · |V(z) − V₁|²)

    with θ the potential temperature, V the horizontal wind, and level 1
    HYSPLIT's lowest internal level. θₑ, the temperature excess of rising
    air, is ``8.5 u* θ* / (u*³ + 0.6 w*³)^(1/3)`` in unstable air and 0
    otherwise, with the friction velocity ``u*`` (``USTR``) and the
    friction temperature ``θ*`` from the sensible heat flux (``SHTF``) at
    the ground. The height is interpolated to where Ri is 0.25.

    HYSPLIT computes the height at each met grid point and met time; a
    particle takes the nearest grid point's height at the met times before
    and after it, linear in time. So does this function, on HYSPLIT's
    internal levels for a run with the top *z_top*. It applies no minimum:
    HYSPLIT's ``kmix0`` is ``pbl_height_min`` in PYSTILT. A column with no
    height where Ri reaches 0.25 gets the second-highest internal level,
    as in HYSPLIT. :doc:`/development/validation` says how close it comes
    to the ``mlht`` HYSPLIT writes.

    Parameters
    ----------
    particles : pandas.DataFrame
        A particle table with ``lon``, ``lat``, and ``time``.
    files : iterable of path-like
        ARL met files that cover the particles' times, with ``TEMP``,
        ``PRES``, ``UWND``, ``VWND`` and ``SHGT``, the sensible heat flux
        ``SHTF``, and the friction velocity ``USTR`` or the momentum fluxes
        ``UMOF`` and ``VMOF``.
    z_top : float, default 25000
        The top of the domain in meters, which sets HYSPLIT's internal
        levels (``z_top`` of :class:`runtraj.models.hysplit.HysplitConfig`).

    Returns
    -------
    numpy.ndarray
        The mixed-layer height at each row, in m above ground. A row off
        the met's grid is NaN.

    Raises
    ------
    ValueError
        If *particles* has no ``time`` column or a row is outside the
        files' times.
    KeyError
        If the files lack a variable it needs.

    Examples
    --------
    >>> meta = runtraj.particles.particles_metadata(
    ...     sim.particles_path
    ... )  # doctest: +SKIP
    >>> mlht = stilt.sampling.calc_pbl_height(
    ...     sim.particles, meta.met_files
    ... )  # doctest: +SKIP
    """
    import arlmet

    if "time" not in particles.columns:
        raise ValueError(
            "calc_pbl_height needs a 'time' column: the release time plus 'age' minutes."
        )
    handles = [arlmet.File(path) for path in files]
    try:
        first = handles[0]
        names = {r.variable for h in handles for r in h.records}
        momentum = ["USTR"] if "USTR" in names else ["UMOF", "VMOF"]
        missing = sorted(
            {"TEMP", "PRES", "UWND", "VWND", "SHGT", "SHTF", *momentum} - names
        )
        if missing:
            raise KeyError(
                f"calc_pbl_height needs the met variables {missing}, which the "
                "files lack."
            )
        met_times = pd.DatetimeIndex(sorted({t for h in handles for t in h.times}))
        before, after, weight = _bracket(met_times, pd.DatetimeIndex(particles["time"]))

        # One column per met grid point and met time, at the nearest grid point.
        x, y = first.grid.fractional_indices(
            particles["lon"].to_numpy(dtype=float),
            particles["lat"].to_numpy(dtype=float),
        )
        rows = pd.DataFrame(
            {
                "i": np.rint(x),
                "j": np.rint(y),
                "lon": particles["lon"].to_numpy(dtype=float),
                "lat": particles["lat"].to_numpy(dtype=float),
            }
        )
        later = weight > 0
        both = pd.concat(
            [rows.assign(time=before), rows[later].assign(time=after[later])],
            ignore_index=True,
        )
        key = ["i", "j", "time"]
        columns = both.dropna(subset=["i", "j"]).drop_duplicates(subset=key)
        columns = columns.reset_index(drop=True)

        levels = arlmet.internal_levels(first, model_top=z_top)
        heights = levels.heights
        ground = arlmet.sample_points(
            handles,
            columns[["lon", "lat", "time"]].assign(z=0.0),
            ["SHGT", "SHTF", *momentum],
            z_kind="agl",
            method="nearest",
            model_top=z_top,
        )
        # The internal levels above each column's ground, (columns, levels).
        zz = levels.heights_at(ground["SHGT"].to_numpy(dtype=float))
        n_col, n_lev = zz.shape
        at_levels = pd.DataFrame(
            {
                "lon": np.repeat(columns["lon"].to_numpy(), n_lev),
                "lat": np.repeat(columns["lat"].to_numpy(), n_lev),
                "time": np.repeat(columns["time"].to_numpy(), n_lev),
                "z": zz.ravel(),
            }
        )
        profile_variables = ["TEMP", "PRES", "UWND", "VWND"]
        if "SPHU" in names:
            profile_variables.append("SPHU")
        profile = arlmet.sample_points(
            handles,
            at_levels,
            profile_variables,
            z_kind="agl",
            method="nearest",
            model_top=z_top,
        )
    finally:
        for handle in handles:
            handle.close()

    def grid(name: str) -> np.ndarray:
        return profile[name].to_numpy(dtype=float).reshape(n_col, n_lev)

    temp, pres, uu, vv = grid("TEMP"), grid("PRES"), grid("UWND"), grid("VWND")
    sphu = grid("SPHU") if "SPHU" in profile_variables else np.zeros_like(temp)
    theta = temp * (1000.0 / pres) ** 0.286
    density = 100.0 * pres / (_R_DRY * temp * (1.0 + 0.61 * sphu))
    ks = max(int(np.sum(heights <= _SURFACE_LAYER_TOP)), 1) - 1
    if "USTR" in momentum:
        ustar = np.maximum(ground["USTR"].to_numpy(dtype=float), 0.0001)
    else:
        # u* from the momentum fluxes, as stbanl does without USTR, with the
        # density of its lowest level above the first.
        def component(flux: str) -> np.ndarray:
            f = ground[flux].to_numpy(dtype=float)
            rho = density[:, min(max(1, ks), n_lev - 1)]
            return np.sign(f) * np.maximum(0.01, np.sqrt(np.abs(f) / rho))

        ustar = np.hypot(component("UMOF"), component("VMOF"))
    heat = ground["SHTF"].to_numpy(dtype=float)
    if first.source.strip() == "ETA":
        heat = -heat  # NAM's (ETA's) flux is positive downward (prfcom.f)
    zmix = _richardson_mixed_layer(zz, theta, uu, vv, density, ustar, heat, ks)

    # Each row takes its grid point's height at the met times around it.
    columns["zmix"] = zmix
    found = both.merge(columns[[*key, "zmix"]], on=key, how="left")["zmix"]
    value = found.to_numpy(dtype=float)
    n = len(particles)
    out = value[:n].copy()
    w = weight[later]
    out[later] = (1.0 - w) * out[later] + w * value[n:]
    return out


def _richardson_mixed_layer(
    zz: np.ndarray,
    theta: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    density: np.ndarray,
    ustar: np.ndarray,
    heat: np.ndarray,
    ks: int,
) -> np.ndarray:
    """
    Return the mixed-layer height of each column, as ``stbanl.f`` finds it for ``kmixd: 3``.

    *zz* (m above ground), *theta* (potential temperature, K), *uu* and
    *vv* (m s⁻¹) and *density* (kg m⁻³) are ``(columns, levels)`` on the
    internal levels; *ustar* (m s⁻¹) and *heat*, the sensible heat flux
    (W m⁻², positive upward), are per column; *ks* is the index of the top
    of the surface layer. The stability at the top of the surface layer
    comes from the fluxes, and the temperature excess from it in unstable
    air.
    """
    n_col, n_lev = zz.shape
    heat = np.maximum(-25.0, heat)
    tstar = np.sign(-heat) * np.maximum(
        0.0001, np.abs(heat / (density[:, ks] * _CP * ustar))
    )
    zl = zz[:, ks] * _VON_KARMAN * _HYSPLIT_GRAV * tstar / (ustar**2 * theta[:, ks])
    unstable = np.clip(zl, -2.0, 15.0) < 0

    # Up the column until Ri passes 0.25, interpolating to where it is 0.25.
    zmix = zz[:, 0].copy()
    ri = np.zeros(n_col)
    done = np.zeros(n_col, dtype=bool)
    for k in range(1, n_lev):
        go = ~done & (ri < _CRITICAL_RICHARDSON)
        done |= ~go
        if not go.any():
            break
        tstk = tstar * density[:, ks] / density[:, k]
        wstar = np.abs(_HYSPLIT_GRAV * ustar * tstk * zz[:, k] / theta[:, 0]) ** (1 / 3)
        excess = np.where(
            unstable,
            np.abs(8.5 * ustar * tstk / (ustar**3 + 0.6 * wstar**3) ** (1 / 3)),
            0.0,
        )
        t1 = theta[:, 0] + excess
        du = np.maximum(0.0001, (uu[:, k] - uu[:, 0]) ** 2 + (vv[:, k] - vv[:, 0]) ** 2)
        old = ri
        new = _HYSPLIT_GRAV * (theta[:, k] - t1) * (zz[:, k] - zz[:, 0]) / t1 / du
        with np.errstate(divide="ignore", invalid="ignore"):
            crossed = (zz[:, k] - zz[:, k - 1]) / (new - old) * (
                _CRITICAL_RICHARDSON - old
            ) + zz[:, k - 1]
        zmix = np.where(go, np.where(new == old, zz[:, k], crossed), zmix)
        ri = np.where(go, new, ri)
    if n_lev >= 2:
        # No crossing below the top: the second-highest level, as in HYSPLIT;
        # and always a little above the lowest level.
        zmix = np.minimum(zmix, zz[:, -2])
        zmix = np.maximum(zmix, zz[:, 0] + (zz[:, 1] - zz[:, 0]) / 1000.0)
    return zmix


__all__ = [
    "calc_pbl_height",
    "sample_met",
]
