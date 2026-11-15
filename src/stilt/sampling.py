"""
Sampling gridded fields, such as a surface flux or a mole fraction, at points.

A field is an :class:`xarray.DataArray` on a regular ``lat``/``lon`` grid
(``y``/``x`` for a projected footprint grid), with an optional vertical
dimension and an optional ``time`` dimension. Each point takes the value of
the cell it falls in. ``foot.stilt.enhancement`` and
``particles.stilt.enhancement`` sample a surface flux this way, taking the
flux step that starts at or before each time, or, when the flux's ``time``
coordinate carries CF time bounds, the step whose bounds hold the time, and
:func:`stilt.particles.background` a mole-fraction field, taking the
nearest time. PYSTILT does
not convert units. A flux in µmol m⁻² s⁻¹ times a footprint in ppm per
(µmol m⁻² s⁻¹) gives ppm.

:func:`sample_met` samples ARL met files at particle positions, for the
particle columns a transport model can write from the met (``mlht``,
``dens``, ``pres``, ``zsfc``), and :func:`calc_pbl_height` finds the
mixed-layer height from the met's profile, as HYSPLIT does by default.
:func:`sample_grib_pressure` reads the pressure from GFS or ERA5 GRIB
files, the met FLEXPART runs on, through cfgrib.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from pandas.errors import InvalidIndexError

from stilt.spatial import horizontal_dims


def nearest_cell(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Return the index of the cell nearest each value, or ``-1`` outside the cells.

    Each cell reaches halfway to its neighbors. The end cells reach the
    same distance beyond their centers. With a single cell, every finite
    value is inside it.

    Parameters
    ----------
    coords : numpy.ndarray
        Cell centers along one axis, ascending or descending.
    values : numpy.ndarray
        Positions to look up.

    Returns
    -------
    numpy.ndarray
        Index into *coords* for each value.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("Field coordinates must be a non-empty 1-D array.")
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


def vertical_dim(field: xr.DataArray) -> str | None:
    """
    Return the name of a field's vertical dimension.

    That is the one dimension that is neither horizontal nor ``time``.
    Returns ``None`` for a field without one, such as a column mean or a
    surface field, and raises if there is more than one candidate.
    """
    y_dim, x_dim = horizontal_dims(field)
    extra = [str(d) for d in field.dims if d not in (y_dim, x_dim, "time")]
    if len(extra) > 1:
        raise ValueError(
            f"Expected at most one vertical dimension besides {y_dim!r}/{x_dim!r} "
            f"and 'time'; got {extra}."
        )
    return extra[0] if extra else None


def _bounds_indexer(
    bounds: pd.IntervalIndex, stamps: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return each stamp's step index from a flux's CF time bounds, and whether it falls in one.

    A stamp before the first bound, after the last, or in a gap between
    bounds falls in none: the bounds say where every step starts and ends,
    so there is nothing to guess at either edge.
    """
    try:
        it = bounds.get_indexer(stamps)
    except InvalidIndexError as e:
        raise ValueError(
            "The field's time bounds overlap, so a time can fall in more than "
            "one step. Two bounds cannot cover the same instant."
        ) from e
    in_step = it >= 0
    return np.where(in_step, it, 0), in_step


def sample_field(
    field: xr.DataArray,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike | None = None,
    times: ArrayLike | None = None,
    fill_value: float = np.nan,
    step_starts: bool = False,
) -> np.ndarray:
    """
    Return the field's value in the cell nearest each point.

    A point outside the field horizontally, or in a missing cell, gives
    ``fill_value``. Longitudes are wrapped to the field's convention (-180
    to 180 or 0 to 360).

    Parameters
    ----------
    field : xarray.DataArray
        Field with horizontal dimensions (``lat`` and ``lon``, or ``y`` and
        ``x``), and optionally a vertical dimension and ``time``.
    x, y : array-like
        Point coordinates: longitude and latitude for a ``lat``/``lon``
        field.
    z : array-like, optional
        Vertical coordinate of each point, in the units of the field's
        vertical dimension. Required when the field has one. Matched to the
        nearest level, so a point above the top level takes the top level.
    times : array-like, optional
        Time of each point. Required when the field has a ``time``
        dimension. By default matched to the nearest time, so times
        outside the field's span take the first or last step. See
        ``step_starts``.
    fill_value : float, default NaN
        Value for points outside the field and for missing cells. Use 0 for
        a surface flux, where no data means no emission. ``NaN`` suits a
        mole fraction, which is unknown there.
    step_starts : bool, default False
        Whether the field's ``time`` labels are the starts of steps, as a
        flux inventory's are. A point then takes the last step that starts
        at or before its time, the rule of ``foot.stilt.aggregate`` and
        ``project.jacobian``. A point before the first label takes
        ``fill_value`` and a point after the last takes the last step, since
        the labels do not say where it ends. The labels must strictly
        increase: a repeated label is an error. When ``field["time"]`` is a
        :class:`pandas.IntervalIndex` (its CF time bounds, read as each
        step's start and end), a point is matched by which step's bounds
        hold it instead: one before the first bound, after the last, or in
        a gap between bounds takes ``fill_value``, since the bounds say
        where every step starts and ends and there is nothing to guess at
        either edge. The bounds must not overlap.

    Returns
    -------
    numpy.ndarray
        One value per point.
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
        index = field.indexes["time"]
        if step_starts:
            if isinstance(index, pd.IntervalIndex):
                it, in_step = _bounds_indexer(index, stamps)
            else:
                if not index.is_monotonic_increasing:
                    raise ValueError(
                        "The field's time labels must strictly increase to be read "
                        "as step starts."
                    )
                if not index.is_unique:
                    repeated = index[index.duplicated()][0]
                    raise ValueError(
                        "The field's time labels must strictly increase to be read "
                        f"as step starts, but {repeated} is repeated. Two steps "
                        "cannot start at the same time; drop one, such as the "
                        "overlap of two concatenated files."
                    )
                it = index.searchsorted(stamps, side="right") - 1
                in_step = it >= 0
                it = np.clip(it, 0, None)
            inside = inside & in_step
        else:
            it = index.get_indexer(stamps, method="nearest")
        indexers["time"] = xr.DataArray(it, dims="points")
    sampled = field.isel(indexers).to_numpy().astype(float)
    return np.where(inside & ~np.isnan(sampled), sampled, fill_value)


def _nearest_level(levels: xr.DataArray, z: np.ndarray) -> np.ndarray:
    """Return the index of the level nearest each ``z``."""
    coords = levels.to_numpy().astype(float)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("The vertical coordinate must be a non-empty 1-D array.")
    return np.abs(coords[None, :] - z[:, None]).argmin(axis=1)


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
        returned, add it: the receptor time plus ``age`` minutes.
    files : iterable of path-like
        ARL met files that cover the particles' times, such as the files the
        run read (``met_files`` of :class:`stilt.transport.ModelRun` or of
        :func:`stilt.particles.particles_metadata`).
    columns : iterable of str, default ("mlht", "dens", "pres", "zsfc")
        The columns to add.
    z_top : float or None, default 25000
        The top of the run's domain in meters (``z_top`` of
        :class:`stilt.transport.hysplit.HysplitConfig`), since HYSPLIT's
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
    >>> meta = stilt.particles.particles_metadata(sim.particles_path)  # doctest: +SKIP
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
            "sample_met needs a 'time' column: the receptor time plus 'age' minutes."
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
        levels (``z_top`` of :class:`stilt.transport.hysplit.HysplitConfig`).

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
    >>> meta = stilt.particles.particles_metadata(sim.particles_path)  # doctest: +SKIP
    >>> mlht = stilt.sampling.calc_pbl_height(
    ...     sim.particles, meta.met_files
    ... )  # doctest: +SKIP
    """
    import arlmet

    if "time" not in particles.columns:
        raise ValueError(
            "calc_pbl_height needs a 'time' column: the receptor time plus 'age' minutes."
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
    "nearest_cell",
    "is_grib",
    "sample_field",
    "sample_grib_pressure",
    "sample_met",
    "vertical_dim",
]


# -- GRIB met (GFS) -------------------------------------------------------------


def is_grib(path: str | os.PathLike[str]) -> bool:
    """
    Return whether *path* is a GRIB file, by the ``GRIB`` its first message starts with.

    A path that is not a file is not GRIB, so the ARL reader reports it.
    """
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        return f.read(4) == b"GRIB"


def sample_grib_pressure(
    particles: pd.DataFrame, files: Iterable[str | os.PathLike[str]]
) -> np.ndarray:
    """
    Return the pressure at each particle's position from GFS or ERA5 GRIB files, in hPa.

    For a transport model that reads GRIB, such as FLEXPART, this is what
    :func:`sample_met` is for ARL files. Each row is sampled at its ``lon``,
    ``lat``, ``zagl``, and ``time``: bilinear across the grid, then linear
    in time between the two files around the particle's. In a column,
    ``ln p`` is linear in height from the surface pressure ``sp`` at the
    terrain, so ``zagl = 0`` gives ``sp`` itself, up through:

    - GFS: the pressure levels whose geopotential height ``gh`` is above the
      terrain ``orog``. Geopotential meters are taken as meters.
    - ERA5, as flex_extract writes it: the model levels' half levels,
      ``p = a + b·sp``, whose heights above the terrain (the surface
      geopotential ``z`` over g) come from the hypsometric equation with
      each level's virtual temperature, ``t (1 + 0.608 q)``.

    Reading needs cfgrib (``pip install "pystilt[grib]"``). Each file is
    read once per call; a GFS 0.5° file takes a few seconds.

    Parameters
    ----------
    particles : pandas.DataFrame
        Rows with ``lon``, ``lat``, ``zagl`` (m above ground), and ``time``.
    files : iterable of path-like
        GRIB files, one valid time each, covering the particles' times: GFS
        (``pgrb2``) with ``sp`` and ``orog`` at the surface and ``gh`` on
        pressure levels, or ERA5 with ``sp`` and ``z`` at the surface and
        ``t`` and ``q`` on model levels.

    Returns
    -------
    numpy.ndarray
        Pressure in hPa, one per row. A row above the highest level, or
        with no height, is NaN.

    Raises
    ------
    ImportError
        If cfgrib is not installed.
    ValueError
        If two files have the same valid time, or a row's time is outside
        the files' times.
    """
    try:
        import cfgrib  # noqa: F401
    except ImportError as err:
        raise ImportError(
            'Reading GRIB met needs cfgrib: pip install "pystilt[grib]"'
        ) from err
    by_time: dict[np.datetime64, str] = {}
    for path in files:
        valid = _grib_valid_time(os.fspath(path))
        if valid in by_time:
            raise ValueError(
                f"{by_time[valid]} and {os.fspath(path)} are both for {valid}."
            )
        by_time[valid] = os.fspath(path)
    times = np.array(sorted(by_time), dtype="datetime64[ns]")
    when = particles["time"].to_numpy(dtype="datetime64[ns]")
    if len(times) == 0 or when.min() < times[0] or when.max() > times[-1]:
        raise ValueError(
            "The particles' times run outside the GRIB files' valid times "
            f"({times[0] if len(times) else None} to "
            f"{times[-1] if len(times) else None})."
        )
    # The file at or before each row, and the weight of the next one.
    # One file covers only its own time, so it is both.
    after = np.clip(
        np.searchsorted(times, when, side="right"),
        min(1, len(times) - 1),
        len(times) - 1,
    )
    before = np.maximum(after - 1, 0)
    span = (times[after] - times[before]).astype(float)
    weight = (when - times[before]).astype(float) / np.where(span > 0, span, 1.0)

    lon = particles["lon"].to_numpy(dtype=float)
    lat = particles["lat"].to_numpy(dtype=float)
    zagl = particles["zagl"].to_numpy(dtype=float)
    out = np.zeros(len(particles))
    columns: dict[int, dict] = {}
    for k in np.unique(np.r_[before, after]):
        rows = (before == k) | (after == k)
        share = np.where(before == k, 1.0 - weight, 0.0) + np.where(
            after == k, weight, 0.0
        )
        if k not in columns:
            columns[k] = _grib_columns(by_time[times[k]])
        ln_p = _grib_ln_pressure(columns[k], lon[rows], lat[rows], zagl[rows])
        out[rows] += share[rows] * ln_p
    return np.exp(out)


def _grib_valid_time(path: str) -> np.datetime64:
    """Return the valid time of a GRIB file's surface pressure."""
    with _open_grib(path, "surface", "sp") as ds:
        return ds["valid_time"].values.astype("datetime64[ns]").reshape(-1)[0]


def _open_grib(path: str, level: str, name: str) -> xr.Dataset:
    """Open one field of a GRIB file with cfgrib, writing no index file beside it."""
    return xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": level, "shortName": name},
            "indexpath": "",
        },
    )


#: Gas constant of dry air, J kg⁻¹ K⁻¹, for ERA5's model-level heights.
_DRY_AIR_GAS_CONSTANT = 287.06


def _grib_columns(path: str) -> dict:
    """
    Read one file's columns: GFS on pressure levels, or ERA5 on model levels.

    GFS gives the grid, ``sp`` (Pa), ``orog`` (m), the levels (hPa), and
    ``gh`` (m). ERA5, as flex_extract writes it, gives ``sp``, the surface
    geopotential ``z`` (as meters), ``t`` and ``q`` on the model levels from
    the top down, and the levels' ``a`` and ``b`` coefficients (Pa, and none).
    """
    with _open_grib(path, "surface", "sp") as ds:
        sp = ds["sp"].values.astype(float)
        lats = ds["latitude"].values.astype(float)
        lons = ds["longitude"].values.astype(float)
    grid = {"lats": lats, "lons": lons, "sp": sp}
    if _has_field(path, "isobaricInhPa", "gh"):
        with _open_grib(path, "surface", "orog") as ds:
            orog = ds["orog"].values.astype(float)
        with _open_grib(path, "isobaricInhPa", "gh") as ds:
            levels = ds["isobaricInhPa"].values.astype(float)
            gh = ds["gh"].values.astype(float)
        order = np.argsort(-levels)  # bottom up: pressure falls with height
        return {
            **grid,
            "kind": "pressure",
            "orog": orog,
            "levels": levels[order],
            "gh": gh[order],
        }
    with _open_grib(path, "surface", "z") as ds:
        orog = ds["z"].values.astype(float) / 9.80665
    with _open_grib(path, "hybrid", "t") as ds:
        model_levels = ds["hybrid"].values.astype(int)
        t = ds["t"].values.astype(float)
    with _open_grib(path, "hybrid", "q") as ds:
        q = ds["q"].values.astype(float)
    a, b = _hybrid_coefficients(path)
    order = np.argsort(model_levels)  # from the top down, as a and b are
    return {
        **grid,
        "kind": "hybrid",
        "orog": orog,
        "t": t[order],
        "q": q[order],
        "levels": model_levels[order],
        "a": a,
        "b": b,
    }


def _has_field(path: str, level: str, name: str) -> bool:
    """Return whether the GRIB file at *path* has the field *name* on *level*."""
    import eccodes

    with open(path, "rb") as f:
        while (handle := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                if (
                    eccodes.codes_get(handle, "shortName") == name
                    and eccodes.codes_get(handle, "typeOfLevel") == level
                ):
                    return True
            finally:
                eccodes.codes_release(handle)
    return False


def _hybrid_coefficients(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the model levels' half-level ``a`` (Pa) and ``b``, from the top, from the first model-level message."""
    import eccodes

    with open(path, "rb") as f:
        while (handle := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                if eccodes.codes_get(handle, "typeOfLevel") == "hybrid":
                    pv = np.asarray(eccodes.codes_get_array(handle, "pv"), dtype=float)
                    half = len(pv) // 2
                    return pv[:half], pv[half:]
            finally:
                eccodes.codes_release(handle)
    raise ValueError(
        f"{path} has no model-level field to read the level coefficients from."
    )


def _profile(
    columns: dict, lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return each point's terrain (m), and its column's heights above sea level (m) and ``ln`` pressures (hPa), from the ground up.

    GFS: the surface, then the pressure levels above the terrain; a level
    below it takes the surface's height and pressure, so the profile only
    rises. ERA5: the half levels, ``p = a + b·sp``, with heights from the
    hypsometric equation on each model level's virtual temperature,
    ``T (1 + 0.608 q)``; the top half level (``p = 0``) is left out.
    """
    corners = _bilinear_corners(columns["lats"], columns["lons"], lat, lon)

    def at(field: np.ndarray) -> np.ndarray:
        out = 0.0
        for i, j, w in corners:
            out = out + (
                w * field[i, j] if field.ndim == 2 else w[:, None] * field[:, i, j].T
            )
        return np.asarray(out)

    sp_at = at(columns["sp"])
    orog_at = at(columns["orog"])
    if columns["kind"] == "pressure":
        gh_at = at(columns["gh"])
        above = gh_at > orog_at[:, None]
        height = np.column_stack([orog_at, np.where(above, gh_at, orog_at[:, None])])
        ln_p = np.column_stack(
            [
                np.log(sp_at / 100.0),
                np.where(
                    above,
                    np.log(columns["levels"])[None, :],
                    np.log(sp_at / 100.0)[:, None],
                ),
            ]
        )
        return orog_at, height, ln_p
    a, b = columns["a"], columns["b"]
    levels = columns[
        "levels"
    ]  # model levels n, from the top; half levels n - 1 and n bound them
    tv = at(columns["t"]) * (1.0 + 0.608 * at(columns["q"]))
    half = a[None, :] + b[None, :] * sp_at[:, None]  # (rows, half levels from the top)
    upper, lower = half[:, levels - 1], half[:, levels]
    with np.errstate(divide="ignore"):
        ratio = np.where(
            upper > 0, np.log(lower / np.where(upper > 0, upper, 1.0)), np.log(2.0)
        )
    thickness = _DRY_AIR_GAS_CONSTANT * tv / 9.80665 * ratio
    # Heights of the lower half levels from the ground up: the ground, then each layer's top.
    tops = orog_at[:, None] + np.cumsum(thickness[:, ::-1], axis=1)
    height = np.column_stack([orog_at, tops[:, :-1]])
    ln_p = np.log(lower[:, ::-1] / 100.0)
    return orog_at, height, ln_p


def _grib_ln_pressure(
    columns: dict, lon: np.ndarray, lat: np.ndarray, zagl: np.ndarray
) -> np.ndarray:
    """Return ``ln`` of the pressure in hPa at each point, linear in height through its column."""
    orog_at, height, ln_p = _profile(columns, lon, lat)
    z = orog_at + zagl
    upper = (height < z[:, None]).sum(axis=1)
    rows = np.arange(len(z))
    top = upper >= height.shape[1]
    upper = np.clip(upper, 1, height.shape[1] - 1)
    h0, h1 = height[rows, upper - 1], height[rows, upper]
    p0, p1 = ln_p[rows, upper - 1], ln_p[rows, upper]
    frac = np.where(h1 > h0, (z - h0) / np.where(h1 > h0, h1 - h0, 1.0), 0.0)
    result = p0 + np.clip(frac, 0.0, 1.0) * (p1 - p0)
    return np.where(top | np.isnan(z), np.nan, result)


def _bilinear_corners(
    lats: np.ndarray, lons: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return the four (lat index, lon index, weight) of each point on a regular grid.

    A global grid wraps across its last longitude; a point off a regional
    grid takes its nearest edge.
    """
    dlat = lats[1] - lats[0]
    dlon = lons[1] - lons[0]
    fi = (lat - lats[0]) / dlat
    fj = ((lon - lons[0]) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lats) - 2)
    wi = np.clip(fi - i0, 0.0, 1.0)
    if np.isclose(len(lons) * dlon, 360.0):
        j0 = np.floor(fj).astype(int) % len(lons)
        j1 = (j0 + 1) % len(lons)
        wj = fj - np.floor(fj)
    else:
        j0 = np.clip(np.floor(fj).astype(int), 0, len(lons) - 2)
        j1 = j0 + 1
        wj = np.clip(fj - j0, 0.0, 1.0)
    return [
        (i0, j0, (1 - wi) * (1 - wj)),
        (i0, j1, (1 - wi) * wj),
        (i0 + 1, j0, wi * (1 - wj)),
        (i0 + 1, j1, wi * wj),
    ]
