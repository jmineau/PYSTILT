"""
Sampling gridded fields, such as a surface flux or a mole fraction, at points.

A field is an :class:`xarray.DataArray` on a regular ``lat``/``lon`` grid
(``y``/``x`` for a projected footprint grid), with an optional vertical
dimension and an optional ``time`` dimension. Each point takes the value of
the cell it falls in. ``foot.stilt.enhancement`` and
``particles.stilt.enhancement`` sample a surface flux this way, taking the
flux step that starts at or before each time, and
:func:`stilt.particles.background` a mole-fraction field, taking the
nearest time. PYSTILT does
not convert units. A flux in µmol m⁻² s⁻¹ times a footprint in ppm per
(µmol m⁻² s⁻¹) gives ppm.

:func:`sample_met` samples ARL met files at particle positions, for the
particle columns a transport model can write from the met (``mlht``,
``dens``, ``pres``, ``zsfc``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

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
        the labels do not say where it ends. The labels must increase.

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
            if not index.is_monotonic_increasing:
                raise ValueError(
                    "The field's time labels must increase to be read as step starts."
                )
            it = index.searchsorted(stamps, side="right") - 1
            inside = inside & (it >= 0)
            it = np.clip(it, 0, None)
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


def sample_met(
    particles: pd.DataFrame,
    files: Iterable[str | os.PathLike[str]],
    columns: Iterable[str] = ("mlht", "dens", "pres", "zsfc"),
) -> pd.DataFrame:
    """
    Return the particles with met columns sampled from ARL files at each particle's position.

    A transport model that writes only positions can get the columns
    HYSPLIT writes from the met this way. Each row is sampled at its
    ``lon``, ``lat``, ``zagl``, and ``time``: bilinear across the met's
    grid and linear in height between its levels
    (:func:`arlmet.sample_points`), then linear in time between the two
    met times around the particle's, as HYSPLIT does. Each column comes
    from these ARL variables:

    - ``mlht`` is ``PBLH``, the met model's boundary-layer height, in m
      above ground.
    - ``dens`` is 100 ``PRES`` / (R ``TEMP`` (1 + 0.61 ``SPHU``)), the
      density of moist air in kg m⁻³, with R = 287.04 J kg⁻¹ K⁻¹.
    - ``pres`` is ``PRES``, the pressure, in hPa.
    - ``zsfc`` is ``SHGT``, the terrain height, in m above sea level.

    The values are the met's, not HYSPLIT's. HYSPLIT's ``mlht`` is the
    met's ``PBLH`` only with ``kmixd: 0``, and then it is at least
    ``kmix0``; the default, ``kmixd: 3``, finds the mixed layer from the
    profile. HYSPLIT also finds the heights of the met's levels its own
    way, so its ``pres`` and ``dens`` differ from these;
    :doc:`/guides/meteorology` says by how much.

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

    Returns
    -------
    pandas.DataFrame
        A copy of *particles* with the columns added. A row off the met's
        grid, below its lowest level, or above its top is NaN.

    Raises
    ------
    ValueError
        If a column is not one of the four or is in *particles* already
        (drop it first), *particles* has no ``time`` column, or a row is
        outside the files' times.
    KeyError
        If the files lack a variable a column needs.

    Examples
    --------
    >>> meta = stilt.particles.particles_metadata(sim.particles_path)  # doctest: +SKIP
    >>> hysplit = sim.particles  # doctest: +SKIP
    >>> sampled = stilt.sampling.sample_met(  # doctest: +SKIP
    ...     hysplit.drop(columns="pres"), meta.met_files, columns=["pres"]
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
        sampled = arlmet.sample_points(handles, points, variables, z_kind="agl")
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


__all__ = [
    "nearest_cell",
    "sample_field",
    "sample_met",
    "vertical_dim",
]
