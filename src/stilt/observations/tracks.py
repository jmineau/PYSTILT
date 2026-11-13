"""
Receptors along the track of a moving platform.

A vehicle, a train, an aircraft, or a ship measures every second or so as it
moves. :func:`receptors_from_track` groups those measurements by time or by
distance along the track and makes a :class:`~stilt.PointReceptor` at the
mean time and position of each group. The table it returns gives each
measurement its receptor id, so the modeled values join back to the
measurements after a run. :data:`TRACK_SCHEMA` lists the columns a track
needs, and :func:`check_track` checks a table against it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from stilt.receptors import PointReceptor, VerticalReference
from stilt.spatial import haversine_km

#: The columns a track needs, as ``{name: meaning}``. A track may have any
#: other columns, such as the measured mole fractions, and
#: :func:`receptors_from_track` keeps them in its table.
TRACK_SCHEMA: dict[str, str] = {
    "time": "Time of the measurement. A time without a time zone is taken as UTC.",
    "longitude": "Longitude where the air was sampled, degrees.",
    "latitude": "Latitude where the air was sampled, degrees.",
    "altitude": (
        "Height where the air was sampled, in meters above ground or above sea "
        "level (the altitude_ref of receptors_from_track)."
    ),
}

_DEFAULT_EVERY = pd.Timedelta("5min")


def check_track(track: pd.DataFrame) -> None:
    """
    Raise unless a track has the columns in :data:`TRACK_SCHEMA`.

    Parameters
    ----------
    track : pandas.DataFrame
        One row per measurement.

    Raises
    ------
    ValueError
        Naming the missing columns.

    Examples
    --------
    >>> check_track(pd.DataFrame(columns=["time", "longitude", "latitude"]))
    Traceback (most recent call last):
    ...
    ValueError: The track lacks the columns ['altitude'] (see TRACK_SCHEMA).
    """
    missing = [name for name in TRACK_SCHEMA if name not in track.columns]
    if missing:
        raise ValueError(f"The track lacks the columns {missing} (see TRACK_SCHEMA).")


def receptors_from_track(
    track: pd.DataFrame,
    *,
    every: str | dt.timedelta | None = None,
    spacing: float | None = None,
    altitude_ref: VerticalReference = "agl",
    gap: str | dt.timedelta = "10min",
) -> tuple[list[PointReceptor], pd.DataFrame]:
    """
    Return a receptor for each group of measurements along a track, and their table.

    The measurements are grouped by time (``every``) or by distance along
    the track (``spacing``), and each group becomes one
    :class:`~stilt.PointReceptor` at the mean time and position of its
    measurements.

    Parameters
    ----------
    track : pandas.DataFrame
        One row per measurement, with the columns in :data:`TRACK_SCHEMA`,
        screened as you like. Its rows may come in any order.
    every : str or timedelta, optional
        Length of each group in time, such as ``"5min"``. Groups start on the
        clock: with ``"5min"``, at 18:00, 18:05, and so on. Five minutes when
        ``spacing`` is not given either.
    spacing : float, optional
        Length of each group along the track, in meters, in place of
        ``every``. The distance is summed between consecutive measurements.
    altitude_ref : {"agl", "msl"}, default "agl"
        Whether ``altitude`` is above ground (an inlet on a vehicle) or
        above sea level (an aircraft's GPS altitude).
    gap : str or timedelta, default "10min"
        A pause in the track longer than this ends a group, so a group never
        holds measurements from both sides of it.

    Returns
    -------
    receptors : list of PointReceptor
        One per group, in time order. Two groups whose rounded time and
        position agree share one receptor.
    table : pandas.DataFrame
        The track, in its own row order and index, with a ``receptor``
        column holding each measurement's receptor id. An existing
        ``receptor`` column is replaced. Keep it in the project with
        ``project.add_table("track", table)``.

    Raises
    ------
    ValueError
        If the track lacks a column in :data:`TRACK_SCHEMA` or a value in
        one, if it has no rows, if both ``every`` and ``spacing`` are given,
        or if ``every``, ``spacing``, or ``gap`` is not positive.

    Notes
    -----
    A receptor's time is its group's mean time, rounded to the nearest
    minute, because a run starts on a whole minute. Its longitude and
    latitude are the mean position, rounded to 5 decimals (about 1 m), and
    its altitude the mean height, rounded to 0.1 m. The mean longitude of a
    group that crosses 180 degrees is taken across it.

    A receptor stands for a stretch of track, so the receptor of a group
    that turns is inside the turn. Groups by ``every`` are as long
    as the distance the platform covers in that time; give an aircraft
    ``spacing`` to keep them short. With ``spacing``, a platform that stands
    still and keeps measuring stays in one group until it moves on or
    pauses for longer than ``gap``.

    Examples
    --------
    Ten minutes of a car driving east at about 10 m/s, measured every
    second:

    >>> track = pd.DataFrame(
    ...     {
    ...         "time": pd.date_range("2024-07-01 18:00", periods=600, freq="1s"),
    ...         "longitude": np.linspace(-111.90, -111.83, 600),
    ...         "latitude": 40.76,
    ...         "altitude": 3.0,
    ...     }
    ... )
    >>> receptors, table = receptors_from_track(track, every="5min")
    >>> [r.id for r in receptors]
    ['202407011802_-111.88253_40.76_3', '202407011807_-111.84747_40.76_3']
    >>> table["receptor"].value_counts().tolist()
    [300, 300]

    In a project:

    >>> project.add_receptors(receptors)  # doctest: +SKIP
    >>> project.add_table("track", table)  # doctest: +SKIP
    """
    check_track(track)
    if every is not None and spacing is not None:
        raise ValueError("Give every or spacing, not both.")
    columns = list(TRACK_SCHEMA)
    blank = [name for name in columns if track[name].isna().any()]
    if blank:
        raise ValueError(
            f"The track has missing values in {blank}. Drop those rows first."
        )
    if track.empty:
        raise ValueError("The track has no rows.")
    pause = pd.Timedelta(gap)
    if not pause > pd.Timedelta(0):
        raise ValueError(f"gap must be positive, not {gap!r}.")

    times = pd.to_datetime(track["time"])
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    order = np.argsort(times.to_numpy(), kind="stable")
    time = pd.Series(times.to_numpy()[order])
    lon = track["longitude"].to_numpy(dtype=float)[order]
    lat = track["latitude"].to_numpy(dtype=float)[order]
    alt = track["altitude"].to_numpy(dtype=float)[order]

    # A stretch is the track between two pauses longer than gap.
    breaks = time.diff().to_numpy()[1:] > pause.to_timedelta64()
    stretch = np.concatenate([[0], np.cumsum(breaks)])

    if spacing is None:
        length = _DEFAULT_EVERY if every is None else pd.Timedelta(every)
        if not length > pd.Timedelta(0):
            raise ValueError(f"every must be positive, not {every!r}.")
        within = time.dt.floor(length).to_numpy()
    else:
        if not spacing > 0:
            raise ValueError(f"spacing must be positive, not {spacing!r}.")
        step = 1000 * haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
        step[breaks] = 0.0  # each stretch starts its distance at zero
        along = np.concatenate([[0.0], np.cumsum(step)])
        start = along[np.searchsorted(stretch, stretch)]  # first row of each stretch
        within = np.floor((along - start) / spacing)
    keys = pd.DataFrame({"stretch": stretch, "within": within})
    group = keys.groupby(["stretch", "within"], sort=True).ngroup().to_numpy()

    # Longitudes relative to each group's first, so a group across 180 averages.
    first = np.unique(group, return_index=True)[1]
    lon0 = lon[first]
    offset = (lon - lon0[group] + 180) % 360 - 180
    means = (
        pd.DataFrame(
            {"group": group, "time": time, "offset": offset, "lat": lat, "alt": alt}
        )
        .groupby("group", sort=True)
        .mean()
    )
    receptor_times = means["time"].dt.round("min")
    receptor_lons = np.round((lon0 + means["offset"].to_numpy() + 180) % 360 - 180, 5)
    receptor_lats = np.round(means["lat"].to_numpy(), 5)
    receptor_alts = np.round(means["alt"].to_numpy(), 1)

    made = [
        PointReceptor(
            time=t,
            longitude=float(x),
            latitude=float(y),
            altitude=float(z),
            altitude_ref=altitude_ref,
        )
        for t, x, y, z in zip(
            receptor_times, receptor_lons, receptor_lats, receptor_alts, strict=True
        )
    ]
    ids = np.array([r.id for r in made], dtype=object)
    # Two receptors with one id have the same minute and position, so are
    # equal: keep the first.
    unique: dict[str, PointReceptor] = {}
    for r in made:
        unique.setdefault(r.id, r)

    receptor = np.empty(len(track), dtype=object)
    receptor[order] = ids[group]
    return list(unique.values()), track.assign(receptor=receptor)


__all__ = ["TRACK_SCHEMA", "check_track", "receptors_from_track"]
