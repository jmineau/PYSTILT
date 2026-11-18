"""
The release table: one row per release point, as ``releases.csv`` holds it.

:func:`release_rows` checks a table quickly, without building releases,
and the functions here convert between tables, CSV files, and releases.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from io import StringIO
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from .models import (
    _TIME_FORMS,
    ColumnRelease,
    MultiPointRelease,
    PointRelease,
    Release,
    _column_location,
    _multipoint_locations,
    _parse_time_text,
    _point_location,
)
from .validation import column_errors, multipoint_errors, point_errors

#: The columns of a release table, one row per release point.
COLUMNS = ("group", "time", "longitude", "latitude", "altitude", "altitude_ref")

#: Column names :func:`read_releases` accepts for each field, in any case.
ALIASES: dict[str, tuple[str, ...]] = {
    "group": ("group",),
    "time": ("time",),
    "longitude": ("longitude", "long", "lon"),
    "latitude": ("latitude", "lati", "lat"),
    "altitude": ("altitude", "zagl", "zmsl", "z"),
    "altitude_ref": ("altitude_ref", "height_ref"),
}


# ---------------------------------------------------------------------------
# The release table
# ---------------------------------------------------------------------------


def releases_to_frame(releases: Iterable[Release]) -> pd.DataFrame:
    """
    Return releases as one table with a row per release point.

    The columns are ``group`` (the release's position, shared by the rows
    of a column or multipoint release), ``time``, ``longitude``,
    ``latitude``, ``altitude``, ``altitude_ref``, and one column per label
    in any release's ``attrs``.
    """
    releases = list(releases)
    labels = list(dict.fromkeys(k for r in releases for k in r.attrs))
    rows = [
        {
            "group": idx,
            "time": r.time,
            "longitude": lon,
            "latitude": lat,
            "altitude": alt,
            "altitude_ref": r.altitude_ref,
            **{k: r.attrs.get(k) for k in labels},
        }
        for idx, r in enumerate(releases)
        for lat, lon, alt in r.coords()
    ]
    frame = pd.DataFrame(rows, columns=[*COLUMNS, *labels])
    return frame.astype({"time": "datetime64[ns]"})


def _normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Return *frame* with the release columns under their standard names.

    Also returns the file's own spelling of each standard column. Any other
    column is kept as it is. A ``zmsl`` altitude column implies ``msl``.

    Raises
    ------
    ValueError
        If a column is named ``r_idx``, the old name of ``group``.
    """
    old = [str(c) for c in frame.columns if str(c).lower() == "r_idx"]
    if old:
        raise ValueError(
            f"The column {old[0]!r} is not read. Rename it to 'group': rows "
            "that share a group make one column or multipoint release."
        )
    spelling: dict[str, str] = {}
    for field, aliases in ALIASES.items():
        for col in frame.columns:
            if str(col).lower() in aliases:
                spelling[field] = str(col)
                break
    frame = frame.rename(columns={name: field for field, name in spelling.items()})
    if "altitude_ref" not in frame.columns:
        implied = "msl" if spelling.get("altitude", "").lower() == "zmsl" else "agl"
        _refuse_misnamed_altitude_ref(frame, implied)
        frame = frame.assign(altitude_ref=implied)
    return frame, spelling


def _refuse_misnamed_altitude_ref(frame: pd.DataFrame, implied: str) -> None:
    """
    Raise if a label column holds only ``agl`` and ``msl``, as an ``altitude_ref`` column would.

    Under another name, such as ``alt_ref``, the column would be kept as a
    label and every height read as *implied*: 1500 m above sea level as
    1500 m above ground.
    """
    for col in frame.columns:
        if col in COLUMNS or not pd.api.types.is_string_dtype(frame[col]):
            continue
        values = pd.unique(frame[col].dropna())
        refs = {str(v).strip().lower() for v in values}
        if refs and refs <= {"agl", "msl"}:
            raise ValueError(
                f"The column {col!r} holds altitude references "
                f"({', '.join(sorted(refs))}), but only a column named "
                f"'altitude_ref' or 'height_ref' is read as one, so every height "
                f"would be taken as {implied}. Did you mean 'altitude_ref'? "
                "Rename the column."
            )


#: Columns :func:`release_rows` adds; a releases file may not use these names.
ROW_COLUMNS = ("release", "kind", "location")


def release_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Check a release table and return it with each row's release id, kind, and location.

    The table has a row per release point, as ``releases.csv`` does, with
    ``time``, ``longitude``, ``latitude``, and ``altitude`` columns in any of
    the accepted spellings (see :func:`releases_from_frame`). The result has
    the standard column names, ``altitude_ref``, and three more columns:
    ``release`` (the id), ``kind`` (``point``, ``column``, or
    ``multipoint``), and ``location`` (the location id). A release listed
    twice keeps its first rows. No release object is built, so this is quick
    on a large file; :func:`releases_from_rows` builds the releases.

    Raises
    ------
    ValueError
        For anything a release would refuse: a missing column, a longitude
        or latitude out of range or empty, a height that is empty or not
        finite, a negative height above ground, rows of one group that
        differ in time or altitude reference, a column whose two heights are
        equal, a multipoint release with two points at one location, or two
        different releases with one id. Also for a misnamed
        ``altitude_ref`` column (:func:`releases_from_frame`).
    """
    frame, _ = _normalize_columns(frame)
    required = ("time", "longitude", "latitude", "altitude")
    if any(c not in frame.columns for c in required):
        raise ValueError(f"Release table must contain columns: {list(required)}")
    clash = [c for c in frame.columns if c in ROW_COLUMNS]
    if clash:
        raise ValueError(
            f"A releases file may not have a column named {clash}; PYSTILT uses "
            "those names. Rename the column."
        )
    n = len(frame)
    lon = frame["longitude"].to_numpy(dtype=float)
    lat = frame["latitude"].to_numpy(dtype=float)
    alt = frame["altitude"].to_numpy(dtype=float)
    ref = frame["altitude_ref"].astype(str).str.lower().to_numpy()
    # Group keys as text: a file that mixes numeric and string ids must not
    # split one release into two. Without a group column, every row is a release.
    keys = (
        frame["group"].astype(str).to_numpy()
        if "group" in frame.columns
        else np.arange(n).astype(str)
    )
    codes, uniques = pd.factorize(keys)

    def fail(mask: np.ndarray, message: str) -> None:
        """Raise *message* for the first row or group where *mask* is set."""
        if mask.any():
            where = int(np.flatnonzero(mask)[0])
            label = (
                f"group={uniques[codes[where]]}" if "group" in frame else f"row {where}"
            )
            raise ValueError(f"{label}: {message}")

    time = _times(frame["time"], fail)
    fail(~np.isin(ref, ("agl", "msl")), "altitude_ref must be 'agl' or 'msl'.")
    for bad, message in point_errors(lon, lat, alt, ref):
        fail(bad, message)

    g = pd.DataFrame(
        {
            "code": codes,
            "time": time.to_numpy(),
            "ref": ref,
            "lon": lon,
            "lat": lat,
            "alt": alt,
        }
    ).groupby("code", sort=False)
    size = g["code"].transform("size").to_numpy()
    fail(
        g["ref"].transform("nunique").to_numpy() > 1,
        "All rows in one release group must share the same altitude_ref.",
    )
    fail(
        g["time"].transform("nunique").to_numpy() > 1,
        "All rows in one release group must share the same release time.",
    )
    same_xy = (g["lon"].transform("nunique").to_numpy() == 1) & (
        g["lat"].transform("nunique").to_numpy() == 1
    )
    kind = np.where(
        size == 1, "point", np.where((size == 2) & same_xy, "column", "multipoint")
    )
    # A column's two rows may come in either order.
    bottom = g["alt"].transform("min").to_numpy()
    top = g["alt"].transform("max").to_numpy()
    for bad, message in column_errors(bottom, top):
        fail((kind == "column") & bad, message)

    # Ids are made once per release, from its first row (or all its rows for
    # a multipoint), then spread to its rows. Python floats round much faster
    # than numpy scalars, and give the same result.
    _, first_rows = np.unique(codes, return_index=True)  # groups are 0, 1, ... in order
    lon_f, lat_f, alt_f = lon.tolist(), lat.tolist(), alt.tolist()
    group_kind = kind[first_rows]
    group_ref = ref[first_rows].tolist()
    bottom_f, top_f = bottom.tolist(), top.tolist()
    group_location = np.empty(len(first_rows), dtype=object)
    for code, row in enumerate(first_rows):
        if group_kind[code] == "point":
            group_location[code] = _point_location(
                lon_f[row], lat_f[row], alt_f[row], group_ref[code]
            )
        elif group_kind[code] == "column":
            group_location[code] = _column_location(
                lon_f[row], lat_f[row], bottom_f[row], top_f[row], group_ref[code]
            )
    multi = np.flatnonzero(kind == "multipoint")
    if len(multi):
        for bad, message in multipoint_errors(lon[multi], lat[multi], codes[multi]):
            dup = np.zeros(n, dtype=bool)
            dup[multi] = bad
            fail(dup, message)
        found = _multipoint_locations(
            codes[multi], lon[multi], lat[multi], alt[multi], group_ref
        )
        for code, location in found.items():
            group_location[code] = location
    stamps = time.iloc[first_rows].dt.strftime("%Y%m%d%H%M").to_numpy(dtype=object)
    group_ids = stamps + "_" + group_location
    location = group_location[codes]
    ids = group_ids[codes]
    out = frame.assign(
        time=time, altitude_ref=ref, release=ids, kind=kind, location=location
    )

    # A release id names its result files: one id, one release.
    repeated = np.flatnonzero(pd.Index(group_ids).duplicated(keep=False))
    if len(repeated):
        # Build every repeated group's release in one call, each group on
        # its own (keyed by its code, not its id), in code order.
        rows = np.isin(codes, repeated)
        built = releases_from_rows(out.loc[rows].assign(release=codes[rows]))
        release_of = dict(zip(repeated.tolist(), built, strict=True))
        by_id: dict[str, list[int]] = {}
        for code in repeated.tolist():
            by_id.setdefault(str(group_ids[code]), []).append(code)
        drop: list[int] = []
        for group in by_id.values():
            check_distinct_ids([release_of[code] for code in group])
            drop.extend(group[1:])  # the same release listed again
        out = out.loc[~np.isin(codes, drop)]
    return out


def _times(column: pd.Series, fail: Callable[[np.ndarray, str], None]) -> pd.Series:
    """
    Return a release table's times as naive UTC times, calling *fail* on a row that has none.

    Text must be ISO 8601 or ``YYYYMMDDHHMM`` (:func:`_parse_time_text`);
    datetimes are taken as they are, an aware one converted to UTC.
    """
    if pd.api.types.is_numeric_dtype(column):
        fail(
            np.ones(len(column), dtype=bool),
            f"time must be {_TIME_FORMS}; got the number {column.iloc[0]!r}.",
        )
    if pd.api.types.is_string_dtype(column):
        time = _parse_time_text(column)
        bad = time.isna().to_numpy()
        if bad.any():
            value = column.iloc[int(np.flatnonzero(bad)[0])]
            what = (
                "empty" if pd.isna(value) else f"{value!r}, which is not {_TIME_FORMS}"
            )
            fail(bad, f"time is {what}.")
    else:
        time = pd.to_datetime(column, utc=True)
        fail(time.isna().to_numpy(), "time is empty.")
    return time.dt.tz_localize(None)


def _build(
    kind: str,
    time: dt.datetime,
    altitude_ref: str,
    attrs: dict[str, Any],
    lon: np.ndarray,
    lat: np.ndarray,
    alt: np.ndarray,
) -> Release:
    """
    Return the release of one group of checked rows, given as plain values.

    The rows passed :func:`release_rows`, which runs every check a release
    runs, so the release is built without running them again.
    """
    common = {"time": time, "altitude_ref": altitude_ref, "attrs": attrs}
    if kind == "point":
        return PointRelease.model_construct(
            longitude=float(lon[0]),
            latitude=float(lat[0]),
            altitude=float(alt[0]),
            **common,
        )
    if kind == "column":
        return ColumnRelease.model_construct(
            longitude=float(lon[0]),
            latitude=float(lat[0]),
            bottom=float(alt.min()),
            top=float(alt.max()),
            **common,
        )
    return MultiPointRelease.model_construct(
        longitudes=tuple(lon.tolist()),
        latitudes=tuple(lat.tolist()),
        altitudes=tuple(alt.tolist()),
        **common,
    )


def releases_from_rows(rows: pd.DataFrame) -> list[Release]:
    """
    Build the releases of a table that :func:`release_rows` checked.

    The rows are not checked again. Each release's labels (``attrs``) are
    the other columns of its first row.

    Returns
    -------
    list of Release
        In table order, one per release id.
    """
    names = [str(c) for c in rows.columns if c not in COLUMNS and c not in ROW_COLUMNS]
    labels = rows[names].astype(object).to_numpy()
    times = pd.DatetimeIndex(rows["time"]).to_pydatetime()
    lon = rows["longitude"].to_numpy(dtype=float)
    lat = rows["latitude"].to_numpy(dtype=float)
    alt = rows["altitude"].to_numpy(dtype=float)
    kind = rows["kind"].to_numpy()
    ref = rows["altitude_ref"].to_numpy()
    releases = []
    for idx in rows.groupby("release", sort=False).indices.values():
        i = idx[0]
        attrs = {
            k: (None if pd.isna(v) else v)
            for k, v in zip(names, labels[i], strict=True)
        }
        releases.append(
            _build(
                str(kind[i]),
                times[i],
                str(ref[i]),
                attrs,
                lon[idx],
                lat[idx],
                alt[idx],
            )
        )
    return releases


def releases_from_frame(frame: pd.DataFrame) -> list[Release]:
    """
    Check a table with a row per release point and build its releases.

    The table needs ``time``, ``longitude``, ``latitude``, and ``altitude``
    columns. Each row is one :class:`PointRelease`, unless a ``group``
    column groups rows into one release: a group of two rows at one
    location is a :class:`ColumnRelease`, any other group a
    :class:`MultiPointRelease`.

    ======================  =============================================
    Field                   Accepted column names (any case)
    ======================  =============================================
    time                    ``time``
    longitude               ``longitude``, ``long``, ``lon``
    latitude                ``latitude``, ``lati``, ``lat``
    altitude (m)            ``altitude``, ``zagl``, ``zmsl``, ``z``
    release group           ``group``
    altitude reference      ``altitude_ref``, ``height_ref``
    ======================  =============================================

    Without an ``altitude_ref`` column, altitudes are above mean sea level
    when the column is named ``zmsl`` and above ground level otherwise. Any
    other column becomes a label in each release's ``attrs``. For a group,
    labels come from its first row. Without an ``altitude_ref`` column, a
    label column that holds only ``agl`` and ``msl``, such as ``alt_ref``,
    is refused, since its heights would be read with the wrong reference.

    Returns
    -------
    list of Release
        In table order. A release listed twice appears once.

    Raises
    ------
    ValueError
        As :func:`release_rows` raises.
    """
    return releases_from_rows(release_rows(frame))


def check_distinct_ids(releases: Iterable[Release]) -> None:
    """
    Raise if two different releases share an id.

    An id names a release's result files, so two releases with one id
    would overwrite each other. Equal releases listed twice are fine.
    """
    seen: dict[str, Release] = {}
    for r in releases:
        other = seen.setdefault(r.id, r)
        if other is not r and other != r:
            raise ValueError(
                f"Two different releases share the id {r.id!r}: {other!r} and {r!r}."
            )


# ---------------------------------------------------------------------------
# CSV files
# ---------------------------------------------------------------------------


def read_release_frame(path: str | Path | IO[str]) -> pd.DataFrame:
    """Read a releases CSV as a table, with ``group`` and ``time`` as text, without building releases."""
    header = pd.read_csv(path, nrows=0).columns
    if hasattr(path, "seek"):
        path.seek(0)
    # With type inference, pandas types each chunk of a large file on its
    # own, so a group column that mixes numbers and text would come back
    # part int and part str.
    # Times are read as text, and parsed by release_rows in the forms the
    # docs give, never guessed: pandas would read 05/07/2023 month-first.
    dtype: dict[Any, Any] = {
        c: str for c in header if str(c).lower() in ("group", "time")
    }
    return pd.read_csv(path, dtype=dtype)


def read_releases(path: str | Path | IO[str]) -> list[Release]:
    """
    Read releases from a CSV file.

    The file needs columns for time, longitude, latitude, and altitude, and
    may group rows into one release with ``group``. See
    :func:`releases_from_frame` for the rules and the accepted column
    names.

    Parameters
    ----------
    path : str, Path or file-like
        CSV file path or open text stream.

    Returns
    -------
    list of Release
        In file order.
    """
    return releases_from_frame(read_release_frame(path))


def _csv_frame(releases: Iterable[Release]) -> pd.DataFrame:
    """Return the release table with times as the text written to CSV."""
    frame = releases_to_frame(releases)
    return frame.assign(
        time=pd.DatetimeIndex(frame["time"]).strftime("%Y-%m-%d %H:%M:%S")
    )


def releases_to_csv(releases: Iterable[Release]) -> str:
    """Return releases as CSV text that :func:`read_releases` reads back."""
    return _csv_frame(releases).to_csv(index=False, lineterminator="\n")


def write_releases(releases: Iterable[Release], path: str | Path) -> Path:
    """Write releases to a CSV file that :func:`read_releases` reads, and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(releases_to_csv(releases))
    return path


def append_releases_csv(text: str, releases: Iterable[Release]) -> str:
    """
    Return the text of a releases CSV with rows for more releases appended.

    The existing header sets the columns and their order, so a hand-written
    file keeps its column names and ``group`` values, and its rows are kept
    as written. Label columns are filled from each release's ``attrs`` or
    left empty. New releases are numbered after the largest ``group`` in
    the file. When a new release's heights are above sea level and the file
    has no ``altitude_ref`` column, the column is added, with ``agl`` on the
    existing rows.

    Raises
    ------
    ValueError
        If the file lacks a time, longitude, latitude, or altitude column,
        has no ``group`` column for a column or multipoint release, or
        names its altitude column ``zagl``/``zmsl`` for a different altitude
        reference than a new release uses.
    """
    releases = list(releases)
    if not text.strip():
        return releases_to_csv(releases)
    columns = list(pd.read_csv(StringIO(text), nrows=0).columns)
    _, spelling = _normalize_columns(pd.DataFrame(columns=columns))
    missing = [
        f for f in ("time", "longitude", "latitude", "altitude") if f not in spelling
    ]
    if missing:
        raise ValueError(f"releases.csv lacks a column for {missing}; cannot append.")
    if "group" not in spelling and any(len(r.coords()) > 1 for r in releases):
        raise ValueError(
            "releases.csv has no group column, so a column or multipoint release "
            "cannot be appended; add a group column to the file."
        )
    if "altitude_ref" not in spelling:
        file_ref = {"zagl": "agl", "zmsl": "msl"}.get(spelling["altitude"].lower())
        for r in releases:
            if file_ref is not None and r.altitude_ref != file_ref:
                raise ValueError(
                    f"releases.csv altitudes are {file_ref}; release {r.id} is "
                    f"{r.altitude_ref}. Add an altitude_ref column to mix them."
                )
        if file_ref is None and any(r.altitude_ref != "agl" for r in releases):
            # The one case that rewrites the existing rows: they gain the column.
            existing = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
            text = existing.assign(altitude_ref="agl").to_csv(
                index=False, lineterminator="\n"
            )
            columns.append("altitude_ref")
            spelling["altitude_ref"] = "altitude_ref"

    new = _csv_frame(releases)
    if "group" in spelling:
        ids = pd.read_csv(
            StringIO(text),
            usecols=[spelling["group"]],
            dtype=str,
            keep_default_na=False,
        ).iloc[:, 0]
        numbers = [int(v) for v in ids if v.strip().lstrip("-").isdigit()]
        new["group"] += max(numbers, default=-1) + 1
    new = new.rename(columns=spelling).reindex(columns=columns)
    rows = new.astype(object).where(new.notna(), "")
    body = text if text.endswith("\n") else text + "\n"
    return body + rows.to_csv(index=False, header=False, lineterminator="\n")
