"""
The receptor table: one row per release point, as ``receptors.csv`` holds it.

:func:`receptor_rows` checks a table quickly, without building receptors,
and the functions here convert between tables, CSV files, and receptors.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from .models import (
    ColumnReceptor,
    MultiPointReceptor,
    PointReceptor,
    Receptor,
    _column_location,
    _multipoint_locations,
    _point_location,
)
from .validation import column_errors, multipoint_errors, point_errors

#: The columns of a receptor table, one row per release point.
COLUMNS = ("r_idx", "time", "longitude", "latitude", "altitude", "altitude_ref")

#: Column names :func:`read_receptors` accepts for each field, in any case.
ALIASES: dict[str, tuple[str, ...]] = {
    "r_idx": ("r_idx",),
    "time": ("time",),
    "longitude": ("longitude", "long", "lon"),
    "latitude": ("latitude", "lati", "lat"),
    "altitude": ("altitude", "zagl", "zmsl", "z"),
    "altitude_ref": ("altitude_ref", "height_ref"),
}


# ---------------------------------------------------------------------------
# The receptor table
# ---------------------------------------------------------------------------


def receptors_to_frame(receptors: Iterable[Receptor]) -> pd.DataFrame:
    """
    Return receptors as one table with a row per release point.

    The columns are ``r_idx`` (the receptor's position, shared by the rows
    of a column or multipoint receptor), ``time``, ``longitude``,
    ``latitude``, ``altitude``, ``altitude_ref``, and one column per label
    in any receptor's ``attrs``.
    """
    receptors = list(receptors)
    labels = list(dict.fromkeys(k for r in receptors for k in r.attrs))
    rows = [
        {
            "r_idx": idx,
            "time": r.time,
            "longitude": lon,
            "latitude": lat,
            "altitude": alt,
            "altitude_ref": r.altitude_ref,
            **{k: r.attrs.get(k) for k in labels},
        }
        for idx, r in enumerate(receptors)
        for lat, lon, alt in r.coords()
    ]
    frame = pd.DataFrame(rows, columns=[*COLUMNS, *labels])
    return frame.astype({"time": "datetime64[ns]"})


def _normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Return *frame* with the receptor columns under their standard names.

    Also returns the file's own spelling of each standard column. Any other
    column is kept as it is. A ``zmsl`` altitude column implies ``msl``.
    """
    spelling: dict[str, str] = {}
    for field, aliases in ALIASES.items():
        for col in frame.columns:
            if str(col).lower() in aliases:
                spelling[field] = str(col)
                break
    frame = frame.rename(columns={name: field for field, name in spelling.items()})
    if "altitude_ref" not in frame.columns:
        implied = "msl" if spelling.get("altitude", "").lower() == "zmsl" else "agl"
        frame = frame.assign(altitude_ref=implied)
    return frame, spelling


#: Columns :func:`receptor_rows` adds; a receptors file may not use these names.
ROW_COLUMNS = ("receptor", "kind", "location")


def receptor_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Check a receptor table and return it with each row's receptor id, kind, and location.

    The table has a row per release point, as ``receptors.csv`` does, with
    ``time``, ``longitude``, ``latitude``, and ``altitude`` columns in any of
    the accepted spellings (see :func:`receptors_from_frame`). The result has
    the standard column names, ``altitude_ref``, and three more columns:
    ``receptor`` (the id), ``kind`` (``point``, ``column``, or
    ``multipoint``), and ``location`` (the location id). A receptor listed
    twice keeps its first rows. No receptor object is built, so this is quick
    on a large file; :func:`receptors_from_rows` builds the receptors.

    Raises
    ------
    ValueError
        For anything a receptor would refuse: a missing column, a longitude
        or latitude out of range, a negative height above ground, rows of one
        group that differ in time or altitude reference, a column whose two
        heights are equal, a multipoint receptor with two points at one
        location, or two different receptors with one id.
    """
    frame, _ = _normalize_columns(frame)
    required = ("time", "longitude", "latitude", "altitude")
    if any(c not in frame.columns for c in required):
        raise ValueError(f"Receptor table must contain columns: {list(required)}")
    clash = [c for c in frame.columns if c in ROW_COLUMNS]
    if clash:
        raise ValueError(
            f"A receptors file may not have a column named {clash}; PYSTILT uses "
            "those names. Rename the column."
        )
    n = len(frame)
    time = pd.to_datetime(frame["time"])
    if getattr(time.dt, "tz", None) is not None:
        time = time.dt.tz_convert("UTC").dt.tz_localize(None)
    lon = frame["longitude"].to_numpy(dtype=float)
    lat = frame["latitude"].to_numpy(dtype=float)
    alt = frame["altitude"].to_numpy(dtype=float)
    ref = frame["altitude_ref"].astype(str).str.lower().to_numpy()
    # Group keys as text: a file that mixes numeric and string ids must not
    # split one receptor into two. Without r_idx, every row is a receptor.
    keys = (
        frame["r_idx"].astype(str).to_numpy()
        if "r_idx" in frame.columns
        else np.arange(n).astype(str)
    )
    codes, uniques = pd.factorize(keys)

    def fail(mask: np.ndarray, message: str) -> None:
        """Raise *message* for the first row or group where *mask* is set."""
        if mask.any():
            where = int(np.flatnonzero(mask)[0])
            label = (
                f"r_idx={uniques[codes[where]]}" if "r_idx" in frame else f"row {where}"
            )
            raise ValueError(f"{label}: {message}")

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
        "All rows in one receptor group must share the same altitude_ref.",
    )
    fail(
        g["time"].transform("nunique").to_numpy() > 1,
        "All rows in one receptor group must share the same release time.",
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

    # Ids are made once per receptor, from its first row (or all its rows for
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
        time=time, altitude_ref=ref, receptor=ids, kind=kind, location=location
    )

    # A receptor id names its result files: one id, one receptor.
    repeated = np.flatnonzero(pd.Index(group_ids).duplicated(keep=False))
    if len(repeated):
        # Build every repeated group's receptor in one call, each group on
        # its own (keyed by its code, not its id), in code order.
        rows = np.isin(codes, repeated)
        built = receptors_from_rows(out.loc[rows].assign(receptor=codes[rows]))
        receptor_of = dict(zip(repeated.tolist(), built, strict=True))
        by_id: dict[str, list[int]] = {}
        for code in repeated.tolist():
            by_id.setdefault(str(group_ids[code]), []).append(code)
        drop: list[int] = []
        for group in by_id.values():
            check_distinct_ids([receptor_of[code] for code in group])
            drop.extend(group[1:])  # the same receptor listed again
        out = out.loc[~np.isin(codes, drop)]
    return out


def _build(
    kind: str,
    time: dt.datetime,
    altitude_ref: str,
    attrs: dict[str, Any],
    lon: np.ndarray,
    lat: np.ndarray,
    alt: np.ndarray,
) -> Receptor:
    """
    Return the receptor of one group of checked rows, given as plain values.

    The rows passed :func:`receptor_rows`, which runs every check a receptor
    runs, so the receptor is built without running them again.
    """
    common = {"time": time, "altitude_ref": altitude_ref, "attrs": attrs}
    if kind == "point":
        return PointReceptor.model_construct(
            longitude=float(lon[0]),
            latitude=float(lat[0]),
            altitude=float(alt[0]),
            **common,
        )
    if kind == "column":
        return ColumnReceptor.model_construct(
            longitude=float(lon[0]),
            latitude=float(lat[0]),
            bottom=float(alt.min()),
            top=float(alt.max()),
            **common,
        )
    return MultiPointReceptor.model_construct(
        longitudes=tuple(lon.tolist()),
        latitudes=tuple(lat.tolist()),
        altitudes=tuple(alt.tolist()),
        **common,
    )


def receptors_from_rows(rows: pd.DataFrame) -> list[Receptor]:
    """
    Build the receptors of a table that :func:`receptor_rows` checked.

    The rows are not checked again. Each receptor's labels (``attrs``) are
    the other columns of its first row.

    Returns
    -------
    list of Receptor
        In table order, one per receptor id.
    """
    names = [str(c) for c in rows.columns if c not in COLUMNS and c not in ROW_COLUMNS]
    labels = rows[names].astype(object).to_numpy()
    times = pd.DatetimeIndex(rows["time"]).to_pydatetime()
    lon = rows["longitude"].to_numpy(dtype=float)
    lat = rows["latitude"].to_numpy(dtype=float)
    alt = rows["altitude"].to_numpy(dtype=float)
    kind = rows["kind"].to_numpy()
    ref = rows["altitude_ref"].to_numpy()
    receptors = []
    for idx in rows.groupby("receptor", sort=False).indices.values():
        i = idx[0]
        attrs = {
            k: (None if pd.isna(v) else v)
            for k, v in zip(names, labels[i], strict=True)
        }
        receptors.append(
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
    return receptors


def receptors_from_frame(frame: pd.DataFrame) -> list[Receptor]:
    """
    Check a table with a row per release point and build its receptors.

    The table needs ``time``, ``longitude``, ``latitude``, and ``altitude``
    columns. Each row is one :class:`PointReceptor`, unless an ``r_idx``
    column groups rows into one receptor: a group of two rows at one
    location is a :class:`ColumnReceptor`, any other group a
    :class:`MultiPointReceptor`.

    ======================  =============================================
    Field                   Accepted column names (any case)
    ======================  =============================================
    time                    ``time``
    longitude               ``longitude``, ``long``, ``lon``
    latitude                ``latitude``, ``lati``, ``lat``
    altitude (m)            ``altitude``, ``zagl``, ``zmsl``, ``z``
    receptor group          ``r_idx``
    altitude reference      ``altitude_ref``, ``height_ref``
    ======================  =============================================

    Without an ``altitude_ref`` column, altitudes are above mean sea level
    when the column is named ``zmsl`` and above ground level otherwise. Any
    other column becomes a label in each receptor's ``attrs``. For a group,
    labels come from its first row.

    Returns
    -------
    list of Receptor
        In table order. A receptor listed twice appears once.

    Raises
    ------
    ValueError
        As :func:`receptor_rows` raises.
    """
    return receptors_from_rows(receptor_rows(frame))


def check_distinct_ids(receptors: Iterable[Receptor]) -> None:
    """
    Raise if two different receptors share an id.

    An id names a receptor's result files, so two receptors with one id
    would overwrite each other. Equal receptors listed twice are fine.
    """
    seen: dict[str, Receptor] = {}
    for r in receptors:
        other = seen.setdefault(r.id, r)
        if other is not r and other != r:
            raise ValueError(
                f"Two different receptors share the id {r.id!r}: {other!r} and {r!r}."
            )


# ---------------------------------------------------------------------------
# CSV files
# ---------------------------------------------------------------------------


def read_receptor_frame(path: str | Path | IO[str]) -> pd.DataFrame:
    """Read a receptors CSV as a table, with ``r_idx`` as text and ``time`` parsed, without building receptors."""
    header = pd.read_csv(path, nrows=0).columns
    if hasattr(path, "seek"):
        path.seek(0)
    # With type inference, pandas types each chunk of a large file on its
    # own, so an r_idx column that mixes numbers and text would come back
    # part int and part str.
    dtype: dict[Any, Any] = {c: str for c in header if str(c).lower() == "r_idx"}
    times = [c for c in header if str(c).lower() == "time"]
    return pd.read_csv(path, dtype=dtype, parse_dates=times)


def read_receptors(path: str | Path | IO[str]) -> list[Receptor]:
    """
    Read receptors from a CSV file.

    The file needs columns for time, longitude, latitude, and altitude, and
    may group rows into one receptor with ``r_idx``. See
    :func:`receptors_from_frame` for the rules and the accepted column
    names.

    Parameters
    ----------
    path : str, Path or file-like
        CSV file path or open text stream.

    Returns
    -------
    list of Receptor
        In file order.
    """
    return receptors_from_frame(read_receptor_frame(path))


def _csv_frame(receptors: Iterable[Receptor]) -> pd.DataFrame:
    """Return the receptor table with times as the text written to CSV."""
    frame = receptors_to_frame(receptors)
    return frame.assign(
        time=pd.DatetimeIndex(frame["time"]).strftime("%Y-%m-%d %H:%M:%S")
    )


def receptors_to_csv(receptors: Iterable[Receptor]) -> str:
    """Return receptors as CSV text that :func:`read_receptors` reads back."""
    return _csv_frame(receptors).to_csv(index=False, lineterminator="\n")


def write_receptors(receptors: Iterable[Receptor], path: str | Path) -> Path:
    """Write receptors to a CSV file that :func:`read_receptors` reads, and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(receptors_to_csv(receptors))
    return path


def append_receptors_csv(text: str, receptors: Iterable[Receptor]) -> str:
    """
    Return the text of a receptors CSV with rows for more receptors appended.

    The existing header sets the columns and their order, so a hand-written
    file keeps its column names and ``r_idx`` values, and its rows are kept
    as written. Label columns are filled from each receptor's ``attrs`` or
    left empty. New receptors are numbered after the largest ``r_idx`` in
    the file. When a new receptor's heights are above sea level and the file
    has no ``altitude_ref`` column, the column is added, with ``agl`` on the
    existing rows.

    Raises
    ------
    ValueError
        If the file lacks a time, longitude, latitude, or altitude column,
        has no ``r_idx`` column for a column or multipoint receptor, or
        names its altitude column ``zagl``/``zmsl`` for a different altitude
        reference than a new receptor uses.
    """
    receptors = list(receptors)
    if not text.strip():
        return receptors_to_csv(receptors)
    columns = list(pd.read_csv(StringIO(text), nrows=0).columns)
    _, spelling = _normalize_columns(pd.DataFrame(columns=columns))
    missing = [
        f for f in ("time", "longitude", "latitude", "altitude") if f not in spelling
    ]
    if missing:
        raise ValueError(f"receptors.csv lacks a column for {missing}; cannot append.")
    if "r_idx" not in spelling and any(len(r.coords()) > 1 for r in receptors):
        raise ValueError(
            "receptors.csv has no r_idx column, so a column or multipoint receptor "
            "cannot be appended; add an r_idx column to the file."
        )
    if "altitude_ref" not in spelling:
        file_ref = {"zagl": "agl", "zmsl": "msl"}.get(spelling["altitude"].lower())
        for r in receptors:
            if file_ref is not None and r.altitude_ref != file_ref:
                raise ValueError(
                    f"receptors.csv altitudes are {file_ref}; receptor {r.id} is "
                    f"{r.altitude_ref}. Add an altitude_ref column to mix them."
                )
        if file_ref is None and any(r.altitude_ref != "agl" for r in receptors):
            # The one case that rewrites the existing rows: they gain the column.
            existing = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
            text = existing.assign(altitude_ref="agl").to_csv(
                index=False, lineterminator="\n"
            )
            columns.append("altitude_ref")
            spelling["altitude_ref"] = "altitude_ref"

    new = _csv_frame(receptors)
    if "r_idx" in spelling:
        ids = pd.read_csv(
            StringIO(text),
            usecols=[spelling["r_idx"]],
            dtype=str,
            keep_default_na=False,
        ).iloc[:, 0]
        numbers = [int(v) for v in ids if v.strip().lstrip("-").isdigit()]
        new["r_idx"] += max(numbers, default=-1) + 1
    new = new.rename(columns=spelling).reindex(columns=columns)
    rows = new.astype(object).where(new.notna(), "")
    body = text if text.endswith("\n") else text + "\n"
    return body + rows.to_csv(index=False, header=False, lineterminator="\n")
