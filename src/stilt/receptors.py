"""
Receptors, the places and times particles are released from.

A :class:`PointReceptor` releases from one point, a :class:`ColumnReceptor`
from a vertical line, and a :class:`MultiPointReceptor` from several points
at once (for example a slanted satellite sounding). :func:`read_receptors`
reads them from a CSV file.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import pandas as pd
from shapely import Geometry, LineString, MultiPoint, Point

from stilt.config import VerticalReference, validate_vertical_reference

if TYPE_CHECKING:
    from stilt.visualization import ReceptorPlotAccessor

TimeLike: TypeAlias = dt.datetime | pd.Timestamp | np.datetime64 | str


def _validate_lon(lon) -> None:
    """Raise if any longitude value falls outside [-180, 180]."""
    arr = np.asarray(lon)
    if np.any((arr < -180) | (arr > 180)):
        raise ValueError("longitude must be within [-180, 180].")


def _validate_lat(lat) -> None:
    """Raise if any latitude value falls outside [-90, 90]."""
    arr = np.asarray(lat)
    if np.any((arr < -90) | (arr > 90)):
        raise ValueError("latitude must be within [-90, 90].")


def _validate_agl(alt, altitude_ref: str) -> None:
    """Raise if any altitude is negative when altitude_ref is 'agl'."""
    if altitude_ref == "agl" and np.any(np.asarray(alt) < 0):
        raise ValueError("AGL altitudes must be >= 0.")


def _validate_distinct_horizontal(lons, lats) -> None:
    """
    Raise if two points share a horizontal location.

    HYSPLIT joins consecutive starting locations at the same latitude and
    longitude into one vertical line source and releases only from the last
    pair. Several heights at one location in a multipoint receptor would
    lose all but the top segment. PYSTILT can also match particles to their
    release point by horizontal position, which cannot tell such points
    apart even when they are not consecutive.
    """
    pts = np.column_stack((np.round(lons, 5), np.round(lats, 5)))
    if len(np.unique(pts, axis=0)) != len(pts):
        raise ValueError(
            "MultiPointReceptor points must have distinct horizontal locations "
            "(HYSPLIT collapses starting locations that share a lat/lon into a "
            "single vertical line source and releases only between the last two "
            "heights). Use ColumnReceptor for a vertical column, or one "
            "PointReceptor per height (distinct r_idx) for discrete release "
            "heights at one location."
        )


def _format_coord(val: float) -> str:
    """Format a coordinate without a decimal point when it is a whole number."""
    return str(int(val)) if val == int(val) else str(val)


def _parse_time(time: TimeLike) -> dt.datetime:
    """Parse any supported time-like value to a naive UTC datetime."""
    if time is None:
        raise ValueError("'time' must be provided for all receptor types.")
    if isinstance(time, (int, float, np.integer, np.floating)):
        raise TypeError(
            "Numeric receptor times are not accepted. Pass a datetime-like value "
            "or a string such as '202301011200' or '2023-01-01T12:00:00Z'."
        )
    if isinstance(time, str) and re.fullmatch(r"\d{12}", time):
        parsed = pd.Timestamp(pd.to_datetime(time, format="%Y%m%d%H%M", utc=True))
    elif isinstance(time, (dt.datetime, pd.Timestamp, np.datetime64, str)):
        parsed = pd.Timestamp(time)
    else:
        raise TypeError(
            "Receptor time must be a datetime-like value or supported time string."
        )
    if str(parsed) == "NaT":
        raise ValueError("Receptor time cannot be NaT.")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return cast(dt.datetime, parsed.to_pydatetime()).replace(tzinfo=None)


class LocationID(str):
    """
    Identifier of a receptor's location.

    ``"<lon>_<lat>_<alt>"`` for a point, ``"<lon>_<lat>_X"`` for a column,
    and ``"multi_<hash>"`` for a multipoint receptor, where ``<hash>`` is 10
    hex characters of a SHA-256 of its points.

    Raises
    ------
    ValueError
        If the string has none of these forms.
    """

    _MULTI_PATTERN = re.compile(r"^multi_[0-9a-f]{10}$")

    def __new__(cls, value: str) -> LocationID:
        if cls._MULTI_PATTERN.fullmatch(value):
            return super().__new__(cls, value)
        parts = value.split("_")
        if len(parts) != 3:
            raise ValueError(
                "LocationID must be 'lon_lat_alt', 'lon_lat_X', or 'multi_<hash>'."
            )
        lon, lat, alt = parts
        try:
            float(lon)
            float(lat)
            if alt != "X":
                float(alt)
        except ValueError as exc:
            raise ValueError(
                "LocationID must be 'lon_lat_alt', 'lon_lat_X', or 'multi_<hash>'."
            ) from exc
        return super().__new__(cls, value)


class ReceptorID(str):
    """
    Identifier of a receptor, ``"<YYYYMMDDHHMM>_<location_id>"``.

    The time is the release time in UTC. The parsed parts are available as
    ``time`` and ``location``.

    Parameters
    ----------
    id_str : str
        The identifier, for example ``"202307151800_-111.848_40.766_10"``.

    Attributes
    ----------
    time : datetime
        Release time (UTC, naive).
    location : LocationID
        Location part of the id.

    Raises
    ------
    ValueError
        If *id_str* does not have this form.
    """

    time: dt.datetime
    location: LocationID

    def __new__(cls, id_str: str) -> ReceptorID:
        match = re.fullmatch(r"(?P<time>\d{12})_(?P<location>.+)", id_str)
        if match is None:
            raise ValueError(
                "ReceptorID must be in format '{YYYYMMDDHHMM}_{location_id}'."
            )
        instance = super().__new__(cls, id_str)
        time_str = match.group("time")
        location_id = match.group("location")
        try:
            instance.time = dt.datetime.strptime(time_str, "%Y%m%d%H%M")
        except ValueError as exc:
            raise ValueError(
                "ReceptorID timestamp must use the '{YYYYMMDDHHMM}' format."
            ) from exc
        instance.location = LocationID(location_id)
        return instance

    @classmethod
    def from_parts(
        cls,
        time: dt.datetime | pd.Timestamp,
        location_id: LocationID,
    ) -> ReceptorID:
        """Build a ReceptorID from a release time and a location id."""
        return cls(f"{pd.Timestamp(time):%Y%m%d%H%M}_{location_id}")


class Receptor(ABC):
    """
    Base class for receptors.

    Iterating a receptor yields ``(lat, lon, alt)`` for each release point.
    Two receptors are equal when their type, time, points, and
    ``altitude_ref`` match.

    Attributes
    ----------
    time : datetime
        Release time (UTC, naive).
    altitude_ref : {"agl", "msl"}
        Whether altitudes are above ground level or above mean sea level.
    attrs : dict
        Extra labels, such as a site or scene name. These are the columns of
        ``receptors.csv`` that PYSTILT does not use. They are kept when the
        receptors are written back to CSV and can be used to filter, as in
        ``model.simulations.sel(where=lambda r: r.attrs["site"] == "WBB")``.
        They are not part of the receptor's id.
    """

    def __init__(self, time: TimeLike, altitude_ref: VerticalReference) -> None:
        self.time = _parse_time(time)
        self.altitude_ref: VerticalReference = validate_vertical_reference(altitude_ref)
        self.attrs: dict[str, Any] = {}
        self._geometry = None
        self._plot: ReceptorPlotAccessor | None = None

    @property
    @abstractmethod
    def location_id(self) -> LocationID:
        """Identifier of this receptor's location."""
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[float, float, float]]:
        """Yield ``(lat, lon, alt)`` for each constituent point."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """Return this receptor as a dict that :meth:`from_dict` reads back."""
        ...

    @abstractmethod
    def _build_geometry(self) -> Geometry:
        """Construct the shapely geometry for this receptor."""
        ...

    @property
    def id(self) -> ReceptorID:
        """Receptor id, the release time followed by the location id."""
        return ReceptorID(f"{self.time:%Y%m%d%H%M}_{self.location_id}")

    @property
    def geometry(self):
        """Shapely geometry of the release points."""
        if self._geometry is None:
            self._geometry = self._build_geometry()
        return self._geometry

    @property
    def points(self) -> list[Point]:
        """Release points as shapely ``Point(lon, lat, alt)`` objects."""
        return [Point(lon, lat, alt) for lat, lon, alt in self]

    @property
    def plot(self) -> ReceptorPlotAccessor:
        """Plotting methods, such as ``receptor.plot.map()``."""
        if self._plot is None:
            from stilt.visualization import ReceptorPlotAccessor

            self._plot = ReceptorPlotAccessor(self)
        return self._plot

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        other = cast(Receptor, other)
        return (
            self.time == other.time
            and self.altitude_ref == other.altitude_ref
            and tuple(self) == tuple(other)
        )

    def __hash__(self) -> int:
        return hash((self.time.isoformat(), tuple(self), self.altitude_ref))

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Receptor:
        """
        Build a receptor from a dict made by :meth:`to_dict`.

        Raises
        ------
        ValueError
            If the dict has no ``"type"`` key or names an unknown type.
        """
        data = dict(d)
        type_str = cast(str | None, data.pop("type", None))
        if not type_str:
            raise ValueError("Dictionary must contain a 'type' key.")
        registry = {sub.__name__: sub for sub in Receptor.__subclasses__()}
        if type_str not in registry:
            raise ValueError(f"Unknown receptor type: '{type_str}'.")
        return registry[type_str](**data)  # type: ignore[arg-type]

    @classmethod
    def from_points(
        cls,
        time: TimeLike,
        points: list[tuple[float, float, float]],
        *,
        altitude_ref: VerticalReference = "agl",
    ) -> Receptor:
        """
        Build the right receptor type for a list of points.

        Parameters
        ----------
        time : datetime-like or str
            Release time (UTC).
        points : list of tuple of float
            ``(longitude, latitude, altitude)`` of each release point.
        altitude_ref : {"agl", "msl"}, default "agl"
            Whether altitudes are above ground level or above mean sea level.

        Returns
        -------
        Receptor
            A :class:`PointReceptor` for one point, a :class:`ColumnReceptor`
            for two points at the same longitude and latitude, and a
            :class:`MultiPointReceptor` otherwise.
        """
        if not points:
            raise ValueError("At least one point must be provided.")
        lons, lats, alts = zip(*points, strict=False)
        lons, lats, alts = list(lons), list(lats), list(alts)
        if len(lons) == 1:
            return PointReceptor(
                time, lons[0], lats[0], alts[0], altitude_ref=altitude_ref
            )
        if len(lons) == 2 and lons[0] == lons[1] and lats[0] == lats[1]:
            bottom, top = (
                (alts[0], alts[1]) if alts[0] < alts[1] else (alts[1], alts[0])
            )
            return ColumnReceptor(
                time, lons[0], lats[0], bottom, top, altitude_ref=altitude_ref
            )
        return MultiPointReceptor(time, lons, lats, alts, altitude_ref=altitude_ref)


class PointReceptor(Receptor):
    """
    Receptor that releases particles from one point.

    Parameters
    ----------
    time : datetime-like or str
        Release time. Time zone aware values are converted to UTC, and naive
        values are taken as UTC. A string may also be ``"YYYYMMDDHHMM"``.
    longitude : float
        Longitude in degrees, from -180 to 180.
    latitude : float
        Latitude in degrees, from -90 to 90.
    altitude : float
        Release height in metres.
    altitude_ref : {"agl", "msl"}, default "agl"
        Whether *altitude* is above ground level or above mean sea level.

    Examples
    --------
    >>> r = PointReceptor("2023-07-15 18:00", -111.848, 40.766, 10)
    >>> r.id
    '202307151800_-111.848_40.766_10'
    """

    def __init__(
        self,
        time: TimeLike,
        longitude: float,
        latitude: float,
        altitude: float,
        *,
        altitude_ref: VerticalReference = "agl",
    ) -> None:
        super().__init__(time, altitude_ref)
        self.longitude = float(longitude)
        self.latitude = float(latitude)
        self.altitude = float(altitude)
        _validate_lon(self.longitude)
        _validate_lat(self.latitude)
        _validate_agl(self.altitude, self.altitude_ref)

    @property
    def location_id(self) -> LocationID:
        """Location id, ``"<lon>_<lat>_<alt>"``."""
        x = _format_coord(self.longitude)
        y = _format_coord(self.latitude)
        z = _format_coord(self.altitude)
        return LocationID(f"{x}_{y}_{z}")

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[tuple[float, float, float]]:
        yield (self.latitude, self.longitude, self.altitude)

    def __repr__(self) -> str:
        return (
            f"PointReceptor(id={self.id!r}, "
            f"lon={self.longitude:.5f}, lat={self.latitude:.5f}, alt={self.altitude:g} {self.altitude_ref})"
        )

    def _build_geometry(self) -> Point:
        """Return a shapely Point at the release point."""
        return Point(self.longitude, self.latitude, self.altitude)

    def to_dict(self) -> dict[str, object]:
        """Return this receptor as a dict with ``type``, ``time``, ``longitude``, ``latitude``, ``altitude``, and ``altitude_ref``."""
        return {
            "type": type(self).__name__,
            "time": self.time.isoformat(),
            "longitude": self.longitude,
            "latitude": self.latitude,
            "altitude": self.altitude,
            "altitude_ref": self.altitude_ref,
        }


class ColumnReceptor(Receptor):
    """
    Receptor that releases particles evenly along a vertical line.

    Parameters
    ----------
    time : datetime-like or str
        Release time, as for :class:`PointReceptor`.
    longitude : float
        Longitude in degrees, from -180 to 180.
    latitude : float
        Latitude in degrees, from -90 to 90.
    bottom : float
        Bottom of the column in metres. Must be less than *top*.
    top : float
        Top of the column in metres.
    altitude_ref : {"agl", "msl"}, default "agl"
        Whether the heights are above ground level or above mean sea level.
    """

    def __init__(
        self,
        time: TimeLike,
        longitude: float,
        latitude: float,
        bottom: float,
        top: float,
        *,
        altitude_ref: VerticalReference = "agl",
    ) -> None:
        super().__init__(time, altitude_ref)
        self.longitude = float(longitude)
        self.latitude = float(latitude)
        self.bottom = float(bottom)
        self.top = float(top)
        _validate_lon(self.longitude)
        _validate_lat(self.latitude)
        if self.bottom >= self.top:
            raise ValueError("'bottom' must be less than 'top'.")
        _validate_agl(self.bottom, self.altitude_ref)

    @property
    def location_id(self) -> LocationID:
        """Location id, ``"<lon>_<lat>_X"``."""
        x = _format_coord(self.longitude)
        y = _format_coord(self.latitude)
        return LocationID(f"{x}_{y}_X")

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[tuple[float, float, float]]:
        yield (self.latitude, self.longitude, self.bottom)
        yield (self.latitude, self.longitude, self.top)

    def __repr__(self) -> str:
        return (
            f"ColumnReceptor(id={self.id!r}, "
            f"lon={self.longitude:.5f}, lat={self.latitude:.5f}, "
            f"bottom={self.bottom:g} {self.altitude_ref}, top={self.top:g} {self.altitude_ref})"
        )

    def _build_geometry(self) -> LineString:
        """Return a shapely LineString from the bottom to the top of the column."""
        return LineString(
            [
                (self.longitude, self.latitude, self.bottom),
                (self.longitude, self.latitude, self.top),
            ]
        )

    def to_dict(self) -> dict[str, object]:
        """Return this receptor as a dict with ``type``, ``time``, ``longitude``, ``latitude``, ``bottom``, ``top``, and ``altitude_ref``."""
        return {
            "type": type(self).__name__,
            "time": self.time.isoformat(),
            "longitude": self.longitude,
            "latitude": self.latitude,
            "bottom": self.bottom,
            "top": self.top,
            "altitude_ref": self.altitude_ref,
        }


class MultiPointReceptor(Receptor):
    """
    Receptor that releases particles from several points at once.

    Use it for a slanted column, such as a satellite sounding sampled at
    several heights along the line of sight. Each point must have its own
    longitude and latitude. For several heights at one location, use a
    :class:`ColumnReceptor` or one :class:`PointReceptor` per height.

    Parameters
    ----------
    time : datetime-like or str
        Release time, as for :class:`PointReceptor`.
    longitudes : array-like of float
        Longitude of each point in degrees.
    latitudes : array-like of float
        Latitude of each point in degrees.
    altitudes : array-like of float
        Height of each point in metres.
    altitude_ref : {"agl", "msl"}, default "agl"
        Whether the heights are above ground level or above mean sea level.

    Raises
    ------
    ValueError
        If the arrays differ in length, a coordinate is out of range, or two
        points share a horizontal location.
    """

    def __init__(
        self,
        time: TimeLike,
        longitudes,
        latitudes,
        altitudes,
        *,
        altitude_ref: VerticalReference = "agl",
    ) -> None:
        super().__init__(time, altitude_ref)
        self.longitudes = np.asarray(longitudes, dtype=float)
        self.latitudes = np.asarray(latitudes, dtype=float)
        self.altitudes = np.asarray(altitudes, dtype=float)
        if not (len(self.longitudes) == len(self.latitudes) == len(self.altitudes)):
            raise ValueError(
                "longitudes, latitudes, and altitudes must have the same length."
            )
        _validate_lon(self.longitudes)
        _validate_lat(self.latitudes)
        _validate_agl(self.altitudes, self.altitude_ref)
        _validate_distinct_horizontal(self.longitudes, self.latitudes)

    @property
    def location_id(self) -> LocationID:
        """
        Location id, ``"multi_<hash>"``.

        The hash covers the sorted points, with coordinates rounded to 5
        decimals and altitudes truncated to whole metres.
        """
        pts_sorted = sorted(
            zip(self.longitudes, self.latitudes, self.altitudes, strict=False)
        )
        canonical = json.dumps(
            [
                [round(float(lon), 5), round(float(lat), 5), int(alt)]
                for lon, lat, alt in pts_sorted
            ],
            separators=(",", ":"),
        )
        hash_str = hashlib.sha256(canonical.encode()).hexdigest()[:10]
        return LocationID(f"multi_{hash_str}")

    def __len__(self) -> int:
        return len(self.longitudes)

    def __iter__(self) -> Iterator[tuple[float, float, float]]:
        yield from zip(self.latitudes, self.longitudes, self.altitudes, strict=False)

    def __repr__(self) -> str:
        return f"MultiPointReceptor(id={self.id!r}, n_points={len(self)}, altitude_ref={self.altitude_ref})"

    def _build_geometry(self) -> MultiPoint:
        """Return a shapely MultiPoint of the release points."""
        return MultiPoint(
            list(zip(self.longitudes, self.latitudes, self.altitudes, strict=False))
        )

    def to_dict(self) -> dict[str, object]:
        """Return this receptor as a dict with ``type``, ``time``, ``longitudes``, ``latitudes``, ``altitudes``, and ``altitude_ref``."""
        return {
            "type": type(self).__name__,
            "time": self.time.isoformat(),
            "longitudes": self.longitudes.tolist(),
            "latitudes": self.latitudes.tolist(),
            "altitudes": self.altitudes.tolist(),
            "altitude_ref": self.altitude_ref,
        }


#: Column names :func:`read_receptors` accepts for each receptor field.
_CSV_ALIASES = {
    "time": ("time",),
    "longitude": ("longitude", "long", "lon"),
    "latitude": ("latitude", "lati", "lat"),
    "altitude": ("altitude", "zagl", "zmsl", "z"),
    "r_idx": ("r_idx",),
    "altitude_ref": ("altitude_ref", "height_ref"),
}

#: Header spellings mapped onto the short names the reader works with.
_CSV_RENAMES = {
    alias: {"longitude": "long", "latitude": "lati", "altitude": "z"}.get(field, field)
    for field, aliases in _CSV_ALIASES.items()
    for alias in aliases
}


def read_receptors(path: str | Path | IO[str]) -> list[Receptor]:
    """
    Read receptors from a CSV file.

    The file needs columns for time, longitude, latitude, and altitude. Each
    row is one :class:`PointReceptor`, unless an ``r_idx`` column groups
    rows into one receptor. A group of two rows at the same location becomes
    a :class:`ColumnReceptor` and any other group a
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
    when the column is named ``zmsl`` and above ground level otherwise.
    Any other columns are kept in each receptor's ``attrs``. For a group,
    they come from its first row.

    Parameters
    ----------
    path : str, Path or file-like
        CSV file path or open text stream.

    Returns
    -------
    list of Receptor
        In file order.

    Raises
    ------
    ValueError
        If a required column is missing, or the rows of one group differ in
        time or altitude reference.
    """
    # Read r_idx as text. With type inference, pandas types each chunk of a
    # large file separately, so in a file that mixes numeric and string ids,
    # a group split across chunks would come back part int and part str and
    # be split into two receptors.
    header = pd.read_csv(path, nrows=0).columns
    if hasattr(path, "seek"):
        path.seek(0)  # type: ignore[union-attr]
    # Annotated loosely because the reader's own signature spells this mapping
    # with invariant value types, which no precise annotation here satisfies.
    dtype: dict[Hashable, Any] = {c: str for c in header if str(c).lower() == "r_idx"}
    df = pd.read_csv(path, parse_dates=["time"], dtype=dtype)

    spelling = {str(col).lower(): str(col) for col in df.columns}
    original_columns = list(spelling)
    inferred_altitude_ref = None
    if "zmsl" in original_columns:
        inferred_altitude_ref = "msl"
    elif "zagl" in original_columns:
        inferred_altitude_ref = "agl"

    df.columns = df.columns.str.lower()
    df = df.rename(columns=_CSV_RENAMES)
    if "altitude_ref" not in df.columns:
        df["altitude_ref"] = inferred_altitude_ref or "agl"

    required_cols = ["time", "lati", "long", "z"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Receptor file must contain columns: {required_cols}")

    # The columns PYSTILT does not use become one ``attrs`` dict per row, in
    # the file's own spelling, with empty cells as None.
    extra = [
        c for c in df.columns if c not in (*required_cols, "r_idx", "altitude_ref")
    ]
    labels = df[extra].astype(object).where(df[extra].notna(), None)
    names = [spelling.get(c, c) for c in extra]
    records = [
        dict(zip(names, values, strict=True))
        for values in labels.itertuples(index=False, name=None)
    ]
    if not extra:  # a frame with no columns iterates as no rows
        records = [{} for _ in df.index]
    df = df.drop(columns=extra).assign(attrs=records)

    def _point_receptor(row: Any) -> PointReceptor:
        receptor = PointReceptor(
            time=row.time,
            longitude=row.long,
            latitude=row.lati,
            altitude=row.z,
            altitude_ref=row.altitude_ref,
        )
        receptor.attrs = row.attrs
        return receptor

    def _point_receptors_from_rows(frame: pd.DataFrame) -> list[Receptor]:
        """Build one PointReceptor per row from a normalised receptor DataFrame."""
        return [_point_receptor(row) for row in frame.itertuples(index=False)]

    if "r_idx" in df.columns:
        group_sizes = df.groupby("r_idx").size()
        multi_keys = [k for k, v in group_sizes.items() if v > 1]

        if not multi_keys:
            return _point_receptors_from_rows(df)

        single_mask = ~df["r_idx"].isin(multi_keys)
        result: dict[object, Receptor] = {}
        for row in df[single_mask].itertuples(index=False):
            result[cast(Any, row).r_idx] = _point_receptor(row)
        for key, g in df[~single_mask].groupby("r_idx"):
            try:
                result[key] = _receptor_from_group(cast(pd.DataFrame, g))
            except ValueError as exc:
                raise ValueError(f"r_idx={key}: {exc}") from exc
            result[key].attrs = g["attrs"].tolist()[0]

        return [result[k] for k in df["r_idx"].unique()]

    return _point_receptors_from_rows(df)


def receptors_to_csv(receptors: Iterable[Receptor]) -> str:
    """
    Return receptors as CSV text that :func:`read_receptors` reads back.

    Each release point is one row, and ``r_idx`` groups the rows of column
    and multipoint receptors. Each receptor's ``attrs`` become extra
    columns.
    """
    import csv
    from io import StringIO

    receptors = list(receptors)
    extra = list(dict.fromkeys(k for r in receptors for k in r.attrs))
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "r_idx",
            "time",
            "longitude",
            "latitude",
            "altitude",
            "altitude_ref",
            *extra,
        ],
    )
    writer.writeheader()
    for idx, receptor in enumerate(receptors):
        for lat, lon, altitude in receptor:
            writer.writerow(
                {
                    "r_idx": idx,
                    "time": receptor.time.isoformat(sep=" "),
                    "longitude": float(lon),
                    "latitude": float(lat),
                    "altitude": float(altitude),
                    "altitude_ref": receptor.altitude_ref,
                    **{k: receptor.attrs.get(k, "") for k in extra},
                }
            )
    return buffer.getvalue()


def append_receptors_csv(text: str, receptors: Iterable[Receptor]) -> str:
    """
    Return the text of a receptors CSV with rows for more receptors appended.

    The existing header sets the columns and their order, so a hand-written
    file keeps its column names and ``r_idx`` values. Extra columns are
    filled from each receptor's ``attrs`` or left empty. New receptors are
    numbered after the largest ``r_idx`` in the file.

    Parameters
    ----------
    text : str
        Contents of the existing CSV.
    receptors : iterable of Receptor
        Receptors to append.

    Returns
    -------
    str
        The CSV text with the new rows.

    Raises
    ------
    ValueError
        If the file lacks a time, longitude, latitude, or altitude column,
        has no ``r_idx`` column for a column or multipoint receptor, or
        names its altitude column ``zagl``/``zmsl`` for a different altitude
        reference than a new receptor uses.
    """
    import csv
    from io import StringIO

    rows = list(csv.reader(StringIO(text)))
    if not rows:
        return receptors_to_csv(receptors)
    header = rows[0]
    lower = [h.strip().lower() for h in header]

    def column(field: str) -> str | None:
        return next(
            (header[i] for i, h in enumerate(lower) if h in _CSV_ALIASES[field]), None
        )

    columns = {field: column(field) for field in _CSV_ALIASES}
    missing = [
        f for f in ("time", "longitude", "latitude", "altitude") if columns[f] is None
    ]
    if missing:
        raise ValueError(f"receptors.csv lacks a column for {missing}; cannot append.")
    c_time, c_lon, c_lat, c_alt = (
        str(columns[f]) for f in ("time", "longitude", "latitude", "altitude")
    )
    file_ref = {"zagl": "agl", "zmsl": "msl"}.get(c_alt.lower())

    receptors = list(receptors)
    idx_column = columns["r_idx"]
    if idx_column is None and any(len(list(r)) > 1 for r in receptors):
        raise ValueError(
            "receptors.csv has no r_idx column, so a column or multipoint receptor "
            "cannot be appended; add an r_idx column to the file."
        )
    next_idx = 0
    if idx_column is not None:
        position = header.index(idx_column)
        numeric = [
            int(r[position])
            for r in rows[1:]
            if r[position].strip().lstrip("-").isdigit()
        ]
        next_idx = max(numeric, default=-1) + 1

    ref_column = columns["altitude_ref"]
    extra = [h for h in header if h not in columns.values()]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, restval="")
    for k, receptor in enumerate(receptors):
        if ref_column is None and file_ref not in (None, receptor.altitude_ref):
            raise ValueError(
                f"receptors.csv altitudes are {file_ref}; receptor {receptor.id} is "
                f"{receptor.altitude_ref}. Add an altitude_ref column to mix them."
            )
        for lat, lon, altitude in receptor:
            row: dict[str, Any] = {
                c_time: receptor.time.isoformat(sep=" "),
                c_lon: float(lon),
                c_lat: float(lat),
                c_alt: float(altitude),
            }
            if idx_column is not None:
                row[idx_column] = next_idx + k
            if ref_column is not None:
                row[ref_column] = receptor.altitude_ref
            row.update({h: receptor.attrs.get(h, "") for h in extra})
            writer.writerow(row)
    body = text if text.endswith("\n") else text + "\n"
    return body + buffer.getvalue()


def write_receptors(receptors: Iterable[Receptor], path: str | Path) -> Path:
    """Write receptors to a CSV file that :func:`read_receptors` reads, and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(receptors_to_csv(receptors))
    return path


def _receptor_from_group(group: pd.DataFrame) -> Receptor:
    """Build one receptor from the rows of one ``r_idx`` group."""
    refs = {str(v).lower() for v in group["altitude_ref"].tolist()}
    if len(refs) != 1:
        raise ValueError(
            "All rows in one receptor group must share the same altitude_ref."
        )
    altitude_ref = validate_vertical_reference(refs.pop())
    lons = group["long"].tolist()
    lats = group["lati"].tolist()
    alts = group["z"].tolist()
    times = pd.to_datetime(group["time"]).unique()
    if len(times) != 1:
        raise ValueError(
            "All rows in one receptor group must share the same release time."
        )
    time = pd.to_datetime(times[0])
    return Receptor.from_points(
        time=time,
        points=list(zip(lons, lats, alts, strict=False)),
        altitude_ref=altitude_ref,
    )


__all__ = [
    "ColumnReceptor",
    "LocationID",
    "MultiPointReceptor",
    "PointReceptor",
    "Receptor",
    "ReceptorID",
    "read_receptors",
    "append_receptors_csv",
    "receptors_to_csv",
    "write_receptors",
]
