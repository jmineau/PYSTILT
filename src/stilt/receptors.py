"""
Receptors, the places and times particles are released from.

A :class:`PointReceptor` releases from one point, a :class:`ColumnReceptor`
from a vertical line, and a :class:`MultiPointReceptor` from several points
at once (for example a slanted satellite sounding). :func:`read_receptors`
reads them from a CSV file, and :func:`receptors_to_frame` gives them as one
table with a row per release point.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from functools import cached_property
from io import StringIO
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAlias,
    cast,
)

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from shapely import Geometry, LineString, MultiPoint, Point

from stilt.config import VerticalReference

if TYPE_CHECKING:
    from stilt.visualization import ReceptorPlotAccessor

TimeLike: TypeAlias = dt.datetime | pd.Timestamp | np.datetime64 | str

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
# Times and ids
# ---------------------------------------------------------------------------


def parse_time(time: TimeLike) -> dt.datetime:
    """
    Return *time* as a naive UTC datetime.

    Accepts datetimes (aware ones are converted to UTC), pandas and numpy
    times, ISO strings, and the compact ``"YYYYMMDDHHMM"`` form.
    """
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


def parse_receptor_id(receptor_id: str) -> tuple[dt.datetime, str]:
    """
    Split a receptor id into its release time and location id.

    An id is ``"<YYYYMMDDHHMM>_<location>"``, where the location is
    ``"<lon>_<lat>_<alt>"`` for a point, ``"<lon>_<lat>_X"`` for a column,
    and ``"multi_<hash>"`` for a multipoint receptor.

    Raises
    ------
    ValueError
        If *receptor_id* does not have this form.
    """
    match = re.fullmatch(r"(?P<time>\d{12})_(?P<location>.+)", str(receptor_id))
    if match is None:
        raise ValueError(
            f"Receptor id {receptor_id!r} is not of the form "
            "'{YYYYMMDDHHMM}_{location_id}'."
        )
    try:
        time = dt.datetime.strptime(match.group("time"), "%Y%m%d%H%M")
    except ValueError as exc:
        raise ValueError(
            f"Receptor id {receptor_id!r} does not start with a YYYYMMDDHHMM time."
        ) from exc
    location = match.group("location")
    if not re.fullmatch(r"multi_[0-9a-f]{10}", location):
        parts = location.split("_")
        try:
            if len(parts) != 3:
                raise ValueError
            float(parts[0])
            float(parts[1])
            if parts[2] != "X":
                float(parts[2])
        except ValueError:
            raise ValueError(
                f"Receptor id {receptor_id!r} has no location of the form "
                "'lon_lat_alt', 'lon_lat_X', or 'multi_<hash>'."
            ) from None
    return time, location


def _format_coord(val: float) -> str:
    """Format a coordinate without a decimal point when it is a whole number."""
    return str(int(val)) if val == int(val) else str(val)


# ---------------------------------------------------------------------------
# Receptors
# ---------------------------------------------------------------------------


class Receptor(BaseModel):
    """
    Base class of the receptor types.

    A receptor is a frozen value: its release time, its release points, and
    the vertical reference of their heights. Iterating one yields
    ``(lat, lon, alt)`` for each release point. Two receptors are equal when
    their type, time, points, and ``altitude_ref`` match; ``attrs`` do not
    count.

    Attributes
    ----------
    time : datetime
        Release time (UTC, naive). Aware values are converted to UTC, and a
        string may be ISO or ``"YYYYMMDDHHMM"``.
    altitude_ref : {"agl", "msl"}
        Whether heights are above ground level or above mean sea level.
    attrs : dict
        Extra labels, such as a site or scene name. These are the columns of
        ``receptors.csv`` that PYSTILT does not use. They are kept when the
        receptors are written back to CSV and can be used to filter, as in
        ``model.receptors.sel(site="WBB")``. They are not part of the
        receptor's id or of equality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    time: dt.datetime = Field(description="Release time, UTC.")
    altitude_ref: VerticalReference = Field(
        "agl", description="Whether heights are above ground (agl) or sea level (msl)."
    )
    attrs: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
        description="Extra labels from the receptor file; not part of the id.",
    )

    @field_validator("time", mode="before")
    @classmethod
    def _parse_time(cls, value: Any) -> dt.datetime:
        return parse_time(value)

    @field_validator("altitude_ref", mode="before")
    @classmethod
    def _lower_ref(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    # -- identity ----------------------------------------------------------

    @property
    def location_id(self) -> str:
        """Identifier of this receptor's location."""
        raise NotImplementedError

    @property
    def id(self) -> str:
        """Receptor id, the release time followed by the location id."""
        return f"{self.time:%Y%m%d%H%M}_{self.location_id}"

    def coords(self) -> list[tuple[float, float, float]]:
        """Return ``(lat, lon, alt)`` of each release point."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[float, float, float]]:  # type: ignore[override]
        """Yield ``(lat, lon, alt)`` for each release point."""
        return iter(self.coords())

    def __len__(self) -> int:
        return len(self.coords())

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        other = cast(Receptor, other)
        return (
            self.time == other.time
            and self.altitude_ref == other.altitude_ref
            and self.coords() == other.coords()
        )

    def __hash__(self) -> int:
        return hash(
            (type(self).__name__, self.time, self.altitude_ref, tuple(self.coords()))
        )

    def __str__(self) -> str:
        return repr(self)

    # -- geometry ----------------------------------------------------------

    def _build_geometry(self) -> Geometry:
        raise NotImplementedError

    @cached_property
    def geometry(self) -> Geometry:
        """Shapely geometry of the release points."""
        return self._build_geometry()

    @property
    def points(self) -> list[Point]:
        """Release points as shapely ``Point(lon, lat, alt)`` objects."""
        return [Point(lon, lat, alt) for lat, lon, alt in self.coords()]

    @cached_property
    def plot(self) -> ReceptorPlotAccessor:
        """Plotting methods, such as ``receptor.plot.map()``."""
        from stilt.visualization import ReceptorPlotAccessor

        return ReceptorPlotAccessor(self)

    # -- dicts -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return this receptor as a plain dict that :meth:`from_dict` reads back."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Receptor:
        """
        Build a receptor of the right type from a dict made by :meth:`to_dict`.

        Dicts written by earlier versions name the type under ``"type"``
        (``"PointReceptor"``). Those load too.

        Raises
        ------
        ValueError
            If the dict names no known receptor kind.
        """
        data = dict(d)
        if "kind" not in data:
            legacy = {
                "PointReceptor": "point",
                "ColumnReceptor": "column",
                "MultiPointReceptor": "multipoint",
            }
            type_name = data.pop("type", None)
            if not type_name:
                raise ValueError("Receptor dict must contain a 'kind' key.")
            if type_name not in legacy:
                raise ValueError(f"Unknown receptor type: {type_name!r}.")
            data["kind"] = legacy[type_name]
        return _ANY_RECEPTOR.validate_python(data)

    @classmethod
    def from_points(
        cls,
        time: TimeLike,
        points: Iterable[tuple[float, float, float]],
        *,
        altitude_ref: VerticalReference = "agl",
        attrs: Mapping[str, Any] | None = None,
    ) -> Receptor:
        """
        Build the right receptor type for a list of points.

        Parameters
        ----------
        time : datetime-like or str
            Release time (UTC).
        points : iterable of tuple of float
            ``(longitude, latitude, altitude)`` of each release point.
        altitude_ref : {"agl", "msl"}, default "agl"
            Whether altitudes are above ground level or above mean sea level.
        attrs : mapping, optional
            Extra labels.

        Returns
        -------
        Receptor
            A :class:`PointReceptor` for one point, a :class:`ColumnReceptor`
            for two points at the same longitude and latitude, and a
            :class:`MultiPointReceptor` otherwise.
        """
        pts = [(float(lon), float(lat), float(alt)) for lon, lat, alt in points]
        if not pts:
            raise ValueError("At least one point must be provided.")
        labels = dict(attrs or {})
        if len(pts) == 1:
            lon, lat, alt = pts[0]
            return PointReceptor(
                time, lon, lat, alt, altitude_ref=altitude_ref, attrs=labels
            )
        if len(pts) == 2 and pts[0][:2] == pts[1][:2]:
            bottom, top = sorted((pts[0][2], pts[1][2]))
            return ColumnReceptor(
                time,
                pts[0][0],
                pts[0][1],
                bottom,
                top,
                altitude_ref=altitude_ref,
                attrs=labels,
            )
        lons, lats, alts = zip(*pts, strict=True)
        return MultiPointReceptor(
            time, lons, lats, alts, altitude_ref=altitude_ref, attrs=labels
        )


def _check_lon(values: Iterable[float]) -> None:
    if any(not -180 <= v <= 180 for v in values):
        raise ValueError("longitude must be within [-180, 180].")


def _check_lat(values: Iterable[float]) -> None:
    if any(not -90 <= v <= 90 for v in values):
        raise ValueError("latitude must be within [-90, 90].")


def _check_agl(values: Iterable[float], altitude_ref: str) -> None:
    if altitude_ref == "agl" and any(v < 0 for v in values):
        raise ValueError("AGL altitudes must be >= 0.")


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
    attrs : dict, optional
        Extra labels.

    Examples
    --------
    >>> r = PointReceptor("2023-07-15 18:00", -111.848, 40.766, 10)
    >>> r.id
    '202307151800_-111.848_40.766_10'
    """

    kind: Literal["point"] = "point"
    longitude: float = Field(description="Longitude of the release point, degrees.")
    latitude: float = Field(description="Latitude of the release point, degrees.")
    altitude: float = Field(description="Release height, metres.")

    def __init__(
        self,
        time: TimeLike,
        longitude: float,
        latitude: float,
        altitude: float,
        *,
        altitude_ref: VerticalReference = "agl",
        attrs: Mapping[str, Any] | None = None,
        kind: Literal["point"] = "point",
    ) -> None:
        BaseModel.__init__(
            self,
            time=time,
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            altitude_ref=altitude_ref,
            attrs=dict(attrs or {}),
            kind=kind,
        )

    @model_validator(mode="after")
    def _check(self) -> PointReceptor:
        _check_lon([self.longitude])
        _check_lat([self.latitude])
        _check_agl([self.altitude], self.altitude_ref)
        return self

    @property
    def location_id(self) -> str:
        """Location id, ``"<lon>_<lat>_<alt>"``."""
        return "_".join(
            _format_coord(v) for v in (self.longitude, self.latitude, self.altitude)
        )

    def coords(self) -> list[tuple[float, float, float]]:
        """Return ``(lat, lon, alt)`` of the release point."""
        return [(self.latitude, self.longitude, self.altitude)]

    def __repr__(self) -> str:
        return (
            f"PointReceptor(id={self.id!r}, lon={self.longitude:.5f}, "
            f"lat={self.latitude:.5f}, alt={self.altitude:g} {self.altitude_ref})"
        )

    def _build_geometry(self) -> Point:
        return Point(self.longitude, self.latitude, self.altitude)


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
    attrs : dict, optional
        Extra labels.
    """

    kind: Literal["column"] = "column"
    longitude: float = Field(description="Longitude of the column, degrees.")
    latitude: float = Field(description="Latitude of the column, degrees.")
    bottom: float = Field(description="Bottom of the column, metres.")
    top: float = Field(description="Top of the column, metres.")

    def __init__(
        self,
        time: TimeLike,
        longitude: float,
        latitude: float,
        bottom: float,
        top: float,
        *,
        altitude_ref: VerticalReference = "agl",
        attrs: Mapping[str, Any] | None = None,
        kind: Literal["column"] = "column",
    ) -> None:
        BaseModel.__init__(
            self,
            time=time,
            longitude=longitude,
            latitude=latitude,
            bottom=bottom,
            top=top,
            altitude_ref=altitude_ref,
            attrs=dict(attrs or {}),
            kind=kind,
        )

    @model_validator(mode="after")
    def _check(self) -> ColumnReceptor:
        _check_lon([self.longitude])
        _check_lat([self.latitude])
        if self.bottom >= self.top:
            raise ValueError("'bottom' must be less than 'top'.")
        _check_agl([self.bottom], self.altitude_ref)
        return self

    @property
    def location_id(self) -> str:
        """Location id, ``"<lon>_<lat>_X"``."""
        return f"{_format_coord(self.longitude)}_{_format_coord(self.latitude)}_X"

    def coords(self) -> list[tuple[float, float, float]]:
        """Return ``(lat, lon, alt)`` of the bottom and the top of the column."""
        return [
            (self.latitude, self.longitude, self.bottom),
            (self.latitude, self.longitude, self.top),
        ]

    def __repr__(self) -> str:
        return (
            f"ColumnReceptor(id={self.id!r}, lon={self.longitude:.5f}, "
            f"lat={self.latitude:.5f}, bottom={self.bottom:g} {self.altitude_ref}, "
            f"top={self.top:g} {self.altitude_ref})"
        )

    def _build_geometry(self) -> LineString:
        return LineString(
            [
                (self.longitude, self.latitude, self.bottom),
                (self.longitude, self.latitude, self.top),
            ]
        )


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
    attrs : dict, optional
        Extra labels.

    Raises
    ------
    ValueError
        If the arrays differ in length, a coordinate is out of range, or two
        points share a horizontal location.
    """

    kind: Literal["multipoint"] = "multipoint"
    longitudes: tuple[float, ...] = Field(
        description="Longitude of each point, degrees."
    )
    latitudes: tuple[float, ...] = Field(description="Latitude of each point, degrees.")
    altitudes: tuple[float, ...] = Field(description="Height of each point, metres.")

    def __init__(
        self,
        time: TimeLike,
        longitudes: Iterable[float],
        latitudes: Iterable[float],
        altitudes: Iterable[float],
        *,
        altitude_ref: VerticalReference = "agl",
        attrs: Mapping[str, Any] | None = None,
        kind: Literal["multipoint"] = "multipoint",
    ) -> None:
        BaseModel.__init__(
            self,
            time=time,
            longitudes=longitudes,
            latitudes=latitudes,
            altitudes=altitudes,
            altitude_ref=altitude_ref,
            attrs=dict(attrs or {}),
            kind=kind,
        )

    @field_validator("longitudes", "latitudes", "altitudes", mode="before")
    @classmethod
    def _as_floats(cls, value: Any) -> tuple[float, ...]:
        return tuple(float(v) for v in np.asarray(value, dtype=float).ravel())

    @model_validator(mode="after")
    def _check(self) -> MultiPointReceptor:
        if not (len(self.longitudes) == len(self.latitudes) == len(self.altitudes)):
            raise ValueError(
                "longitudes, latitudes, and altitudes must have the same length."
            )
        _check_lon(self.longitudes)
        _check_lat(self.latitudes)
        _check_agl(self.altitudes, self.altitude_ref)
        # HYSPLIT joins consecutive starting locations at one latitude and
        # longitude into a vertical line source and releases only from the
        # last pair, and PYSTILT matches particles to their release point by
        # horizontal position.
        horizontal = {
            (round(lon, 5), round(lat, 5))
            for lon, lat in zip(self.longitudes, self.latitudes, strict=True)
        }
        if len(horizontal) != len(self.longitudes):
            raise ValueError(
                "MultiPointReceptor points must have distinct horizontal locations "
                "(HYSPLIT collapses starting locations that share a lat/lon into a "
                "single vertical line source and releases only between the last two "
                "heights). Use ColumnReceptor for a vertical column, or one "
                "PointReceptor per height (distinct r_idx) for discrete release "
                "heights at one location."
            )
        return self

    @property
    def location_id(self) -> str:
        """
        Location id, ``"multi_<hash>"``.

        The hash covers the sorted points, with longitudes and latitudes
        rounded to 5 decimals and altitudes to 0.01 m. A whole-metre
        altitude hashes as its integer.
        """
        points = sorted(
            zip(self.longitudes, self.latitudes, self.altitudes, strict=True)
        )
        canonical = json.dumps(
            [
                [round(lon, 5), round(lat, 5), _hash_altitude(alt)]
                for lon, lat, alt in points
            ],
            separators=(",", ":"),
        )
        return "multi_" + hashlib.sha256(canonical.encode()).hexdigest()[:10]

    def coords(self) -> list[tuple[float, float, float]]:
        """Return ``(lat, lon, alt)`` of each release point."""
        return list(zip(self.latitudes, self.longitudes, self.altitudes, strict=True))

    def __repr__(self) -> str:
        return (
            f"MultiPointReceptor(id={self.id!r}, n_points={len(self)}, "
            f"altitude_ref={self.altitude_ref})"
        )

    def _build_geometry(self) -> MultiPoint:
        return MultiPoint(
            list(zip(self.longitudes, self.latitudes, self.altitudes, strict=True))
        )


def _hash_altitude(alt: float) -> int | float:
    """Altitude as hashed in a multipoint id: an int when whole, else to 0.01 m."""
    rounded = round(alt, 2)
    return int(rounded) if rounded == int(rounded) else rounded


#: Any receptor, for validating a dict of unknown kind.
AnyReceptor = Annotated[
    PointReceptor | ColumnReceptor | MultiPointReceptor, Field(discriminator="kind")
]
_ANY_RECEPTOR: TypeAdapter[Receptor] = TypeAdapter(AnyReceptor)


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


def receptors_from_frame(frame: pd.DataFrame) -> list[Receptor]:
    """
    Build receptors from a table with a row per release point.

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
        In table order.

    Raises
    ------
    ValueError
        If a required column is missing, the rows of one group differ in
        time or altitude reference, or two different receptors get the same
        id.
    """
    frame, spelling = _normalize_columns(frame)
    required = ("time", "longitude", "latitude", "altitude")
    if any(c not in frame.columns for c in required):
        raise ValueError(f"Receptor table must contain columns: {list(required)}")
    labels = [c for c in frame.columns if c not in COLUMNS]
    names = [str(c) for c in labels]
    values = frame[labels].astype(object).where(frame[labels].notna(), None)
    attrs = [
        dict(zip(names, row, strict=True))
        for row in values.itertuples(index=False, name=None)
    ]
    if not labels:  # a frame with no columns iterates as no rows
        attrs = [{} for _ in frame.index]
    time = pd.to_datetime(frame["time"])
    lon = frame["longitude"].to_numpy(dtype=float)
    lat = frame["latitude"].to_numpy(dtype=float)
    alt = frame["altitude"].to_numpy(dtype=float)
    ref = frame["altitude_ref"].astype(str).str.lower().to_numpy()

    def point(i: int) -> Receptor:
        return PointReceptor(
            time.iloc[i], lon[i], lat[i], alt[i], altitude_ref=ref[i], attrs=attrs[i]
        )

    if "r_idx" not in frame.columns:
        receptors = [point(i) for i in range(len(frame))]
    else:
        # Group keys as text: a file that mixes numeric and string ids must
        # not split one receptor into two.
        keys = frame["r_idx"].astype(str).to_numpy()
        rows: dict[str, list[int]] = {}
        for i, key in enumerate(keys):
            rows.setdefault(key, []).append(i)
        receptors = []
        for key, idx in rows.items():
            if len(idx) == 1:
                receptors.append(point(idx[0]))
                continue
            try:
                if len(set(ref[idx])) != 1:
                    raise ValueError(
                        "All rows in one receptor group must share the same altitude_ref."
                    )
                if time.iloc[idx].nunique() != 1:
                    raise ValueError(
                        "All rows in one receptor group must share the same release time."
                    )
                receptors.append(
                    Receptor.from_points(
                        time.iloc[idx[0]],
                        [(lon[i], lat[i], alt[i]) for i in idx],
                        altitude_ref=ref[idx[0]],
                        attrs=attrs[idx[0]],
                    )
                )
            except ValueError as exc:
                raise ValueError(f"r_idx={key}: {_message(exc)}") from exc
    check_distinct_ids(receptors)
    return receptors


def _message(exc: ValueError) -> str:
    """Return the message of *exc* on one line, without pydantic's framing."""
    if isinstance(exc, ValidationError):
        return "; ".join(
            str(e["msg"]).removeprefix("Value error, ") for e in exc.errors()
        )
    return str(exc)


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


def _read_frame(path: str | Path | IO[str]) -> pd.DataFrame:
    """Read a receptors CSV with ``r_idx`` as text and ``time`` parsed."""
    header = pd.read_csv(path, nrows=0).columns
    if hasattr(path, "seek"):
        path.seek(0)  # type: ignore[union-attr]
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
    return receptors_from_frame(_read_frame(path))


def _csv_frame(receptors: Iterable[Receptor]) -> pd.DataFrame:
    """Return the receptor table with times as the text written to CSV."""
    frame = receptors_to_frame(receptors)
    return frame.assign(time=frame["time"].dt.strftime("%Y-%m-%d %H:%M:%S"))


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
    existing = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
    _, spelling = _normalize_columns(existing.iloc[:0])
    missing = [
        f for f in ("time", "longitude", "latitude", "altitude") if f not in spelling
    ]
    if missing:
        raise ValueError(f"receptors.csv lacks a column for {missing}; cannot append.")
    if "r_idx" not in spelling and any(len(r) > 1 for r in receptors):
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
            existing = existing.assign(altitude_ref="agl")
            spelling["altitude_ref"] = "altitude_ref"

    new = _csv_frame(receptors)
    if "r_idx" in spelling:
        numbers = [
            int(v)
            for v in existing[spelling["r_idx"]]
            if v.strip().lstrip("-").isdigit()
        ]
        new["r_idx"] += max(numbers, default=-1) + 1
    new = new.rename(columns=spelling)
    new = new.reindex(columns=existing.columns)
    new = new.astype(object).where(new.notna(), "")
    combined = pd.concat([existing, new.astype(str)], ignore_index=True)
    return combined.to_csv(index=False, lineterminator="\n")


__all__ = [
    "ALIASES",
    "COLUMNS",
    "AnyReceptor",
    "ColumnReceptor",
    "MultiPointReceptor",
    "PointReceptor",
    "Receptor",
    "append_receptors_csv",
    "check_distinct_ids",
    "parse_receptor_id",
    "parse_time",
    "read_receptors",
    "receptors_from_frame",
    "receptors_to_csv",
    "receptors_to_frame",
    "write_receptors",
]
