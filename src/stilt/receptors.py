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
from collections.abc import Iterable, Mapping
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
        If *receptor_id* does not start with a ``YYYYMMDDHHMM`` time and an
        underscore.
    """
    stamp, _, location = str(receptor_id).partition("_")
    try:
        if len(stamp) != 12 or not stamp.isdigit() or not location:
            raise ValueError
        time = dt.datetime.strptime(stamp, "%Y%m%d%H%M")
    except ValueError:
        raise ValueError(
            f"Receptor id {receptor_id!r} is not of the form "
            "'{YYYYMMDDHHMM}_{location_id}'."
        ) from None
    return time, location


def _format_coord(val: float) -> str:
    """Format a coordinate without a decimal point when it is a whole number."""
    return str(int(val)) if val == int(val) else str(val)


def _point_location(lon: float, lat: float, alt: float) -> str:
    """Location id of a point receptor, ``"<lon>_<lat>_<alt>"``."""
    return f"{_format_coord(lon)}_{_format_coord(lat)}_{_format_coord(alt)}"


def _column_location(lon: float, lat: float) -> str:
    """Location id of a column receptor, ``"<lon>_<lat>_X"``."""
    return f"{_format_coord(lon)}_{_format_coord(lat)}_X"


def _multipoint_location(
    lons: Iterable[float], lats: Iterable[float], alts: Iterable[float]
) -> str:
    """
    Location id of a multipoint receptor, ``"multi_<hash>"``.

    The hash covers the sorted points, with longitudes and latitudes rounded
    to 5 decimals and altitudes to 0.01 m. A whole-metre altitude hashes as
    its integer, so ids made before heights were kept to 0.01 m still match.
    """
    points = sorted(zip(lons, lats, alts, strict=True))
    canonical = json.dumps(
        [
            [round(float(lon), 5), round(float(lat), 5), _hash_altitude(float(alt))]
            for lon, lat, alt in points
        ],
        separators=(",", ":"),
    )
    return "multi_" + hashlib.sha256(canonical.encode()).hexdigest()[:10]


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

    kind: str = Field(description="Receptor type: point, column, or multipoint.")
    time: dt.datetime = Field(description="Release time, UTC.")
    altitude_ref: VerticalReference = Field(
        default="agl",
        description="Whether heights are above ground (agl) or sea level (msl).",
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
        """Parse the release time to a naive UTC datetime."""
        return parse_time(value)

    @field_validator("altitude_ref", mode="before")
    @classmethod
    def _lower_ref(cls, value: Any) -> Any:
        """Accept the altitude reference in any case."""
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
        """Return the shapely geometry of the release points."""
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
        time = parse_time(time)
        if len(pts) == 1:
            lon, lat, alt = pts[0]
            return PointReceptor(
                time=time,
                longitude=lon,
                latitude=lat,
                altitude=alt,
                altitude_ref=altitude_ref,
                attrs=labels,
            )
        if len(pts) == 2 and pts[0][:2] == pts[1][:2]:
            bottom, top = sorted((pts[0][2], pts[1][2]))
            return ColumnReceptor(
                time=time,
                longitude=pts[0][0],
                latitude=pts[0][1],
                bottom=bottom,
                top=top,
                altitude_ref=altitude_ref,
                attrs=labels,
            )
        lons, lats, alts = zip(*pts, strict=True)
        return MultiPointReceptor(
            time=time,
            longitudes=lons,
            latitudes=lats,
            altitudes=alts,
            altitude_ref=altitude_ref,
            attrs=labels,
        )


def _check_lon(values: Iterable[float]) -> None:
    """Raise if a longitude is outside [-180, 180]."""
    if any(not -180 <= v <= 180 for v in values):
        raise ValueError("longitude must be within [-180, 180].")


def _check_lat(values: Iterable[float]) -> None:
    """Raise if a latitude is outside [-90, 90]."""
    if any(not -90 <= v <= 90 for v in values):
        raise ValueError("latitude must be within [-90, 90].")


def _check_agl(values: Iterable[float], altitude_ref: str) -> None:
    """Raise if a height above ground is negative."""
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

    @model_validator(mode="after")
    def _check(self) -> PointReceptor:
        """Check the coordinates and the height."""
        _check_lon([self.longitude])
        _check_lat([self.latitude])
        _check_agl([self.altitude], self.altitude_ref)
        return self

    @property
    def location_id(self) -> str:
        """Location id, ``"<lon>_<lat>_<alt>"``."""
        return _point_location(self.longitude, self.latitude, self.altitude)

    def coords(self) -> list[tuple[float, float, float]]:
        """Return ``(lat, lon, alt)`` of the release point."""
        return [(self.latitude, self.longitude, self.altitude)]

    def __repr__(self) -> str:
        return (
            f"PointReceptor(id={self.id!r}, lon={self.longitude:.5f}, "
            f"lat={self.latitude:.5f}, alt={self.altitude:g} {self.altitude_ref})"
        )

    def _build_geometry(self) -> Point:
        """Return a shapely Point at the release point."""
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

    @model_validator(mode="after")
    def _check(self) -> ColumnReceptor:
        """Check the coordinates and that the bottom is below the top."""
        _check_lon([self.longitude])
        _check_lat([self.latitude])
        if self.bottom >= self.top:
            raise ValueError("'bottom' must be less than 'top'.")
        _check_agl([self.bottom], self.altitude_ref)
        return self

    @property
    def location_id(self) -> str:
        """Location id, ``"<lon>_<lat>_X"``."""
        return _column_location(self.longitude, self.latitude)

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
        """Return a shapely LineString from the bottom to the top."""
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

    @field_validator("longitudes", "latitudes", "altitudes", mode="before")
    @classmethod
    def _as_floats(cls, value: Any) -> tuple[float, ...]:
        """Return the values as a tuple of floats."""
        return tuple(float(v) for v in np.asarray(value, dtype=float).ravel())

    @model_validator(mode="after")
    def _check(self) -> MultiPointReceptor:
        """Check the points, and that no two share a horizontal location."""
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
        return _multipoint_location(self.longitudes, self.latitudes, self.altitudes)

    def coords(self) -> list[tuple[float, float, float]]:
        """Return ``(lat, lon, alt)`` of each release point."""
        return list(zip(self.latitudes, self.longitudes, self.altitudes, strict=True))

    def __repr__(self) -> str:
        return (
            f"MultiPointReceptor(id={self.id!r}, n_points={len(self.coords())}, "
            f"altitude_ref={self.altitude_ref})"
        )

    def _build_geometry(self) -> MultiPoint:
        """Return a shapely MultiPoint of the release points."""
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
    on a large file; :func:`receptor_from_rows` builds one from its rows.

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
    fail((lon < -180) | (lon > 180), "longitude must be within [-180, 180].")
    fail((lat < -90) | (lat > 90), "latitude must be within [-90, 90].")
    fail((ref == "agl") & (alt < 0), "AGL altitudes must be >= 0.")

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
    fail(
        (kind == "column") & (g["alt"].transform("nunique").to_numpy() == 1),
        "'bottom' must be less than 'top'.",
    )

    # Ids are made once per receptor, from its first row (or all its rows for
    # a multipoint), then spread to its rows. Python floats round much faster
    # than numpy scalars, and give the same result.
    _, first_rows = np.unique(codes, return_index=True)  # groups are 0, 1, ... in order
    lon_f, lat_f, alt_f = lon.tolist(), lat.tolist(), alt.tolist()
    group_kind = kind[first_rows]
    group_location = np.empty(len(first_rows), dtype=object)
    for code, row in enumerate(first_rows):
        if group_kind[code] == "point":
            group_location[code] = _point_location(lon_f[row], lat_f[row], alt_f[row])
        elif group_kind[code] == "column":
            group_location[code] = _column_location(lon_f[row], lat_f[row])
    multi = np.flatnonzero(kind == "multipoint")
    if len(multi):
        # HYSPLIT joins consecutive starting locations at one latitude and
        # longitude into a vertical line source, and particles are matched
        # to their release point by horizontal position.
        horizontal = pd.DataFrame(
            {
                "code": codes[multi],
                "lon": [round(lon_f[i], 5) for i in multi],
                "lat": [round(lat_f[i], 5) for i in multi],
            }
        )
        dup = np.zeros(n, dtype=bool)
        dup[multi] = horizontal.duplicated().to_numpy()
        fail(
            dup,
            "MultiPointReceptor points must have distinct horizontal locations "
            "(HYSPLIT collapses starting locations that share a lat/lon into a "
            "single vertical line source and releases only between the last two "
            "heights). Use ColumnReceptor for a vertical column, or one "
            "PointReceptor per height (distinct r_idx) for discrete release "
            "heights at one location.",
        )
        members: dict[int, list[int]] = {}
        for i, code in zip(multi.tolist(), codes[multi].tolist(), strict=True):
            members.setdefault(code, []).append(i)
        for code, idx in members.items():
            group_location[code] = _multipoint_location(
                [lon_f[i] for i in idx],
                [lat_f[i] for i in idx],
                [alt_f[i] for i in idx],
            )
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
        by_id: dict[str, list[int]] = {}
        for code in repeated:
            by_id.setdefault(str(group_ids[code]), []).append(int(code))
        drop: list[int] = []
        for group in by_id.values():
            check_distinct_ids(
                [receptor_from_rows(out.loc[codes == code]) for code in group]
            )
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
    """Return the receptor of one group of rows, given as plain values."""
    common = {"time": time, "altitude_ref": altitude_ref, "attrs": attrs}
    if kind == "point":
        return PointReceptor(
            longitude=float(lon[0]),
            latitude=float(lat[0]),
            altitude=float(alt[0]),
            **common,
        )
    if kind == "column":
        return ColumnReceptor(
            longitude=float(lon[0]),
            latitude=float(lat[0]),
            bottom=float(alt.min()),
            top=float(alt.max()),
            **common,
        )
    return MultiPointReceptor(
        longitudes=tuple(lon.tolist()),
        latitudes=tuple(lat.tolist()),
        altitudes=tuple(alt.tolist()),
        **common,
    )


def _labels(rows: pd.DataFrame) -> list[str]:
    """Return the label columns of a :func:`receptor_rows` table."""
    return [str(c) for c in rows.columns if c not in COLUMNS and c not in ROW_COLUMNS]


def _attrs(values: Iterable[Any], names: list[str]) -> dict[str, Any]:
    """Return one row's labels as a dict, with empty cells as ``None``."""
    return {k: (None if pd.isna(v) else v) for k, v in zip(names, values, strict=True)}


def receptor_from_rows(rows: pd.DataFrame) -> Receptor:
    """
    Build the receptor of its rows of a :func:`receptor_rows` table.

    Its labels (``attrs``) are the other columns of its first row.
    """
    names = _labels(rows)
    return _build(
        str(rows["kind"].iloc[0]),
        parse_time(rows["time"].iloc[0]),
        str(rows["altitude_ref"].iloc[0]),
        _attrs(rows[names].astype(object).iloc[0].tolist(), names),
        rows["longitude"].to_numpy(dtype=float),
        rows["latitude"].to_numpy(dtype=float),
        rows["altitude"].to_numpy(dtype=float),
    )


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
        In table order. A receptor listed twice appears once.

    Raises
    ------
    ValueError
        As :func:`receptor_rows` raises.
    """
    rows = receptor_rows(frame)
    names = _labels(rows)
    labels = rows[names].astype(object).to_numpy()
    times = rows["time"].dt.to_pydatetime()
    lon = rows["longitude"].to_numpy(dtype=float)
    lat = rows["latitude"].to_numpy(dtype=float)
    alt = rows["altitude"].to_numpy(dtype=float)
    kind = rows["kind"].to_numpy()
    ref = rows["altitude_ref"].to_numpy()
    receptors = []
    for idx in rows.groupby("receptor", sort=False).indices.values():
        i = idx[0]
        receptors.append(
            _build(
                str(kind[i]),
                times[i],
                str(ref[i]),
                _attrs(labels[i], names),
                lon[idx],
                lat[idx],
                alt[idx],
            )
        )
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


def read_receptor_frame(path: str | Path | IO[str]) -> pd.DataFrame:
    """Read a receptors CSV as a table, with ``r_idx`` as text and ``time`` parsed, without building receptors."""
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
    return receptors_from_frame(read_receptor_frame(path))


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


__all__ = [
    "ColumnReceptor",
    "MultiPointReceptor",
    "PointReceptor",
    "Receptor",
    "append_receptors_csv",
    "parse_receptor_id",
    "read_receptor_frame",
    "read_receptors",
    "receptor_from_rows",
    "receptor_rows",
    "receptors_from_frame",
    "receptors_to_csv",
    "receptors_to_frame",
    "write_receptors",
]
