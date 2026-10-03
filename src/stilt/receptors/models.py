"""
The receptor models, their times, and their ids.

A :class:`PointReceptor` releases from one point, a :class:`ColumnReceptor`
from a vertical line, and a :class:`MultiPointReceptor` from several points
at once (for example a slanted satellite sounding).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import (
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
    field_validator,
    model_validator,
)
from shapely import Geometry, LineString, MultiPoint, Point

if TYPE_CHECKING:
    from stilt.visualization import ReceptorPlotAccessor

from .validation import _raise_first, column_errors, multipoint_errors, point_errors

TimeLike: TypeAlias = dt.datetime | pd.Timestamp | np.datetime64 | str

#: Whether an altitude is above ground level or above mean sea level.
VerticalReference: TypeAlias = Literal["agl", "msl"]


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
    ``"<lon>_<lat>_<alt>"`` for a point, ``"<lon>_<lat>_X<bottom>-<top>"``
    for a column, and ``"multi_<hash>"`` for a multipoint receptor. Heights
    above mean sea level end the location with ``msl``.

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


def _ref_suffix(altitude_ref: str) -> str:
    """Return the end of a location id for a vertical reference: ``"msl"`` or nothing."""
    return "msl" if altitude_ref == "msl" else ""


def _point_location(lon: float, lat: float, alt: float, altitude_ref: str) -> str:
    """Location id of a point receptor, ``"<lon>_<lat>_<alt>"`` (``msl`` appended above sea level)."""
    return (
        f"{_format_coord(lon)}_{_format_coord(lat)}_{_format_coord(alt)}"
        f"{_ref_suffix(altitude_ref)}"
    )


def _column_location(
    lon: float, lat: float, bottom: float, top: float, altitude_ref: str
) -> str:
    """Location id of a column receptor, ``"<lon>_<lat>_X<bottom>-<top>"`` (``msl`` appended above sea level)."""
    return (
        f"{_format_coord(lon)}_{_format_coord(lat)}"
        f"_X{_format_coord(bottom)}-{_format_coord(top)}{_ref_suffix(altitude_ref)}"
    )


def _multipoint_location(
    lons: Iterable[float],
    lats: Iterable[float],
    alts: Iterable[float],
    altitude_ref: str,
) -> str:
    """
    Location id of a multipoint receptor, ``"multi_<hash>"`` (``msl`` appended above sea level).

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
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:10]
    return f"multi_{digest}{_ref_suffix(altitude_ref)}"


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
        receptors are written back to CSV, and are columns of
        ``project.receptors`` and ``project.simulations`` to select on. They
        are not part of the receptor's id or of equality.
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r})"

    def __str__(self) -> str:
        return repr(self)

    # -- geometry ----------------------------------------------------------

    @cached_property
    def geometry(self) -> Geometry:
        """
        Shapely geometry of the release points, in ``(lon, lat, alt)``.

        A ``Point`` for a point receptor, a vertical ``LineString`` from the
        bottom to the top of a column, and a ``MultiPoint`` otherwise.
        """
        points = [(lon, lat, alt) for lat, lon, alt in self.coords()]
        if self.kind == "point":
            return Point(points[0])
        if self.kind == "column":
            return LineString(points)
        return MultiPoint(points)

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

    def to_json(self) -> str:
        """Return this receptor as JSON, as result files record it; :meth:`from_json` reads it back."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str | bytes) -> Receptor:
        """Build a receptor from the JSON :meth:`to_json` wrote."""
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Receptor:
        """
        Build a receptor of the right type from a dict made by :meth:`to_dict`.

        Raises
        ------
        ValueError
            If the dict has no ``kind`` key, or names no known kind.
        """
        data = dict(d)
        if "kind" not in data:
            raise ValueError("Receptor dict must contain a 'kind' key.")
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
        _raise_first(
            point_errors(
                [self.longitude], [self.latitude], [self.altitude], self.altitude_ref
            )
        )
        return self

    @property
    def location_id(self) -> str:
        """Location id, ``"<lon>_<lat>_<alt>"``, ending in ``msl`` above sea level."""
        return _point_location(
            self.longitude, self.latitude, self.altitude, self.altitude_ref
        )

    def coords(self) -> list[tuple[float, float, float]]:
        """Return ``(lat, lon, alt)`` of the release point."""
        return [(self.latitude, self.longitude, self.altitude)]


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
        _raise_first(
            [
                *point_errors(
                    [self.longitude], [self.latitude], [self.bottom], self.altitude_ref
                ),
                *column_errors([self.bottom], [self.top]),
            ]
        )
        return self

    @property
    def location_id(self) -> str:
        """Location id, ``"<lon>_<lat>_X<bottom>-<top>"``, ending in ``msl`` above sea level."""
        return _column_location(
            self.longitude, self.latitude, self.bottom, self.top, self.altitude_ref
        )

    def coords(self) -> list[tuple[float, float, float]]:
        """Return ``(lat, lon, alt)`` of the bottom and the top of the column."""
        return [
            (self.latitude, self.longitude, self.bottom),
            (self.latitude, self.longitude, self.top),
        ]


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
        _raise_first(
            [
                *point_errors(
                    self.longitudes, self.latitudes, self.altitudes, self.altitude_ref
                ),
                *multipoint_errors(
                    self.longitudes, self.latitudes, [0] * len(self.longitudes)
                ),
            ]
        )
        return self

    @property
    def location_id(self) -> str:
        """
        Location id, ``"multi_<hash>"``, ending in ``msl`` above sea level.

        The hash covers the sorted points, with longitudes and latitudes
        rounded to 5 decimals and altitudes to 0.01 m. A whole-metre
        altitude hashes as its integer.
        """
        return _multipoint_location(
            self.longitudes, self.latitudes, self.altitudes, self.altitude_ref
        )

    def coords(self) -> list[tuple[float, float, float]]:
        """Return ``(lat, lon, alt)`` of each release point."""
        return list(zip(self.latitudes, self.longitudes, self.altitudes, strict=True))


def _hash_altitude(alt: float) -> int | float:
    """Altitude as hashed in a multipoint id: an int when whole, else to 0.01 m."""
    rounded = round(alt, 2)
    return int(rounded) if rounded == int(rounded) else rounded


#: Any receptor, for validating a dict of unknown kind.
AnyReceptor = Annotated[
    PointReceptor | ColumnReceptor | MultiPointReceptor, Field(discriminator="kind")
]
_ANY_RECEPTOR: TypeAdapter[Receptor] = TypeAdapter(AnyReceptor)
