"""Receptor data models for STILT simulations."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

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


def _format_coord(val: float) -> str:
    """Format a coordinate float as an integer string when it is whole, else as-is."""
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
    Unique spatial location identifier.

    Format: ``"{lon}_{lat}_{alt}"`` for points, ``"{lon}_{lat}_X"`` for
    columns, or ``"multi_{10-char-sha256}"`` for multipoint receptors.
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
    Unique receptor identifier: ``{YYYYMMDDHHMM}_{location_id}``.

    Parameters
    ----------
    id_str : str
        Full receptor ID string in ``{YYYYMMDDHHMM}_{location_id}`` format.
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
        """Build a ReceptorID from separate time and location components."""
        return cls(f"{pd.Timestamp(time):%Y%m%d%H%M}_{location_id}")


class Receptor(ABC):
    """Abstract base for all STILT receptor types."""

    def __init__(self, time: TimeLike, altitude_ref: VerticalReference) -> None:
        self.time = _parse_time(time)
        self.altitude_ref: VerticalReference = validate_vertical_reference(altitude_ref)
        self._geometry = None
        self._plot: ReceptorPlotAccessor | None = None

    @property
    @abstractmethod
    def location_id(self) -> LocationID:
        """Spatial location identifier that uniquely describes this receptor's position."""
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[float, float, float]]:
        """Yield ``(lat, lon, alt)`` for each constituent point."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """Serialize this receptor to a round-trippable dict including a ``"type"`` key."""
        ...

    @abstractmethod
    def _build_geometry(self) -> Geometry:
        """Construct the shapely geometry for this receptor."""
        ...

    @property
    def id(self) -> ReceptorID:
        """Receptor identifier composed of timestamp and location."""
        return ReceptorID(f"{self.time:%Y%m%d%H%M}_{self.location_id}")

    @property
    def geometry(self):
        """Lazily derived shapely geometry."""
        if self._geometry is None:
            self._geometry = self._build_geometry()
        return self._geometry

    @property
    def points(self) -> list[Point]:
        """All constituent shapely Points."""
        return [Point(lon, lat, alt) for lat, lon, alt in self]

    @property
    def plot(self) -> ReceptorPlotAccessor:
        """Plotting namespace (e.g. ``receptor.plot.map()``)."""
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
        """Reconstruct a receptor from a dict produced by ``to_dict``."""
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
        Build a receptor from ``(longitude, latitude, altitude)`` tuples.

        Returns :class:`PointReceptor` for one point, :class:`ColumnReceptor`
        when two points share the same horizontal location, and
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
    A single-point STILT receptor.

    Parameters
    ----------
    time : datetime-like or str
        Timestamp associated with the receptor.
    longitude : float
        Longitude [-180, 180].
    latitude : float
        Latitude [-90, 90].
    altitude : float
        Altitude interpreted according to ``altitude_ref``.
    altitude_ref : {"agl", "msl"}, default "agl"
        Vertical reference for the altitude.
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
        """``"{lon}_{lat}_{alt}"`` formatted location identifier."""
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
        """Return a shapely Point at this receptor's location."""
        return Point(self.longitude, self.latitude, self.altitude)

    def to_dict(self) -> dict[str, object]:
        """Return a dict with keys ``type``, ``time``, ``longitude``, ``latitude``, ``altitude``, ``altitude_ref``."""
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
    A vertical-column STILT receptor (two altitudes, shared horizontal location).

    Parameters
    ----------
    time : datetime-like or str
        Timestamp associated with the receptor.
    longitude : float
        Longitude [-180, 180].
    latitude : float
        Latitude [-90, 90].
    bottom : float
        Lower altitude bound (must be less than ``top``).
    top : float
        Upper altitude bound.
    altitude_ref : {"agl", "msl"}, default "agl"
        Vertical reference for the altitudes.
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
        """``"{lon}_{lat}_X"`` formatted location identifier (altitude replaced by ``X``)."""
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
        """Return a shapely LineString spanning bottom to top at this receptor's location."""
        return LineString(
            [
                (self.longitude, self.latitude, self.bottom),
                (self.longitude, self.latitude, self.top),
            ]
        )

    def to_dict(self) -> dict[str, object]:
        """Return a dict with keys ``type``, ``time``, ``longitude``, ``latitude``, ``bottom``, ``top``, ``altitude_ref``."""
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
    A multi-point STILT receptor with arbitrary spatial coordinates.

    Parameters
    ----------
    time : datetime-like or str
        Timestamp associated with the receptor.
    longitudes : array-like of float
        Longitudes of each constituent point.
    latitudes : array-like of float
        Latitudes of each constituent point.
    altitudes : array-like of float
        Altitudes of each constituent point.
    altitude_ref : {"agl", "msl"}, default "agl"
        Vertical reference for the altitudes.
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

    @property
    def location_id(self) -> LocationID:
        """``"multi_{sha256[:10]}"`` identifier derived from a sorted canonical hash of all points."""
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
        """Return a shapely MultiPoint covering all constituent locations."""
        return MultiPoint(
            list(zip(self.longitudes, self.latitudes, self.altitudes, strict=False))
        )

    def to_dict(self) -> dict[str, object]:
        """Return a dict with keys ``type``, ``time``, ``longitudes``, ``latitudes``, ``altitudes``, ``altitude_ref``."""
        return {
            "type": type(self).__name__,
            "time": self.time.isoformat(),
            "longitudes": self.longitudes.tolist(),
            "latitudes": self.latitudes.tolist(),
            "altitudes": self.altitudes.tolist(),
            "altitude_ref": self.altitude_ref,
        }


def read_receptors(path: str | Path) -> list[Receptor]:
    """Load receptors from a CSV file."""
    df = pd.read_csv(path, parse_dates=["time"])

    original_columns = [str(col).lower() for col in df.columns]
    inferred_altitude_ref = None
    if "zmsl" in original_columns:
        inferred_altitude_ref = "msl"
    elif "zagl" in original_columns:
        inferred_altitude_ref = "agl"

    cols = {
        "latitude": "lati",
        "longitude": "long",
        "altitude": "z",
        "zagl": "z",
        "zmsl": "z",
        "lat": "lati",
        "lon": "long",
        "height_ref": "altitude_ref",
    }
    df.columns = df.columns.str.lower()
    df = df.rename(columns=cols)
    if "altitude_ref" not in df.columns:
        df["altitude_ref"] = inferred_altitude_ref or "agl"

    required_cols = ["time", "lati", "long", "z"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Receptor file must contain columns: {required_cols}")

    def _point_receptors_from_rows(frame: pd.DataFrame) -> list[Receptor]:
        """Build one PointReceptor per row from a normalised receptor DataFrame."""
        return [
            PointReceptor(
                time=cast(Any, row).time,
                longitude=cast(Any, row).long,
                latitude=cast(Any, row).lati,
                altitude=cast(Any, row).z,
                altitude_ref=cast(Any, row).altitude_ref,
            )
            for row in frame.itertuples(index=False)
        ]

    if "r_idx" in df.columns:
        group_sizes = df.groupby("r_idx").size()
        multi_keys = [k for k, v in group_sizes.items() if v > 1]

        if not multi_keys:
            return _point_receptors_from_rows(df)

        single_mask = ~df["r_idx"].isin(multi_keys)
        result: dict[object, Receptor] = {}
        for row in df[single_mask].itertuples(index=False):
            r = cast(Any, row)
            result[r.r_idx] = PointReceptor(
                time=r.time,
                longitude=r.long,
                latitude=r.lati,
                altitude=r.z,
                altitude_ref=r.altitude_ref,
            )
        for key, g in df[~single_mask].groupby("r_idx"):
            result[key] = _receptor_from_group(cast(pd.DataFrame, g))

        return [result[k] for k in df["r_idx"].unique()]

    return _point_receptors_from_rows(df)


def _receptor_from_group(group: pd.DataFrame) -> Receptor:
    """Build one receptor from a grouped receptor CSV slice."""
    refs = {str(v).lower() for v in group["altitude_ref"].tolist()}
    if len(refs) != 1:
        raise ValueError(
            "All rows in one receptor group must share the same altitude_ref."
        )
    altitude_ref = validate_vertical_reference(refs.pop())
    lons = group["long"].tolist()
    lats = group["lati"].tolist()
    alts = group["z"].tolist()
    time = pd.to_datetime(np.atleast_1d(group["time"])[0])
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
]
