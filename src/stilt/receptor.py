"""Receptor data model used by STILT simulations."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import pandas as pd
from shapely import LineString, MultiPoint, Point

from stilt.config import VerticalReference, validate_vertical_reference

if TYPE_CHECKING:
    from stilt.visualization import ReceptorPlotAccessor

TimeLike: TypeAlias = dt.datetime | pd.Timestamp | np.datetime64 | str
CoordLike: TypeAlias = float | Iterable[float]


def _format_coord(val: float) -> str:
    """Format coordinate as int if whole number, otherwise as float."""
    return str(int(val)) if val == int(val) else str(val)


def _parse_time(time: object) -> dt.datetime:
    """Parse a time value into one naive UTC datetime."""
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


class ReceptorKind(str, Enum):
    """Stable receptor-kind discriminator."""

    POINT = "point"
    COLUMN = "column"
    MULTIPOINT = "multipoint"


class LocationID(str):
    """Unique identifier for a receptor's spatial location."""

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

    @classmethod
    def from_receptor(cls, receptor: Receptor) -> LocationID:
        """Create a LocationID from a Receptor instance."""
        if receptor.kind == ReceptorKind.POINT:
            x = _format_coord(receptor.longitudes[0])
            y = _format_coord(receptor.latitudes[0])
            z = _format_coord(receptor.altitudes[0])
            return cls(f"{x}_{y}_{z}")
        if receptor.kind == ReceptorKind.COLUMN:
            x = _format_coord(receptor.longitudes[0])
            y = _format_coord(receptor.latitudes[0])
            return cls(f"{x}_{y}_X")
        pts_sorted = sorted(
            zip(
                receptor.longitudes,
                receptor.latitudes,
                receptor.altitudes,
                strict=False,
            )
        )
        canonical = json.dumps(
            [
                [round(float(lon), 5), round(float(lat), 5), int(altitude)]
                for lon, lat, altitude in pts_sorted
            ],
            separators=(",", ":"),
        )
        hash_str = hashlib.sha256(canonical.encode()).hexdigest()[:10]
        return cls(f"multi_{hash_str}")


class ReceptorID(str):
    """Unique identifier for one receptor timestamp plus location."""

    time: dt.datetime
    location: LocationID

    def __new__(cls, id_str: str):
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
    def from_receptor(cls, receptor: Receptor) -> ReceptorID:
        """Create a ReceptorID from a Receptor instance."""
        return cls.from_parts(receptor.time, LocationID.from_receptor(receptor))

    @classmethod
    def from_parts(
        cls,
        time: dt.datetime | pd.Timestamp,
        location_id: LocationID,
    ) -> ReceptorID:
        """Create a ReceptorID from separate time and location_id components."""
        return cls(f"{pd.Timestamp(time):%Y%m%d%H%M}_{location_id}")


class Receptor:
    """A STILT receptor: a spatial location associated with a timestamp."""

    def __init__(
        self,
        time: TimeLike,
        longitude: CoordLike,
        latitude: CoordLike,
        altitude: CoordLike,
        *,
        altitude_ref: VerticalReference = "agl",
    ):
        """
        Parameters
        ----------
        time : datetime-like or str
            The timestamp associated with the receptor.
        longitude : float or array-like
            Longitude(s) of the receptor.
        latitude : float or array-like
            Latitude(s) of the receptor.
        altitude : float or array-like
            Altitude(s) interpreted according to ``altitude_ref``.
        altitude_ref : {"agl", "msl"}, default "agl"
            Vertical reference for the receptor altitudes.
        """
        self.time = _parse_time(time)
        self.altitude_ref: VerticalReference = validate_vertical_reference(altitude_ref)

        altitude_values = np.atleast_1d(np.asarray(altitude, dtype=float))
        longitude_values: CoordLike = longitude
        latitude_values: CoordLike = latitude
        if (
            np.isscalar(longitude)
            and np.isscalar(latitude)
            and altitude_values.size == 2
        ):
            lon_scalar = float(cast(Any, longitude))
            lat_scalar = float(cast(Any, latitude))
            longitude_values = [lon_scalar, lon_scalar]
            latitude_values = [lat_scalar, lat_scalar]

        self.longitudes = np.atleast_1d(np.asarray(longitude_values, dtype=float))
        self.latitudes = np.atleast_1d(np.asarray(latitude_values, dtype=float))
        self.altitudes = altitude_values

        if not (len(self.longitudes) == len(self.latitudes) == len(self.altitudes)):
            raise ValueError(
                "longitude, latitude, and altitude must have the same length."
            )
        if np.any((self.longitudes < -180) | (self.longitudes > 180)):
            raise ValueError("longitude must be within [-180, 180].")
        if np.any((self.latitudes < -90) | (self.latitudes > 90)):
            raise ValueError("latitude must be within [-90, 90].")
        if self.altitude_ref == "agl" and np.any(self.altitudes < 0):
            raise ValueError("AGL altitudes must be >= 0.")

        n = len(self.longitudes)
        if n == 1:
            self._kind: ReceptorKind = ReceptorKind.POINT
        elif (
            n == 2
            and self.longitudes[0] == self.longitudes[1]
            and self.latitudes[0] == self.latitudes[1]
        ):
            self._kind = ReceptorKind.COLUMN
        else:
            self._kind = ReceptorKind.MULTIPOINT

        self._geometry = None
        self._points = None
        self._plot: ReceptorPlotAccessor | None = None

    @property
    def plot(self) -> ReceptorPlotAccessor:
        """Plotting namespace (e.g. ``receptor.plot.map()``)."""
        if self._plot is None:
            from stilt.visualization import ReceptorPlotAccessor

            self._plot = ReceptorPlotAccessor(self)
        return self._plot

    @property
    def kind(self) -> ReceptorKind:
        """One of ``ReceptorKind.POINT``, ``COLUMN``, or ``MULTIPOINT``."""
        return self._kind

    @property
    def id(self) -> ReceptorID:
        """Receptor identifier composed of timestamp and location ID."""
        return ReceptorID.from_receptor(self)

    @property
    def geometry(self):
        """Lazily derived shapely geometry."""
        if self._geometry is None:
            if self._kind == ReceptorKind.POINT:
                self._geometry = Point(
                    self.longitudes[0], self.latitudes[0], self.altitudes[0]
                )
            elif self._kind == ReceptorKind.COLUMN:
                self._geometry = LineString(
                    list(
                        zip(
                            self.longitudes,
                            self.latitudes,
                            self.altitudes,
                            strict=False,
                        )
                    )
                )
            else:
                self._geometry = MultiPoint(
                    list(
                        zip(
                            self.longitudes,
                            self.latitudes,
                            self.altitudes,
                            strict=False,
                        )
                    )
                )
        return self._geometry

    @property
    def points(self) -> list[Point]:
        """All constituent shapely Points."""
        if self._points is None:
            self._points = [
                Point(lon, lat, altitude)
                for lon, lat, altitude in zip(
                    self.longitudes, self.latitudes, self.altitudes, strict=False
                )
            ]
        return self._points

    @property
    def longitude(self) -> float:
        """Longitude of the receptor. Raises for multipoint."""
        if self._kind == ReceptorKind.MULTIPOINT:
            raise AttributeError("Multipoint receptors do not have a single longitude.")
        return float(self.longitudes[0])

    @property
    def latitude(self) -> float:
        """Latitude of the receptor. Raises for multipoint."""
        if self._kind == ReceptorKind.MULTIPOINT:
            raise AttributeError("Multipoint receptors do not have a single latitude.")
        return float(self.latitudes[0])

    @property
    def altitude(self) -> float:
        """Point altitude. Raises for non-point receptors."""
        if self._kind != ReceptorKind.POINT:
            raise AttributeError("Only point receptors have a single altitude.")
        return float(self.altitudes[0])

    @property
    def bottom(self) -> float:
        """Bottom altitude. Raises for non-column receptors."""
        if self._kind != ReceptorKind.COLUMN:
            raise AttributeError("Only column receptors have a bottom altitude.")
        return float(self.altitudes.min())

    @property
    def top(self) -> float:
        """Top altitude. Raises for non-column receptors."""
        if self._kind != ReceptorKind.COLUMN:
            raise AttributeError("Only column receptors have a top altitude.")
        return float(self.altitudes.max())

    def __len__(self) -> int:
        """Number of constituent points."""
        return len(self.longitudes)

    def __eq__(self, other: object) -> bool:
        """Return True when both timestamp and location are equal."""
        if not isinstance(other, Receptor):
            return False
        return (
            self.time == other.time
            and np.array_equal(self.longitudes, other.longitudes)
            and np.array_equal(self.latitudes, other.latitudes)
            and self.altitude_ref == other.altitude_ref
            and np.array_equal(self.altitudes, other.altitudes)
        )

    def __hash__(self) -> int:
        """Hash receptor identity from time, coordinates, and altitude reference."""
        return hash(
            (
                self.time.isoformat(),
                tuple(np.asarray(self.longitudes, dtype=float)),
                tuple(np.asarray(self.latitudes, dtype=float)),
                tuple(np.asarray(self.altitudes, dtype=float)),
                self.altitude_ref,
            )
        )

    def __repr__(self) -> str:
        """Compact developer-facing receptor representation."""
        if self.kind == ReceptorKind.POINT:
            location = f"lon={self.longitude:.5f}, lat={self.latitude:.5f}, alt={self.altitude:g}"
        elif self.kind == ReceptorKind.COLUMN:
            location = (
                f"lon={self.longitude:.5f}, lat={self.latitude:.5f}, "
                f"bottom={self.bottom:g}, top={self.top:g}"
            )
        else:
            location = f"n_points={len(self)}"
        return f"Receptor(kind={self.kind.value!r}, id={self.id!r}, {location})"

    def to_dict(self) -> dict[str, object]:
        """JSON-round-trippable dict. Handles point/column/multipoint."""
        return {
            "time": self.time.isoformat(),
            "longitude": self.longitudes.tolist(),
            "latitude": self.latitudes.tolist(),
            "altitude": self.altitudes.tolist(),
            "altitude_ref": self.altitude_ref,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Receptor:
        """Reconstruct a :class:`Receptor` from a dict."""
        return cls(
            time=cast(TimeLike, d["time"]),
            longitude=cast(CoordLike, d["longitude"]),
            latitude=cast(CoordLike, d["latitude"]),
            altitude=cast(CoordLike, d["altitude"]),
            altitude_ref=cast(VerticalReference, d.get("altitude_ref", "agl")),
        )

    @classmethod
    def from_column(
        cls,
        time: TimeLike,
        longitude: float,
        latitude: float,
        bottom: float,
        top: float,
        *,
        altitude_ref: VerticalReference = "agl",
    ) -> Receptor:
        """Build a vertical-column receptor."""
        if bottom >= top:
            raise ValueError("'bottom' must be less than 'top'.")
        return cls(
            time=time,
            longitude=[longitude, longitude],
            latitude=[latitude, latitude],
            altitude=[bottom, top],
            altitude_ref=altitude_ref,
        )

    @classmethod
    def from_points(
        cls,
        time: TimeLike,
        points: Iterable[tuple[float, float, float]],
        *,
        altitude_ref: VerticalReference = "agl",
    ) -> Receptor:
        """Build a receptor from a sequence of (longitude, latitude, altitude) tuples."""
        pts = list(points)
        if not pts:
            raise ValueError("At least one point must be provided.")
        lons, lats, altitudes = zip(*pts, strict=False)
        return cls(
            time=time,
            longitude=list(lons),
            latitude=list(lats),
            altitude=list(altitudes),
            altitude_ref=altitude_ref,
        )


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
        return [
            Receptor(
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
            result[r.r_idx] = Receptor(
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
    return Receptor(
        time=pd.to_datetime(np.atleast_1d(group["time"])[0]),
        longitude=group["long"],
        latitude=group["lati"],
        altitude=group["z"],
        altitude_ref=altitude_ref,
    )


__all__ = [
    "LocationID",
    "Receptor",
    "ReceptorID",
    "ReceptorKind",
    "read_receptors",
]
