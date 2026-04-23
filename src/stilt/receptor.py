"""Receptor data model used by STILT simulations."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from shapely import LineString, MultiPoint, Point

from stilt.config import VerticalReference, validate_vertical_reference

if TYPE_CHECKING:
    from stilt.visualization import ReceptorPlotAccessor


def _format_coord(val: float) -> str:
    """Format coordinate as int if whole number, otherwise as float."""
    return str(int(val)) if val == int(val) else str(val)


def _parse_time(time):
    """Parse a time value into a datetime object."""
    if time is None:
        raise ValueError("'time' must be provided for all receptor types.")
    if isinstance(time, pd.Timestamp):
        return time.to_pydatetime()
    if isinstance(time, str):
        if "-" in time:
            return dt.datetime.fromisoformat(time)
        return dt.datetime.strptime(time, "%Y%m%d%H%M")
    if isinstance(time, dt.datetime):
        return time
    # Fallback: try pandas conversion (handles numpy datetime64, etc.)
    return pd.to_datetime(time).to_pydatetime()


class LocationID(str):
    """Unique identifier for a receptor's spatial location.

    For point receptors: 'lon_lat_alt' (e.g. '-120.5_35.2_100').
    For column receptors: 'lon_lat_X' (e.g. '-120.5_35.2_X').
    For multipoint receptors: 'multi_{hash}' where {hash} is a SHA-256 hash of the sorted coordinates.
    """

    @classmethod
    def from_receptor(cls, receptor: Receptor) -> LocationID:
        """Create a LocationID from a Receptor instance."""
        if receptor._kind == "point":
            x = _format_coord(receptor.longitudes[0])
            y = _format_coord(receptor.latitudes[0])
            z = _format_coord(receptor.altitudes[0])
            return cls(f"{x}_{y}_{z}")
        elif receptor._kind == "column":
            x = _format_coord(receptor.longitudes[0])
            y = _format_coord(receptor.latitudes[0])
            return cls(f"{x}_{y}_X")
        else:
            # Multipoint: SHA-256 of canonical JSON (sorted by lon, lat, altitude).
            # Matches R-STILT implementation in simulation_step.r.
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
    """
    Unique identifier for a receptor, composed of timestamp and location ID.

    Format: '{YYYYMMDDHHMM}_{location_id}', where location_id is either:
      - 'lon_lat_alt' for point receptors
      - 'lon_lat_X' for column receptors
      - 'multi_{hash}' for multipoint receptors (SHA-256 hash of sorted coordinates)
    """

    time: dt.datetime
    location: LocationID

    def __new__(cls, id_str: str):
        if "_" not in id_str:
            raise ValueError(
                "ReceptorID must be in format '{YYYYMMDDHHMM}_{location_id}'"
            )
        instance = super().__new__(cls, id_str)
        time_str, location_id = id_str.split("_", 1)
        time = dt.datetime(
            int(time_str[0:4]),  # year
            int(time_str[4:6]),  # month
            int(time_str[6:8]),  # day
            int(time_str[8:10]),  # hour
            int(time_str[10:12]),  # minute
        )
        instance.time = time
        instance.location = LocationID(location_id)
        return instance

    @classmethod
    def from_receptor(cls, receptor: Receptor) -> ReceptorID:
        """Create a ReceptorID from a Receptor instance."""
        return cls.from_parts(receptor.time, LocationID.from_receptor(receptor))

    @classmethod
    def from_parts(cls, time: dt.datetime, location_id: LocationID) -> ReceptorID:
        """Create a ReceptorID from separate time and location_id components."""
        return cls(f"{time:%Y%m%d%H%M}_{location_id}")


class Receptor:
    """A STILT receptor: a spatial location associated with a timestamp.

    The receptor kind (point, column, multipoint) is inferred from the
    coordinate arrays.
    """

    def __init__(
        self,
        time,
        longitude,
        latitude,
        altitude=None,
        *,
        altitude_ref: VerticalReference = "agl",
    ):
        """
        Parameters
        ----------
        time : datetime, str, or Timestamp
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
        if altitude is None:
            raise ValueError("'altitude' must be provided for all receptor types.")

        # If altitude is a 2-element sequence and lon/lat are scalars,
        # expand lon/lat to form a column receptor.
        if (
            np.isscalar(longitude)
            and np.isscalar(latitude)
            and hasattr(altitude, "__len__")
            and len(altitude) == 2
        ):
            longitude = [longitude, longitude]
            latitude = [latitude, latitude]

        self.longitudes = np.atleast_1d(np.asarray(longitude, dtype=float))
        self.latitudes = np.atleast_1d(np.asarray(latitude, dtype=float))
        self.altitudes = np.atleast_1d(np.asarray(altitude, dtype=float))

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

        # Infer kind from array shape and values
        n = len(self.longitudes)
        if n == 1:
            self._kind = "point"
        elif (
            n == 2
            and self.longitudes[0] == self.longitudes[1]
            and self.latitudes[0] == self.latitudes[1]
        ):
            self._kind = "column"
        else:
            self._kind = "multipoint"

        # Lazy caches
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
    def kind(self) -> str:
        """One of 'point', 'column', or 'multipoint'."""
        return self._kind

    @property
    def id(self) -> ReceptorID:
        """Receptor identifier composed of timestamp and location ID."""
        return ReceptorID.from_receptor(self)

    @property
    def geometry(self):
        """Lazily derived shapely geometry."""
        if self._geometry is None:
            if self._kind == "point":
                self._geometry = Point(
                    self.longitudes[0], self.latitudes[0], self.altitudes[0]
                )
            elif self._kind == "column":
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
    def points(self):
        """All constituent shapely Points."""
        if self._points is None:
            self._points = [
                Point(lon, lat, altitude)
                for lon, lat, altitude in zip(
                    self.longitudes, self.latitudes, self.altitudes, strict=False
                )
            ]
        return self._points

    # -- Coordinate accessors --------------------------------------------------

    @property
    def longitude(self) -> float:
        """Longitude of the receptor. Raises for multipoint."""
        if self._kind == "multipoint":
            raise AttributeError("Multipoint receptors do not have a single longitude.")
        return float(self.longitudes[0])

    @property
    def latitude(self) -> float:
        """Latitude of the receptor. Raises for multipoint."""
        if self._kind == "multipoint":
            raise AttributeError("Multipoint receptors do not have a single latitude.")
        return float(self.latitudes[0])

    @property
    def altitude(self) -> float:
        """Point altitude. Raises for non-point receptors."""
        if self._kind != "point":
            raise AttributeError("Only point receptors have a single altitude.")
        return float(self.altitudes[0])

    @property
    def bottom(self) -> float:
        """Bottom altitude. Raises for non-column receptors."""
        if self._kind != "column":
            raise AttributeError("Only column receptors have a bottom altitude.")
        return float(self.altitudes.min())

    @property
    def top(self) -> float:
        """Top altitude. Raises for non-column receptors."""
        if self._kind != "column":
            raise AttributeError("Only column receptors have a top altitude.")
        return float(self.altitudes.max())

    def __len__(self) -> int:
        """Number of constituent points."""
        return len(self.longitudes)

    # -- Equality --------------------------------------------------------------

    def __eq__(self, other) -> bool:
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

    def __repr__(self) -> str:
        """Compact developer-facing receptor representation."""
        return f"Receptor(kind={self.kind!r}, id={self.id!r})"

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-round-trippable dict. Handles point/column/multipoint."""
        return {
            "time": self.time.isoformat(),
            "longitude": self.longitudes.tolist(),
            "latitude": self.latitudes.tolist(),
            "altitude": self.altitudes.tolist(),
            "altitude_ref": self.altitude_ref,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Receptor:
        """Reconstruct a :class:`Receptor` from a dict (inverse of :meth:`to_dict`).

        Parameters
        ----------
        d : dict
            Dictionary with keys ``time``, ``longitude``, ``latitude``,
            and ``altitude``.

        Returns
        -------
        Receptor
        """
        return cls(
            time=d["time"],
            longitude=d["longitude"],
            latitude=d["latitude"],
            altitude=d["altitude"],
            altitude_ref=d.get("altitude_ref", "agl"),
        )

    # -- Factory methods -------------------------------------------------------

    @classmethod
    def from_column(
        cls,
        time,
        longitude,
        latitude,
        bottom,
        top,
        *,
        altitude_ref: VerticalReference = "agl",
    ) -> Receptor:
        """Build a vertical-column receptor.

        Parameters
        ----------
        time : datetime, str, or Timestamp
        longitude, latitude : float
        bottom, top : float
            Bottom and top heights above ground level. bottom must be < top.
        """
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
        time,
        points,
        *,
        altitude_ref: VerticalReference = "agl",
    ) -> Receptor:
        """Build a receptor from a sequence of (longitude, latitude, altitude) tuples.

        A single point produces a point receptor; two points at the same
        lon/lat produce a column; everything else produces a multipoint.

        Parameters
        ----------
        time : datetime, str, or Timestamp
        points : sequence of (lon, lat, altitude) tuples
        """
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
    """Load receptors from a CSV file.

    Parameters
    ----------
    path : str or Path
        CSV path containing receptor definitions.

    Returns
    -------
    list[Receptor]
        Parsed receptors. If an ``r_idx`` column exists, one receptor is
        created per unique index value (supports column/multipoint receptors);
        otherwise one receptor is created per row.
    """
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
        """Fast path: one Receptor per row (all point receptors)."""
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

    # If an "r_idx" column exists, group rows and create a single Receptor per
    # index value (supports column / multipoint receptors).  Use itertuples for
    # single-row groups and groupby only for multi-row groups — groupby().apply()
    # over many unique keys is very slow in pandas.
    if "r_idx" in df.columns:
        group_sizes = df.groupby("r_idx").size()
        multi_keys = [k for k, v in group_sizes.items() if v > 1]

        if not multi_keys:
            return _point_receptors_from_rows(df)

        single_mask = ~df["r_idx"].isin(multi_keys)
        result: dict = {}
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
