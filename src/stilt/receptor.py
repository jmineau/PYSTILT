"""Receptor data model used by STILT simulations."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

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
    def timestr(self) -> str:
        """Time in 'YYYYMMDDHHMM' format."""
        return self.time.strftime("%Y%m%d%H%M")

    @property
    def location_id(self) -> str:
        """Unique identifier for this receptor's spatial location."""
        if self._kind == "point":
            x = _format_coord(self.longitudes[0])
            y = _format_coord(self.latitudes[0])
            z = _format_coord(self.altitudes[0])
            return f"{x}_{y}_{z}"
        elif self._kind == "column":
            x = _format_coord(self.longitudes[0])
            y = _format_coord(self.latitudes[0])
            return f"{x}_{y}_X"
        else:
            # Multipoint: SHA-256 of canonical JSON (sorted by lon, lat, altitude).
            # Matches R-STILT implementation in simulation_step.r.
            pts_sorted = sorted(
                zip(self.longitudes, self.latitudes, self.altitudes, strict=False)
            )
            canonical = json.dumps(
                [
                    [round(float(lon), 5), round(float(lat), 5), int(altitude)]
                    for lon, lat, altitude in pts_sorted
                ],
                separators=(",", ":"),
            )
            hash_str = hashlib.sha256(canonical.encode()).hexdigest()[:10]
            return f"multi_{hash_str}"

    @property
    def id(self) -> str:
        """Receptor identifier composed of timestamp and location ID."""
        return f"{self.timestr}_{self.location_id}"

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
        Parsed receptors. If a ``group`` column exists, one receptor is
        created per group; otherwise one per row.
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

    # If "group" column exists, group rows and create a single Receptor
    # per group. Otherwise, treat each row as a separate point receptor.
    key = "group" if "group" in df.columns else df.index

    return (
        df.groupby(key)
        .apply(
            lambda x: _receptor_from_group(x),
            include_groups=False,
        )
        .to_list()
    )


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
