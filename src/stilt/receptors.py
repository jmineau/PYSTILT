import datetime as dt
import hashlib
import json
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import Geometry, LineString, MultiPoint, Point


def _format_coord(val: float) -> str:
    """Format coordinate as int if whole number, otherwise as float."""
    return str(int(val)) if val == int(val) else str(val)


class Location:
    """
    Represents a spatial location for STILT models, independent of time.
    Can be used to generate consistent location IDs and create receptors when combined with time.
    """

    def __init__(self, geometry: Geometry):
        """
        Initialize a location with a shapely geometry.

        Parameters
        ----------
        geometry : shapely.Geometry
            A geometric object (e.g., Point, MultiPoint, LineString).
        """
        self._geometry = geometry

        if isinstance(geometry, Point):
            self._lons = np.array([geometry.x])
            self._lats = np.array([geometry.y])
            self._hgts = np.array([geometry.z])
        elif isinstance(geometry, MultiPoint):
            self._lons = np.array([pt.x for pt in geometry.geoms])
            self._lats = np.array([pt.y for pt in geometry.geoms])
            self._hgts = np.array([pt.z for pt in geometry.geoms])
        elif isinstance(geometry, LineString):
            self._lons = np.array([coord[0] for coord in geometry.coords])
            self._lats = np.array([coord[1] for coord in geometry.coords])
            self._hgts = np.array([coord[2] for coord in geometry.coords])
        else:
            raise TypeError("Unsupported geometry type for Location.")

        self._coords = None
        self._points = None

    @property
    def geometry(self):
        """
        Location geometry.
        """
        return self._geometry

    @property
    def id(self) -> str:
        """
        Generate a unique identifier for this location based on its geometry.
        """
        if isinstance(self.geometry, Point):
            x = _format_coord(self.geometry.x)
            y = _format_coord(self.geometry.y)
            z = _format_coord(self.geometry.z)
            return f"{x}_{y}_{z}"
        elif isinstance(self.geometry, LineString):
            # For column locations
            coords = list(self.geometry.coords)
            if not (
                len(coords) == 2
                and coords[0][0] == coords[1][0]
                and coords[0][1] == coords[1][1]
            ):
                raise ValueError(
                    "LineString must represent a vertical column with two points at the same (lon, lat)."
                )
            x = _format_coord(coords[0][0])
            y = _format_coord(coords[0][1])
            return f"{x}_{y}_X"

        # For MultiPoint geometries: SHA-256 of canonical JSON (sorted by lon, lat, zagl)
        # Matches R-STILT implementation in simulation_step.r.
        points_sorted = sorted(zip(self._lons, self._lats, self._hgts, strict=False))
        canonical = json.dumps(
            [
                [round(lon, 5), round(lat, 5), int(zagl)]
                for lon, lat, zagl in points_sorted
            ],
            separators=(",", ":"),
        )
        hash_str = hashlib.sha256(canonical.encode()).hexdigest()[:10]
        return f"multi_{hash_str}"

    @property
    def coords(self) -> pd.DataFrame:
        """
        Returns the location's coordinates as a pandas DataFrame.
        """
        if self._coords is None:
            self._coords = pd.DataFrame(
                {"longitude": self._lons, "latitude": self._lats, "height": self._hgts}
            )
        return self._coords

    @property
    def points(self) -> list[Point]:
        """
        Returns a list of shapely Point objects representing the location's coordinates.
        """
        if self._points is None:
            self._points = self.coords.apply(
                lambda row: Point(row["longitude"], row["latitude"], row["height"]),
                axis=1,
            ).to_list()
        return self._points

    @classmethod
    def from_point(cls, longitude, latitude, height) -> "Location":
        """
        Create a Location from a single point.

        Parameters
        ----------
        longitude : float
            Longitude coordinate
        latitude : float
            Latitude coordinate
        height : float
            Height above ground level

        Returns
        -------
        Location
            A point location
        """
        return cls(Point(longitude, latitude, height))

    @classmethod
    def from_column(cls, longitude, latitude, bottom, top) -> "Location":
        """
        Create a Location representing a vertical column.

        Parameters
        ----------
        longitude : float
            Longitude coordinate
        latitude : float
            Latitude coordinate
        bottom : float
            Bottom height of column
        top : float
            Top height of column

        Returns
        -------
        Location
            A column location
        """
        if not (bottom < top):
            raise ValueError("'bottom' height must be less than 'top' height.")
        return cls(
            LineString([(longitude, latitude, bottom), (longitude, latitude, top)])
        )

    @classmethod
    def from_points(cls, points) -> "Location":
        """
        Create a Location from multiple points.

        Parameters
        ----------
        points : list of tuple
            List of (lon, lat, height) tuples

        Returns
        -------
        Location
            A multi-point location
        """
        if len(points) == 0:
            raise ValueError("At least one point must be provided.")
        elif len(points) == 1:
            return cls.from_point(*points[0])
        elif len(points) == 2:
            p1, p2 = points
            if p1[0] == p2[0] and p1[1] == p2[1]:
                bottom = min(p1[2], p2[2])
                top = max(p1[2], p2[2])
                return cls.from_column(
                    longitude=p1[0], latitude=p1[1], bottom=bottom, top=top
                )
        return cls(MultiPoint(points))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Location):
            return False
        return self.geometry == other.geometry


class Receptor:
    """
    A receptor that wraps a geometric object (Point, MultiPoint, LineString)
    and associates it with a timestamp.

    The receptor kind (point, column, multipoint) is derived from the
    underlying geometry — there are no subclasses.
    """

    def __init__(self, time, location: Location):
        """
        Parameters
        ----------
        time : datetime or str
            The timestamp associated with the receptor.
        location : Location
            A location object representing the receptor's spatial position.
        """
        if time is None:
            raise ValueError("'time' must be provided for all receptor types.")
        elif isinstance(time, str):
            if "-" in time:
                time = dt.datetime.fromisoformat(time)
            else:
                time = dt.datetime.strptime(time, "%Y%m%d%H%M")
        elif not isinstance(time, dt.datetime):
            raise TypeError("'time' must be a datetime object.")

        self.time = time
        self.location = location

    @property
    def kind(self) -> str:
        """
        The receptor kind, derived from the underlying geometry.

        Returns
        -------
        str
            One of 'point', 'column', or 'multipoint'.
        """
        if isinstance(self.location.geometry, Point):
            return "point"
        elif isinstance(self.location.geometry, LineString):
            return "column"
        elif isinstance(self.location.geometry, MultiPoint):
            return "multipoint"
        return "unknown"

    @property
    def geometry(self) -> Geometry:
        """Receptor geometry."""
        return self.location.geometry

    @property
    def timestr(self) -> str:
        """Time in 'YYYYMMDDHHMM' format."""
        return self.time.strftime("%Y%m%d%H%M")

    @property
    def id(self) -> str:
        return f"{self.timestr}_{self.location.id}"

    @property
    def is_vertical(self) -> bool:
        return self.kind == "column"

    # -- Coordinate accessors --------------------------------------------------

    @property
    def longitude(self) -> float:
        """Longitude of the receptor (first point for multipoint)."""
        return self.location._lons[0]

    @property
    def latitude(self) -> float:
        """Latitude of the receptor (first point for multipoint)."""
        return self.location._lats[0]

    @property
    def height(self) -> float:
        """Height AGL of the receptor (first point for multipoint)."""
        return self.location._hgts[0]

    @property
    def points(self) -> list[Point]:
        """All constituent shapely Points."""
        return self.location.points

    @property
    def bottom(self) -> float | None:
        """Bottom height for column receptors, None otherwise."""
        if self.kind != "column":
            return None
        return float(min(self.location._hgts))

    @property
    def top(self) -> float | None:
        """Top height for column receptors, None otherwise."""
        if self.kind != "column":
            return None
        return float(max(self.location._hgts))

    # -- Iteration -------------------------------------------------------------

    def __iter__(self) -> Generator[float | Point, None, None]:
        """
        Iterate over the receptor's coordinates.

        For point receptors, yields (longitude, latitude, height).
        For multipoint/column receptors, yields constituent Points.
        """
        if self.kind == "point":
            yield self.longitude
            yield self.latitude
            yield self.height
        else:
            yield from self.points

    def __len__(self) -> int:
        """Number of constituent points."""
        return len(self.location._lons)

    # -- Equality --------------------------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, Receptor):
            return False
        return self.time == other.time and self.location == other.location

    def __repr__(self) -> str:
        return f"Receptor(kind={self.kind!r}, id={self.id!r})"

    # -- Factory methods -------------------------------------------------------

    @staticmethod
    def build(time, longitude, latitude, height) -> "Receptor":
        """
        Build a receptor from time and coordinates.

        Parameters
        ----------
        time : datetime
            Timestamp of the receptor.
        longitude : float | list[float]
            Longitude(s) of the receptor.
        latitude : float | list[float]
            Latitude(s) of the receptor.
        height : float | list[float]
            Height(s) above ground level of the receptor.

        Returns
        -------
        Receptor
        """
        time = pd.to_datetime(np.atleast_1d(time)[0])

        # If height is a list/array of length 2 and lon/lat are scalars,
        # repeat lon/lat to form a column receptor.
        if (
            np.isscalar(longitude)
            and np.isscalar(latitude)
            and hasattr(height, "__len__")
            and len(height) == 2
        ):
            longitude = [longitude, longitude]
            latitude = [latitude, latitude]

        location = Location.from_points(
            list(
                zip(
                    np.atleast_1d(longitude),
                    np.atleast_1d(latitude),
                    np.atleast_1d(height),
                    strict=False,
                )
            )
        )
        return Receptor(time=time, location=location)

    @staticmethod
    def load_receptors_from_csv(path: str | Path) -> list["Receptor"]:
        """
        Load receptors from a CSV file.
        """
        df = pd.read_csv(path, parse_dates=["time"])

        cols = {
            "latitude": "lati",
            "longitude": "long",
            "height": "zagl",
            "lat": "lati",
            "lon": "long",
        }
        df.columns = df.columns.str.lower()
        df = df.rename(columns=cols)

        required_cols = ["time", "lati", "long", "zagl"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Receptor file must contain columns: {required_cols}")

        # If "group" column exists, group rows and create a single Receptor
        # per group. Otherwise, treat each row as a separate point receptor.
        key = "group" if "group" in df.columns else df.index

        receptors = (
            df.groupby(key)
            .apply(
                lambda x: Receptor.build(
                    time=x["time"],
                    longitude=x["long"],
                    latitude=x["lati"],
                    height=x["zagl"],
                ),
                include_groups=False,
            )
            .to_list()
        )

        return receptors
