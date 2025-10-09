
from abc import ABC
import datetime as dt
import hashlib
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from shapely import Geometry, Point, MultiPoint, LineString


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
            return f"{self.geometry.x}_{self.geometry.y}_{self.geometry.z}"
        elif isinstance(self.geometry, LineString):
            # For column locations
            coords = list(self.geometry.coords)
            if not (len(coords) == 2 and coords[0][0] == coords[1][0] and coords[0][1] == coords[1][1]):
                raise ValueError("LineString must represent a vertical column with two points at the same (lon, lat).")
            return f"{coords[0][0]}_{coords[0][1]}_X"

        # For MultiPoint geometries
        wkt_string = self.geometry.wkt
        hash_str = hashlib.md5(wkt_string.encode('utf-8')).hexdigest()
        return f"multi_{hash_str}"

    @property
    def coords(self) -> pd.DataFrame:
        """
        Returns the location's coordinates as a pandas DataFrame.
        """
        if self._coords is None:
            self._coords = pd.DataFrame({
                'longitude': self._lons,
                'latitude': self._lats,
                'height': self._hgts
            })
        return self._coords

    @property
    def points(self) -> list[Point]:
        """
        Returns a list of shapely Point objects representing the location's coordinates.
        """
        if self._points is None:
            self._points = self.coords.apply(lambda row: Point(row['longitude'],
                                                               row['latitude'],
                                                               row['height']),
                                               axis=1).to_list()
        return self._points

    @classmethod
    def from_point(cls, longitude, latitude, height) -> 'Location':
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
    def from_column(cls, longitude, latitude, bottom, top) -> 'Location':
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
        return cls(LineString([(longitude, latitude, bottom), (longitude, latitude, top)]))

    @classmethod
    def from_points(cls, points) -> 'Location':
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
                return cls.from_column(longitude=p1[0], latitude=p1[1], bottom=bottom, top=top)
        return cls(MultiPoint(points))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Location):
            return False
        return self.geometry == other.geometry


class Receptor(ABC):
    def __init__(self, time, location: Location):
        """
        A receptor that wraps a geometric object (Point, MultiPoint, etc.)
        and associates it with a timestamp.

        Parameters
        ----------
        time : datetime
            The timestamp associated with the receptor.
        location : Location
            A location object representing the receptor's spatial position.
        """
        if time is None:
            raise ValueError("'time' must be provided for all receptor types.")
        elif isinstance(time, str):
            if '-' in time:
                time = dt.datetime.fromisoformat(time)
            else:
                time = dt.datetime.strptime(time, '%Y%m%d%H%M')
        elif not isinstance(time, dt.datetime):
            raise TypeError("'time' must be a datetime object.")

        self.time = time
        self.location = location

    @property
    def geometry(self) -> Geometry:
        """
        Receptor geometry.
        """
        return self.location.geometry

    def __eq__(self, other) -> bool:
        if not isinstance(other, Receptor):
            return False
        return (self.time == other.time and
                self.location == other.location)

    @property
    def timestr(self) -> str:
        """
        Get the time as an ISO formatted string.

        Returns
        -------
        str
            Time in 'YYYYMMDDHHMM' format.
        """
        return self.time.strftime('%Y%m%d%H%M')

    @property
    def id(self) -> str:
        return f"{self.timestr}_{self.location.id}"

    @property
    def is_vertical(self) -> bool:
        raise NotImplementedError
        # TODO : when a receptor is created from metadata, it is not currently possible
        # to distinguish a SlantReceptor from a MultiPoint receptor
        return isinstance(self, (ColumnReceptor, SlantReceptor))

    @staticmethod
    def build(time, longitude, latitude, height) -> 'Receptor':
        """
        Build a receptor object from time, latitude, longitude, and height.

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
            The constructed receptor object.
        """
        # Get receptor time
        time = pd.to_datetime(np.atleast_1d(time)[0])

        # If height is a list/array of length 2 and longitude/latitude are scalars, repeat lon/lat
        if np.isscalar(longitude) and np.isscalar(latitude) and hasattr(height, '__len__') and len(height) == 2:
            longitude = [longitude, longitude]
            latitude = [latitude, latitude]
        # Build location object to determine geometry type
        location = Location.from_points(list(zip(np.atleast_1d(longitude),
                                                        np.atleast_1d(latitude),
                                                        np.atleast_1d(height))))
        # Build appropriate receptor subclass based on geometry type
        if isinstance(location.geometry, Point):
            return PointReceptor(time=time,
                                 longitude=location._lons[0],
                                 latitude=location._lats[0],
                                 height=location._hgts[0])
        elif isinstance(location.geometry, MultiPoint):
            return MultiPointReceptor(time=time,
                                      points=location.points)
        elif isinstance(location.geometry, LineString):
            return ColumnReceptor.from_points(time=time,
                                              points=location.points)
        else:
            raise ValueError("Unsupported geometry type for receptor.")

    @staticmethod
    def load_receptors_from_csv(path: str | Path) -> list['Receptor']:
        """
        Load receptors from a CSV file.
        """
        # Read the CSV file
        df = pd.read_csv(path, parse_dates=['time'])

        # Map columns
        cols = {
            'latitude': 'lati',
            'longitude': 'long',
            'height': 'zagl',
            'lat': 'lati',
            'lon': 'long',
        }
        df.columns = df.columns.str.lower()
        df = df.rename(columns=cols)

        # Check for required columns
        required_cols = ['time', 'lati', 'long', 'zagl']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Receptor file must contain columns: {required_cols}")

        # Determine grouping key
        if 'group' in df.columns:
            # Group rows and create a single Receptor for each group
            key = 'group'
        else:
            # Treat each row as a separate PointReceptor
            key = df.index

        # Build receptors
        receptors = df.groupby(key).apply(lambda x:
                                          Receptor.build(time=x['time'],
                                                         longitude=x['long'],
                                                         latitude=x['lati'],
                                                         height=x['zagl']),
                                          include_groups=False).to_list()

        return receptors


class PointReceptor(Receptor):
    """
    Represents a single receptor at a specific 3D point (latitude, longitude, height) in space and time.
    """

    def __init__(self, time, longitude, latitude, height):
        location = Location.from_point(longitude=longitude,
                                       latitude=latitude,
                                       height=height)
        super().__init__(time=time, location=location)

    @property
    def longitude(self) -> float:
        return self.location._lons[0]

    @property
    def latitude(self) -> float:
        return self.location._lats[0]

    @property
    def height(self) -> float:
        return self.location._hgts[0]

    def __iter__(self) -> Generator[float, None, None]:
        """
        Allow unpacking of PointReceptor into (lon, lat, height).
        """
        yield self.longitude
        yield self.latitude
        yield self.height


class MultiPointReceptor(Receptor):
    """
    Represents a receptor composed of multiple 3D points, all at the same time.
    """

    def __init__(self, time, points):
        location = Location.from_points(points)
        super().__init__(time=time, location=location)

    @property
    def points(self) -> list[Point]:
        return self.location.points

    def __iter__(self) -> Generator[Point, None, None]:
        """
        Allow unpacking of MultiPointReceptor into its constituent Points.
        """
        yield from self.points

    def __len__(self) -> int:
        return len(self.points)


class ColumnReceptor(Receptor):
    """
    Represents a vertical column receptor at a single (x, y) location,
    defined by a bottom and top height.
    """

    def __init__(self, time, longitude, latitude, bottom, top):
        location = Location.from_column(longitude=longitude,
                                        latitude=latitude,
                                        bottom=bottom,
                                        top=top)
        super().__init__(time=time, location=location)

        self._longitude = longitude
        self._latitude = latitude
        self._top = top
        self._bottom = bottom

    @property
    def longitude(self) -> float:
        return self._longitude

    @property
    def latitude(self) -> float:
        return self._latitude

    @property
    def top(self) -> float:
        return self._top

    @property
    def bottom(self) -> float:
        return self._bottom

    @classmethod
    def from_points(cls, time, points):
        p1, p2 = points

        lon = p1[0]
        lat = p1[1]
        if lon != p2[0]:
            raise ValueError("For a column receptor, the longitude must be the same for both points.")
        if lat != p2[1]:
            raise ValueError("For a column receptor, the latitude must be the same for both points.")

        top = max(p1[2], p2[2])
        bottom = min(p1[2], p2[2])
        if not (bottom < top):
            raise ValueError("'bottom' height must be less than 'top' height.")

        return cls(time=time, longitude=lon, latitude=lat, bottom=bottom, top=top)


class SlantReceptor(MultiPointReceptor):
    """
    Represents a slanted column receptor, defined by multiple points along the slant.
    """
    
    @classmethod
    def from_top_and_bottom(cls, time, bottom, top, numpar, weights=None):
        """
        Parameters
        ----------
        time : any
            Timestamp.
        bottom : tuple
            (lon, lat, height) tuple for the bottom of the slant.
        top : tuple
            (lon, lat, height) tuple for the top of the slant.
        numpar : int
            Number of points along the slant.
        weights : list of float, optional
            Weights for each point along the slant. Must be the same length as `numpar`.
        """
        raise NotImplementedError("SlantReceptor is not fully implemented yet.")

        if len(bottom) != 3 or len(top) != 3:
            raise ValueError("'bottom' and 'top' must be (lon, lat, height) tuples.")
        if numpar < 2:
            raise ValueError("'numpar' must be at least 2 to define a slant.")

        # Generate intermediate points along the slant
        # TODO :
        # - Implement the logic to create slant receptors from the endpoints.
        #   - There are various difficulties in determining the correct slant path
        #     including determining the appropriate height above ground.
        #   - Aaron is working on this. 
        lon_step = (top[0] - bottom[0]) / (numpar - 1)
        lat_step = (top[1] - bottom[1]) / (numpar - 1)
        height_step = (top[2] - bottom[2]) / (numpar - 1)
        points = [
            (bottom[0] + i * lon_step, bottom[1] + i * lat_step, bottom[2] + i * height_step)
            for i in range(numpar)
        ]

        # Initialize as a MultiPointReceptor
        super().__init__(time=time, points=points)