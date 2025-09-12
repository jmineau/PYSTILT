"""
Stochastic Time-Inverted Lagrangian Transport (STILT) Model.

A python implementation of the [R-STILT](https://github.com/jmineau/stilt) model framework.

> Inspired by https://github.com/uataq/air-tracker-stiltctl
"""

from abc import ABC, abstractmethod
from collections import UserDict
import datetime as dt
from functools import cached_property
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, ClassVar, Generator, List, Literal, Tuple, Type
from typing_extensions import \
    Self  # requires python 3.11 to import from typing

import f90nml
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import Point, LineString, MultiPoint
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, field_validator, model_validator, Field
import pyproj
import xarray as xr
import yaml


def stilt_init(project: str | Path, branch: str | None = None,
               repo: str | None = None):
    '''
    Initialize STILT project

    Python implementation of Rscript -e "uataq::stilt_init('project')"

    Parameters
    ----------
    project : str
        Name/path of STILT project. If path is not provided,
        project will be created in current working directory.
    branch : str, optional
        Branch of STILT project repo. The default is jmineau.
    repo : str, optional
        URL of STILT project repo. The default is jmineau/stilt.
    '''
    if branch is None:
        branch = 'jmineau'
    if repo is None:
        repo = 'https://github.com/jmineau/stilt'
    elif 'uataq' in repo and branch == 'jmineau':
        raise ValueError("The 'uataq' repo does not have a 'jmineau' branch. ")

    # Extract project name and working directory
    project = Path(project)
    name = project.name
    wd = project.parent
    if wd == Path('.'):
        wd = Path.cwd()

    if project.exists():
        raise FileExistsError(f'{project} already exists')

    # Clone git repository
    cmd = f'git clone -b {branch} --single-branch --depth=1 {repo} {project}'
    subprocess.check_call(cmd, shell=True)

    # Run setup executable
    project.joinpath('setup').chmod(0o755)
    subprocess.check_call('./setup', cwd=project)

    # Render run_stilt.r template with project name and working directory
    run_stilt_path = project.joinpath('r/run_stilt.r')
    run_stilt = run_stilt_path.read_text()
    run_stilt = run_stilt.replace('{{project}}', name)
    run_stilt = run_stilt.replace('{{wd}}', str(wd))
    run_stilt_path.write_text(run_stilt)


class Meteorology:
    """Meteorological data files for STILT simulations."""
    def __init__(self, path: str | Path, format: str, tres: str | pd.Timedelta):
        self.path = Path(path)
        self.format = format
        self.tres = pd.to_timedelta(tres)

        # Initialize available files list
        self._available_files = []

    @property
    def available_files(self) -> list[Path]:
        if not self._available_files:
            self._available_files = list(self.path.glob(self.format))
        return self._available_files

    def get_files(self, r_time, n_hours) -> list[Path]:
        # Implement logic to retrieve meteorological files based on the parameters
        raise NotImplementedError

    def calc_subgrids(self, files, out_dir, exe_dir,
                      projection, xmin, xmax, ymin, ymax,
                      levels=None, buffer=0.1) -> Self:
        # I think we want to return a new Meteorology instance
        # with a new path that we can check for files
        raise NotImplementedError


class Location:
    """
    Represents a spatial location for STILT models, independent of time.
    Can be used to generate consistent location IDs and create receptors when combined with time.
    """
    def __init__(self, geometry: shapely.Geometry):
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
    def points(self) -> List[Point]:
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
    def geometry(self) -> shapely.Geometry:
        """
        Receptor geometry.
        """
        return self._location.geometry

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
    def load_receptors_from_csv(path: str | Path) -> List['Receptor']:
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
    def points(self) -> List[Point]:
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


class Resolution(BaseModel):
    xres: float
    yres: float

    def __str__(self) -> str:
        return f"{self.xres}x{self.yres}"


class Control(BaseModel):
    """HYSPLIT control parameters."""
    receptor: Receptor
    emisshrs: float
    n_hours: int
    w_option: int
    z_top: float
    met_files: List[Path]

    class Config:
        # Allows Pydantic to work with custom classes like Receptor
        arbitrary_types_allowed = True

    def to_file(self, path):
        raise NotImplementedError

    @classmethod
    def from_path(cls, path):
        """
        Build Control object from HYSPLIT control file.

        Returns
        -------
        Control
            Control object with parsed parameters.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError('CONTROL file not found. Has the simulation been ran?')

        with path.open() as f:
            lines = f.readlines()

        # Parse receptor time
        time = dt.datetime.strptime(lines[0].strip(), '%y %m %d %H %M')

        # Parse receptors
        n_receptors = int(lines[1].strip())
        cursor = 2
        lats, lons, zagls = [], [], []
        for i in range(cursor, cursor + n_receptors):
            lat, lon, zagl = map(float, lines[i].strip().split())
            lats.append(lat)
            lons.append(lon)
            zagls.append(zagl)

        # Build receptor from receptors
        receptor = Receptor.build(time=time, latitude=lats, longitude=lons, height=zagls)

        cursor += n_receptors
        n_hours = int(lines[cursor].strip())
        w_option = int(lines[cursor + 1].strip())
        z_top = float(lines[cursor + 2].strip())

        # Parse met files
        n_met_files = int(lines[cursor + 3].strip())
        cursor += 4
        met_files = []
        for i in range(n_met_files):
            dir_index = cursor + (i * 2)
            file_index = dir_index + 1
            met_file = Path(lines[dir_index].strip()) / lines[file_index].strip()
            met_files.append(met_file)

        cursor += 2 * n_met_files
        emisshrs = float(lines[cursor].strip())

        return cls(
            receptor=receptor,
            emisshrs=emisshrs,
            n_hours=n_hours,
            w_option=w_option,
            z_top=z_top,
            met_files=met_files
        )


class SystemParams(BaseModel):
    stilt_wd: Path
    output_wd: Path | None = None
    lib_loc: Path | int | None = None

    @model_validator(mode="after")
    def _set_system_defaults(self) -> Self:
        """Set default values for system parameters."""

        if self.output_wd is None:
            self.output_wd = self.stilt_wd / "out"

        return self


class FootprintParams(BaseModel):
    hnf_plume: bool = True
    projection: str = '+proj=longlat'
    smooth_factor: float = 1.0
    time_integrate: bool = False
    xmn: float | None = None
    xmx: float | None = None
    xres: float | List[float] | None = None
    ymn: float | None = None
    ymx: float | None = None
    yres: float | List[float] | None = None

    @model_validator(mode="after")
    def _set_footprint_defaults(self) -> Self:
        """Set default values for footprint parameters."""
        if self.yres is None:
            self.yres = self.xres
        return self

    @model_validator(mode="after")
    def _validate_footprint_params(self) -> Self:
        """Validate footprint parameters."""

        if type(self.xres) != type(self.yres):
            raise ValueError("xres and yres must both be of the same type.")

        def length(res):
            if res is None:
                return 0
            if isinstance(res, list):
                return len(res)
            return 1

        xlen = length(self.xres)
        ylen = length(self.yres)

        if xlen != ylen:
            raise ValueError("xres and yres must have the same length.")

        return self

    @property
    def resolutions(self) -> List[Resolution] | None:
        """Get the x and y resolutions as a list of tuples."""
        if self.xres is None:
            return None
        if not isinstance(self.xres, list):
            self.xres = [self.xres]
            self.yres = [self.yres]
        return [Resolution(xres=xres, yres=yres) for xres, yres
                in zip(self.xres, self.yres)]


class MetParams(BaseModel):
    met_path: Path
    met_file_format: str
    met_file_tres: str
    met_subgrid_buffer: float = 0.1
    met_subgrid_enable: bool = False
    met_subgrid_levels: int | None = None
    n_met_min: int = 1


class ModelParams(BaseModel):
    n_hours: int = -24
    numpar: int = 1000
    rm_dat: bool = True
    run_foot: bool = True
    run_trajec: bool = True
    simulation_id: str | List[str] | None = None
    timeout: int = 3600
    varsiwant: List[Literal[
        'time', 'indx', 'long', 'lati', 'zagl', 'sigw', 'tlgr', 'zsfc', 'icdx',
        'temp', 'samt', 'foot', 'shtf', 'tcld', 'dmas', 'dens', 'rhfr', 'sphu',
        'lcld', 'zloc', 'dswf', 'wout', 'mlht', 'rain', 'crai', 'pres', 'whtf',
        'temz', 'zfx1'
    ]] = Field(default_factory=lambda: [
        'time', 'indx', 'long', 'lati', 'zagl', 'foot', 'mlht', 'pres',
        'dens', 'samt', 'sigw', 'tlgr'
    ])

    @model_validator(mode='after')
    def _validate_run_flags(self) -> Self:
        """Ensure at least one of `run_trajec` or `run_foot` is True."""
        if not self.run_trajec and not self.run_foot:
            raise ValueError("Nothing to do: set `run_trajec` or `run_foot` to True")
        return self


class TransportParams(BaseModel):
    capemin: float = -1.0
    cmass: int = 0
    conage: int = 48
    cpack: int = 1
    delt: int = 1
    dxf: int = 1
    dyf: int = 1
    dzf: float = 0.01
    efile: str = ''
    emisshrs: float = 0.01
    frhmax: float = 3.0
    frhs: float = 1.0
    frme: float = 0.1
    frmr: float = 0.0
    frts: float = 0.1
    frvs: float = 0.1
    hscale: int = 10800
    ichem: int = 8
    idsp: int = 2
    initd: int = 0
    k10m: int = 1
    kagl: int = 1
    kbls: int = 1
    kblt: int = 5
    kdef: int = 0
    khinp: int = 0
    khmax: int = 9999
    kmix0: int = 250
    kmixd: int = 3
    kmsl: int = 0
    kpuff: int = 0
    krand: int = 4
    krnd: int = 6
    kspl: int = 1
    kwet: int = 1
    kzmix: int = 0
    maxdim: int = 1
    maxpar: int | None = None
    mgmin: int = 10
    mhrs: int = 9999
    nbptyp: int = 1
    ncycl: int = 0
    ndump: int = 0
    ninit: int = 1
    nstr: int = 0
    nturb: int = 0
    nver: int = 0
    outdt: int = 0
    p10f: int = 1
    pinbc: str = ''
    pinpf: str = ''
    poutf: str = ''
    qcycle: int = 0
    rhb: float = 80.0
    rht: float = 60.0
    splitf: int = 1
    tkerd: float = 0.18
    tkern: float = 0.18
    tlfrac: float = 0.1
    tout: float = 0.0
    tratio: float = 0.75
    tvmix: float = 1.0
    veght: float = 0.5
    vscale: int = 200
    vscaleu: int = 200
    vscales: int = -1
    w_option: int = 0
    wbbh: int = 0
    wbwf: int = 0
    wbwr: int = 0
    wvert: bool = False
    z_top: float = 25000.0
    zicontroltf: int = 0
    ziscale: int | List[int] = 0


class ErrorParams(BaseModel):
    siguverr: float | None = None
    tluverr: float | None = None
    zcoruverr: float | None = None
    horcoruverr: float | None = None
    sigzierr: float | None = None
    tlzierr: float | None = None
    horcorzierr: float | None = None

    XYERR_PARAMS: ClassVar[Tuple[str, ...]] = ('siguverr', 'tluverr', 'zcoruverr', 'horcoruverr')
    ZIERR_PARAMS: ClassVar[Tuple[str, ...]] = ('sigzierr', 'tlzierr', 'horcorzierr')

    @model_validator(mode='after')
    def _validate_error_params(self) -> Self:
        """
        Validate error parameters to ensure they are either all set or all None
        """
        xy_params = self.xyerr_params()
        zi_params = self.zierr_params()

        for name, params in [("XY", xy_params), ("ZI", zi_params)]:
            is_na = [pd.isna(v) for v in params.values()]
            if any(is_na):
                if not all(is_na):
                    raise ValueError(f"Inconsistent {name} error parameters: all must be set or all must be None")

        return self

    def xyerr_params(self) -> dict[str, float | None]:
        """
        Get the XY error parameters as a dictionary.
        """
        return {param: getattr(self, param) for param in self.XYERR_PARAMS}

    def zierr_params(self) -> dict[str, float | None]:
        """
        Get the ZI error parameters as a dictionary.
        """
        return {param: getattr(self, param) for param in self.ZIERR_PARAMS}

    @property
    def winderrtf(self) -> int:
        """
        Determine the winderrtf flag based on the presence of error parameters.

        Returns
        -------
        int
            Wind error control flag.
                0 : No error parameters are set
                1 : ZI error parameters are set
                2 : XY error parameters are set
                3 : Both XY and ZI error parameters are set
        """
        xyerr = all(self.xyerr_params().values())
        zierr = all(self.zierr_params().values())

        return 2 * xyerr + zierr


class UserFuncParams(BaseModel):
    before_footprint: Callable | Path | None = None

    @field_validator('before_footprint', mode='before')
    @classmethod
    def _load_before_footprint(cls, v: Any) -> Any:
        """Ensure before_footprint is a callable or None."""
        if isinstance(v, (str, Path)):
            # Load the function from the specified path
            p = Path(v)

            if p.suffix.lower().endswith('r'):
                # Pass the R path
                return v
            elif p.suffix.lower().endswith('py'):
                # Load the Python function
                raise NotImplementedError("Loading Python functions from file is not implemented yet.")
            else:
                raise ValueError(f"Unsupported file type: {p.suffix}")
        return v


class BaseConfig(ABC, SystemParams, FootprintParams, MetParams, ModelParams,
                 TransportParams, ErrorParams, UserFuncParams):
    """
    STILT Configuration

    This class consolidates all configuration parameters for the STILT model,
    including system settings, footprint parameters, meteorological data,
    model specifics, transport settings, error handling, and user-defined
    functions.
    """

    class Config:
        # Allows Pydantic to work with custom classes like Receptor
        arbitrary_types_allowed = True

    @staticmethod
    def _load_yaml_params(path: str | Path) -> dict[str, Any]:
        """
        Load a YAML config file and return its contents as a dictionary.
        """
        with Path(path).open() as f:
            config = yaml.safe_load(f)

        # Flatten the config dictionary
        params = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    params[f"{subkey}"] = subvalue
            else:
                params[key] = value

        return params

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """
        Load STILT configuration from a YAML file.
        """
        params = cls._load_yaml_params(path)
        return cls(**params)

    @model_validator(mode='after')
    def _validate_base_config(self) -> Self:
        """Perform validation that depends on multiple fields."""

        # Check if there's anything to run
        if not self.run_trajec and not self.run_foot:
            raise ValueError("Nothing to do: set run_trajec or run_foot to True")

        # Check for grid parameters if running footprint or subgrid met
        if self.run_foot or self.met_subgrid_enable:
            required_grid_params = ['xmn', 'xmx', 'xres', 'ymn', 'ymx']
            if any(getattr(self, arg) is None for arg in required_grid_params):
                raise ValueError(
                    "xmn, xmx, xres, ymn, and ymx must be specified when "
                    "met_subgrid_enable or run_foot is True"
                )

        return self

    @model_validator(mode='after')
    def _set_config_defaults(self) -> Self:
        """Set default values for configuration parameters."""

        # Set default for maxpar if not provided
        if self.maxpar is None:
            self.maxpar = self.numpar

        return self

    def system_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr)
                for attr in SystemParams.model_fields.keys()}

    def footprint_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr)
                for attr in FootprintParams.model_fields.keys()}

    def met_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr)
                for attr in MetParams.model_fields.keys()}

    def model_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr)
                for attr in ModelParams.model_fields.keys()}

    def transport_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr)
                for attr in TransportParams.model_fields.keys()}

    def error_params(self) -> dict[str, Any]:
        return {attr: getattr(self, attr)
                for attr in ErrorParams.model_fields.keys()}

    def user_funcs(self) -> dict[str, Any]:
        return {attr: getattr(self, attr)
                for attr in UserFuncParams.model_fields.keys()}


class SimulationConfig(BaseConfig):

    receptor: Receptor

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        # Open simulation config like a model config
        model_config = ModelConfig.from_path(path)
        # Then extract the receptor
        receptor = model_config.receptors[0]
        return cls(receptor=receptor, **model_config.model_dump())

    @field_validator('simulation_id', mode='after')
    @classmethod
    def _validate_simulation_id(cls, simulation_id) -> str:
        if not simulation_id:
            simulation_id = cls.receptor.id
        elif not isinstance(simulation_id, str):
            raise TypeError("simulation_id must be a string")
        return simulation_id

    def to_model_config(self) -> 'ModelConfig':
        config = self.model_dump()
        receptor = config.pop('receptor')
        return ModelConfig(
            receptors=[receptor],
            **config
        )


class ModelConfig(BaseConfig):

    receptors: List[Receptor]

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        params = cls._load_yaml_params(path)
        if 'stilt_wd' not in params:
            params['stilt_wd'] = Path(path).parent
        return cls(**params)

    @model_validator(mode='before')
    @classmethod
    def _load_receptors(cls, data) -> Self:
        """
        Validates and loads receptors. If a path is provided, it loads
        receptors from the corresponding CSV file.
        """
        receptors = data.get('receptors')
        if isinstance(receptors, (str, Path)):
            # If the input is a path, load from the file.
            receptor_path = Path(receptors)
            if not receptor_path.is_absolute():
                receptor_path = Path(data.get('stilt_wd')) / receptor_path
            data['receptors'] = Receptor.load_receptors_from_csv(receptor_path)
        return data

    @model_validator(mode='after')
    def _validate_model_config(self) -> Self:
        """Validate the model configuration."""

        # Check if simulation_id is set
        if isinstance(self.simulation_id, str) and len(self.receptors) > 1:
            raise ValueError("Simulation ID must be specified for each receptor or be left blank.")

        return self

    def to_file(self):
        # Write out receptor information to csv
        # Write out config
        raise NotImplementedError

    def build_simulation_configs(self) -> List[SimulationConfig]:
        """
        Build a list of SimulationConfig objects, one for each receptor.
        """
        raise NotImplementedError
        config = self.model_dump()
        receptors = config.pop('receptors')
        simulation_id = config.pop('simulation_id')
        if isinstance(simulation_id, list):
            # TODO
            pass
            
        return [
            SimulationConfig(receptor=receptor, **config)
            for receptor in receptors
        ]


class Output(ABC):
    """Abstract base class for STILT model outputs."""
    def __init__(self, simulation_id, receptor, data):
        self.simulation_id = simulation_id
        self.receptor = receptor
        self.data = data

    @property
    @abstractmethod
    def id(self) -> str:
        pass


class Trajectory(Output):
    """STILT trajectory."""
    def __init__(self, simulation_id: str, receptor: Receptor,
                 data: pd.DataFrame, n_hours: int,
                 met_files: list[Path], params: dict):
        super().__init__(simulation_id=simulation_id,
                         receptor=receptor, data=data)
        self.n_hours = n_hours
        self.met_files = met_files
        self.params = params

    @property
    def id(self) -> str:
        """Generate the ID for the trajectory."""
        return f"{self.simulation_id}_{'error' if self.is_error else 'trajec'}"

    @property
    def is_error(self) -> bool:
        """Determine if the trajectory has errors based on the wind error flag."""
        winderrtf = self.params.get('winderrtf', 0)
        return winderrtf > 0

    @classmethod
    def calculate(cls, simulation_dir, control, namelist: f90nml.Namelist,
                  timeout=3600, rm_dat=True,
                  file=None):
        raise NotImplementedError
        simulation_dir = Path(simulation_dir)
        simulation_id = simulation_dir.name

         # Write files needed for hysplit binary
        # - CONTROL
        control.write(simulation_dir / 'CONTROL')
        # - SETUP.CFG
        setup = f90nml.Namelist({'setup': namelist})
        setup.write(simulation_dir / 'SETUP.CFG')
        # - Optional[ZICONTROL]
        # TODO

        # Call hysplit binary to calculate trajectory
        hycs_std = simulation_dir / "hycs_std"
        # TODO
        # Exit if not backwards

        # Read PARTICLE_STILT.DAT file
        data = pd.read_csv(simulation_dir / "PARTICLE_STILT.DAT", skiprows=1)
        # Delete if selected
        if rm_dat:
            # TODO
            pass

        # Calculate `xhgt` (original release height) for Column/MultiPoint simulations

        # Write data to parquet
        if file is True:
            # Use default name
            file = simulation_dir / f"{simulation_id}_trajec.parquet"
        if file:
            data.to_parquet(file)

        return cls(
            simulation_id=simulation_dir.name,
            receptor=control.receptor,
            data=data,
            n_hours=control.n_hours,
            met_files=control.met_files,
            params=namelist,
        )

    @classmethod
    def from_path(cls, path):
        # Read config and control files
        config = SimulationConfig.from_path(path.parent / 'config.yaml')
        control = Control.from_path(path.parent / 'CONTROL')

        # Read data from parquet file
        data = Trajectory.read_parquet(path, r_time=control.receptor.time,
                                       outdt=config.outdt)

        return cls(
            simulation_id=config.simulation_id,
            receptor=config.receptor,
            data=data,
            n_hours=config.n_hours,
            met_files=control.met_files,
            params=config.transport_params(),
        )

    @staticmethod
    def read_parquet(path: str | Path,
                     r_time: dt.datetime,
                     outdt: int = 0, **kwargs) -> pd.DataFrame:
        data = pd.read_parquet(path, **kwargs)

        unit = 'min' if outdt == 0 else str(outdt) + 'min'
        data['datetime'] =  r_time + pd.to_timedelta(data['time'], unit=unit)
        return data


class Footprint(Output):
    """STILT footprint."""

    # Maybe in the future we will inherit or replicate BaseGrid functionality
    # super(BaseGrid).__init__(data=self.data, crs=self.crs)
    # thinking now that this would be implmented in a subclass to keep this class cleaner

    def __init__(self,
                 simulation_id: str, receptor: Receptor,
                 data: xr.DataArray,
                 xmin: float, xmax: float,
                 ymin: float, ymax: float,
                 xres: float, yres: float,
                 projection: str = '+proj=longlat',
                 smooth_factor: float = 1.0,
                 time_integrate: bool = False,
                 ):
        super().__init__(simulation_id=simulation_id, receptor=receptor,
                         data=data)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xres = xres
        self.yres = yres
        self.projection = projection
        self.smooth_factor = smooth_factor
        self.time_integrate = time_integrate

    @property
    def resolution(self) -> str:
        return f"{self.xres}x{self.yres}"

    @property
    def id(self) -> str:
        return f"{self.simulation_id}_{self.resolution}_foot"

    @property
    def time_range(self) -> tuple[dt.datetime, dt.datetime]:
        """
        Get time range of footprint data.

        Returns
        -------
        tuple[pd.Timestamp, pd.Timestamp]
            Time range of footprint data.
        """
        times = sorted(self.data.time.values)
        start = pd.Timestamp(times[0]).to_pydatetime()
        stop = pd.Timestamp(times[-1]) + pd.Timedelta(hours=1)
        return start, stop.to_pydatetime()

    @classmethod
    def from_path(cls, path: str | Path,
                  **kwargs) -> Self:
        """
        Create Footprint object from netCDF file.

        Parameters
        ----------
        path : str | Path
            Path to netCDF file.
        **kwargs : dict
            Additional keyword arguments for xr.open_dataset.

        Returns
        -------
        Footprint
            Footprint object.
        """
        # Resolve the file path
        path = Path(path).resolve()

        # Build the configuration from the file path
        config = SimulationConfig.from_path(path.parent)

        # Read the netCDF file, parsing the receptor
        data = cls.read_netcdf(path, parse_receptor=True, **kwargs)

        # Assert that the footprint receptor matches the config receptor
        if data.attrs['receptor'] != config.receptor:
            raise ValueError(f"Receptor mismatch: {data.attrs['receptor']} != {config.receptor}")

        # Redefine the attributes of the data
        attrs = {
            'time_created': dt.datetime.fromisoformat(data.attrs.pop('time_created'))
        }
        data.attrs = attrs

        return cls(simulation_id=config.simulation_id,
                   receptor=config.receptor,
                   data=data.foot,
                   xmin=config.xmn, xmax=config.xmx,
                   ymin=config.ymn, ymax=config.ymx,
                   xres=config.xres, yres=config.yres,
                   projection=config.projection,
                   smooth_factor=config.smooth_factor,
                   time_integrate=config.time_integrate
                   )

    @staticmethod
    def get_res_from_file(file: str | Path) -> str:
        """
        Extract resolution from the file name.

        Parameters
        ----------
        file : str | Path
            Path to the footprint netCDF file.

        Returns
        -------
        str
            resolution.
        """
        file = Path(file).resolve()
        simulation_id = Simulation.get_sim_id_from_path(file)
        pattern = fr'{simulation_id}_?(.*)_foot\.nc$'
        match = re.search(pattern, file.name)
        if match:
            res = match.group(1)
        else:
            raise ValueError(f"Unable to extract resolution from file name: {file.name}")
        return res

    @classmethod
    def calculate(cls, particles: pd.DataFrame, xmin: float, xmax: float,
                  ymin: float, ymax: float, xres: float, yres: float,
                  projection: str = '+proj=longlat',
                  smooth_factor: float = 1.0, time_integrate: bool = False,
                  file: str | Path | None = None) -> Self:
        raise NotImplementedError
        wrap_lons_antimeridian = partial(wrap_lons, base=0)

        def make_gauss_kernel(rs: tuple[float, float], sigma: float):
            '''
            Replicate Ben's make_gauss_kernel function in Python

            .. note::
                No need for projection parameter as we don't need the raster package
            '''
            if sigma == 0:
                return np.array([[1]])

            rs = np.array(rs)

            d = 3 * sigma
            nx = 1 + 2 * int(d / rs[0])
            ny = 1 + 2 * int(d / rs[1])
            m = np.zeros((ny, nx))

            half_rs = rs / 2
            xr = nx * half_rs[0]
            yr = ny * half_rs[1]

            # Create a grid of coordinates
            x_coords = np.linspace(-xr + half_rs[0], xr - half_rs[0], nx)
            y_coords = np.linspace(-yr + half_rs[1], yr - half_rs[1], ny)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)

            # Calculate the Gaussian kernel
            p = x_grid**2 + y_grid**2
            m = 1 / (2 * np.pi * sigma**2) * np.exp(-p / (2 * sigma**2))
            w = m / np.sum(m)
            w[np.isnan(w)] = 1
            return w

        p = particles.copy()

        nparticles = len(p['indx'].unique())
        time_sign = np.sign(np.median(p['time']))
        is_longlat = '+proj=longlat' in projection

        if is_longlat:
            # Determine longitude wrapping behavior for grid extents containing anti
            # meridian, including partial wraps (e.g. 20deg from 170:-170) and global
            # coverage (e.g. 360deg from -180:180)
            xdist = ((180 - xmn) - (-180 - xmx)) % 360
            if xdist == 0:
                xdist = 360
                xmn, xmx = -180, 180
            elif xmx < xmn or xmx > 180:
                p['long'] = wrap_lons_antimeridian(p['long'])
                xmn = wrap_lons_antimeridian(xmn)
                xmx = wrap_lons_antimeridian(xmx)

            xres_deg = xres
            yres_deg = yres
        else:
            # Convert grid resolution to degrees using pyproj
            proj = pyproj.Proj(projection)
            xres_deg, yres_deg = proj(xres, 0, inverse=True)[0], proj(0, yres, inverse=True)[1]

        # Interpolate particle locations during first 100 minutes of simulation if
        # median distance traveled per time step is larger than grid resolution
        aptime = p['time'].abs()
        distances = p[aptime < 100].groupby('indx').agg(
            dx=pd.NamedAgg(column='long', aggfunc=lambda x: x.diff().abs().median()),
            dy=pd.NamedAgg(column='lati', aggfunc=lambda x: x.diff().abs().median())
        ).reset_index()
        should_interpolate = (distances.dx.median() > xres_deg) or (distances.dy.median() > yres_deg)

        if should_interpolate:
            times = time_sign * np.concatenate([np.arange(0, 10.1, 0.1),
                                                np.arange(10.2, 20.2, 0.2),
                                                np.arange(20.5, 100.5, 0.5)])

            # Preserve total field prior to split-interpolating particle positions
            foot_0_10_sum = p.foot[aptime <= 10].sum()
            foot_10_20_sum = p.foot[(aptime > 10) & (aptime <= 20)].sum()
            foot_20_100_sum = p.foot[(aptime > 20) & (aptime <= 100)].sum()

            # Split particle influence along linear trajectory to sub-minute timescales
            time_indx_grid = pd.DataFrame(list(itertools.product(times, p['indx'].unique())))
            p = p.merge(time_indx_grid, on=['indx', 'time'], how='outer')
            def interpolate_group(group):
                group = group.sort_values(by='time', ascending=False)
                group['long'] = group['long'].interpolate(method='linear')
                group['lati'] = group['lati'].interpolate(method='linear')
                group['foot'] = group['foot'].interpolate(method='linear')
                return group

            p = p.groupby('indx').apply(interpolate_group).reset_index(drop=True)
            p = p.dropna()
            p['time'] = p['time'].round(1)  # FIX ME I AM HERE checking on the interpolation

        p['rtime'] = p.groupby('indx')['time'].transform(lambda x: x - time_sign * x.abs().min())

        if not is_longlat:
            from pyproj import Proj, transform
            proj = Proj(projection)
            p['long'], p['lati'] = transform(Proj(init='epsg:4326'), proj, p['long'].values, p['lati'].values)
            xmn, xmx, ymn, ymx = transform(Proj(init='epsg:4326'), proj, [xmn, xmx], [ymn, ymx])

        glong = np.arange(xmn, xmx, xres)
        glati = np.arange(ymn, ymx, yres)

        kernel = p.groupby('rtime').agg({'long': 'var', 'lati': 'var', 'lati': 'mean'}).reset_index()
        kernel['varsum'] = kernel['long'] + kernel['lati']
        kernel['di'] = kernel['varsum'] ** 0.25
        kernel['ti'] = (kernel['rtime'].abs() / 1440) ** 0.5
        kernel['grid_conv'] = np.cos(np.deg2rad(kernel['lati'])) if is_longlat else 1
        kernel['w'] = smooth_factor * 0.06 * kernel['di'] * kernel['ti'] / kernel['grid_conv']

        max_k = make_gauss_kernel([xres, yres], kernel['w'].max())

        xbuf = max_k.shape[1]
        ybuf = max_k.shape[0]
        glong_buf = np.arange(xmn - xbuf * xres, xmx + xbuf * xres, xres)
        glati_buf = np.arange(ymn - ybuf * yres, ymx + ybuf * yres, yres)

        p = p[(p['foot'] > 0) & (p['long'] >= xmn - xbuf * xres) & (p['long'] < xmx + xbuf * xres) & (p['lati'] >= ymn - ybuf * yres) & (p['lati'] < ymx + ybuf * yres)]
        if p.empty:
            return None

        p['loi'] = np.searchsorted(glong_buf, p['long']) - 1
        p['lai'] = np.searchsorted(glati_buf, p['lati']) - 1

        nx, ny = len(glong_buf), len(glati_buf)
        grd = np.zeros((ny, nx))

        interval = 3600
        interval_mins = interval / 60
        p['layer'] = 0 if time_integrate else np.floor(p['time'] / interval_mins).astype(int)

        layers = p['layer'].unique()
        foot = np.zeros((ny, nx, len(layers)))

        for i, layer in enumerate(layers):
            layer_subset = p[p['layer'] == layer]
            for rtime in layer_subset['rtime'].unique():
                step = layer_subset[layer_subset['rtime'] == rtime]
                step_w = kernel.loc[kernel['rtime'].sub(rtime).abs().idxmin(), 'w']
                k = make_gauss_kernel([xres, yres], step_w)
                for _, row in step.iterrows():
                    loi, lai = int(row['loi']), int(row['lai'])
                    foot[lai:lai+k.shape[0], loi:loi+k.shape[1], i] += row['foot'] * k

        foot = foot[xbuf:-xbuf, ybuf:-ybuf, :] / nparticles

        time_out = self.receptor.time if time_integrate else self.receptor.time + pd.to_timedelta(layers * interval, unit='s')

        da = xr.DataArray(foot, coords=[glati, glong, time_out], dims=['lat', 'lon', 'time'])
        return da

    @staticmethod
    def read_netcdf(file: str | Path, parse_receptor: bool = True, **kwargs) -> xr.Dataset:
        """
        Read netCDF file and return xarray Dataset.

        Parameters
        ----------
        file : str | Path
            Path to netCDF file.
        parse_receptor : bool, optional
            Whether to parse receptor coordinates. Default is True.
        **kwargs : dict
            Additional keyword arguments for xr.open_dataset.

        Returns
        -------
        xr.Dataset
            Footprint data as an xarray Dataset.
        """
        ds = xr.open_dataset(Path(file).resolve(), **kwargs)

        if parse_receptor:
            receptor = Receptor.build(
                time=ds.attrs.pop('r_time'),
                longitude=ds.attrs.pop('r_long'),
                latitude=ds.attrs.pop('r_lati'),
                height=ds.attrs.pop('r_zagl'),
            )
            ds.attrs['receptor'] = receptor
            # ds = ds.assign_coords({'receptor': receptor.id})

        return ds

    @staticmethod
    def _integrate_over_time(data: xr.DataArray, start: dt.datetime | None = None,
                             end: dt.datetime | None = None) -> xr.DataArray:
        """
        Integrate footprint dataarray over time.
        """
        return data.sel(time=slice(start, end)).sum('time')

    def integrate_over_time(self, start: dt.datetime | None = None,
                            end: dt.datetime | None = None) -> xr.DataArray:
        """
        Integrate footprint over time.

        Parameters
        ----------
        start : datetime, optional
            Start time of integration. The default is None.
        end : datetime, optional
            End time of integration. The default is None.

        Returns
        -------
        xr.DataArray
            Time-integrated footprint
        """
        return self._integrate_over_time(self.data, start=start, end=end)


class FootprintCollection(UserDict):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

        self.resolutions = simulation.config.resolutions

    def __getitem__(self, key):
        # If the footprint for the given resolution is not loaded, load it
        if key not in self.data:
            path = self.simulation.paths['footprints'].get(key)
            if path and path.exists():
                xres, yres = map(float, key.split('x'))
                self.data[key] = Footprint(
                    simulation_id=self.simulation.id,
                    receptor=self.simulation.receptor,
                    data=Footprint.read_netcdf(path, parse_receptor=False).foot,
                    xmin=self.simulation.config.xmn, xmax=self.simulation.config.xmx,
                    ymin=self.simulation.config.ymn, ymax=self.simulation.config.ymx,
                    xres=xres, yres=yres,
                    projection=self.simulation.config.projection,
                    smooth_factor=self.simulation.config.smooth_factor,
                    time_integrate=self.simulation.config.time_integrate,
                )
            else:
                raise KeyError(f"Footprint for resolution '{key}' not found.")
        return self.data[key]

    def get(self, resolution: str) -> Footprint | None:
        """
        Get footprint for a specific resolution.

        Parameters
        ----------
        resolution : str
            Resolution string (e.g., '1x1').

        Returns
        -------
        Footprint | None
            Footprint object if available, else None.
        """
        try:
            return self[resolution]
        except KeyError:
            return None

    def __repr__(self):
        return f"FootprintCollection(simulation_id={self.simulation.id}, resolutions={self.resolutions})"

class Simulation:

    PATHS = {
        'config': 'config.yaml',
        'control': 'CONTROL',
        'error': 'error.parquet',
        'footprints': '*_foot.nc',
        'log': 'stilt.log',
        'params': 'CONC.CFG',
        'receptors': 'receptors.csv',
        'setup': 'SETUP.CFG',
        'trajectory': 'trajec.parquet',
        'winderr': 'WINDERR',
        'zicontrol': 'ZICONTROL',
        'zierr': 'ZIERR',
    }

    FAILURE_PHRASES = {
        'Insufficient number of meteorological files found': 'MISSING_MET_FILES',
        'meteorological data time interval varies': 'VARYING_MET_INTERVAL',
        'PARTICLE_STILT.DAT does not contain any trajectory data': 'NO_TRAJECTORY_DATA',
        'Fortran runtime error': 'FORTRAN_RUNTIME_ERROR',
    }

    def __init__(self,
                 config: SimulationConfig,
                 ):
        self.config = config

        self.id = self.config.simulation_id
        self.path = self.config.output_wd / 'by-id' / self.id
        self.receptor = self.config.receptor

        # Lazy loading
        self._paths = None
        self._meteorology = None
        self._met_files = None
        self._control = None
        self._setup = None
        self._trajectory = None
        self._error = None
        self._footprints = None

    @property
    def paths(self) -> dict[str, Path | dict[str, Path]]:
        if self._paths is None:
            paths = {}

            paths['config'] = self.path / f"{self.id}_{self.PATHS['config']}"
            paths['control'] = self.path / self.PATHS['control']
            paths['error'] = self.path / f"{self.id}_{self.PATHS['error']}"
            paths['log'] = self.path / self.PATHS['log']
            paths['params'] = self.path / self.PATHS['params']
            paths['receptors'] = self.path / self.PATHS['receptors']
            paths['setup'] = self.path / self.PATHS['setup']
            paths['trajectory'] = self.path / f"{self.id}_{self.PATHS['trajectory']}"
            paths['winderr'] = self.path / self.PATHS['winderr']
            paths['zicontrol'] = self.path / self.PATHS['zicontrol']
            paths['zierr'] = self.path / self.PATHS['zierr']

            # Build footprint paths based on resolutions
            paths['footprints'] = {}
            resolutions = self.config.resolutions
            if resolutions is not None:
                for res in resolutions:
                    paths['footprints'][str(res)] = self.path / f"{self.id}_{res}_foot.nc"

            self._paths = paths

        return self._paths

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """
        Load simulation from a directory containing config.yaml.

        Parameters
        ----------
        path : str | Path
            Path to the simulation directory.

        Returns
        -------
        Self
            Instance of the simulation.
        """
        simulation_dir = Path(path).resolve()
        config_path = simulation_dir / f"{simulation_dir.name}_config.yaml"
        config = SimulationConfig.from_path(config_path)
        return cls(config=config)

    @property
    def is_backward(self) -> bool:
        """
        Check if the simulation is backward in time.
        """
        return self.config.n_hours < 0

    @property
    def time_range(self) -> tuple[dt.datetime, dt.datetime]:
        """
        Get time range of simulation.

        Returns
        -------
        TimeRange
            Time range of simulation.
        """
        r_time = self.receptor.time
        if self.is_backward:
            start = r_time + dt.timedelta(hours=self.config.n_hours)
            stop = r_time
        else:
            start = r_time
            stop = r_time + dt.timedelta(hours=self.config.n_hours + 1)
        return start, stop

    @property
    def status(self) -> str | None:
        """
        Get the status of the simulation.

        Returns
        -------
        str
            Status of the simulation.
        """
        if not self.path.exists():
            return None

        if self.config.run_trajec and not self.paths['trajectory'].exists():
            status = 'FAILURE'
        elif self.config.run_foot and not all(path.exists()
                                              for path in self.paths['footprints'].values()):
            status = 'FAILURE'
        else:
            status = 'SUCCESS'

        if status.lower().startswith('fail'):
            status += f':{Simulation.identify_failure_reason(self.path)}'
        return status

    @property
    def meteorology(self) -> Meteorology:
        if not self._meteorology:
            self._meteorology = Meteorology(path=self.config.met_path,
                                            format=self.config.met_file_format,
                                            tres=self.config.met_file_tres)
        return self._meteorology

    @property
    def met_files(self) -> List[Path]:
        if not self._met_files:
            if self.paths['control'].exists():
                control = self.control
                self._met_files = control.met_files
            else:
                # Get meteorology files from the meteorology object
                self._met_files = self.meteorology.get_files(
                    r_time=self.receptor.time,
                    n_hours=self.config.n_hours
                    )
                if self.config.met_subgrid_enable:
                    # Build subgrid meteorology
                    self._meteorology = self.meteorology.calc_subgrids(
                        files=self._met_files,
                        out_dir=self.config.output_wd / 'met',
                        exe_dir=self.config.stilt_wd / 'exe',
                        projection=self.config.projection,
                        xmin=self.config.xmn,
                        xmax=self.config.xmx,
                        ymin=self.config.ymn,
                        ymax=self.config.ymx,
                        levels=self.config.met_subgrid_levels,
                        buffer=self.config.met_subgrid_buffer
                        )
                    # Get subgrid meteorology files
                    self._met_files = self._meteorology.get_files(
                        t_start=self.receptor.time + pd.Timedelta(hours=self.config.n_hours),
                        n_hours=self.config.n_hours
                    )
                if len(self._met_files) < self.config.n_met_min:
                    raise ValueError(f"Insufficient meteorological files found. "
                                    f"Found: {len(self._met_files)}, "
                                    f"Required: {self.config.n_met_min}")
        return self._met_files

    @property
    def control(self) -> Control:
        if not self._control:
            if self.paths['control'].exists():
                self._control = Control.from_path(self.paths['control'])
            else:
                self._control = Control(receptor=self.receptor,
                                        n_hours=self.config.n_hours,
                                        emisshrs=self.config.emisshrs,
                                        w_option=self.config.w_option,
                                        z_top=self.config.z_top,
                                        met_files=self.met_files,
                                        )
        return self._control

    @property
    def setup(self) -> f90nml.Namelist:
        """Setup namelist."""
        if not self._setup:
            if self.paths['setup'].exists():
                self._setup = f90nml.read(self.paths['setup'])['setup']
            else:
                names = ['capemin', 'cmass', 'conage', 'cpack', 'delt', 'dxf',
                         'dyf', 'dzf', 'efile', 'frhmax', 'frhs', 'frme', 'frmr',
                         'frts', 'frvs', 'hscale', 'ichem', 'idsp', 'initd',
                         'k10m', 'kagl', 'kbls', 'kblt', 'kdef', 'khinp',
                         'khmax', 'kmix0', 'kmixd', 'kmsl', 'kpuff', 'krand',
                         'krnd', 'kspl', 'kwet', 'kzmix', 'maxdim', 'maxpar',
                         'mgmin', 'mhrs', 'nbptyp', 'ncycl', 'ndump', 'ninit',
                         'nstr', 'nturb', 'numpar', 'nver', 'outdt', 'p10f',
                         'pinbc', 'pinpf', 'poutf', 'qcycle', 'rhb', 'rht',
                         'splitf', 'tkerd', 'tkern', 'tlfrac', 'tout', 'tratio',
                         'tvmix', 'varsiwant', 'veght', 'vscale', 'vscaleu',
                         'vscales', 'wbbh', 'wbwf', 'wbwr', 'winderrtf', 'wvert',
                         'zicontroltf']
                namelist = {name: getattr(self, name) for name in names if hasattr(self, name)}
                self._setup = f90nml.Namelist(namelist)
        return self._setup

    def write_xyerr(self, path: str | Path) -> None:
        """
        Write the XY error parameters to a file.
        """
        raise NotImplementedError

    def write_zierr(self, path: str | Path) -> None:
        raise NotImplementedError

    def _load_trajectory(self, path: str | Path) -> Trajectory:
        """Load trajectory from parquet file."""
        trajectory = Trajectory(
            simulation_id=self.id,
            receptor=self.receptor,
            data=Trajectory.read_parquet(
                path=path,
                r_time=self.receptor.time,
                outdt=self.config.outdt,
                ),
            n_hours=self.config.n_hours,
            params=self.config.transport_params(),
            met_files=self.met_files,
        )
        return trajectory

    @property
    def trajectory(self) -> Trajectory | None:
        """STILT particle trajectories."""
        if not self._trajectory:
            path = self.paths['trajectory']
            if path.exists():
                self._trajectory = self._load_trajectory(path)
            else:
                print("Trajectory file not found. Has the simulation been run?")
            return self._trajectory

    @property
    def error(self) -> Trajectory | None:
        """STILT particle error trajectories."""
        if self.has_error and not self._error:
            path = self.paths['error']
            if path.exists():
                self._error = self._load_trajectory(path)
        return self._error

    @property
    def footprints(self) -> FootprintCollection:
        """
        Dictionary of STILT footprints.

        Returns
        -------
        FootprintCollection
            Collection of Footprint objects.
        """
        if self._footprints is None:
            self._footprints = FootprintCollection(simulation=self)
        return self._footprints

    @property
    def footprint(self) -> Footprint | None:
        """
        Load the default footprint from the simulation directory.

        The default footprint is the one with the highest resolution
        if multiple footprints exist, otherwise it is the only footprint.

        Returns
        -------
        Footprint
            Footprint object.
        """
        resolutions = self.config.resolutions
        if resolutions is None:
            return None
        num_foots = len(resolutions)

        def area(r: Resolution) -> float:
            return r.xres * r.yres

        if num_foots == 0:
            return None
        elif num_foots == 1:
            return self.footprints[resolutions[0]]
        else:
            # Find the resolution with the smallest area (xres * yres)
            smallest = min(resolutions, key=area)
            return self.footprints[str(smallest)]

    @cached_property  # TODO i think i need to set a setter
    def log(self) -> str:
        """
        STILT log.

        Returns
        -------
        str
            STILT log contents.
        """
        if self.path:
            log_file = self.path / 'stilt.log'
            if not log_file.exists():
                raise FileNotFoundError(f'Log file not found: {log_file}')
            log = log_file.read_text()
        else:
            log = ''
        return log

    @staticmethod
    def identify_failure_reason(path: str | Path) -> str:
        """
        Identify reason for simulation failure.

        Parameters
        ----------
        path : str | Path
            Path to the STILT simulation directory.

        Returns
        -------
        str
            Reason for simulation failure.
        """
        path = Path(path)
        if not Simulation.is_sim_path(path):
            raise ValueError(f"Path '{path}' is not a valid STILT simulation directory.")

        if not path.glob('*config.json'):
            raise ValueError('Simulation was successful')

        # Check log file for errors
        if not (path / 'stilt.log').exists():
            return 'EMPTY_LOG'
        for phrase, reason in Simulation.FAILURE_PHRASES.items():
            if phrase in (path / 'stilt.log').read_text():
                return reason
        return 'UNKNOWN'

    @staticmethod
    def get_sim_id_from_path(path: str | Path) -> str:
        """
        Extract simulation ID from the path.

        Parameters
        ----------
        path : str | Path
            Path within the STILT output directory.

        Returns
        -------
        str
            Simulation ID.
        """
        path = Path(path).resolve()
        
        # anything beyond by-id/ is considered part of the simulation ID (not including the file name)
        if not 'by-id' in path.parent.parts:
            raise ValueError("Unable to extract simulation ID from path. 'by-id' directory not found in parent path.")
        id_index = path.parts.index('by-id') + 1
        sim_id_parts = path.parts[id_index:]
        if not sim_id_parts:
            raise ValueError("No simulation ID found in path.")
        return os.sep.join(sim_id_parts)

    @staticmethod
    def is_sim_path(path: str | Path) -> bool:
        """
        Check if the path is a valid STILT simulation directory.

        Parameters
        ----------
        path : str | Path
            Path to check.

        Returns
        -------
        bool
            True if the path is a valid STILT simulation directory, False otherwise.
        """
        path = Path(path)
        if not path.is_dir():
            return False
        exe_exists = (path / 'hycs_std').exists()
        is_exe_dir = path.name == 'exe'
        return exe_exists and not is_exe_dir


class SimulationCollection:

    COLUMNS = ['id', 'location_id', 'status',
               'r_time', 'r_long', 'r_lati', 'r_zagl',
               't_start', 't_end',
               'path', 'simulation']

    def __init__(self, sims: list[Simulation] | None = None):
        """
        Initialize SimulationCollection.

        Parameters
        ----------
        sims : list[Simulation]
            List of Simulation objects to add to the collection.
            If None, an empty collection is created.
        """
        # Initialize an empty DataFrame with the required columns
        self._df = pd.DataFrame(columns=self.COLUMNS)

        # Add simulations to the collection if provided
        if sims:
            rows = [self._prepare_simulation_row(sim) for sim in sims]
            self._df = pd.DataFrame(rows, columns=self.COLUMNS)
            self._df.set_index('id', inplace=True)

    @staticmethod
    def _prepare_simulation_row(sim: Simulation) -> dict[str, Any]:
        """
        Prepare a dictionary row for a Simulation object.

        Parameters
        ----------
        sim : Simulation
            Simulation object to prepare.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the Simulation object.
        """
        if isinstance(sim, dict) and 'id' in sim:
            # Assume dictionaries with 'id' key are failed simulations
            return sim

        return {
            'id': sim.id,
            'location_id': sim.receptor.location.id,
            'status': sim.status,
            'r_time': sim.receptor.time,
            'r_long': sim.receptor.location._lons,
            'r_lati': sim.receptor.location._lats,
            'r_zagl': sim.receptor.location._hgts,
            't_start': sim.time_range[0],
            't_end': sim.time_range[1],
            'path': sim.path,
            'simulation': sim,
        }

    @classmethod
    def from_paths(cls, paths: List[Path | str]) -> Self:
        """
        Create SimulationCollection from a list of simulation paths.

        Parameters
        ----------
        paths : list[Path | str]
            List of paths to STILT simulation directories or files.

        Returns
        -------
        SimulationCollection
            Collection of Simulations.
        """
        sims = []
        for path in paths:
            path = Path(path)
            if not Simulation.is_sim_path(path):
                raise ValueError(f"Path '{path}' is not a valid STILT simulation directory.")
            try:
                sim = Simulation.from_path(path)
            except Exception as e:
                failure_reason = Simulation.identify_failure_reason(path)
                sim = {
                    'id': Simulation.get_sim_id_from_path(path=path),
                    'status': f'FAILURE:{failure_reason}',
                    'path': path,
                }
            sims.append(sim)
        return cls(sims=sims)

    @property
    def df(self) -> pd.DataFrame:
        """
        Get the underlying DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing simulation metadata.
        """
        return self._df

    def __getitem__(self, key: str) -> Simulation:
        """
        Get a Simulation object by its ID.

        Parameters
        ----------
        key : str
            Simulation ID.

        Returns
        -------
        Simulation
            Simulation object corresponding to the given ID.
        """
        if key not in self._df.index:
            raise KeyError(f"Simulation with ID '{key}' not found in the collection.")
        return self._df.loc[key, 'simulation']

    def __setitem__(self, key: str, value: Simulation) -> None:
        """
        Set a Simulation object by its ID.

        Parameters
        ----------
        key : str
            Simulation ID.
        value : Simulation
            Simulation object to set.
        """
        if not isinstance(value, Simulation):
            raise TypeError(f"Value must be a Simulation object, got {type(value)}.")
        row = self._prepare_simulation_row(value)
        if key in self._df.index:
            raise KeyError(f"Simulation with ID '{key}' already exists in the collection.")
        self._df.loc[key] = row

    def __contains__(self, key: str) -> bool:
        """
        Check if a Simulation ID exists in the collection.

        Parameters
        ----------
        key : str
            Simulation ID.

        Returns
        -------
        bool
            True if the Simulation ID exists, False otherwise.
        """
        return key in self._df.index

    def __iter__(self):
        """
        Iterate over Simulations in the collection.

        Returns
        -------
        Iterator[Simulation]
            Iterator over Simulation objects.
        """
        return iter(self._df.simulation)

    def __len__(self) -> int:
        """
        Get the number of Simulations in the collection.

        Returns
        -------
        int
            Number of Simulations in the collection.
        """
        return len(self._df)

    def __repr__(self) -> str:
        return repr(self._df)

    def load_trajectories(self) -> None:
        """
        Load trajectories for all simulations in the collection.

        Returns
        -------
        None
            The trajectories are loaded into the 'simulation' column of the DataFrame.
        """
        self._df['trajectory'] = self._df['simulation'].apply(
            lambda sim: sim.trajectory if isinstance(sim, Simulation) else None
        )
        return None

    def load_footprints(self, resolutions: List[str] | None = None) -> None:
        """
        Load footprints for simulations in the collection.

        Parameters
        ----------
        resolutions : list[str], optional
            Resolutions to filter footprints. If None, all footprints are loaded.

        Returns
        -------
        None
            The footprints are loaded into the 'footprints' column of the DataFrame.
        """
        if isinstance(resolutions, str):
            resolutions = [resolutions]

        sims = self._df['simulation']

        # Collect all unique resolutions across simulations
        if resolutions is None:
            resolutions = set()
            for sim in sims:
                if isinstance(sim, Simulation):
                    sim_resolutions = sim.config.resolutions
                    if sim_resolutions is not None:
                        resolutions.update(map(str, sim.config.resolutions))

        if not resolutions:
            return None

        # Populate the footprint columns
        for idx, sim in sims.items():
            if isinstance(sim, Simulation):
                for res in resolutions:
                    col_name = f"footprint_{res}"
                    footprint = sim.footprints.get(res)
                    if footprint is not None:
                        if col_name not in self._df.columns:
                            # Add columns for each resolution
                            self._df[col_name] = None
                        self._df.at[idx, col_name] = footprint

        # If only one resolution exists, rename the column to "footprint"
        if len(resolutions) == 1:
            single_res_col = f"footprint_{resolutions[0]}"
            self._df.rename(columns={single_res_col: "footprint"}, inplace=True)

        return None

    @classmethod
    def merge(cls, collections: Self | list[Self]) -> Self:
        """
        Merge multiple SimulationCollections into one.

        Parameters
        ----------
        collections : list[SimulationCollection]
            List of SimulationCollections to merge.

        Returns
        -------
        SimulationCollection
            Merged SimulationCollection.
        """
        if not isinstance(collections, list):
            collections = [collections]

        merged_sims = pd.concat([collection._df for collection in collections])
        if merged_sims.index.has_duplicates:
            raise ValueError("Merged simulations contain duplicate IDs. Ensure unique simulation IDs across collections.")
        collection = cls()
        collection._df = merged_sims
        return collection

    def get_missing(self, in_receptors: str | Path | pd.DataFrame, include_failed: bool = False) -> pd.DataFrame:
        """
        Find simulations in csv that are missing from simulation collection.

        Parameters
        ----------
        in_receptors : str | Path | pd.DataFrame
            Path to csv file containing receptor configuration or a DataFrame directly.
        include_failed : bool, optional
            Include failed simulations in output. The default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame of missing simulations.
        """
        # Use receptor info to match simulations
        cols = ['time', 'long', 'lati', 'zagl']

        # Load dataframes
        if isinstance(in_receptors, (str, Path)):
            in_df = pd.read_csv(in_receptors)
        elif isinstance(in_receptors, pd.DataFrame):
            in_df = in_receptors.copy()
        else:
            raise TypeError("in_receptors must be a path to a csv file or a pandas DataFrame.")
        in_df['time'] = pd.to_datetime(in_df['time'])

        sim_df = self.df.copy()
        if include_failed:
            # Drop failed simulations from the sim df so that when doing an outer join with the input receptors,
            # they appear in the input receptors but not in the simulation collection
            sim_df = sim_df[sim_df['status'] == 'SUCCESS']
        r_cols = {f'r_{col}': col for col in cols}
        sim_df = sim_df[list(r_cols.keys())].rename(columns=r_cols).reset_index(drop=True)

        # Merge dataframes on receptor info
        merged = pd.merge(in_df, sim_df, on=cols, how='outer', indicator=True)
        missing = merged[merged['_merge'] == 'left_only']
        return missing.drop(columns='_merge')

    def plot_availability(self, ax: plt.Axes | None = None, **kwargs) -> plt.Axes:
        """
        Plot availability of simulations over time.

        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib Axes to plot on. If None, a new figure and axes are created.
        **kwargs : dict
            Additional keyword arguments for the scatter plot.

        Returns
        -------
        plt.Axes
            Matplotlib Axes with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        df = self.df.copy()
        df['status'] = df['status'].fillna('MISSING')

        # Iterate through each row of the DataFrame to plot the rectangles
        for index, row in df.iterrows():
            # Calculate the duration of the event
            duration = row['t_end'] - row['t_start']
            
            # Plot a horizontal bar (gantt bar)
            ax.barh(
                y=row['location_id'],       # Y-axis is the location
                width=duration,             # Width is the time duration
                left=row['t_start'],        # Start position on the X-axis
                height=0.6,                 # Height of the bar
                align='center',
                color='green' if row['status'] == 'SUCCESS' else 'red',
                edgecolor='black',
                alpha=0.6,
                **kwargs
            )
        
        fig.autofmt_xdate()
        
        ax.set(
            title='Simulation Availability',
            xlabel='Time',
            ylabel='Location ID'
        )

        return ax


class Model:
    def __init__(self,
                 project: str | Path,
                 **kwargs):

        # Extract project name and working directory
        project = Path(project)
        self.project = project.name

        if project.exists():
            # Build model config from existing project
            config = ModelConfig.from_path(project / 'config.yaml')

        else:  # Create a new project
            # Build model config
            config = kwargs.pop('config', None)
            if config is None:
                config = self.initialize(project, **kwargs)
            elif not isinstance(config, ModelConfig):
                raise TypeError("config must be a ModelConfig instance.")

        self.config = config

        self._simulations = None  # Lazy loading

    @staticmethod
    def initialize(project: Path, **kwargs) -> ModelConfig:
        # Determine working directory
        wd = project.parent
        if wd == Path('.'):
            wd = Path.cwd()
        stilt_wd = wd / project
        del kwargs['stilt_wd']

        # Call stilt_init
        repo = kwargs.pop('repo', None)
        branch = kwargs.pop('branch', None)
        stilt_init(project=project, branch=branch, repo=repo)

        # Build config overriding default values with kwargs
        config = ModelConfig(stilt_wd=stilt_wd, **kwargs)

        return config

    @property
    def simulations(self) -> SimulationCollection | None:
        """
        Load all simulations from the output working directory.
        """
        if self._simulations is None:
            output_wd = self.config.output_wd
            if output_wd.exists():
                paths = list(self.config.output_wd.glob('by-id/*'))
                self._simulations = SimulationCollection.from_paths(paths)
        return self._simulations

    def run(self):
        # Run the STILT model
        # TODO Dont have time to implement python calculations
        self._run_rscript()

    def _run_rscript(self):
        # In the meantime, we can call the R execultable
        raise NotImplementedError
