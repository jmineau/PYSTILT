
from abc import ABC, abstractmethod
from collections import UserDict
import datetime as dt
from functools import cached_property
import os
from pathlib import Path
import re

import f90nml
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel
from typing_extensions import Self
import xarray as xr

from stilt.config import Resolution, SimulationConfig
from stilt.meteorology import Meteorology
from stilt.receptors import Receptor


class Control(BaseModel):
    """HYSPLIT control parameters."""
    receptor: Receptor
    emisshrs: float
    n_hours: int
    w_option: int
    z_top: float
    met_files: list[Path]

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
        config = SimulationConfig.from_path(path.parent / f'{path.parent.name}_config.yaml')
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
        config = SimulationConfig.from_path(path.parent / f'{path.parent.name}_config.yaml')

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
    def met_files(self) -> list[Path]:
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
