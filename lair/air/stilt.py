"""
Stochastic Time-Inverted Lagrangian Transport (STILT) Model.

Inspired by https://github.com/uataq/air-tracker-stiltctl
"""

from collections import UserDict
import datetime as dt
import os
import subprocess
from functools import cached_property
import json
from pathlib import Path
import re
from typing import Any, Literal
from typing_extensions import \
    Self  # requires python 3.11 to import from typing

import pandas as pd
import pyarrow.parquet as pq
import pyproj
import sparse
import xarray as xr

from lair.utils.clock import TimeRange
from lair.utils.geo import BaseGrid, CRS, clip, write_rio_crs

# TODO:
# - Support multiple receptors
#   - ColumnReceptor
# - Run stilt_cli.r from python
#   - stilt_cli only runs one simulation at a time via simulation step
#   - want to be able to access slurm, but seems hard to manipulate run_stilt.r
# - Trajectory gif
# - Footprint plot


def stilt_init(project: str | Path, branch='jmineau',
               repo: str = 'https://github.com/jmineau/stilt'):
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


def fix_sim_links(old_out_dir, new_out_dir):
    '''
    Fix sim symlinks if stilt wd was changed

    Parameters
    ----------
    old_out_dir : str
        old out directory where symlinks currently point to and shouldnt.
    new_out_dir : str
        new out directory where symlinks should point to.

    Returns
    -------
    None.

    '''

    for subdir in ['footprints', 'particles']:
        for file in os.listdir(os.path.join(new_out_dir, subdir)):
            filepath = os.path.join(new_out_dir, subdir, file)

            # Check if file is link
            if os.path.islink(filepath):
                old_link = os.readlink(filepath)

                # Check if old_link is to file in old_out_dir
                if old_link.startswith(old_out_dir):
                    # Get sim_id of simulation
                    sim_id = os.path.basename(os.path.dirname(old_link))

                    # Remove old_link
                    os.remove(filepath)

                    # TODO this should be a relative path
                    # Create new link to new_out_dir
                    new_link = os.path.join(new_out_dir, 'by-id', sim_id, file)
                    os.symlink(new_link, filepath)


class _UniqueReceptorMeta(type):
    _receptors = {}

    def __call__(cls, longitude, latitude, height, *args, **kwargs):
        key = (longitude, latitude, height)
        if key in cls._receptors:
            return cls._receptors[key]
        receptor = super().__call__(*key, *args, **kwargs)
        cls._receptors[key] = receptor
        return receptor

class Receptor(metaclass=_UniqueReceptorMeta):

    stilt_mapper = {
        'lati': 'latitude',
        'long': 'longitude',
        'zagl': 'height',
    }

    def __init__(self, longitude: Any, latitude: Any, height: Any,
                 name: str | None = None):
        self.longitude = float(longitude)
        self.latitude = float(latitude)
        self.height = float(height)
        self.name = name

    @classmethod
    def from_simulation_id(cls, sim_id: str) -> Self:
        """
        Extract receptor metadata from simulation ID.
        Requires simulation ID to be in the format '%Y%m%d%H%M_long_lati_zagl'.

        Parameters
        ----------
        sim_id : str
            Simulation ID.

        Returns
        -------
        Receptor
            Receptor object.
        """
        parts = sim_id.split('_')
        assert len(parts) == 4, 'Invalid simulation ID format'
        run_time, long, lati, zagl = parts
        return cls(longitude=long, latitude=lati, height=zagl)

    @property
    def id(self) -> str:
        return f'{self.longitude}_{self.latitude}_{self.height}'

    def __repr__(self) -> str:
        return f'Receptor(longitude={self.longitude}, latitude={self.latitude}, height={self.height})'

    def __str__(self) -> str:
        x = self.name if self.name else self.id
        return f'Receptor({x})'

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class Trajectory:

    def __init__(self, simulation_id: str,
                 run_time: dt.datetime, params: dict,
                 receptor: Receptor, particles: pd.DataFrame,
                 particle_error: pd.DataFrame | None = None,
                 error_params: dict | None = None):
        self.simulation_id = simulation_id
        self.run_time = run_time
        self.params = params
        self.receptor = receptor
        self.particles = particles
        self.particle_error = particle_error
        self.error_params = error_params

    @classmethod
    def from_parquet(cls, file: str | Path) -> Self:
        # Extract simulation ID from file name
        simulation_id = Path(file).stem.replace('_traj', '')

        # Read particle parquet data
        particles = pd.read_parquet(file)
        meta = pq.read_metadata(file).metadata

        # Check for particle error data
        err_cols = particles.columns.str.endswith('_err')
        if any(err_cols):
            # Split error columns from main data
            particle_error = particles.loc[:, err_cols]
            particles = particles.loc[:, ~err_cols]

            # Extract error parameters
            err_params = Trajectory._extract_meta(meta, 'err_param')
        else:
            particle_error = None
            err_params = None

        # Build receptor object from metadata
        receptor_meta = Trajectory._extract_meta(meta, 'receptor')
        receptor = Receptor(
            latitude = receptor_meta['lati'],
            longitude = receptor_meta['long'],
            height = receptor_meta['zagl']
        )

        # Assign run time as attribute
        run_time = dt.datetime.fromisoformat(receptor_meta['run_time'])

        # Extract trajectory parameters
        params = Trajectory._extract_meta(meta, 'param')

        return cls(simulation_id, run_time, params, receptor, particles,
                   particle_error, err_params)

    @staticmethod
    def _extract_meta(meta: dict, group: str
                      ) -> dict[str, Any]:
        def parse_val(v: bytes) -> Any:
            val = v.decode()
            try:
                return int(val)
            except ValueError:
                try:
                    return float(val)
                except ValueError:
                    val = {
                        '.TRUE.': True,
                        '.FALSE.': False,
                        'NA': None
                    }[val]
                    return val

        data = {}
        for k, v in meta.items():
            key = k.decode()
            if key.startswith(group):
                data[key.split(':')[1]] = parse_val(v)

        return data

    def to_xarray(self, error: bool = False) -> xr.Dataset:
        """
        Convert trajectory data to xarray.Dataset
        with time and particle index as coordinates.

        Parameters
        ----------
        error : bool
            Output particle error data if available.
        """
        def pivot(df: pd.DataFrame) -> xr.Dataset:
            ds = xr.Dataset.from_dataframe(
                df.set_index(['indx', 'time'])
            )
            return ds

        if not error:
            ds = pivot(self.particles)
            ds.attrs = self.params
        elif self.particle_error is not None:
            ds = pivot(self.particle_error)
            ds.attrs = self.error_params or {}
        else:
            raise ValueError('No error data available')

        # TODO add receptor as coords?

        return ds


class Footprint(BaseGrid):
    """
    Footprint object containing STILT simulation footprint data.
    """
    def __init__(self, simulation_id: str,
                 run_time: dt.datetime,
                 receptor: Receptor,
                 data: xr.DataArray,
                 crs: Any = 4326,
                 time_created: str | None = None):
        self.simulation_id = simulation_id
        self.run_time = run_time
        self.receptor = receptor
        self.data = data
        self.crs = CRS(crs)
        self.time_created = time_created

    def __repr__(self):
        return f'Footprint({self.simulation_id})'

    @staticmethod
    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        """
        Preprocess footprint data to add run_time and receptor dimensions.

        Parameters
        ----------
        ds : xr.Dataset
            Footprint data.

        Returns
        -------
        xr.Dataset
            Preprocessed footprint data.
        """
        # Extract metadata
        run_time = dt.datetime.fromisoformat(ds.attrs.pop('r_run_time'))
        receptor = Receptor(longitude=ds.attrs.pop('r_long'),
                            latitude=ds.attrs.pop('r_lati'),
                            height=ds.attrs.pop('r_zagl'))

        # Add run_time and receptor dimensions
        ds = ds.expand_dims(receptor=[receptor.id],
                            run_time=[run_time])

        return ds

    @classmethod
    def from_netcdf(cls, file: str | Path, chunks: str | dict | None = None) -> Self:
        """
        Create Footprint object from netCDF file.

        Parameters
        ----------
        file : str | Path
            Path to netCDF file.
        chunks : str | dict, optional
            Chunks for dask array. The default is None.

        Returns
        -------
        Footprint
            Footprint object.
        """
        simulation_id = Path(file).stem.replace('_foot', '')
        ds = xr.open_dataset(file, chunks=chunks)
        ds = cls._preprocess(ds)

        time_created = ds.attrs.pop('time_created')
        run_time = pd.Timestamp(ds.run_time.item()).to_pydatetime()
        receptor = Receptor(*ds.receptor.item().split('_'))
        crs = pyproj.CRS.from_proj4(ds.attrs['crs'])
        ds = write_rio_crs(ds, crs)

        return cls(simulation_id, run_time=run_time, receptor=receptor,
                   data=ds.foot, crs=crs, time_created=time_created)

    @property
    def sparse(self) -> xr.DataArray:
        """
        Convert footprint to sparse representation.

        Returns
        -------
        xr.DataArray
            Sparse footprint.
        """
        sparse_arr = sparse.COO.from_numpy(self.data.values)
        return xr.DataArray(sparse_arr, name='foot', coords=self.data.coords,
                            dims=self.data.dims, attrs=self.data.attrs)

    @property
    def time_range(self) -> TimeRange:
        """
        Get time range of footprint data.

        Returns
        -------
        TimeRange
            Time range of footprint data.
        """
        times = sorted(self.data.time.values)
        return TimeRange(start=times[0], stop=pd.Timestamp(times[-1]) + pd.Timedelta(hours=1))

    def time_integrate(self, start: dt.datetime | None = None,
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
        return self.data.sel(time=slice(start, end)).sum('time')

    def clip_to_grid(self, grid, buffer: float = 0.1):
        """
        Clip footprint to grid extent with buffer.

        Parameters
        ----------
        grid : BaseGrid
            Grid to clip footprint to. Must have dims 'lon' and 'lat'.
        buffer : float, optional
            Buffer around grid extent as a fraction of grid size. The default is 0.1.

        Returns
        -------
        Footprint
            Clipped footprint.
        """
        # Determine bounds to clip footprints from out_grid
        # Clip footprints to slightly larger than the output grid
        xmin, xmax = grid['lon'].min(), grid['lon'].max()
        ymin, ymax = grid['lat'].min(), grid['lat'].max()

        # Calculate buffer
        xbuffer = (xmax - xmin) * buffer
        ybuffer = (ymax - ymin) * buffer

        # Build clip bbox
        bbox = [xmin - xbuffer, ymin - ybuffer,
                xmax + xbuffer, ymax + ybuffer]

        clipped = clip(self.data, bbox=bbox, crs=self.crs)

        return Footprint(simulation_id=self.simulation_id, run_time=self.run_time, receptor=self.receptor,
                         data=clipped, crs=self.crs, time_created=self.time_created)


class Simulation:

    FAILURE_PHRASES = {
        'Insufficient number of meteorological files found': 'MISSING_MET_FILES',
        'meteorological data time interval varies': 'VARYING_MET_INTERVAL',
        'PARTICLE_STILT.DAT does not contain any trajectory data': 'NO_TRAJECTORY_DATA',
        'Fortran runtime error': 'FORTRAN_RUNTIME_ERROR',
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        assert self.path.exists(), f'Simulation not found: {self.path}'
        self._path = self.path.resolve()

        # Extract simulation ID from path
        by_id_dir = Simulation.find_by_id_directory(self._path)
        self.id = str(self._path.relative_to(by_id_dir))

        # Paths to output files
        self.log_path = self.path / 'stilt.log'
        self.config_path = self.path / f'{self.id}_config.json'
        self.traj_path = self.path / f'{self.id}_traj.parquet'
        self.foot_path = self.path / f'{self.id}_foot.nc'

        # Get run time and receptor from simulation ID (requires default sim_id format)
        self.run_time = dt.datetime.strptime(self.id.split('_')[0], '%Y%m%d%H%M')
        self.receptor = Receptor.from_simulation_id(self.id)

    def __repr__(self) -> str:
        return f'Simulation({self.id})'

    @staticmethod
    def find_by_id_directory(path: Path) -> Path:
        for parent in path.resolve().parents:
            if parent.name == 'by-id':
                return parent
        raise FileNotFoundError('by-id directory not found in the path hierarchy')

    @cached_property
    def control(self) -> dict[str, Any]:
        """
        Parse CONTROL file for simulation metadata.

        Returns
        -------
        dict
            Dictionary of simulation metadata.
        """
        file = self.path / 'CONTROL'

        if not file.exists():
            raise FileNotFoundError('CONTROL file not found. Has the simulation been ran?')

        with file.open() as f:
            lines = f.readlines()

        control = {}

        control['run_time'] = dt.datetime.strptime(lines[0].strip(), '%y %m %d %H %M')

        control['receptors'] = []
        control['n_receptors'] = int(lines[1].strip())
        cursor = 2
        for i in range(cursor, cursor + control['n_receptors']):
            lat, lon, zagl = map(float, lines[i].strip().split())
            control['receptors'].append(Receptor(latitude=lat, longitude=lon, height=zagl))

        cursor += control['n_receptors']
        control['n_hours'] = int(lines[cursor].strip())
        control['w_option'] = int(lines[cursor + 1].strip())
        control['z_top'] = float(lines[cursor + 2].strip())

        control['met_files'] = []
        control['n_met_files'] = int(lines[cursor + 3].strip())
        cursor += 4
        for i in range(control['n_met_files']):
            dir_index = cursor + (i * 2)
            file_index = dir_index + 1
            met_file = Path(lines[dir_index].strip()) / lines[file_index].strip()
            control['met_files'].append(met_file)

        cursor += 3 + 2 * control['n_met_files']
        control['emisshrs'] = float(lines[cursor].strip())

        return control

    @property
    def is_backward(self) -> bool:
        return self.control['n_hours'] < 0

    @property
    def time_range(self) -> TimeRange:
        """
        Get time range of simulation.

        Returns
        -------
        TimeRange
            Time range of simulation.
        """
        if self.is_backward:
            start = self.run_time + dt.timedelta(hours=self.control['n_hours'])
            stop = self.run_time
        else:
            start = self.run_time
            stop = self.run_time + dt.timedelta(hours=self.control['n_hours'] + 1)
        return TimeRange(start=start, stop=stop)

    @cached_property
    def log(self) -> str:
        """
        Read STILT log file.

        Returns
        -------
        str
            STILT log file contents.
        """
        if not self.log_path.exists():
            raise FileNotFoundError(f'Log file not found: {self.log_path}')
        return self.log_path.read_text()

    @property
    def was_successful(self) -> bool:
        """
        Check if simulation was successful by checking for the config json.

        Returns
        -------
        bool
            True if simulation was successful, False otherwise.
        """
        # TODO: is this the best way to do this?
        # Other ideas include checking the stilt.log for keywords like 'STOP' or 'ERROR'
        return self.config_path.exists()

    def identify_failure_reason(self):
        """
        Identify reason for simulation failure.

        Returns
        -------
        str
            Reason for simulation failure.
        """
        if self.was_successful:
            raise ValueError('Simulation was successful')

        # Check log file for errors
        if self.log == '':
            return 'EMPTY_LOG'
        for phrase, reason in self.FAILURE_PHRASES.items():
            if phrase in self.log:
                return reason
        return 'UNKNOWN'

    @cached_property
    def config(self) -> dict[str, Any]:
        if not self.was_successful:
            raise FileNotFoundError(f'Config file not found: {self.config_path}')
        with self.config_path.open() as config_file:
            return json.load(config_file)

    @property
    def has_trajectory(self) -> bool:
        return self.traj_path.exists()

    @cached_property
    def trajectory(self) -> Trajectory:
        if not self.has_trajectory:
            raise FileNotFoundError(f'Trajectory not found: {self.traj_path}')
        return Trajectory.from_parquet(self.traj_path)

    @property
    def has_footprint(self) -> bool:
        return self.foot_path.exists()

    @cached_property
    def footprint(self) -> Footprint:
        if not self.has_footprint:
            raise FileNotFoundError(f'Footprint not found: {self.foot_path}')
        foot = Footprint.from_netcdf(self.foot_path)
        assert foot.run_time == self.run_time, f'Footprint run time {foot.run_time} does not match simulation run time {self.run_time}'
        assert foot.receptor == self.receptor, f'Footprint receptor {foot.receptor} does not match simulation receptor {self.receptor}'
        return foot


class SimulationCollection(UserDict):
    """
    Collection of Simulation objects.
    Inherits from dict with additional categorization of successful and failed simulations.

    Attributes
    ----------
    successful : dict[simulation.id, Simulation]
        Dictionary of successful simulations.
    failed : dict[simulation.id, Simulation]
        Dictionary of failed simulations

    Methods
    -------
    from_output_wd(output_wd)
        Create a SimulationCollection from an output directory.
    merge(collections)
        Merge multiple SimulationCollections into one.
    get_missing(in_csv, include_failed)
        Find simulations in csv that are missing from output directory.
    to_dataframe(subset)
        Convert simulation metadata to pandas DataFrame.
    to_csv(path, subset)
        Save simulation metadata to csv.
    """

    def __init__(self, sims: list[Simulation] | dict[str, Simulation] | None = None):
        """
        Initialize SimulationCollection.

        Parameters
        ----------
        sims : list[Simulation] | dict[str, Simulation], optional
            List or dictionary of Simulation objects. The default is None.
            If a dict, keys must match simulation IDs.
        """
        super().__init__()  # initialize dict

        # Define additional dictionaries for successful and failed simulations
        self.failed = {}
        self.successful = {}

        # Add simulations to collection
        if sims:
            # When subclassing dict, self.update would not call __setitem__
            # If we subclass UserDict, self.update would call __setitem__
            # However, self.data.update does not call __setitem__ for UserDict
            # Therefore we don't want to set on self.data directly, rather just self
            if isinstance(sims, dict):
                self.update(sims)
            else:
                # Loop through simulations and add to collection
                for sim in sims:
                    self[sim.id] = sim

    def __repr__(self) -> str:
        return f'SimulationCollection({len(self)} simulations)'

    def __setitem__(self, key: str, sim: Simulation) -> None:
        """
        Set a simulation in the collection and categorize it.

        Parameters
        ----------
        key : str
            Simulation ID.
        sim : Simulation
            Simulation object.
        """
        assert key == sim.id, f'Key {key} does not match simulation ID {sim.id}'
        assert isinstance(sim, Simulation), f'Value must be a Simulation object'
        super().__setitem__(key, sim)
        if sim.was_successful:
            self.successful[key] = sim
        else:
            self.failed[key] = sim

    @classmethod
    def from_output_wd(cls, output_wd: str | Path) -> Self:
        """
        Create a SimulationCollection from an output directory.

        Parameters
        ----------
        output_wd : str | Path
            Path to output directory.

        Returns
        -------
        SimulationCollection
            SimulationCollection object.
        """
        sims = {}
        by_id_dir = Path(output_wd) / 'by-id'
        for root, dirs, files in os.walk(by_id_dir, followlinks=True):
            # Identify simulations using 'stilt.log' files
            if 'stilt.log' in files:
                sim = Simulation(root)
                sims[sim.id] = sim
        return cls(sims)

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
        merged_sims = {}
        if isinstance(collections, cls):
            collections = [collections]
        for collection in collections:
            merged_sims.update(collection)
        return cls(merged_sims)

    def get_missing(self, in_csv: str | Path, include_failed: bool = False) -> pd.DataFrame:
        """
        Find simulations in csv that are missing from output directory.

        Parameters
        ----------
        in_csv : str | Path
            Path to csv file containing simulation metadata.
        include_failed : bool, optional
            Include failed simulations in output. The default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame of missing simulations.
        """
        # Load dataframes
        in_df = pd.read_csv(in_csv)
        sim_df = self.to_dataframe(subset='successful' if include_failed else None)

        # Parse run_time as datetime
        in_df['run_time'] = pd.to_datetime(in_df['run_time'])
        sim_df['run_time'] = pd.to_datetime(sim_df['run_time'])

        # Use run_time & receptor info to match simulations
        cols = ['run_time', 'long', 'lati', 'zagl']

        # Merge dataframes on run_time & receptor info
        merged = pd.merge(in_df, sim_df[cols], on=cols, how='outer', indicator=True)
        missing = merged[merged['_merge'] == 'left_only']
        return missing.drop(columns='_merge')

    def to_dataframe(self, subset: Literal['successful', 'failed'] | None = None) -> pd.DataFrame:
        """
        Convert simulation metadata to pandas DataFrame.

        Parameters
        ----------
        subset : str, optional
            Subset of simulations to include. The default is None.

        Returns
        -------
        pd.DataFrame
            DataFrame of simulation metadata.
        """
        if subset is None:
            sims = self
        else:
            sims = {'successful': self.successful, 'failed': self.failed}[subset]

        data = []
        for sim in sims.values():
            data.append({
                'sim_id': sim.id,
                'run_time': sim.run_time,
                'long': sim.receptor.longitude,
                'lati': sim.receptor.latitude,
                'zagl': sim.receptor.height
            })
        return pd.DataFrame(data).set_index('sim_id').sort_values('run_time')

    def to_csv(self, path: str | Path, subset: Literal['successful', 'failed'] | None = None) -> None:
        """
        Save simulation metadata to csv.
        Contains columns for 'run_time', 'long', 'lati', and 'zagl'.

        Returns
        -------
        None
        """
        df = self.to_dataframe(subset=subset)
        df.to_csv(path, index=False)
        return None


class RunScript(UserDict):
    """
    STILT run script object containing configuration information.
    Inherits from dict with additional grouping of configuration options.

    Parameters
    ----------
    path : str | Path
        Path to run script.

    Attributes
    ----------
    path : Path
        Path to run script.
    model_options : dict
        Model configuration options.
    parallel_options : dict
        Parallelization options.
    footprint_options : dict
        Footprint calculation options.
    meteorological_options : dict
        Meteorological data options.
    transport_options : dict
        Transport & dispersion options.
    error_options : dict
        Transport error options.
    """
    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        assert self.path.exists(), f'Run script not found: {self.path}'
        self._parse()

        # Build stilt and output working directories from file.path calls
        self._stilt_wd = self.data.pop('stilt_wd')
        self._output_wd = self.data.pop('output_wd')
        stilt_wd = self._parse_file_path(self._stilt_wd)
        output_wd = self._parse_file_path(self._output_wd)
        stilt_wd = stilt_wd.replace('project', self.data['project'])
        output_wd = output_wd.replace('stilt_wd', stilt_wd)
        self.stilt_wd = Path(stilt_wd)
        self.output_wd = Path(output_wd)

    def __repr__(self) -> str:
        return f'RunScript({self.path})'

    def _parse(self) -> None:
        """
        Parse the run script for relevant configuration information.

        Returns
        -------
        None
        """
        pattern = re.compile(r"(\w+\.?\w+)\s*<-\s*([^#]*)")
        multiline_var = None
        multiline_value = []

        with self.path.open() as f:
            for line in f:
                if line.startswith('#'):
                    continue

                if multiline_var:
                    multiline_value.append(line.strip())
                    if line.count("(") < line.count(")"):
                        self.data[multiline_var] = self._parse_value(" ".join(multiline_value))
                        multiline_var = None
                        multiline_value = []
                    continue
                else:
                    if line.startswith(' '):
                        continue

                match = pattern.search(line)
                if match:
                    key, value = match.groups()
                    value = value.strip()

                    # Determine if a variable is multiline by checking if there are more ( than )
                    if value.count("(") > value.count(")"):
                        multiline_var = key
                        multiline_value.append(value)
                    else:
                        self.data[key] = self._parse_value(value)
        return None

    @staticmethod
    def _parse_value(value: str) -> Any:
        """
        Parse a string value into its appropriate type.

        Parameters
        ----------
        value : str
            String value to parse.

        Returns
        -------
        Any
            Parsed value.
        """
        value = value.strip()
        if value in ['NA', 'NULL']:
            return None
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        try:
            return float(value)
        except ValueError:
            if value.lower() in ['true', 't']:
                return True
            elif value.lower() in ['false', 'f']:
                return False
        return re.sub(r'\s+', ' ', value)

    @staticmethod
    def _parse_file_path(value: str) -> str:
        """
        Parse a call to R's file.path() function.
        
        Parameters
        ----------
        value : str
            String value to parse.

        Returns
        -------
        str
            Parsed file path.
        """
        if 'file.path' not in value:
            # If not a file.path call, return value
            return value
        # Extract file path from file.path() call
        path = re.search(r'file.path\((.*)\)', value).group(1)
        # Split the path on commas and strip whitespace and quotes
        parts = [part.strip().strip('"').strip("'") for part in path.split(',')]
        # Join the parts into a single path
        return os.path.join(*parts)

    @cached_property
    def model_options(self) -> dict[str, Any]:
        return {
            'n_hours': self.data.get('n_hours'),
            'numpar': self.data.get('numpar'),
            'reset_output_wd': self.data.get('reset_output_wd'),
            'rm_dat': self.data.get('rm_dat'),
            'run_foot': self.data.get('run_foot'),
            'run_trajec': self.data.get('run_trajec'),
            'simulation_id': self.data.get('simulation_id'),
            'timeout': self.data.get('timeout'),
            'varsiwant': self.data.get('varsiwant'),
            'write_trajec': self.data.get('write_trajec')
        }

    @cached_property
    def parallel_options(self) -> dict[str, Any]:
        return {
            'n_cores': self.data.get('n_cores'),
            'n_nodes': self.data.get('n_nodes'),
            'processes_per_node': self.data.get('processes_per_node'),
            'slurm': self.data.get('slurm'),
            'slurm_options': self.data.get('slurm_options')
        }

    @cached_property
    def footprint_options(self) -> dict[str, Any]:
        return {
            'hnf_plume': self.data.get('hnf_plume'),
            'projection': self.data.get('projection'),
            'smooth_factor': self.data.get('smooth_factor'),
            'time_integrate': self.data.get('time_integrate'),
            'xmn': self.data.get('xmn'),
            'xmx': self.data.get('xmx'),
            'ymn': self.data.get('ymn'),
            'ymx': self.data.get('ymx'),
            'xres': self.data.get('xres'),
            'yres': self.data.get('yres')
        }

    @cached_property
    def meteorological_options(self) -> dict[str, Any]:
        return {
            'met_path': self.data.get('met_path'),
            'met_file_format': self.data.get('met_file_format'),
            'n_hours_per_met_file': self.data.get('n_hours_per_met_file'),
            'met_subgrid_buffer': self.data.get('met_subgrid_buffer'),
            'met_subgrid_enable': self.data.get('met_subgrid_enable'),
            'met_subgrid_levels': self.data.get('met_subgrid_levels'),
            'n_met_min': self.data.get('n_met_min')
        }

    @cached_property
    def transport_options(self) -> dict[str, Any]:
        return {
            'capemin': self.data.get('capemin'),
            'cmass': self.data.get('cmass'),
            'conage': self.data.get('conage'),
            'cpack': self.data.get('cpack'),
            'delt': self.data.get('delt'),
            'dxf': self.data.get('dxf'),
            'dyf': self.data.get('dyf'),
            'dzf': self.data.get('dzf'),
            'efile': self.data.get('efile'),
            'emisshrs': self.data.get('emisshrs'),
            'frhmax': self.data.get('frhmax'),
            'frhs': self.data.get('frhs'),
            'frme': self.data.get('frme'),
            'frmr': self.data.get('frmr'),
            'frts': self.data.get('frts'),
            'frvs': self.data.get('frvs'),
            'hscale': self.data.get('hscale'),
            'ichem': self.data.get('ichem'),
            'idsp': self.data.get('idsp'),
            'initd': self.data.get('initd'),
            'k10m': self.data.get('k10m'),
            'kagl': self.data.get('kagl'),
            'kbls': self.data.get('kbls'),
            'kblt': self.data.get('kblt'),
            'kdef': self.data.get('kdef'),
            'khinp': self.data.get('khinp'),
            'khmax': self.data.get('khmax'),
            'kmix0': self.data.get('kmix0'),
            'kmixd': self.data.get('kmixd'),
            'kmsl': self.data.get('kmsl'),
            'kpuff': self.data.get('kpuff'),
            'krand': self.data.get('krand'),
            'krnd': self.data.get('krnd'),
            'kspl': self.data.get('kspl'),
            'kwet': self.data.get('kwet'),
            'kzmix': self.data.get('kzmix'),
            'maxdim': self.data.get('maxdim'),
            'maxpar': self.data.get('maxpar'),
            'mgmin': self.data.get('mgmin'),
            'mhrs': self.data.get('mhrs'),
            'nbptyp': self.data.get('nbptyp'),
            'ncycl': self.data.get('ncycl'),
            'ndump': self.data.get('ndump'),
            'ninit': self.data.get('ninit'),
            'nstr': self.data.get('nstr'),
            'nturb': self.data.get('nturb'),
            'nver': self.data.get('nver'),
            'outdt': self.data.get('outdt'),
            'p10f': self.data.get('p10f'),
            'pinbc': self.data.get('pinbc'),
            'pinpf': self.data.get('pinpf'),
            'poutf': self.data.get('poutf'),
            'qcycle': self.data.get('qcycle'),
            'rhb': self.data.get('rhb'),
            'rht': self.data.get('rht'),
            'splitf': self.data.get('splitf'),
            'tkerd': self.data.get('tkerd'),
            'tkern': self.data.get('tkern'),
            'tlfrac': self.data.get('tlfrac'),
            'tout': self.data.get('tout'),
            'tratio': self.data.get('tratio'),
            'tvmix': self.data.get('tvmix'),
            'veght': self.data.get('veght'),
            'vscale': self.data.get('vscale'),
            'vscaleu': self.data.get('vscaleu'),
            'vscales': self.data.get('vscales'),
            'w_option': self.data.get('w_option'),
            'zicontroltf': self.data.get('zicontroltf'),
            'ziscale': self.data.get('ziscale'),
            'z_top': self.data.get('z_top')
        }

    @cached_property
    def error_options(self) -> dict[str, Any]:
        return {
            'horcoruverr': self.data.get('horcoruverr'),
            'siguverr': self.data.get('siguverr'),
            'tluverr': self.data.get('tluverr'),
            'zcoruverr': self.data.get('zcoruverr'),
            'horcorzierr': self.data.get('horcorzierr'),
            'sigzierr': self.data.get('sigzierr'),
            'tlzierr': self.data.get('tlzierr')
        }


class Model:
    """
    STILT model object containing configuration information and simulation data.
    """
    def __init__(self, project: str | Path,
                 output_wd=None, model_options=None,
                 parallel_options=None, footprint_options=None,
                 meteorological_options=None, transport_options=None,
                 error_options=None):
        # Extract project name and working directory
        project = Path(project)
        self.project = project.name
        wd = project.parent
        if wd == Path('.'):
            wd = Path.cwd()
        self.stilt_wd = wd / project
        self.output_wd = output_wd or self.stilt_wd / 'out'

        # Check if project exists
        assert self.stilt_wd.exists(), f'Project not found: {self.stilt_wd}'

        self.model_options = model_options or {}
        self.parallel_options = parallel_options or {}
        self.footprint_options = footprint_options or {}
        self.meteorological_options = meteorological_options or {}
        self.transport_options = transport_options or {}
        self.error_options = error_options or {}

    @classmethod
    def from_run_script(cls, script: str | Path | RunScript):
        """
        Initialize STILT model from run script.

        Parameters
        ----------
        script : str | Path
            Path to STILT run script.

        Returns
        -------
        Model
            STILT model object.
        """
        run_script = script if isinstance(script, RunScript) else RunScript(script)
        return cls(
            project=run_script.stilt_wd,
            output_wd=run_script.output_wd,
            model_options=run_script.model_options,
            parallel_options=run_script.parallel_options,
            footprint_options=run_script.footprint_options,
            meteorological_options=run_script.meteorological_options,
            transport_options=run_script.transport_options,
            error_options=run_script.error_options
        )

    @property
    def n_hours(self) -> int:
        return self.model_options.get('n_hours', None)

    @property
    def numpar(self) -> int:
        return self.model_options.get('numpar', None)

    @cached_property
    def simulations(self) -> SimulationCollection:
        return SimulationCollection.from_output_wd(self.output_wd)

    @property
    def run_times(self) -> set[dt.datetime]:
        return {sim.run_time for sim in self.simulations.successful.values()}

    @property
    def receptors(self) -> set[Receptor]:
        return {sim.receptor for sim in self.simulations.successful.values()}

    @property
    def footprints(self) -> dict[str, Footprint]:
        return {sim.id: sim.footprint
                for sim in self.simulations.successful.values()
                if sim.has_footprint}

    def failed_sims_by_reason(self) -> dict[str, list[Simulation]]:
        """
        Group failed simulations by reason for failure.

        Returns
        -------
        dict
            Dictionary of failed simulations grouped by reason.
        """
        reasons = {}
        for sim in self.simulations.failed.values():
            reason = sim.identify_failure_reason()
            if reason not in reasons:
                reasons[reason] = []
            reasons[reason].append(sim)
        return reasons

    def __getstate__(self):
        state = self.__dict__.copy()
        # Instead of making Simulation & SimulationCollection objects picklable,
        # we'll just store the simulation paths
        if 'simulations' in state:
            state['simulations'] = dict(state['simulations'])
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore the simulations attribute as a SimulationCollection
        if 'simulations' in state:
            self.simulations = SimulationCollection(state['simulations'])