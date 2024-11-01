"""
Stochastic Time-Inverted Lagrangian Transport (STILT) Model.

Inspired by https://github.com/uataq/air-tracker-stiltctl
"""

import datetime as dt
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
from typing import Any, Literal
from typing_extensions import \
    Self  # requires python 3.11 to import from typing

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
import sparse
import xarray as xr

from lair.utils.geo import BaseGrid, CRS, clip, write_rio_crs

# TODO:
# - Support multiple receptors
#   - ColumnReceptor
# - Run stilt_cli.r from python
#   - stilt_cli only runs one simulation at a time via simulation step
#   - want to be able to access slurm, but seems hard to manipulate run_stilt.r
# - Footprint inherit from BaseGrid
#   - refactor from inventories.py
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

        self.footprints = []  # ? keep track of footprints associated with receptor

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
        if isinstance(other, Receptor):
            return all([
                self.longitude == other.longitude,
                self.latitude == other.latitude,
                self.height == other.height
            ])
        elif isinstance(other, (tuple, list)):
            return all([
                self.longitude == other[0],
                self.latitude == other[1],
                self.height == other[2]
            ])
        elif isinstance(other, str):
            return other in {self.id, self.name} if self.name else self.id == other
        return False

    def __hash__(self) -> int:
        return hash((self.longitude, self.latitude, self.height))


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
        ds = ds.expand_dims(receptor=[receptor],
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
        receptor = ds.receptor.item()
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
        for phrase, reason in self.FAILURE_PHRASES.items():
            if phrase in self.log:
                return reason
            else:
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


class SimulationCollection(dict):
    def __init__(self, out_dirs: str | Path | list[str | Path] | None = None):
        super().__init__()  # initialize dict
        self._failed = {}
        self._successful = {}

        if out_dirs is None:
            self.out_dirs = []
        elif isinstance(out_dirs, (str, Path)):
            self.out_dirs = [Path(out_dirs)]
        else:
            self.out_dirs = [Path(d) for d in out_dirs]
        self._load_simulations()

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
        super().__setitem__(key, sim)
        if sim.was_successful:
            self._successful[key] = sim
        else:
            self._failed[key] = sim

    def _load_simulations(self) -> None:
        for out_dir in self.out_dirs:
            for log_file in out_dir.rglob('stilt.log'):
                sim = Simulation(log_file.parent)
                self[sim.id] = sim

    @property
    def successful(self) -> dict[str, Simulation]:
        return self._successful

    @property
    def failed(self) -> dict[str, Simulation]:
        return self._failed

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
        sim_df = self.as_dataframe(subset='successful' if include_failed else None)

        # Parse run_time as datetime
        in_df['run_time'] = pd.to_datetime(in_df['run_time'])
        sim_df['run_time'] = pd.to_datetime(sim_df['run_time'])

        # Use run_time & receptor info to match simulations
        cols = ['run_time', 'long', 'lati', 'zagl']

        # Merge dataframes on run_time & receptor info
        merged = pd.merge(in_df, sim_df[cols], on=cols, how='outer', indicator=True)
        missing = merged[merged['_merge'] == 'left_only']
        return missing.drop(columns='_merge')

    def as_dataframe(self, subset: Literal['successful', 'failed'] | None = None) -> pd.DataFrame:
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
            sims = {'successful': self._successful, 'failed': self._failed}[subset]

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
        df = self.as_dataframe(subset=subset)
        df.to_csv(path, index=False)
        return None


class Model:

    def __init__(self, path: str | Path,
                 out_dirs: str | Path | list[str | Path] | None = None):
        self.path = Path(path)
        if isinstance(out_dirs, (str, Path)):
            self.out_dirs = [Path(out_dirs)]
        elif isinstance(out_dirs, list):
            self.out_dirs = [Path(d) for d in out_dirs]
        else:
            self.out_dirs = [self.path / 'out']

    @cached_property
    def simulations(self) -> SimulationCollection:
        return SimulationCollection(out_dirs=self.out_dirs)

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
