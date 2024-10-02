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
from pathlib import Path
from typing import Any, Self

import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

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
            print('Receptor already exists')
            return cls._receptors[key]
        print('Creating new receptor')
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
        print('Init receptor')
        self.longitude = longitude
        self.latitude = latitude
        self.height = height
        self.name = name

        self.footprints = []  # ? keep track of footprints associated with receptor

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

    @staticmethod
    def _pivot(df: pd.DataFrame) -> xr.Dataset:
        ds = xr.Dataset.from_dataframe(
            df.set_index(['indx', 'time'])
        )
        return ds

    def to_xarray(self, error: bool = False) -> xr.Dataset:
        """
        Convert trajectory data to xarray.Dataset
        with time and particle index as coordinates.

        Parameters
        ----------
        error : bool
            Output particle error data if available.
        """
        if not error:
            ds = self._pivot(self.particles)
            ds.attrs = self.params
        elif self.particle_error is not None:
            ds = self._pivot(self.particle_error)
            ds.attrs = self.error_params or {}
        else:
            raise ValueError('No error data available')

        # TODO add receptor as coords?

        return ds


class Footprint:
    """
    Footprint object containing STILT simulation footprint data.
    """
    def __init__(self, simulation_id: str,
                 foot: xr.DataArray):
        self.simulation_id = simulation_id
        self.foot = foot

    def __repr__(self):
        return f'Footprint({self.simulation_id})'

    @classmethod
    def from_netcdf(cls, file: str | Path) -> Self:
        """
        Create Footprint object from netCDF file.
        
        Parameters
        ----------
        file : str | Path
            Path to netCDF file.

        Returns
        -------
        Footprint
            Footprint object.
        """
        simulation_id = Path(file).stem.replace('_foot', '')
        foot = xr.open_dataset(file).foot

        return cls(simulation_id, foot)

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
        return self.foot.sel(time=slice(start, end)).sum('time')


class Simulation:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        assert self.path.exists(), f'Simulation not found: {self.path}'

        by_id_dir = Simulation.find_by_id_directory(self.path)
        self.id = str(self.path.relative_to(by_id_dir))

        self.run_time = self.control.pop('run_time')
        receptors = self.control['receptors']
        self.receptor = receptors[0]  # TODO: support multiple receptors

    def __repr__(self) -> str:
        return f'Simulation({self.id})'

    @staticmethod
    def find_by_id_directory(path: Path) -> Path:
        for parent in path.parents:
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
    def trajectory(self) -> Trajectory | None:
        file = self.path / f'{self.id}_traj.parquet'
        if not file.exists():
            return None
        return Trajectory.from_parquet(file)

    @cached_property
    def footprint(self) -> Footprint | None:
        file = self.path / f'{self.id}_foot.nc'
        if not file.exists():
            return None
        footprint = Footprint.from_netcdf(file)
        footprint.foot = footprint.foot.expand_dims(run_time=[self.run_time],)
                                                    # receptor=[self.receptor])
                                                    # r_long=[self.receptor.longitude],
                                                    # r_lati=[self.receptor.latitude],
                                                    # r_zagl=[self.receptor.height])
        return footprint


class STILT:

    def __init__(self, stilt_wd: str | Path,
                 output_wd: str | Path | None = None):
        self.stilt_wd = Path(stilt_wd)
        self.output_wd = Path(output_wd) if output_wd else self.stilt_wd / 'out'

    @property
    def simulations(self) -> dict[str, Simulation]:
        sims = {}
        for control_file in (self.output_wd / 'by-id').rglob('CONTROL'):
            sim_dir = control_file.parent
            sims[sim_dir.stem] = Simulation(sim_dir)
        return sims

    @property
    def run_times(self) -> set[dt.datetime]:
        return {sim.run_time for sim in self.simulations.values()}

    @property
    def receptors(self) -> set[Receptor]:
        return {sim.receptor for sim in self.simulations.values()}

    @cached_property
    def footprints(self) -> Footprint | None:
        footprints = []
        for i, sim in enumerate(self.simulations.values()):
            if i == 100:
                break
            if sim.footprint:
                footprints.append(sim.footprint.foot)
                del sim.footprint  # free memory & reset cache
        if not footprints:
            return None
        return Footprint(simulation_id=self.stilt_wd.stem,
                         foot=xr.merge(footprints).foot)
