"""
Stochastic Time-Inverted Lagrangian Transport (STILT) Model.

A python implementation of the [R-STILT](https://github.com/jmineau/stilt) model framework.

> Inspired by https://github.com/uataq/air-tracker-stiltctl
"""

from pathlib import Path
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from typing_extensions import \
    Self  # requires python 3.11 to import from typing

from stilt.config import ModelConfig
from stilt.simulation import Simulation


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
    def from_paths(cls, paths: list[Path | str]) -> Self:
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
