"""
Stochastic Time-Inverted Lagrangian Transport (STILT) Model.

A python implementation of the [R-STILT](https://github.com/jmineau/stilt) model framework.

> Inspired by https://github.com/uataq/air-tracker-stiltctl
"""

import os
import subprocess
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from stilt.config import ModelConfig
from stilt.simulation import Simulation


def parse_sim_id(sim_id: str) -> tuple[pd.Timestamp, str]:
    """Parse a simulation ID into (time, location_id).

    Handles all ID formats:
        202301150600_-111.5_40.5_100      → point
        202301150600_-111.5_40.5_X        → column
        202301150600_multi_abc123def456   → multipoint

    Parameters
    ----------
    sim_id : str
        Simulation ID string.

    Returns
    -------
    tuple[pd.Timestamp, str]
        (time, location_id) where location_id is everything after the
        12-character timestamp and its trailing underscore.

    Raises
    ------
    ValueError
        If the sim_id doesn't start with a valid 12-digit timestamp.
    """
    if len(sim_id) < 13 or sim_id[12] != "_":
        raise ValueError(f"Invalid sim_id format: {sim_id!r}")

    time_str = sim_id[:12]
    location_id = sim_id[13:]

    try:
        time = pd.to_datetime(time_str, format="%Y%m%d%H%M")
    except ValueError:
        raise ValueError(f"Cannot parse timestamp from sim_id: {sim_id!r}") from None

    return time, location_id


def _find_resolutions(sim_dir: Path, sim_id: str) -> list[str]:
    """Discover footprint resolutions by globbing *_foot.nc files."""
    prefix = f"{sim_id}_"
    resolutions = []
    for f in sim_dir.glob(f"{sim_id}_*_foot.nc"):
        stem = f.stem.removesuffix("_foot")
        res = stem[len(prefix) :]
        if res:
            resolutions.append(res)
    return resolutions


def stilt_init(project: str | Path, branch: str | None = None, repo: str | None = None):
    """
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
    """
    if branch is None:
        branch = "jmineau"
    if repo is None:
        repo = "https://github.com/jmineau/stilt"
    elif "uataq" in repo and branch == "jmineau":
        raise ValueError("The 'uataq' repo does not have a 'jmineau' branch. ")

    # Extract project name and working directory
    project = Path(project)
    name = project.name
    wd = project.parent
    if wd == Path("."):
        wd = Path.cwd()

    if project.exists():
        raise FileExistsError(f"{project} already exists")

    # Clone git repository
    cmd = f"git clone -b {branch} --single-branch --depth=1 {repo} {project}"
    subprocess.check_call(cmd, shell=True)

    # Run setup executable
    project.joinpath("setup").chmod(0o755)
    subprocess.check_call("./setup", cwd=project)

    # Render run_stilt.r template with project name and working directory
    run_stilt_path = project.joinpath("r/run_stilt.r")
    run_stilt = run_stilt_path.read_text()
    run_stilt = run_stilt.replace("{{project}}", name)
    run_stilt = run_stilt.replace("{{wd}}", str(wd))
    run_stilt_path.write_text(run_stilt)


class Model:
    """STILT project interface for managing simulations.

    Config loading is lazy — accessing ``model.simulations`` only requires
    the project directory to exist with the ``out/by-id/`` convention.
    The full ``ModelConfig`` is only parsed when ``model.config`` is accessed.
    """

    def __init__(self, project: str | Path):
        self.project = Path(project)

    @cached_property
    def config(self) -> ModelConfig:
        """Load the project ModelConfig (parses config.yaml + receptor CSV)."""
        config_path = self.project / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config.yaml found in {self.project}. "
                "Use stilt_init() to create a new project."
            )
        return ModelConfig.from_path(config_path)

    @staticmethod
    def initialize(project: Path, **kwargs) -> "Model":
        wd = project.parent
        if wd == Path("."):
            wd = Path.cwd()
        stilt_wd = wd / project
        del kwargs["stilt_wd"]

        repo = kwargs.pop("repo", None)
        branch = kwargs.pop("branch", None)
        stilt_init(project=project, branch=branch, repo=repo)

        # Write config so future Model(project) can load it
        ModelConfig(stilt_wd=stilt_wd, **kwargs)
        return Model(project)

    @cached_property
    def simulations(self) -> pd.DataFrame:
        """Lightweight simulation index built from directory names.

        Only performs a single directory listing — no file existence checks
        or YAML parsing. Fast even on network filesystems with 100K+ sims.

        Returns
        -------
        pd.DataFrame
            Index is sim_id. Columns: time, location_id, path.
        """
        columns = ["time", "location_id", "path"]
        by_id = self.project / "out" / "by-id"
        if not by_id.exists():
            return pd.DataFrame(columns=columns).rename_axis("sim_id")

        rows = []
        for entry in os.scandir(by_id):
            if not entry.is_dir():
                continue
            sim_id = entry.name
            try:
                time, location_id = parse_sim_id(sim_id)
            except ValueError:
                continue
            rows.append(
                {
                    "sim_id": sim_id,
                    "time": time,
                    "location_id": location_id,
                    "path": Path(entry.path),
                }
            )

        if not rows:
            return pd.DataFrame(columns=columns).rename_axis("sim_id")

        return pd.DataFrame(rows).set_index("sim_id").sort_index()

    def invalidate_cache(self):
        """Clear the cached simulations index so it is rebuilt on next access."""
        self.__dict__.pop("simulations", None)

    def get_simulations(
        self,
        resolution: str | None = None,
        time_range: tuple | None = None,
        location_ids: set[str] | None = None,
    ) -> list[Path]:
        """Filtered list of simulation paths for downstream consumers.

        Index-only filters (time_range, location_ids) are applied first with
        no file I/O. File existence checks (footprint resolution) are only
        performed on the remaining rows.

        Parameters
        ----------
        resolution : str, optional
            Only include simulations that have a footprint at this resolution
            (e.g. '0.01x0.01').
        time_range : tuple, optional
            (start, end) datetime tuple to filter by receptor time.
        location_ids : set[str], optional
            Only include simulations whose location_id is in this set
            (e.g. {"-111.85_40.77_4", "-111.847672_40.766189_35"}).

        Returns
        -------
        list[Path]
            Paths to simulation directories matching the filters.
        """
        df = self.simulations
        if df.empty:
            return []
        if time_range:
            df = df[df.time.between(*time_range)]
        if location_ids is not None:
            df = df[df.location_id.isin(location_ids)]
        if resolution:
            df = df[
                df.apply(
                    lambda row: (
                        row.path / f"{row.name}_{resolution}_foot.nc"
                    ).exists(),
                    axis=1,
                )
            ]
        return df.path.tolist()

    def get_missing(
        self,
        receptors: str | Path | pd.DataFrame,
        include_failed: bool = False,
    ) -> pd.DataFrame:
        """Find receptors that don't have completed simulations.

        Parameters
        ----------
        receptors : str | Path | pd.DataFrame
            Receptor CSV path or DataFrame. Must contain a 'sim_id' column,
            or 'time', 'long', 'lati', 'zagl' columns from which sim_ids
            will be built.
        include_failed : bool
            If True, treat failed simulations (no trajectory) as missing.
            Checks for trajectory file existence only on sims in the index.

        Returns
        -------
        pd.DataFrame
            Subset of receptors that are missing from the simulation index.
        """
        if isinstance(receptors, (str, Path)):
            receptors = pd.read_csv(receptors, parse_dates=["time"])
        else:
            receptors = receptors.copy()

        # Build sim_id if not present
        if "sim_id" not in receptors.columns:
            receptors["sim_id"] = _build_sim_ids(receptors)

        # Determine which existing sims count as "done"
        df = self.simulations
        if df.empty:
            return receptors

        if include_failed:
            # Only count sims with trajectory files as done
            has_traj = df.apply(
                lambda row: (row.path / f"{row.name}_traj.parquet").exists(),
                axis=1,
            )
            existing = set(df[has_traj].index)
        else:
            # Any directory present counts as done
            existing = set(df.index)

        return receptors[~receptors.sim_id.isin(existing)]

    def load_simulation(self, sim_id: str) -> Simulation:
        """Load a single Simulation object on demand (parses YAML).

        Parameters
        ----------
        sim_id : str
            Simulation ID.

        Returns
        -------
        Simulation
        """
        return Simulation.from_path(self.project / "out" / "by-id" / sim_id)

    def plot_availability(self, ax: plt.Axes | None = None, **kwargs) -> plt.Axes:
        """Plot simulation availability over time as a Gantt chart.

        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib Axes to plot on.

        Returns
        -------
        plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        df = self.simulations.copy()
        if df.empty:
            return ax

        for _, row in df.iterrows():
            ax.barh(
                y=row.location_id,
                width=pd.Timedelta(hours=1),
                left=row.time,
                height=0.6,
                align="center",
                edgecolor="black",
                alpha=0.6,
                **kwargs,
            )

        fig.autofmt_xdate()
        ax.set(title="Simulation Availability", xlabel="Time", ylabel="Location ID")
        return ax

    def run(self):
        # TODO: implement Python execution
        self._run_rscript()

    def _run_rscript(self):
        raise NotImplementedError


def _build_sim_ids(df: pd.DataFrame) -> pd.Series:
    """Build sim_id strings from a receptor DataFrame.

    Expects columns: time (or Time_UTC), long, lati, zagl.
    Only handles point receptors (the common case).
    """
    time_col = "time"
    if time_col not in df.columns and "Time_UTC" in df.columns:
        time_col = "Time_UTC"

    time = pd.to_datetime(df[time_col])

    return (
        time.dt.strftime("%Y%m%d%H%M")
        + "_"
        + df["long"].astype(str)
        + "_"
        + df["lati"].astype(str)
        + "_"
        + df["zagl"].astype(str)
    )
