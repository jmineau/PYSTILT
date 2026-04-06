import datetime as dt
import os
from functools import cached_property
from pathlib import Path

import f90nml
import pandas as pd
from typing_extensions import Self

from stilt.config import Resolution, SimulationConfig
from stilt.footprint import Footprint, FootprintCollection
from stilt.hysplit import Control
from stilt.meteorology import Meteorology
from stilt.receptors import Receptor
from stilt.trajectory import Trajectory


class Simulation:
    PATHS = {
        "config": "config.yaml",
        "control": "CONTROL",
        "error": "error.parquet",
        "footprints": "*_foot.nc",
        "log": "stilt.log",
        "params": "CONC.CFG",
        "receptors": "receptors.csv",
        "setup": "SETUP.CFG",
        "trajectory": "traj.parquet",
        "winderr": "WINDERR",
        "zicontrol": "ZICONTROL",
        "zierr": "ZIERR",
    }

    FAILURE_PHRASES = {
        "Insufficient number of meteorological files found": "MISSING_MET_FILES",
        "meteorological data time interval varies": "VARYING_MET_INTERVAL",
        "PARTICLE_STILT.DAT does not contain any trajectory data": "NO_TRAJECTORY_DATA",
        "Fortran runtime error": "FORTRAN_RUNTIME_ERROR",
    }

    def __init__(
        self,
        config: SimulationConfig,
    ):
        self.config = config

        self.id = self.config.simulation_id
        self.path = self.config.output_wd / "by-id" / self.id
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

            paths["config"] = self.path / f"{self.id}_{self.PATHS['config']}"
            paths["control"] = self.path / self.PATHS["control"]
            paths["error"] = self.path / f"{self.id}_{self.PATHS['error']}"
            paths["log"] = self.path / self.PATHS["log"]
            paths["params"] = self.path / self.PATHS["params"]
            paths["receptors"] = self.path / self.PATHS["receptors"]
            paths["setup"] = self.path / self.PATHS["setup"]
            paths["trajectory"] = self.path / f"{self.id}_{self.PATHS['trajectory']}"
            paths["winderr"] = self.path / self.PATHS["winderr"]
            paths["zicontrol"] = self.path / self.PATHS["zicontrol"]
            paths["zierr"] = self.path / self.PATHS["zierr"]

            # Build footprint paths based on resolutions
            paths["footprints"] = {}
            resolutions = self.config.resolutions
            if resolutions is not None:
                for res in resolutions:
                    paths["footprints"][str(res)] = (
                        self.path / f"{self.id}_{res}_foot.nc"
                    )

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

        if (
            self.config.run_trajec
            and not self.paths["trajectory"].exists()
            or self.config.run_foot
            and not all(path.exists() for path in self.paths["footprints"].values())
        ):
            status = "FAILURE"
        else:
            status = "SUCCESS"

        if status.lower().startswith("fail"):
            status += f":{Simulation.identify_failure_reason(self.path)}"
        return status

    @property
    def meteorology(self) -> Meteorology:
        if not self._meteorology:
            self._meteorology = Meteorology(
                path=self.config.met_path,
                format=self.config.met_file_format,
                tres=self.config.met_file_tres,
            )
        return self._meteorology

    @property
    def met_files(self) -> list[Path]:
        if not self._met_files:
            if self.paths["control"].exists():
                control = self.control
                self._met_files = control.met_files
            else:
                # Get meteorology files from the meteorology object
                self._met_files = self.meteorology.get_files(
                    r_time=self.receptor.time, n_hours=self.config.n_hours
                )
                if self.config.met_subgrid_enable:
                    # Build subgrid meteorology
                    self._meteorology = self.meteorology.calc_subgrids(
                        files=self._met_files,
                        out_dir=self.config.output_wd / "met",
                        exe_dir=self.config.stilt_wd / "exe",
                        projection=self.config.projection,
                        xmin=self.config.xmn,
                        xmax=self.config.xmx,
                        ymin=self.config.ymn,
                        ymax=self.config.ymx,
                        levels=self.config.met_subgrid_levels,
                        buffer=self.config.met_subgrid_buffer,
                    )
                    # Get subgrid meteorology files
                    self._met_files = self._meteorology.get_files(
                        t_start=self.receptor.time
                        + pd.Timedelta(hours=self.config.n_hours),
                        n_hours=self.config.n_hours,
                    )
                if len(self._met_files) < self.config.n_met_min:
                    raise ValueError(
                        f"Insufficient meteorological files found. "
                        f"Found: {len(self._met_files)}, "
                        f"Required: {self.config.n_met_min}"
                    )
        return self._met_files

    @property
    def control(self) -> Control:
        if not self._control:
            if self.paths["control"].exists():
                self._control = Control.from_path(self.paths["control"])
            else:
                self._control = Control(
                    receptor=self.receptor,
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
            if self.paths["setup"].exists():
                self._setup = f90nml.read(self.paths["setup"])["setup"]
            else:
                names = [
                    "capemin",
                    "cmass",
                    "conage",
                    "cpack",
                    "delt",
                    "dxf",
                    "dyf",
                    "dzf",
                    "efile",
                    "frhmax",
                    "frhs",
                    "frme",
                    "frmr",
                    "frts",
                    "frvs",
                    "hscale",
                    "ichem",
                    "idsp",
                    "initd",
                    "k10m",
                    "kagl",
                    "kbls",
                    "kblt",
                    "kdef",
                    "khinp",
                    "khmax",
                    "kmix0",
                    "kmixd",
                    "kmsl",
                    "kpuff",
                    "krand",
                    "krnd",
                    "kspl",
                    "kwet",
                    "kzmix",
                    "maxdim",
                    "maxpar",
                    "mgmin",
                    "mhrs",
                    "nbptyp",
                    "ncycl",
                    "ndump",
                    "ninit",
                    "nstr",
                    "nturb",
                    "numpar",
                    "nver",
                    "outdt",
                    "p10f",
                    "pinbc",
                    "pinpf",
                    "poutf",
                    "qcycle",
                    "rhb",
                    "rht",
                    "splitf",
                    "tkerd",
                    "tkern",
                    "tlfrac",
                    "tout",
                    "tratio",
                    "tvmix",
                    "varsiwant",
                    "veght",
                    "vscale",
                    "vscaleu",
                    "vscales",
                    "wbbh",
                    "wbwf",
                    "wbwr",
                    "winderrtf",
                    "wvert",
                    "zicontroltf",
                ]
                namelist = {
                    name: getattr(self, name) for name in names if hasattr(self, name)
                }
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
            path = self.paths["trajectory"]
            if path.exists():
                self._trajectory = self._load_trajectory(path)
            else:
                print("Trajectory file not found. Has the simulation been run?")
            return self._trajectory

    @property
    def error(self) -> Trajectory | None:
        """STILT particle error trajectories."""
        if self.has_error and not self._error:
            path = self.paths["error"]
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
            log_file = self.path / "stilt.log"
            if not log_file.exists():
                raise FileNotFoundError(f"Log file not found: {log_file}")
            log = log_file.read_text()
        else:
            log = ""
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
            raise ValueError(
                f"Path '{path}' is not a valid STILT simulation directory."
            )

        if not path.glob("*config.json"):
            raise ValueError("Simulation was successful")

        # Check log file for errors
        if not (path / "stilt.log").exists():
            return "EMPTY_LOG"
        for phrase, reason in Simulation.FAILURE_PHRASES.items():
            if phrase in (path / "stilt.log").read_text():
                return reason
        return "UNKNOWN"

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
        if "by-id" not in path.parent.parts:
            raise ValueError(
                "Unable to extract simulation ID from path. 'by-id' directory not found in parent path."
            )
        id_index = path.parts.index("by-id") + 1
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
        exe_exists = (path / "hycs_std").exists()
        is_exe_dir = path.name == "exe"
        return exe_exists and not is_exe_dir
