import datetime as dt
from abc import ABC, abstractmethod
from pathlib import Path

import f90nml
import pandas as pd
from typing_extensions import Self

from stilt.config import SimulationConfig
from stilt.hysplit import Control
from stilt.receptors import Receptor


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

    def __init__(
        self,
        simulation_id: str,
        receptor: Receptor,
        data: pd.DataFrame,
        n_hours: int,
        met_files: list[Path],
        params: dict,
    ):
        super().__init__(simulation_id=simulation_id, receptor=receptor, data=data)
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
        winderrtf = self.params.get("winderrtf", 0)
        return winderrtf > 0

    @classmethod
    def calculate(
        cls,
        simulation_dir,
        control,
        namelist: f90nml.Namelist,
        timeout=3600,
        rm_dat=True,
        file=None,
    ):
        raise NotImplementedError

    @classmethod
    def from_path(cls, path) -> Self:
        # Resolve the file path
        path = Path(path).resolve()

        # Read config and control files
        config = SimulationConfig.from_path(
            path.parent / f"{path.parent.name}_config.yaml"
        )
        control = Control.from_path(path.parent / "CONTROL")

        # Read data from parquet file
        data = Trajectory.read_parquet(
            path, r_time=control.receptor.time, outdt=config.outdt
        )

        return cls(
            simulation_id=config.simulation_id,
            receptor=config.receptor,
            data=data,
            n_hours=config.n_hours,
            met_files=control.met_files,
            params=config.transport_params(),
        )

    @staticmethod
    def read_parquet(
        path: str | Path, r_time: dt.datetime, outdt: int = 0, **kwargs
    ) -> pd.DataFrame:
        data = pd.read_parquet(path, **kwargs)

        unit = "min" if outdt == 0 else str(outdt) + "min"
        data["datetime"] = r_time + pd.to_timedelta(data["time"], unit=unit)
        return data
