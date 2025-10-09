
from pathlib import Path
from typing_extensions import Self

import pandas as pd


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
