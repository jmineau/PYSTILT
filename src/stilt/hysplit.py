import datetime as dt
from pathlib import Path

from pydantic import BaseModel

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
            raise FileNotFoundError(
                "CONTROL file not found. Has the simulation been ran?"
            )

        with path.open() as f:
            lines = f.readlines()

        # Parse receptor time
        time = dt.datetime.strptime(lines[0].strip(), "%y %m %d %H %M")

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
        receptor = Receptor.build(
            time=time, latitude=lats, longitude=lons, height=zagls
        )

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
            met_files=met_files,
        )
