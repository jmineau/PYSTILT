"""HYSPLIT CONTROL file model."""

import datetime as dt
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from stilt.config import VerticalReference
from stilt.receptors import Receptor


class ControlFile(BaseModel):
    """HYSPLIT control parameters."""

    receptor: Receptor
    emisshrs: float
    n_hours: int
    w_option: int
    z_top: float
    met_files: list[Path]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def write(self, path: str | Path) -> None:
        """
        Write the HYSPLIT CONTROL file.

        Format is positional text: receptor time, receptor coordinates,
        run parameters, met files, emission hours.

        Parameters
        ----------
        path : str or Path
            Destination path. Parent directories are created if absent.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []

        # Receptor time - 2-digit year (HYSPLIT convention)
        lines.append(self.receptor.time.strftime("%y %m %d %H %M"))

        # Receptor coordinates (one line per point)
        n_points = len(self.receptor)
        lines.append(str(n_points))
        for lat, lon, altitude in self.receptor:
            lines.append(f"{lat} {lon} {altitude}")

        # Run parameters
        lines.append(str(self.n_hours))  # (4) total run time (hours)
        lines.append(str(self.w_option))  # (5) vertical motion option
        lines.append(
            str(self.z_top)
        )  # (6) top of model domain (internal coordinates, m-agl)

        # Met files - directory and filename on separate lines each
        lines.append(str(len(self.met_files)))  # (7) number of met files
        for met_file in self.met_files:
            lines.append(
                str(met_file.parent) + "/"
            )  # (8) met file directory (trailing slash important for HYSPLIT)
            lines.append(met_file.name)  # (9) met file name

        # Concentration / Pollutant Definition
        lines.append("1")  # (10) number of different pollutants
        lines.append("test")  # (11) pollutant 4-char identifier
        lines.append("1")  # (12) emission rate (per hour)
        lines.append(str(self.emisshrs))  # (13) hours of emission (can be fractional)
        lines.append(
            "00 00 00 00 00"
        )  # (14) release start (YY MM DD HH MM), 0s for start of met file

        # Concentration / Grid  Definition (hardcoded - required by HYSPLIT parser, ignored by STILT)
        lines.append("1")  # (15) number of simultaneous concentration grids
        lines.append("0.0 0.0")  # (16) grid center (lat, lon), 0s for receptor location
        lines.append("0.5 0.5")  # (17) grid spacing (degrees)
        lines.append("30.0 30.0")  # (18) grid span (degrees)
        lines.append("./")  # (19) output directory
        lines.append("cdump")  # (20) output filename (STILT reads PARTICLE_STILT.DAT)
        lines.append("1")  # (21) number of vertical concentration levels
        lines.append("100")  # (22) height of each level (m)
        lines.append("00 00 00 00 00")  # (23) sampling start
        lines.append("00 00 00 00 00")  # (24) sampling stop
        lines.append("00 2 00")  # (25) sampling interval (type 2 = maximum)

        # Concentration / Deposition Definition (all zeros = passive tracer gas, no deposition)
        lines.append("1")  # (26) number of pollutants depositing
        lines.append(
            "0.0 0.0 0.0"
        )  # (27) particle: diameter (um), density (g/cm3), shape
        lines.append(
            "0.0 0.0 0.0 0.0 0.0"
        )  # (28) deposition velocity (m/s), pollutant molecular weight (g/mol), surface reactivity ratio, diffusivity ratio, effective Henry's constant
        lines.append(
            "0.0 0.0 0.0"
        )  # (29) wet removal: actual Henry's constant, in-cloud, below-cloud (1/s)
        lines.append("0.0")  # (30) radioactive decay half-life (days)
        lines.append("0.0")  # (31) pollutant resuspension (1/m)

        path.write_text("\n".join(lines) + "\n")

    @classmethod
    def read(cls, path, *, altitude_ref: VerticalReference = "agl"):
        """
        Build ControlFile object from HYSPLIT control file.

        Returns
        -------
        ControlFile
            ControlFile object with parsed parameters.
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
        lats, lons, altitudes = [], [], []
        for i in range(cursor, cursor + n_receptors):
            lat, lon, altitude = map(float, lines[i].strip().split())
            lats.append(lat)
            lons.append(lon)
            altitudes.append(altitude)

        receptor = Receptor.from_points(
            time=time,
            points=list(zip(lons, lats, altitudes, strict=False)),
            altitude_ref=altitude_ref,
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
        # Pollutant section: n_pollutant (0), species name (1), emission count (2),
        # then emisshrs at offset 3
        emisshrs = float(lines[cursor + 3].strip())

        return cls(
            receptor=receptor,
            emisshrs=emisshrs,
            n_hours=n_hours,
            w_option=w_option,
            z_top=z_top,
            met_files=met_files,
        )
