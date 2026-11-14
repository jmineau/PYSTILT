"""The HYSPLIT ``CONTROL`` file."""

import datetime as dt
from pathlib import Path

from pydantic import BaseModel


class ControlFile(BaseModel):
    """
    Contents of a HYSPLIT ``CONTROL`` file for one run of ``hycs_std``.

    Holds the release time and the release locations, the run time, the
    vertical motion option, the model top, and the met files. The pollutant,
    concentration grid, and deposition sections HYSPLIT requires are written
    with fixed values that the particle output does not use.

    Attributes
    ----------
    release_time : datetime
        The release time, in UTC, written to the minute.
    locations : list of tuple of float
        The release locations, each ``(latitude, longitude, height)`` in
        degrees and meters, in the order ``CONTROL`` lists them. Whether the
        heights are above ground or above sea level is ``SETUP.CFG``'s
        ``KMSL``, not in this file.
    emisshrs : float
        Hours of emission.
    n_hours : int
        Total run time, in hours, negative for a backward run.
    w_option : int
        Vertical motion option.
    z_top : float
        Top of the model domain, in meters above ground.
    met_files : list of Path
        The meteorology files, in the order HYSPLIT reads them.
    """

    release_time: dt.datetime
    locations: list[tuple[float, float, float]]
    emisshrs: float
    n_hours: int
    w_option: int
    z_top: float
    met_files: list[Path]

    def write(self, path: str | Path) -> None:
        """
        Write the ``CONTROL`` file.

        Parameters
        ----------
        path : str or Path
            File to write.
        """
        lines: list[str] = []

        # (1) release time, with a 2-digit year (HYSPLIT convention)
        lines.append(self.release_time.strftime("%y %m %d %H %M"))

        # (2) number of release locations, (3) one line for each
        lines.append(str(len(self.locations)))
        for lat, lon, height in self.locations:
            lines.append(f"{lat} {lon} {height}")

        # Run parameters
        lines.append(str(self.n_hours))  # (4) total run time (hours)
        lines.append(str(self.w_option))  # (5) vertical motion option
        lines.append(
            str(self.z_top)
        )  # (6) top of model domain (internal coordinates, m-agl)

        # Met files - directory and filename on separate lines each. (7) is
        # "1 N": one grid with N files in time, at most 128 (mtim in
        # HYSPLIT's DEFGRID.INC). A bare "N", as STILT-R writes, makes each
        # file its own grid: at most 12 in NOAA's builds and 99 in the
        # bundled one (mgrd). Both give the same particles.
        lines.append(f"1 {len(self.met_files)}")
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
        # (14) release start (YY MM DD HH MM). Zeros mean the simulation
        # start, which keeps the release time's minute.
        lines.append("00 00 00 00 00")

        # Concentration / Grid  Definition (hardcoded - required by HYSPLIT's parser, not used by the particle output)
        lines.append("1")  # (15) number of simultaneous concentration grids
        lines.append(
            "0.0 0.0"
        )  # (16) grid center (lat, lon), 0s for the release location
        lines.append("0.5 0.5")  # (17) grid spacing (degrees)
        lines.append("30.0 30.0")  # (18) grid span (degrees)
        lines.append("./")  # (19) output directory
        lines.append(
            "cdump"
        )  # (20) output filename (the particles are in PARTICLE_STILT.DAT)
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

        Path(path).write_text("\n".join(lines) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> "ControlFile":
        """
        Read a ``CONTROL`` file written by :meth:`write`.

        Parameters
        ----------
        path : str or Path
            File to read.

        Returns
        -------
        ControlFile
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(
                "CONTROL file not found. Has the simulation been ran?"
            )

        with path.open() as f:
            lines = f.readlines()

        # Parse the release time
        release_time = dt.datetime.strptime(lines[0].strip(), "%y %m %d %H %M")

        # Parse the release locations
        n_locations = int(lines[1].strip())
        cursor = 2
        locations = []
        for i in range(cursor, cursor + n_locations):
            lat, lon, height = map(float, lines[i].strip().split())
            locations.append((lat, lon, height))

        cursor += n_locations
        n_hours = int(lines[cursor].strip())
        w_option = int(lines[cursor + 1].strip())
        z_top = float(lines[cursor + 2].strip())

        # Parse met files: "N" (one file per grid) or "grids files"
        counts = [int(value) for value in lines[cursor + 3].split()]
        n_met_files = counts[0] * (counts[1] if len(counts) > 1 else 1)
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
            release_time=release_time,
            locations=locations,
            emisshrs=emisshrs,
            n_hours=n_hours,
            w_option=w_option,
            z_top=z_top,
            met_files=met_files,
        )
