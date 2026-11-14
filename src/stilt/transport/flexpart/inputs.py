"""
FLEXPART's input files for one receptor: ``COMMAND``, ``RELEASES``, ``PARTOPTIONS``, ``OUTGRID``, ``AVAILABLE``, and ``pathnames``.

FLEXPART keeps each path in 120 characters, so it runs inside the workdir
with relative paths, and the met files a run reads are linked into
``met/`` there.

The layout of a workdir::

    pathnames        ./options/  ./output/  ./met/  ./AVAILABLE
    AVAILABLE        the met files by valid time
    options/         COMMAND, RELEASES, PARTOPTIONS, OUTGRID, SPECIES/SPECIES_001,
                     and links to the checkout's land-use and surface tables
    met/             links to the met files
    output/          where FLEXPART writes partoutput_*.nc
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor

if TYPE_CHECKING:
    from stilt.receptors import Receptor

    from .config import FlexpartConfig, FlexpartMetConfig

#: The data files FLEXPART reads from ``options/`` on every run, linked from the checkout.
DATA_FILES = ("IGBP_int1.dat", "sfcdata.t", "sfcdepo.t")

#: The species every release carries: a passive air tracer, from the checkout.
SPECIES_FILE = "SPECIES_AIRTRACER"

#: The PARTOPTIONS fields PYSTILT turns on: position, and what ``foot`` and
#: the particle table read (``mlht``, ``dens``, ``pres``, ``zsfc``, ``temp``,
#: ``sphu``). Every other field is off.
PARTOPTIONS_ON = (
    "LONGITUDE",
    "LATITUDE",
    "HEIGHT",
    "MIXINGHEIGHT",
    "DENSITY",
    "PRESSURE",
    "TOPOGRAPHY",
    "TEMPERATURE",
    "QV",
)

#: Every field PARTOPTIONS can turn on, in FLEXPART's order.
PARTOPTIONS_ALL = (
    "LONGITUDE",
    "LONGITUDE_AVERAGE",
    "LATITUDE",
    "LATITUDE_AVERAGE",
    "HEIGHT",
    "HEIGHT_AVERAGE",
    "PV",
    "PV_AVERAGE",
    "QV",
    "QV_AVERAGE",
    "DENSITY",
    "DENSITY_AVERAGE",
    "TEMPERATURE",
    "TEMPERATURE_AVERAGE",
    "PRESSURE",
    "PRESSURE_AVERAGE",
    "MIXINGHEIGHT",
    "MIXINGHEIGHT_AVERAGE",
    "TROPOPAUSE",
    "TROPOPAUSE_AVERAGE",
    "TOPOGRAPHY",
    "TOPOGRAPHY_AVERAGE",
    "MASS",
    "MASS_AVERAGE",
    "U",
    "U_AVERAGE",
    "V",
    "V_AVERAGE",
    "W",
    "W_AVERAGE",
    "VSETTLING",
    "VSETTLING_AVERAGE",
    "WETDEPOSITION",
    "DRYDEPOSITION",
)


def namelist(name: str, entries: dict[str, Any]) -> str:
    """Return a Fortran namelist group, one ``KEY=value,`` per line."""

    def value(v: Any) -> str:
        if isinstance(v, bool):
            return ".true." if v else ".false."
        if isinstance(v, str):
            return f'"{v}"'
        return f"{v}"

    lines = [f"&{name}"]
    lines += [f" {key}={value(v)}," for key, v in entries.items()]
    lines.append(" /")
    return "\n".join(lines) + "\n"


def _date(t: dt.datetime) -> int:
    """Return *t*'s date as FLEXPART's ``YYYYMMDD``."""
    return int(t.strftime("%Y%m%d"))


def _time(t: dt.datetime) -> int:
    """Return *t*'s time of day as FLEXPART's ``HHMISS``, an integer (06:00 is 60000)."""
    return int(t.strftime("%H%M%S"))


def command(
    config: FlexpartConfig,
    met: FlexpartMetConfig,
    window: tuple[dt.datetime, dt.datetime],
) -> str:
    """
    Return the ``COMMAND`` file for a run over *window*.

    The run goes backward when ``n_hours`` is negative. Particles are
    written every ``loutstep`` seconds (``IPOUT`` 1); there is no gridded
    output (``IOUT`` 0), no restart file, and no domain filling.
    """
    start, end = window
    return namelist(
        "COMMAND",
        {
            "LDIRECT": -1 if config.n_hours < 0 else 1,
            "IBDATE": _date(start),
            "IBTIME": _time(start),
            "IEDATE": _date(end),
            "IETIME": _time(end),
            "LOUTSTEP": config.loutstep,
            "LOUTAVER": config.loutstep,
            "LOUTSAMPLE": config.lsynctime,
            "LOUTRESTART": -1,
            "LRECOUTSTEP": config.loutstep,
            "LRECOUTAVER": config.loutstep,
            "LRECOUTSAMPLE": config.lsynctime,
            "LSYNCTIME": config.lsynctime,
            "CTL": config.ctl,
            "IFINE": config.ifine,
            "IOUT": 0,
            "IPOUT": 1,
            "LSUBGRID": config.lsubgrid,
            "LCONVECTION": config.lconvection,
            "LTURBULENCE": config.lturbulence,
            "LTURBULENCE_MESO": 0,
            "LAGESPECTRA": 0,
            "IPIN": 0,
            # FLEXPART refuses a backward run without it.
            "IOUTPUTFOREACHRELEASE": 1,
            "IFLUX": 0,
            "MDOMAINFILL": 0,
            "IND_SOURCE": 1,
            "IND_RECEPTOR": 1,
            "MQUASILAG": 0,
            "NESTED_OUTPUT": 0,
            "LNETCDFOUT": 1,
            "LINIT_COND": 0,
            "SFC_ONLY": 0,
            "CBLFLAG": config.cblflag,
            "NXSHIFT": met.nxshift,
            "MAXTHREADGRID": 1,
            "MAXFILESIZE": 10000,
            "LOGVERTINTERP": 0,
            "LCMOUTPUT": 0,
        },
    )


def releases(receptor: Receptor, numpar: int) -> str:
    """
    Return the ``RELEASES`` file for *receptor*: where and when its particles start.

    Every release is at the receptor's time. A point is one release. A
    column is ``numpar`` releases of one particle each, in slabs of equal
    depth from the bottom up: FLEXPART numbers particles in release order,
    so particle *k* starts in slab *k*, which is what
    :func:`stilt.particles.add_release_heights` takes for a model that
    writes no release row. A multipoint receptor is one release per point,
    the particles split as evenly as they go, the first points taking any
    extra. Heights are above ground (``ZKIND`` 1) or above sea level
    (``ZKIND`` 2), as the receptor's ``altitude_ref`` says.

    Raises
    ------
    TypeError
        If *receptor* is not a point, column, or multipoint receptor.
    """
    zkind = 2 if receptor.altitude_ref == "msl" else 1
    boxes: list[tuple[float, float, float, float, int]] = []
    if isinstance(receptor, PointReceptor):
        boxes.append(
            (
                receptor.longitude,
                receptor.latitude,
                receptor.altitude,
                receptor.altitude,
                numpar,
            )
        )
    elif isinstance(receptor, ColumnReceptor):
        step = (receptor.top - receptor.bottom) / numpar
        for k in range(numpar):
            bottom = receptor.bottom + k * step
            boxes.append(
                (receptor.longitude, receptor.latitude, bottom, bottom + step, 1)
            )
    elif isinstance(receptor, MultiPointReceptor):
        n = len(receptor.longitudes)
        share, extra = divmod(numpar, n)
        for i, (lon, lat, alt) in enumerate(
            zip(
                receptor.longitudes, receptor.latitudes, receptor.altitudes, strict=True
            )
        ):
            boxes.append((lon, lat, alt, alt, share + (1 if i < extra else 0)))
    else:
        raise TypeError(f"FLEXPART cannot release from a {type(receptor).__name__}.")

    text = namelist("RELEASES_CTRL", {"NSPEC": 1, "SPECNUM_REL": 1})
    for i, (lon, lat, z1, z2, parts) in enumerate(boxes, start=1):
        text += namelist(
            "RELEASE",
            {
                "IDATE1": _date(receptor.time),
                "ITIME1": _time(receptor.time),
                "IDATE2": _date(receptor.time),
                "ITIME2": _time(receptor.time),
                "LON1": lon,
                "LON2": lon,
                "LAT1": lat,
                "LAT2": lat,
                "Z1": z1,
                "Z2": z2,
                "ZKIND": zkind,
                "MASS": 1.0,
                "PARTS": parts,
                "COMMENT": f"RELEASE {i}",
            },
        )
    return text


def partoptions() -> str:
    """Return the ``PARTOPTIONS`` file: the fields of :data:`PARTOPTIONS_ON` on, every other off."""
    return namelist(
        "PARTOPTIONS", {name: name in PARTOPTIONS_ON for name in PARTOPTIONS_ALL}
    )


def outgrid(receptor: Receptor) -> str:
    """
    Return an ``OUTGRID`` file of one cell at the receptor.

    FLEXPART reads the file even with gridded output off (``IOUT`` 0), and
    the grid must lie inside the met's domain.
    """
    lat, lon, _ = receptor.coords()[0]
    return namelist(
        "OUTGRID",
        {
            "OUTLON0": round(lon - 0.5, 4),
            "OUTLAT0": round(min(max(lat - 0.5, -89.0), 88.0), 4),
            "NUMXGRID": 1,
            "NUMYGRID": 1,
            "DXOUT": 1.0,
            "DYOUT": 1.0,
            "OUTHEIGHTS": 100.0,
        },
    )


def available(met_files: dict[dt.datetime, Path]) -> str:
    """Return the ``AVAILABLE`` file: three header lines, then each met file's valid time and name."""
    lines = [
        "XXXXXX EMPTY LINES XXXXXXXXX",
        "XXXXXX EMPTY LINES XXXXXXXX",
        "YYYYMMDD HHMMSS   name of the file(up to 80 characters)",
    ]
    lines += [
        f"{t:%Y%m%d %H%M%S}      {path.name}    ON DISK"
        for t, path in sorted(met_files.items())
    ]
    return "\n".join(lines) + "\n"


def write_inputs(
    workdir: Path,
    receptor: Receptor,
    config: FlexpartConfig,
    met: FlexpartMetConfig,
    window: tuple[dt.datetime, dt.datetime],
    met_files: dict[dt.datetime, Path],
) -> None:
    """
    Write FLEXPART's input files for *receptor* into *workdir* (the layout in this module's notes).

    Raises
    ------
    FileNotFoundError
        If the checkout lacks a data file or the species file.
    """
    checkout = config.checkout / "options"
    options = workdir / "options"
    (options / "SPECIES").mkdir(parents=True, exist_ok=True)
    (workdir / "output").mkdir(exist_ok=True)
    (workdir / "met").mkdir(exist_ok=True)
    for name in DATA_FILES:
        source = checkout / name
        if not source.is_file():
            raise FileNotFoundError(f"The FLEXPART checkout has no options/{name}.")
        os.symlink(source, options / name)
    species = checkout / "SPECIES" / SPECIES_FILE
    if not species.is_file():
        raise FileNotFoundError(
            f"The FLEXPART checkout has no options/SPECIES/{SPECIES_FILE}."
        )
    shutil.copyfile(species, options / "SPECIES" / "SPECIES_001")
    (options / "COMMAND").write_text(command(config, met, window))
    (options / "RELEASES").write_text(releases(receptor, config.numpar))
    (options / "PARTOPTIONS").write_text(partoptions())
    (options / "OUTGRID").write_text(outgrid(receptor))
    for path in met_files.values():
        os.symlink(Path(path).resolve(), workdir / "met" / Path(path).name)
    (workdir / "AVAILABLE").write_text(available(met_files))
    (workdir / "pathnames").write_text("./options/\n./output/\n./met/\n./AVAILABLE\n")


__all__ = ["available", "command", "outgrid", "partoptions", "releases", "write_inputs"]
