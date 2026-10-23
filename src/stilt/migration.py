"""
Reading what STILT-R and X-STILT wrote.

STILT-R writes each simulation to its own folder of ``out/by-id``, named by
the simulation id. X-STILT runs STILT-R's code and writes the same layout::

    my_project/
      out/by-id/
        201507050000_-111.85_40.77_5/
          201507050000_-111.85_40.77_5_traj.rds   receptor time, particles
          201507050000_-111.85_40.77_5_foot.nc    footprint
          CONTROL                                 the release points HYSPLIT read
      _rslurm_my_project/
        params.RDS                                every run's settings (Slurm runs only)

:func:`read_stilt_r` reads these folders as PYSTILT's receptors, particle
tables, and footprints. The ``.rds`` files are read with the
`rdata <https://github.com/vnmabus/rdata>`_ package, which needs no R
(``pip install "pystilt[stilt-r]"``).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from stilt.footprint.config import FootprintConfig
from stilt.footprint.io import _footprint_array
from stilt.particles import check_particles
from stilt.receptors import Receptor
from stilt.spatial import Grid

if TYPE_CHECKING:
    from stilt.transport.hysplit.control import ControlFile

#: PYSTILT's names for STILT-R's particle columns. HYSPLIT's driver renames
#: the first four the same way; ``xhgt`` is the release height STILT-R adds
#: for a column receptor.
_COLUMNS = {
    "indx": "particle",
    "time": "age",
    "long": "lon",
    "lati": "lat",
    "xhgt": "release_height",
}

_TRAJ = "_traj.rds"
_FOOT = "_foot.nc"
_CONTROL = "CONTROL"

#: The columns of rslurm's ``params.RDS`` that are read: the footprint's
#: smoothing, and two settings that must match the run's own files.
_PARAMS_COLUMNS = ("smooth_factor", "n_hours", "numpar")


def read_stilt_r(
    by_id_dir: str | Path,
    smooth_factor: float | None = None,
) -> Iterator[tuple[Receptor, pd.DataFrame, xr.DataArray | None]]:
    """
    Read STILT-R or X-STILT ``by-id`` output, one simulation at a time.

    Each folder of *by_id_dir* that holds ``<id>_traj.rds`` is a
    simulation. Its receptor, particles, and footprint come back as
    PYSTILT's: a :class:`~stilt.Receptor`, a particle table
    (:doc:`/reference/particles`), and a footprint array, as
    :func:`stilt.read_particles` and :func:`stilt.read_footprint` return
    them. The folders are read in name order, which is time order, and each
    one only when the loop reaches it.

    The receptor's release points are the ones in the folder's ``CONTROL``
    file, the numbers HYSPLIT read, so a point receptor's id is the folder's
    name. Its time is the ``.rds`` file's, which keeps the seconds that
    ``CONTROL`` drops. A receptor with one release point is a
    :class:`~stilt.PointReceptor`, two heights at one place are a
    :class:`~stilt.ColumnReceptor`, and release points at different places,
    such as X-STILT's slant columns, are a :class:`~stilt.MultiPointReceptor`.

    The particle columns get PYSTILT's names: ``indx``, ``time``, ``long``,
    ``lati``, and ``xhgt`` become ``particle``, ``age`` (minutes since
    release), ``lon``, ``lat``, and ``release_height``. ``time`` is then the
    UTC time of each row, the receptor time plus ``age``. Every other column
    keeps its STILT-R name, ``foot_no_hnf_dilution`` included: the
    near-field correction STILT-R applied stays as it is.

    The footprint is ``<id>_foot.nc``, on the grid its file gives, with
    layers stamped at the receptor time plus whole hours, as STILT-R stamps
    them. Its settings hold that grid, ``time_integrate`` (true for a
    backward run with one layer at the receptor time), and
    ``smooth_factor``. The footprint file does not record ``smooth_factor``.
    It comes from ``params.RDS``, which STILT-R's Slurm runs (rslurm) leave
    in ``_rslurm_<project>/`` beside ``out/``, or else from the
    *smooth_factor* argument. ``params.RDS`` is used only when its
    ``n_hours`` and ``numpar`` are the run's (``CONTROL``, and the HYSPLIT
    settings the ``.rds`` saved). Other ``_foot.nc`` files in the folder,
    such as X-STILT's footprints at other resolutions, are not read.

    Parameters
    ----------
    by_id_dir : str or Path
        The ``by-id`` folder, such as ``my_project/out/by-id``.
    smooth_factor : float, optional
        The ``smooth_factor`` STILT-R made the footprints with, for a project
        without ``params.RDS`` (one run without Slurm). Not needed to read
        folders without a footprint.

    Yields
    ------
    receptor : Receptor
        Where and when the particles were released.
    particles : pandas.DataFrame
        One row per particle per output step, as STILT-R saved them.
    footprint : xarray.DataArray or None
        The footprint, or ``None`` when the folder has no ``<id>_foot.nc``:
        STILT-R made particles only, or no particle reached the grid.

    Raises
    ------
    ImportError
        If the rdata package is not installed.
    FileNotFoundError
        If *by_id_dir* does not exist, no folder in it holds a ``_traj.rds``
        file, or a folder that does has no ``CONTROL`` file.
    ValueError
        When a simulation is read, if its files are not as STILT-R writes
        them: a ``_traj.rds`` without the particle columns or the receptor
        time, a ``CONTROL`` file that cannot be read, or a footprint on a
        projected grid (only longitude/latitude grids are read) or with
        layers that are not whole hours from the receptor time. When a
        footprint is read, if its ``smooth_factor`` is not known (no
        ``params.RDS`` and no *smooth_factor*), if ``params.RDS`` disagrees
        with the run's ``n_hours`` or ``numpar``, or if it records a
        ``smooth_factor`` other than *smooth_factor* or more than one.

    Warns
    -----
    UserWarning
        When folders hold a ``_foot.nc`` but no ``_traj.rds``. They are
        skipped: the particles are read from the ``.rds``.

    Notes
    -----
    A run with transport error stores a second set of particles in the same
    ``.rds`` (``particle_error``); only the first set is read. The HYSPLIT
    settings in the ``.rds`` (``params``) are not returned. X-STILT weights
    ``foot`` by the averaging kernel and pressure weighting after it saves
    the ``.rds``, so its particles are unweighted and its footprint is
    weighted.

    Examples
    --------
    >>> for receptor, particles, foot in stilt.migration.read_stilt_r("out/by-id"):
    ...     print(receptor.id, len(particles), foot is not None)

    A project run on one machine has no ``params.RDS``; give the
    ``smooth_factor`` from its ``run_stilt.r``:

    >>> sims = stilt.migration.read_stilt_r("out/by-id", smooth_factor=1)
    """
    root = Path(by_id_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a folder.")
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    runs = [p for p in folders if (p / f"{p.name}{_TRAJ}").is_file()]
    if not runs:
        raise FileNotFoundError(
            f"No folder in {root} holds a STILT-R <id>/<id>{_TRAJ} file. Give "
            "the by-id folder of a STILT-R or X-STILT project, such as "
            "my_project/out/by-id."
        )
    orphans = [p for p in folders if p not in runs and any(p.glob(f"*{_FOOT}"))]
    if orphans:
        warnings.warn(
            f"{len(orphans)} folders of {root} hold a footprint but no {_TRAJ} "
            f"file, and are skipped (the first is {orphans[0].name}).",
            stacklevel=2,
        )
    no_control = [p for p in runs if not (p / _CONTROL).is_file()]
    if no_control:
        raise FileNotFoundError(
            f"{len(no_control)} folders of {root} have no {_CONTROL} file, the "
            f"first {no_control[0]}. The receptor is read from the {_CONTROL} "
            "file STILT-R writes in every by-id folder."
        )
    rdata = _rdata()
    params = _rslurm_params(root, rdata)
    return (_read_simulation(folder, rdata, params, smooth_factor) for folder in runs)


def _rdata() -> Any:
    """Return the rdata package, or raise with the install hint."""
    try:
        import rdata
    except ImportError:
        raise ImportError(
            "Reading STILT-R output needs the rdata package: "
            'pip install "pystilt[stilt-r]"'
        ) from None
    return rdata


def _posixct(obj: Any, attrs: Mapping[str, Any]) -> pd.DatetimeIndex:
    """
    Return R ``POSIXct`` values, seconds since 1970-01-01 UTC, as naive UTC times.

    The value is UTC whatever its ``tzone`` attribute, which only sets how R
    prints it. Times are rounded to the millisecond, below which a float64
    count of seconds holds no more.
    """
    millis = np.round(np.asarray(obj, dtype=float) * 1000)
    return pd.DatetimeIndex(pd.to_datetime(millis, unit="ms"))


def _read_rds(path: Path, rdata: Any) -> dict[str, Any]:
    """Return a ``_traj.rds`` file as a dict, with its times as times and its particles as a DataFrame."""
    classes = rdata.conversion.DEFAULT_CLASS_MAP
    constructors = {
        **classes,
        # dplyr leaves the particles a tibble, a data frame to Python.
        "tbl_df": classes["data.frame"],
        "POSIXct": _posixct,
    }
    output = rdata.read_rds(path, constructor_dict=constructors)
    if not isinstance(output, Mapping) or not isinstance(
        output.get("particle"), pd.DataFrame
    ):
        raise ValueError(
            f"{path} is not a STILT-R trajectory file: it holds no particle table."
        )
    return {str(k): v for k, v in output.items()}


def _rslurm_params(
    by_id: Path, rdata: Any
) -> tuple[Path, dict[str, set[float]]] | None:
    """
    Return rslurm's ``params.RDS`` of a STILT-R project, and the values of its columns that are read.

    ``stilt_apply`` hands rslurm one row per simulation, holding every
    argument of ``simulation_step``, and rslurm saves them in
    ``_rslurm_<job name>/params.RDS``. The job name is the project folder's
    name, the folder ``out/`` is in. ``None`` when there is no such file, as
    after a run without Slurm. Columns of R functions (the
    ``before_trajec`` and ``before_footprint`` hooks) cannot be converted,
    so only the columns needed are.
    """
    project = by_id.resolve().parent.parent
    path = project / f"_rslurm_{project.name}" / "params.RDS"
    if not path.is_file():
        return None
    table = rdata.parser.parse_file(path, extension=".rds").object
    convert = rdata.conversion.SimpleConverter().convert
    attrs = rdata.conversion.convert_attrs(table, convert)
    names = [str(name) for name in attrs.get("names", [])]
    if "data.frame" not in [str(c) for c in attrs.get("class", [])]:
        raise ValueError(f"{path} is not a table of STILT-R runs.")
    values = {
        name: {float(v) for v in np.ravel(convert(table.value[names.index(name)]))}
        for name in _PARAMS_COLUMNS
        if name in names
    }
    return path, values


def _smooth_factor(
    foot: Path,
    n_hours: int,
    stored_params: Any,
    rslurm: tuple[Path, dict[str, set[float]]] | None,
    given: float | None,
) -> float:
    """
    Return the ``smooth_factor`` a STILT-R footprint was made with.

    It is rslurm's ``params.RDS`` value when that file's ``n_hours`` is the
    run's ``CONTROL`` value and its ``numpar`` the one HYSPLIT ran (the
    ``.rds`` file's ``params``, when it has them); otherwise the value given.
    """
    if rslurm is None:
        if given is None:
            raise ValueError(
                f"{foot}: the footprint's smooth_factor is not recorded. STILT-R's "
                "Slurm runs record it in _rslurm_<project>/params.RDS beside out/, "
                "and this project has none. Give the value from run_stilt.r: "
                "read_stilt_r(by_id_dir, smooth_factor=1)."
            )
        return given
    path, values = rslurm
    stored = {str(k): v for k, v in (stored_params or {}).items()}
    ran = {"n_hours": float(n_hours)}
    if "numpar" in stored:
        ran["numpar"] = float(np.ravel(stored["numpar"])[0])
    for name, value in ran.items():
        if values.get(name) != {value}:
            recorded = ", ".join(f"{v:g}" for v in sorted(values.get(name, ())))
            raise ValueError(
                f"{foot}: {path} and this run disagree: it gives {name} "
                f"{recorded or 'no value'}, and the run's files give {value:g}. "
                "That params.RDS is not from this run."
            )
    smooth = sorted(values.get("smooth_factor", ()))
    if len(smooth) != 1:
        raise ValueError(
            f"{path} gives smooth_factor {', '.join(f'{v:g}' for v in smooth) or 'no value'}; "
            "one value is needed to read the footprints."
        )
    if given is not None and given != smooth[0]:
        raise ValueError(
            f"smooth_factor={given:g} was given, but {path} records "
            f"{smooth[0]:g} for this project's runs."
        )
    return smooth[0]


def _control(path: Path) -> ControlFile:
    """Return a run's ``CONTROL`` file, read as PYSTILT reads its own."""
    # STILT-R runs HYSPLIT, so its CONTROL file is HYSPLIT's.
    from stilt.transport.hysplit.control import ControlFile

    try:
        return ControlFile.read(path)
    except (ValueError, IndexError) as error:
        raise ValueError(f"{path} cannot be read as a CONTROL file: {error}") from None


def _receptor(stored: Any, control: ControlFile, path: Path) -> Receptor:
    """
    Return the receptor of a run: its release points from ``CONTROL``, its time from the ``.rds``.

    STILT-R writes ``CONTROL``'s time to the minute; the ``.rds`` keeps the
    seconds. Every point is released at the first ``run_time``
    (``write_control``).
    """
    stored = {str(k): v for k, v in (stored or {}).items()}
    if "run_time" not in stored:
        raise ValueError(f"{path}: the receptor has no run_time.")
    time = pd.DatetimeIndex(stored["run_time"])[0]
    points = [(lon, lat, alt) for lat, lon, alt in control.receptor.coords()]
    return Receptor.from_points(time, points)


def _particles(stored: pd.DataFrame, receptor: Receptor, path: Path) -> pd.DataFrame:
    """Return STILT-R's particle table with PYSTILT's column names and a UTC ``time`` column."""
    particles = stored.rename(columns=_COLUMNS).reset_index(drop=True)
    try:
        check_particles(particles)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    for name in ("particle", "age"):
        particles[name] = particles[name].astype("float64")
    particles["time"] = pd.Timestamp(receptor.time) + pd.to_timedelta(
        particles["age"].to_numpy(), unit="min"
    )
    return particles


def _axis(centers: np.ndarray, name: str, path: Path) -> tuple[float, float, float]:
    """Return the lower edge, upper edge, and cell size of an axis of cell centers."""
    if centers.size < 2:
        raise ValueError(f"{path}: the footprint needs two {name} cells or more.")
    res = float((centers[-1] - centers[0]) / (centers.size - 1))
    return (
        round(float(centers[0]) - res / 2, 10),
        round(float(centers[-1]) + res / 2, 10),
        round(res, 10),
    )


def _footprint(
    path: Path, receptor: Receptor, backward: bool, smooth_factor: float
) -> xr.DataArray:
    """
    Return a STILT-R ``_foot.nc`` as a PYSTILT footprint.

    STILT-R writes ``foot`` with dimensions ``lon``, ``lat``, and ``time``
    (seconds since 1970-01-01 UTC), at cell centers. The grid is rebuilt from
    the centers and checked against them, so the footprint has the
    coordinates of a PYSTILT footprint on the same grid.
    """
    with xr.open_dataset(path, decode_times=False) as ds:
        foot = ds["foot"].load()
        crs = str(ds.attrs.get("crs", "+proj=longlat"))
    if not {"lon", "lat"} <= set(foot.dims):
        raise ValueError(
            f"{path}: only footprints on a longitude/latitude grid are read, and "
            f"this one is on {crs!r}. Open it with xarray.open_dataset."
        )
    foot = foot.transpose("time", "lat", "lon")
    lon, lat = foot["lon"].to_numpy(), foot["lat"].to_numpy()
    xmin, xmax, xres = _axis(lon, "lon", path)
    ymin, ymax, yres = _axis(lat, "lat", path)
    grid = Grid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xres=xres, yres=yres)
    x, y = grid.axes
    if not (
        x.shape == lon.shape
        and y.shape == lat.shape
        and np.allclose(x, lon, rtol=0, atol=1e-6 * xres)
        and np.allclose(y, lat, rtol=0, atol=1e-6 * yres)
    ):
        raise ValueError(f"{path}: the footprint's cells are not evenly spaced.")

    # Seconds since 1970 in UTC; the units say so with a trailing "Z".
    times = _posixct(foot["time"].to_numpy(), {})
    offsets = (times - pd.Timestamp(receptor.time)) / pd.Timedelta(hours=1)
    hours = np.round(offsets.to_numpy(dtype=float))
    if not np.allclose(offsets, hours, rtol=0, atol=1e-6):
        raise ValueError(
            f"{path}: the footprint's layers are not whole hours from the "
            f"receptor time {receptor.time}."
        )
    integrated = backward and hours.size == 1 and hours[0] == 0
    config = FootprintConfig(
        grid=grid, smooth_factor=smooth_factor, time_integrate=bool(integrated)
    )
    values = foot.to_numpy().astype("float64")
    return _footprint_array(values, hours.astype(int), receptor, config, "", x, y)


def _read_simulation(
    folder: Path,
    rdata: Any,
    rslurm: tuple[Path, dict[str, set[float]]] | None,
    smooth_factor: float | None,
) -> tuple[Receptor, pd.DataFrame, xr.DataArray | None]:
    """Return the receptor, particles, and footprint of one ``by-id`` folder."""
    traj = folder / f"{folder.name}{_TRAJ}"
    output = _read_rds(traj, rdata)
    control = _control(folder / _CONTROL)
    receptor = _receptor(output.get("receptor"), control, traj)
    particles = _particles(output["particle"], receptor, traj)
    foot_path = folder / f"{folder.name}{_FOOT}"
    if not foot_path.is_file():
        return receptor, particles, None
    smooth = _smooth_factor(
        foot_path, control.n_hours, output.get("params"), rslurm, smooth_factor
    )
    # STILT-R reads the run's direction the same way (calc_footprint).
    backward = bool(np.median(particles["age"]) < 0)
    return receptor, particles, _footprint(foot_path, receptor, backward, smooth)


__all__ = ["read_stilt_r"]
