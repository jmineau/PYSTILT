"""
Reading what STILT-R and X-STILT wrote.

STILT-R writes each simulation to its own folder of ``out/by-id``, named by
the simulation id. X-STILT runs STILT-R's code and writes the same layout::

    my_project/
      out/by-id/
        201507050000_-111.85_40.77_5/
          201507050000_-111.85_40.77_5_traj.rds   receptor time, particles
          201507050000_-111.85_40.77_5_foot.nc    footprint
          CONTROL                                 the release points, run length, met files
          SETUP.CFG                               HYSPLIT's other settings
          stilt.log                               HYSPLIT's log, and STILT-R's notes
      _rslurm_my_project/
        params.RDS                                every run's settings (Slurm runs only)

:func:`read_stilt_r` reads these folders as PYSTILT's receptors, particle
tables, and footprints. :func:`import_stilt_r` writes them into an output
directory, in folders named by the settings the files record, so a project
reads them as its own (:class:`stilt.transport.stilt_r.StiltRModel`). The
``.rds`` files are read with the `rdata <https://github.com/vnmabus/rdata>`_
package, which needs no R (``pip install "pystilt[stilt-r]"``).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from stilt.config import VARIANT_NAME_RE, Variant
from stilt.footprint.config import FootprintConfig
from stilt.footprint.io import _footprint_array
from stilt.identity import footprint_settings
from stilt.output import Kind, Output
from stilt.particles import check_particles
from stilt.receptors import Receptor
from stilt.spatial import Grid
from stilt.transport import ModelInfo
from stilt.transport.hysplit.config import WIND_ERROR_SETTINGS, ZI_ERROR_SETTINGS
from stilt.transport.hysplit.namelist import read_namelist
from stilt.transport.stilt_r import StiltRConfig, StiltRMetConfig, StiltRModel

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
_SETUP = "SETUP.CFG"
_LOG = "stilt.log"

#: What STILT-R's simulation_step writes to stilt.log when no particle reached
#: the footprint grid, and it writes no footprint.
_EMPTY_FOOTPRINT = "No non-zero footprint values found within the footprint domain."

#: The entries of STILT-R's SETUP.CFG that are no setting of their own: the
#: number of VARSIWANT columns, the flags for the WINDERR and ZICONTROL
#: files, and KMSL, which the import requires to be 0 (heights above
#: ground).
_SETUP_ONLY = ("ivmax", "winderrtf", "zicontroltf", "kmsl")

#: The transport-error settings, as HYSPLIT, STILT-R and PYSTILT name them.
_ERROR_SETTINGS = WIND_ERROR_SETTINGS + ZI_ERROR_SETTINGS

#: The columns of rslurm's ``params.RDS`` that are read: the footprint's
#: smoothing, and two settings that must match the run's own files.
_PARAMS_COLUMNS = ("smooth_factor", "n_hours", "numpar")


def read_stilt_r(
    by_id_dir: str | Path,
    smooth_factor: float | None = None,
    error: bool = False,
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
    error : bool, default False
        Read the second set of particles a run with transport error saved
        (``particle_error``), the run with perturbed winds, in place of the
        first. STILT-R makes no footprint of it, so the footprint is
        ``None``.

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
        ``smooth_factor`` other than *smooth_factor* or more than one. With
        *error*, if a ``.rds`` holds no ``particle_error``.

    Warns
    -----
    UserWarning
        When folders hold a ``_foot.nc`` but no ``_traj.rds``. They are
        skipped: the particles are read from the ``.rds``.

    Notes
    -----
    A run with transport error stores a second set of particles in the same
    ``.rds`` (``particle_error``), which *error* reads. The HYSPLIT
    settings in the ``.rds`` (``params``) are not returned;
    :func:`import_stilt_r` records the settings of each run. X-STILT weights
    ``foot`` by the averaging kernel and pressure weighting after it saves
    the ``.rds``, so its particles are unweighted and its footprint is
    weighted.

    Examples
    --------
    >>> for receptor, particles, foot in stilt.migration.read_stilt_r(
    ...     "out/by-id"
    ... ):  # doctest: +SKIP
    ...     print(receptor.id, len(particles), foot is not None)

    A project run on one machine has no ``params.RDS``; give the
    ``smooth_factor`` from its ``run_stilt.r``:

    >>> sims = stilt.migration.read_stilt_r(
    ...     "out/by-id", smooth_factor=1
    ... )  # doctest: +SKIP

    The particles of the transport-error run:

    >>> for receptor, perturbed, _ in stilt.migration.read_stilt_r(  # doctest: +SKIP
    ...     "out/by-id", error=True
    ... ):
    ...     print(receptor.id, len(perturbed))
    """
    root, runs = _runs(by_id_dir)
    rdata = _rdata()
    params = _rslurm_params(root, rdata)
    return (
        _read_simulation(folder, rdata, params, smooth_factor, error) for folder in runs
    )


def _runs(by_id_dir: str | Path) -> tuple[Path, list[Path]]:
    """
    Return a ``by-id`` folder and its run folders, those that hold ``<id>_traj.rds``, in name order.

    Raises
    ------
    FileNotFoundError
        If *by_id_dir* is not a folder, holds no run, or a run has no
        ``CONTROL`` file.
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
            stacklevel=3,
        )
    no_control = [p for p in runs if not (p / _CONTROL).is_file()]
    if no_control:
        raise FileNotFoundError(
            f"{len(no_control)} folders of {root} have no {_CONTROL} file, the "
            f"first {no_control[0]}. The receptor is read from the {_CONTROL} "
            "file STILT-R writes in every by-id folder."
        )
    return root, runs


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
    error: bool = False,
) -> tuple[Receptor, pd.DataFrame, xr.DataArray | None]:
    """Return the receptor, particles, and footprint of one ``by-id`` folder, or the receptor and its error particles."""
    traj = folder / f"{folder.name}{_TRAJ}"
    output = _read_rds(traj, rdata)
    control = _control(folder / _CONTROL)
    receptor = _receptor(output.get("receptor"), control, traj)
    if error:
        return receptor, _error_particles(output, receptor, traj), None
    particles = _particles(output["particle"], receptor, traj)
    foot = _folder_footprint(
        folder, receptor, particles, control, output, rslurm, smooth_factor
    )
    return receptor, particles, foot


def _error_particles(
    output: Mapping[str, Any], receptor: Receptor, path: Path
) -> pd.DataFrame:
    """Return the particles of a run's transport-error set, with PYSTILT's column names."""
    stored = output.get("particle_error")
    if not isinstance(stored, pd.DataFrame):
        raise ValueError(
            f"{path} holds no transport-error particles (particle_error): "
            "STILT-R ran it without wind or mixed-layer errors."
        )
    return _particles(stored, receptor, path)


def _folder_footprint(
    folder: Path,
    receptor: Receptor,
    particles: pd.DataFrame,
    control: ControlFile,
    output: Mapping[str, Any],
    rslurm: tuple[Path, dict[str, set[float]]] | None,
    smooth_factor: float | None,
) -> xr.DataArray | None:
    """Return a folder's ``<id>_foot.nc`` as a PYSTILT footprint, or ``None`` when it has none."""
    foot_path = folder / f"{folder.name}{_FOOT}"
    if not foot_path.is_file():
        return None
    smooth = _smooth_factor(
        foot_path, control.n_hours, output.get("params"), rslurm, smooth_factor
    )
    # STILT-R reads the run's direction the same way (calc_footprint).
    backward = bool(np.median(particles["age"]) < 0)
    return _footprint(foot_path, receptor, backward, smooth)


def import_stilt_r(
    by_id_dir: str | Path,
    output_dir: str | Path,
    version: str,
    *,
    name: str = "stilt-r",
    smooth_factor: float | None = None,
    met_path: str | None = None,
) -> list[Receptor]:
    """
    Write the runs of a STILT-R or X-STILT ``by-id`` folder into a PYSTILT output directory.

    Each run's particles and footprint, as :func:`read_stilt_r` reads them,
    go to the folders of a variant with ``model: stilt-r``
    (:class:`~stilt.transport.stilt_r.StiltRModel`). A project whose
    ``config.yaml`` has that variant reads them as its own results:
    ``project.status``, ``project.particles``, ``project.footprints``, and
    ``project.jacobian``. The import prints the variant and its met, to paste
    into ``config.yaml``, and returns the receptors, for ``receptors.csv``.

    Every folder of an output directory is named by the hash of its
    settings, and an imported run's settings are only what STILT-R's files
    record, never a default standing in for a value they do not give:

    - HYSPLIT's settings from each run's ``SETUP.CFG``; ``n_hours``,
      ``w_option``, ``z_top``, and ``emisshrs`` from its ``CONTROL``; and
      ``ziscale`` from its ``ZICONTROL`` file when ``SETUP.CFG`` says
      STILT-R wrote one (``ZICONTROLTF``), else 1 (no scaling).
    - ``hnf_plume`` from the particles: STILT-R's near-field correction adds
      the ``foot_no_hnf_dilution`` column.
    - The met: the directory the ``CONTROL`` files name, as they name it.
      Each run's met files are recorded in its particle file, and need not
      exist any more.
    - The footprint's grid and ``time_integrate`` from ``<id>_foot.nc``, and
      its ``smooth_factor`` from ``params.RDS`` or *smooth_factor*, as
      :func:`read_stilt_r` reads them.
    - The STILT-R version, which no file records: *version*.

    The second set of particles of a run with transport error
    (``particle_error``) goes to the folder of a second variant,
    ``<name>-err``, with the wind and mixed-layer error settings STILT-R
    saved with it (``particle_error_params``). It has particles only:
    STILT-R makes no footprint of them. A run whose ``stilt.log`` says that
    no particle reached the footprint grid gets an empty footprint, as a
    PYSTILT run would. Results already in the output directory are kept, so
    an import can be run again after more STILT-R runs finish.

    Parameters
    ----------
    by_id_dir : str or Path
        The ``by-id`` folder, such as ``my_project/out/by-id``.
    output_dir : str or Path
        The output directory, as ``output:`` in ``config.yaml`` names it.
    version : str
        The STILT-R version that made the runs, such as the commit of
        uataq/stilt the project was made from (``git rev-parse --short
        HEAD`` in a STILT-R project folder).
    name : str, default "stilt-r"
        Name of the variant and of its met. The transport-error variant is
        ``<name>-err``.
    smooth_factor : float, optional
        The ``smooth_factor`` STILT-R made the footprints with, for a project
        without ``params.RDS``, as for :func:`read_stilt_r`.
    met_path : str, optional
        The met directory to record when the ``CONTROL`` files name more than
        one, such as STILT-R's ``met_path`` with its files in yearly
        folders. Every directory they name must be in it.

    Returns
    -------
    list of Receptor
        The receptors of the runs, in folder order, for
        :meth:`stilt.Project.add_receptors`.

    Raises
    ------
    ImportError
        If the rdata package is not installed.
    FileNotFoundError
        As :func:`read_stilt_r` raises it, or if a run has no ``SETUP.CFG``
        or no ``ZICONTROL`` file that its ``SETUP.CFG`` says it has.
    ValueError
        As :func:`read_stilt_r` raises it, or if *name* is not a variant
        name; if two runs were made with different settings, which is said
        by name (import each set from a ``by-id`` folder of its own, under a
        name of its own); if ``SETUP.CFG`` holds a setting PYSTILT does not
        have, or ``KMSL`` other than 0 (heights above sea level); if the
        ``CONTROL`` files name several met directories and *met_path* is not
        given or does not hold them all; or if ``SETUP.CFG`` and the
        ``.rds`` disagree on whether a run had transport error.

    Examples
    --------
    >>> receptors = stilt.migration.import_stilt_r(  # doctest: +SKIP
    ...     "my_project/out/by-id", "output", version="1d936c1"
    ... )
    mets:
      stilt-r:
        met_path: /data/hrrr
    variants:
      stilt-r:
        model: stilt-r
        met: stilt-r
        version: 1d936c1
        capemin: -1
        ...
    >>> project = stilt.Project.init("my_project_py", receptors, output="output", ...)  # doctest: +SKIP
    """
    if not VARIANT_NAME_RE.fullmatch(name):
        raise ValueError(
            f"name {name!r} must be a variant name, matching {VARIANT_NAME_RE.pattern}."
        )
    root, runs = _runs(by_id_dir)
    controls = {folder: _control(folder / _CONTROL) for folder in runs}
    recorded = {folder: _hysplit_settings(folder, controls[folder]) for folder in runs}
    first = runs[0]
    for folder in runs[1:]:
        _agree(
            recorded[first][0], recorded[folder][0], "HYSPLIT settings", first, folder
        )
    met = StiltRMetConfig(met_path=_met_path(list(controls.values()), met_path))
    rdata = _rdata()
    rslurm = _rslurm_params(root, rdata)
    output = Output(output_dir)

    # The first run's variants, and its folder; every later run must have
    # the same settings.
    made: dict[str, tuple[Variant, Path]] = {}
    # The first footprint's settings, and its folder.
    first_foot: tuple[FootprintConfig, Path] | None = None
    empty: list[Receptor] = []
    receptors = []
    for folder in runs:
        hysplit, winderrtf = recorded[folder]
        control = controls[folder]
        traj = folder / f"{folder.name}{_TRAJ}"
        stored = _read_rds(traj, rdata)
        receptor = _receptor(stored.get("receptor"), control, traj)
        receptors.append(receptor)
        log = folder / _LOG

        particles = _particles(stored["particle"], receptor, traj)
        settings = {
            **hysplit,
            **dict.fromkeys(_ERROR_SETTINGS),
            "hnf_plume": "foot_no_hnf_dilution" in particles.columns,
        }
        base = _variant(made, name, name, met, settings, version, folder)
        _write_particles(output, base, receptor, particles, control, log)

        foot = _folder_footprint(
            folder, receptor, particles, control, stored, rslurm, smooth_factor
        )
        if foot is not None:
            first_foot = first_foot or (foot.stilt.config, folder)
            _agree(
                footprint_settings(first_foot[0], None),
                footprint_settings(foot.stilt.config, None),
                "footprint settings",
                first_foot[1],
                folder,
            )
            with_foot = dataclasses.replace(base, footprint=first_foot[0])
            if _missing(output, "footprints", with_foot, receptor.id):
                output.write_footprint(with_foot, foot)
        elif log.is_file() and _EMPTY_FOOTPRINT in log.read_text(errors="replace"):
            empty.append(receptor)

        if stored.get("particle_error") is None:
            if winderrtf:
                raise ValueError(
                    f"{folder / _SETUP} says the run had transport error "
                    f"(WINDERRTF={winderrtf}), and {traj.name} holds no "
                    "particle_error."
                )
            continue
        errors, recorded_tf = _error_settings(stored.get("particle_error_params"), traj)
        if recorded_tf != winderrtf:
            raise ValueError(
                f"{folder / _SETUP} gives WINDERRTF={winderrtf}, and the "
                f"particle_error_params of {traj.name} {recorded_tf}."
            )
        perturbed = _error_particles(stored, receptor, traj)
        error_settings = {
            **hysplit,
            **errors,
            "hnf_plume": "foot_no_hnf_dilution" in perturbed.columns,
        }
        error = _variant(
            made, f"{name}-err", name, met, error_settings, version, folder
        )
        _write_particles(output, error, receptor, perturbed, control, log)

    printed = [made[name][0]]
    if first_foot is not None:
        printed[0] = dataclasses.replace(printed[0], footprint=first_foot[0])
        for receptor in empty:
            if _missing(output, "footprints", printed[0], receptor.id):
                output.write_empty_footprint(printed[0], receptor)
    if f"{name}-err" in made:
        printed.append(made[f"{name}-err"][0])
    config = {
        "mets": {name: met.settings()},
        "variants": {v.name: _variant_entry(v) for v in printed},
    }
    print(yaml.safe_dump(config, sort_keys=False, default_flow_style=None), end="")
    return receptors


def _hysplit_settings(folder: Path, control: ControlFile) -> tuple[dict[str, Any], int]:
    """
    Return the HYSPLIT settings a run's ``SETUP.CFG``, ``CONTROL``, and ``ZICONTROL`` record, and its ``WINDERRTF``.

    STILT-R writes ``SETUP.CFG`` again for its transport-error run, the same
    but for ``WINDERRTF``, so that flag is returned on its own. STILT-R
    writes no ``SEED``: HYSPLIT seeded its random numbers itself, which is
    ``seed: null``.
    """
    path = folder / _SETUP
    if not path.is_file():
        raise FileNotFoundError(
            f"{folder} has no {_SETUP}, which holds the HYSPLIT settings of its run."
        )
    setup = read_namelist(path)
    unknown = sorted(set(setup) - set(StiltRConfig.model_fields) - set(_SETUP_ONLY))
    if unknown:
        raise ValueError(
            f"{path} holds {', '.join(k.upper() for k in unknown)}, which "
            "PYSTILT has no setting for, so the run's settings cannot be "
            "recorded in full."
        )
    if setup.get("kmsl", 0) != 0:
        raise ValueError(
            f"{path} gives KMSL={setup['kmsl']}: release heights above sea "
            "level. The import reads heights above ground only."
        )
    settings = {k: v for k, v in setup.items() if k not in _SETUP_ONLY}
    settings.setdefault("seed", None)
    settings.update(
        n_hours=control.n_hours,
        w_option=control.w_option,
        z_top=control.z_top,
        emisshrs=control.emisshrs,
        ziscale=_ziscale(folder, setup) if setup.get("zicontroltf") else 1.0,
    )
    return settings, int(setup.get("winderrtf", 0))


def _ziscale(folder: Path, setup: Mapping[str, Any]) -> list[float]:
    """Return the hourly mixed-layer factors of a run's ``ZICONTROL`` file: a count, then the factors."""
    path = folder / "ZICONTROL"
    if not path.is_file():
        raise FileNotFoundError(
            f"{folder / _SETUP} gives ZICONTROLTF={setup['zicontroltf']}, and "
            f"{folder} has no ZICONTROL file."
        )
    values = path.read_text().split()
    factors = [float(v) for v in values[1:]]
    if not values or int(float(values[0])) != len(factors):
        raise ValueError(
            f"{path} does not give as many factors as its first line says."
        )
    return factors


def _error_settings(params: Any, path: Path) -> tuple[dict[str, float | None], int]:
    """
    Return the transport-error settings STILT-R saved with a run (``particle_error_params``), and its ``winderrtf``.

    An unset setting is R's ``NA``, which is ``None``.
    """
    stored = {str(k): v for k, v in (params or {}).items()}
    missing = [k for k in (*_ERROR_SETTINGS, "winderrtf") if k not in stored]
    if missing:
        raise ValueError(f"{path}: particle_error_params has no {', '.join(missing)}.")
    values: dict[str, float | None] = {}
    for key in (*_ERROR_SETTINGS, "winderrtf"):
        value = np.ma.masked_invalid(
            np.ma.ravel(np.ma.asarray(stored[key], dtype=float))
        )
        values[key] = None if value.size == 0 or value.mask.any() else float(value[0])
    winderrtf = values.pop("winderrtf")
    return values, 0 if winderrtf is None else int(winderrtf)


def _met_path(controls: list[ControlFile], given: str | None) -> str:
    """Return the met directory to record: the one the runs' ``CONTROL`` files name, or *given*, which must hold them all."""
    directories = sorted({str(f.parent) for c in controls for f in c.met_files})
    if given is None:
        if len(directories) == 1:
            return directories[0]
        raise ValueError(
            f"The runs' CONTROL files name {len(directories)} met directories "
            f"({', '.join(directories[:3])}{', ...' if len(directories) > 3 else ''}). "
            "Give the folder that holds them as met_path=, such as the "
            "met_path of run_stilt.r."
        )
    outside = [d for d in directories if not Path(d).is_relative_to(given)]
    if outside:
        raise ValueError(
            f"{outside[0]}, a met directory a CONTROL file names, is not in "
            f"met_path {given}."
        )
    return str(Path(given))


def _variant(
    made: dict[str, tuple[Variant, Path]],
    name: str,
    met_name: str,
    met: StiltRMetConfig,
    settings: dict[str, Any],
    version: str,
    folder: Path,
) -> Variant:
    """
    Return the variant of a run's settings, checked against the first run's.

    The first run's variant is kept in *made* under *name*.

    Raises
    ------
    ValueError
        If the settings are not valid, or not the first run's.
    """
    try:
        config = StiltRConfig.model_validate({**settings, "version": version})
    except ValueError as error:
        raise ValueError(
            f"{folder}: the run's settings are not valid HYSPLIT settings: {error}"
        ) from None
    variant = Variant(
        name=name,
        met=met_name,
        met_config=met,
        transport=config,
        model=ModelInfo(name=StiltRModel.name, version=StiltRModel().version(config)),
    )
    if name not in made:
        made[name] = (variant, folder)
    first, first_folder = made[name]
    _agree(first.run_settings, variant.run_settings, "settings", first_folder, folder)
    return first


def _agree(
    first: Mapping[str, Any],
    other: Mapping[str, Any],
    what: str,
    first_folder: Path,
    folder: Path,
) -> None:
    """Raise when two runs' settings differ, naming the first setting that does."""
    keys = sorted(
        k for k in first.keys() | other.keys() if first.get(k) != other.get(k)
    )
    if keys:
        key = keys[0]
        raise ValueError(
            f"{folder.name} and {first_folder.name} were made with different "
            f"{what}: {key} is {other.get(key)!r} and {first.get(key)!r}. The "
            "runs of one import share their settings: import each set from a "
            "by-id folder of its own, under a name of its own."
        )


def _missing(output: Output, kind: Kind, variant: Variant, receptor_id: str) -> bool:
    """Whether a receptor's result of *kind* is not in the output directory yet."""
    path = output.path(kind, variant, receptor_id)
    return path is None or not path.exists()


def _write_particles(
    output: Output,
    variant: Variant,
    receptor: Receptor,
    particles: pd.DataFrame,
    control: ControlFile,
    log: Path,
) -> None:
    """Write a run's particles, with the met files its ``CONTROL`` names and its ``stilt.log``, unless they are there."""
    if not _missing(output, "particles", variant, receptor.id):
        return
    output.write_particles(variant, receptor, particles, control.met_files)
    if log.is_file():
        output.write_log(variant, receptor.id, log.read_text(errors="replace"))


def _variant_entry(variant: Variant) -> dict[str, Any]:
    """Return a variant as ``config.yaml`` names its folders: the keys of ``_settings.yaml``, flat."""
    entry: dict[str, Any] = {
        "model": variant.model.name,
        "met": variant.met,
        "version": variant.model.version,
    }
    settings = variant.transport.settings()
    entry.update({k: settings[k] for k in sorted(settings)})
    if variant.footprint is None:
        entry["grid"] = None
    else:
        entry.update(variant.footprint.model_dump(mode="json"))
    return entry


__all__ = ["import_stilt_r", "read_stilt_r"]
