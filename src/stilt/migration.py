"""
Reading what STILT-R and X-STILT wrote.

STILT-R writes each simulation to its own folder of ``out/by-id``, named by
the simulation id. X-STILT runs STILT-R's code and writes the same layout::

    out/by-id/
      201507050000_-111.85_40.77_5/
        201507050000_-111.85_40.77_5_traj.rds   receptor, particles, HYSPLIT settings
        201507050000_-111.85_40.77_5_foot.nc    footprint

:func:`read_stilt_r` reads these folders as PYSTILT's receptors, particle
tables, and footprints. The ``.rds`` files are read with the
`rdata <https://github.com/vnmabus/rdata>`_ package, which needs no R
(``pip install "pystilt[stilt-r]"``).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from stilt.footprint.config import FootprintConfig
from stilt.footprint.io import _footprint_array
from stilt.particles import check_particles
from stilt.receptors import ColumnReceptor, Receptor
from stilt.spatial import Grid

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


def read_stilt_r(
    by_id_dir: str | Path,
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

    The particle columns get PYSTILT's names: ``indx``, ``time``, ``long``,
    ``lati``, and ``xhgt`` become ``particle``, ``age`` (minutes since
    release), ``lon``, ``lat``, and ``release_height``. ``time`` is then the
    UTC time of each row, the receptor time plus ``age``. Every other column
    keeps its STILT-R name, ``foot_no_hnf_dilution`` included: the
    near-field correction STILT-R applied stays as it is.

    The footprint is ``<id>_foot.nc``, on the grid its file gives, with
    layers stamped at the receptor time plus whole hours, as STILT-R stamps
    them. Its settings hold that grid and ``time_integrate`` (true for a
    backward run with one layer at the receptor time). The file does not
    record ``smooth_factor``, so the settings give the default of 1, which
    is STILT-R's default too. Other ``_foot.nc`` files in the folder, such as
    X-STILT's footprints at other resolutions, are not read.

    The receptor's coordinates are the ones HYSPLIT ran, to the 15
    significant digits STILT-R writes in its CONTROL file and simulation id,
    so a point receptor's id is the folder's name. A receptor with one
    release point is a :class:`~stilt.PointReceptor`. Several heights at one
    place are a :class:`~stilt.ColumnReceptor` from the lowest to the
    highest, as STILT-R spreads its particles; release points at different
    places, such as X-STILT's slant columns, are a
    :class:`~stilt.MultiPointReceptor`.

    Parameters
    ----------
    by_id_dir : str or Path
        The ``by-id`` folder, such as ``my_project/out/by-id``.

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
        If *by_id_dir* does not exist, or no folder in it holds a
        ``_traj.rds`` file.
    ValueError
        When a simulation is read, if its files are not as STILT-R writes
        them: a ``_traj.rds`` without the particle columns or the receptor,
        or a footprint on a projected grid (only longitude/latitude grids
        are read) or with layers that are not whole hours from the
        receptor time.

    Warns
    -----
    UserWarning
        When folders hold a ``_foot.nc`` but no ``_traj.rds``. They are
        skipped: the receptor is read from the ``.rds``.

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
    rdata = _rdata()
    return (_read_simulation(folder, rdata) for folder in runs)


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
    if not isinstance(output, Mapping) or "particle" not in output:
        raise ValueError(
            f"{path} is not a STILT-R trajectory file: it holds no particle table."
        )
    return {str(k): v for k, v in output.items()}


def _values(value: Any) -> np.ndarray:
    """
    Return a receptor coordinate as HYSPLIT got it: a flat float array, to 15 significant digits.

    STILT-R writes the release points into HYSPLIT's CONTROL file, and its
    simulation id, with R's ``paste``, which keeps 15 significant digits.
    X-STILT may store a coordinate as a list of vectors.
    """
    if isinstance(value, (list, tuple)):
        parts = [np.ravel(np.asarray(v, dtype=float)) for v in value]
        flat = np.concatenate(parts) if parts else np.array([], dtype=float)
    else:
        flat = np.ravel(np.asarray(value, dtype=float))
    return np.array([float(f"{v:.15g}") for v in flat])


def _receptor(stored: Any, path: Path) -> Receptor:
    """Return the receptor a ``_traj.rds`` file records (``run_time``, ``long``, ``lati``, ``zagl``)."""
    if not isinstance(stored, Mapping):
        raise ValueError(f"{path} records no receptor.")
    stored = {str(k): v for k, v in stored.items()}
    missing = [k for k in ("run_time", "long", "lati", "zagl") if k not in stored]
    if missing:
        raise ValueError(f"{path}: the receptor has no {', '.join(missing)}.")
    times = pd.DatetimeIndex(stored["run_time"])
    if len(times) == 0 or (times != times[0]).any():
        raise ValueError(f"{path}: the receptor needs one run_time, not {times}.")
    lon, lat, zagl = (_values(stored[k]) for k in ("long", "lati", "zagl"))
    n = max(lon.size, lat.size, zagl.size)
    try:
        lon, lat, zagl = (np.broadcast_to(v, n) for v in (lon, lat, zagl))
    except ValueError:
        raise ValueError(
            f"{path}: the receptor's long, lati, and zagl have {lon.size}, "
            f"{lat.size}, and {zagl.size} values."
        ) from None
    time = times[0]
    if n > 1 and np.unique(lon).size == 1 and np.unique(lat).size == 1:
        # STILT-R releases a column's particles evenly from the lowest
        # height to the highest (its xhgt), however many heights it is given.
        return ColumnReceptor(
            time=time,
            longitude=float(lon[0]),
            latitude=float(lat[0]),
            bottom=float(zagl.min()),
            top=float(zagl.max()),
        )
    return Receptor.from_points(time, zip(lon, lat, zagl, strict=True))


def _particles(stored: Any, receptor: Receptor, path: Path) -> pd.DataFrame:
    """Return STILT-R's particle table with PYSTILT's column names and a UTC ``time`` column."""
    if not isinstance(stored, pd.DataFrame):
        raise ValueError(f"{path}: the particles are not a table.")
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


def _footprint(path: Path, receptor: Receptor, backward: bool) -> xr.DataArray:
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
    config = FootprintConfig(grid=grid, time_integrate=bool(integrated))
    values = foot.to_numpy().astype("float64")
    return _footprint_array(values, hours.astype(int), receptor, config, "", x, y)


def _read_simulation(
    folder: Path, rdata: Any
) -> tuple[Receptor, pd.DataFrame, xr.DataArray | None]:
    """Return the receptor, particles, and footprint of one ``by-id`` folder."""
    traj = folder / f"{folder.name}{_TRAJ}"
    output = _read_rds(traj, rdata)
    receptor = _receptor(output.get("receptor"), traj)
    particles = _particles(output["particle"], receptor, traj)
    foot_path = folder / f"{folder.name}{_FOOT}"
    if not foot_path.is_file():
        return receptor, particles, None
    # STILT-R reads the run's direction the same way (calc_footprint).
    backward = bool(np.median(particles["age"]) < 0)
    return receptor, particles, _footprint(foot_path, receptor, backward)


__all__ = ["read_stilt_r"]
