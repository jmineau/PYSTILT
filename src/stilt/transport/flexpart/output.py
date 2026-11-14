"""
Read FLEXPART 11's particle output (``partoutput_*.nc``) as a particle table.

FLEXPART 11 writes its particles to NetCDF when ``IPOUT`` is 1 (every
output step) or 2 (the end only) in ``COMMAND``. Each variable on
(particle, time) is one field that PARTOPTIONS turned on: FLEXPART writes
(time, particle) in Fortran's order, which NetCDF stores reversed. The file is
``partoutput_<YYYYMMDDHHMMSS>_init.nc``, and a long run starts another,
``partoutput_<YYYYMMDDHHMMSS>.nc``, when the first one grows past FLEXPART's
size limit or at a restart, so a run can leave several.

What FLEXPART's code (``netcdf_output_mod.f90``, ``output_mod.f90``) does,
which the reader relies on:

- ``time`` is whole seconds from the start of the run, which is ``IEDATE``
  ``IETIME`` for a backward run, so its values are zero or negative there.
- ``particle`` numbers the particles from 1 in the order they were released,
  across all the files of a run.
- Every field is float32 with ``_FillValue`` -1. A particle not yet
  released, released in that same step, or gone for more than one output
  step has -1 in every field at that time.
- ``z`` is height above ground in meters, ``prs`` pressure in Pa.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr

#: FLEXPART's names for the particle table's columns, and for the columns a
#: HYSPLIT run names otherwise; other variables keep FLEXPART's names.
PARTICLE_COLUMNS = {
    "lon": "lon",
    "lat": "lat",
    "z": "zagl",
    "hmix": "mlht",
    "rho": "dens",
    "prs": "pres",
    "T": "temp",
    "sh": "sphu",
    "to": "zsfc",
}

#: Variables whose units the particle table states otherwise, with the factor
#: from FLEXPART's units: ``pres`` is in hPa.
_SCALE = {"prs": 0.01}

#: The fill value of every FLEXPART particle field.
_FILL = -1.0


def read_partoutput(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    receptor_time: dt.datetime,
) -> pd.DataFrame:
    """
    Read FLEXPART 11 ``partoutput`` files as one particle table.

    Each row is a particle at an output time when FLEXPART wrote it, so
    a particle not yet released or already gone has no row. FLEXPART skips
    a particle in the step it is released, so there is no release row at
    ``age = 0``.

    ``lon``, ``lat``, and ``z`` become ``lon``, ``lat``, and ``zagl``; the
    other fields that HYSPLIT also writes take HYSPLIT's names
    (:data:`PARTICLE_COLUMNS`), so ``hmix`` is ``mlht`` and ``rho`` is
    ``dens``, and ``pres`` is in hPa. Every other field keeps FLEXPART's
    name, such as ``pv`` or ``hmix_av``, with ``NaN`` where FLEXPART wrote
    its fill value.

    Parameters
    ----------
    paths : path-like or iterable of path-like
        The ``partoutput_*.nc`` files of one run, in any order.
    receptor_time : datetime.datetime
        Time the particles were released, which ``age`` counts from. For a
        backward run it is usually the run's ``IEDATE`` and ``IETIME``.

    Returns
    -------
    pandas.DataFrame
        ``particle``, ``age`` (minutes, negative for a backward run), ``lon``,
        ``lat``, ``zagl``, then the other fields, ordered by output time
        as FLEXPART wrote them and by particle within a time.

    Raises
    ------
    FileNotFoundError
        If a file does not exist.
    ValueError
        If no file is given, a file has no ``lon``, ``lat``, or ``z``, a
        field is on FLEXPART's grid rather than per particle (the output
        of a domain-filling run, ``MDOMAINFILL`` 1), or an output time is
        not a whole number of minutes from *receptor_time*.

    Examples
    --------
    >>> particles = read_partoutput(
    ...     "output/partoutput_20230601180000_init.nc",
    ...     receptor_time=dt.datetime(2023, 6, 1, 18),
    ... )  # doctest: +SKIP
    """
    if isinstance(paths, str | os.PathLike):
        paths = [paths]
    tables = [_read_one(os.fspath(p), receptor_time) for p in paths]
    if not tables:
        raise ValueError("No partoutput files to read.")
    particles = pd.concat(tables, ignore_index=True)
    # Files of one run cover times in the order they were written; sort so
    # the order does not depend on the order of *paths*.
    order = np.lexsort(
        (particles["particle"].to_numpy(), np.abs(particles["age"].to_numpy()))
    )
    return particles.take(order).reset_index(drop=True)


def _read_one(path: str, receptor_time: dt.datetime) -> pd.DataFrame:
    """Read one ``partoutput`` file, leaving out rows FLEXPART filled."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # mask_and_scale=False keeps the -1 fill values: a longitude can be -1,
    # so which rows are filled is read from z alone.
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        missing = [name for name in ("lon", "lat", "z") if name not in ds]
        if missing:
            raise ValueError(
                f"{os.path.basename(path)} has no {', '.join(missing)}: set "
                "LONGITUDE, LATITUDE, and HEIGHT to .true. in PARTOPTIONS."
            )
        fields: dict[str, np.ndarray] = {}
        for key, var in ds.data_vars.items():
            name = str(key)
            if var.dims == ("particle", "time"):
                fields[name] = var.values.T
            elif {"longitude", "latitude"} & set(var.dims):
                raise ValueError(
                    f"{os.path.basename(path)} has {name!r} on FLEXPART's grid, "
                    "as a domain-filling run (MDOMAINFILL 1) writes it. "
                    "PYSTILT reads per-particle fields only."
                )
        times = ds["time"].values.astype("datetime64[ns]")
        numbers = ds["particle"].values

    age = (times - np.datetime64(receptor_time, "ns")) / np.timedelta64(1, "m")
    if not np.allclose(age, np.round(age)):
        raise ValueError(
            f"{os.path.basename(path)} has output times that are not whole "
            f"minutes from the receptor time {receptor_time}."
        )
    kept = fields["z"] != _FILL
    rows_time, rows_particle = np.nonzero(kept)

    columns: dict[str, np.ndarray] = {
        "particle": numbers[rows_particle].astype(np.int32),
        "age": np.round(age[rows_time]).astype(np.int32),
    }
    for name, values in fields.items():
        out = values[kept].astype(np.float64)
        if name not in ("lon", "lat", "z"):
            out[out == _FILL] = np.nan
        columns[PARTICLE_COLUMNS.get(name, name)] = out * _SCALE.get(name, 1.0)
    first = ["particle", "age", "lon", "lat", "zagl"]
    rest = [c for c in columns if c not in first]
    return pd.DataFrame({c: columns[c] for c in [*first, *rest]})
