"""TCCON GGG2020 public site files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .ggg import read_ggg_netcdf


def read_tccon(
    path: str | Path,
    species: str = "xco2",
    *,
    time_range: tuple[Any, Any] | None = None,
) -> pd.DataFrame:
    """
    Read a TCCON GGG2020 public file into a table of soundings.

    Works on the ``*.public.nc`` and ``*.public.qc.nc`` site files from
    CaltechDATA. These are GGG2020 netCDF files with the kernels already
    expanded per spectrum, so this is :func:`~stilt.observations.read_ggg_netcdf`
    under the network's name; see it for the columns. ``species`` is the
    column variable to read: ``xco2``, ``xch4``, ``xco``, ``xn2o`` or
    ``xh2o``. ``time_range`` keeps only the spectra between two times, which
    a multi-year site file needs.
    """
    return read_ggg_netcdf(path, species, time_range=time_range)
