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

    Reads the ``*.public.nc`` and ``*.public.qc.nc`` site files from
    CaltechDATA. These are GGG2020 netCDF files, so this is
    :func:`~stilt.observations.read_ggg_netcdf` under the network's name.
    See it for the parameters and columns.
    """
    return read_ggg_netcdf(path, species, time_range=time_range)
