"""Small helpers shared by the product readers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _float(var: Any, idx: Any = Ellipsis) -> np.ndarray:
    """Read a netCDF variable as float with fill values as NaN."""
    a = var[idx]
    if np.ma.isMaskedArray(a):
        return np.ma.filled(a.astype(float), np.nan)
    return np.asarray(a, dtype=float)


def _wrap_azimuth(a: np.ndarray) -> np.ndarray:
    """Azimuth in [0, 360) clockwise from north."""
    return np.mod(a, 360.0)


def _in_ranges(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_range: tuple[float, float] | None,
    lat_range: tuple[float, float] | None,
) -> np.ndarray:
    keep = np.isfinite(lon) & np.isfinite(lat)
    if lon_range is not None:
        keep &= (lon >= lon_range[0]) & (lon <= lon_range[1])
    if lat_range is not None:
        keep &= (lat >= lat_range[0]) & (lat <= lat_range[1])
    return keep


def _seconds_since(var: Any, idx: Any = Ellipsis) -> pd.DatetimeIndex:
    """Times from a ``seconds since <origin>`` variable, naive UTC."""
    units = str(getattr(var, "units", "seconds since 1970-01-01 00:00:00"))
    origin = units.split("since", 1)[1].strip() if "since" in units else "1970-01-01"
    return pd.to_datetime(_float(var, idx), unit="s", origin=pd.Timestamp(origin))


def _rows(arr: np.ndarray) -> list[np.ndarray]:
    """One array per row, for an object column."""
    return [np.asarray(r) for r in arr]


def _orbit_from_name(path: Path) -> str:
    """The orbit number in an S5P file name, or the file stem."""
    m = re.search(r"_(\d{5})_\d{2}_\d{6}_", path.name)
    return m.group(1) if m else path.stem
