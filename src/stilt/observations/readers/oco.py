"""OCO-2 and OCO-3 Lite XCO2 files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from ._common import _float, _in_ranges, _rows, _seconds_since, _span, _wrap_azimuth


def read_oco2(
    path: str | Path,
    *,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Read an OCO-2 or OCO-3 L2 Lite XCO2 file into a table of soundings.

    Reads the Lite FP v10 and v11 files (``oco2_LtCO2_*.nc4``,
    ``oco3_LtCO2_*.nc4``).

    Parameters
    ----------
    path : str or Path
        The Lite file.
    lon_range, lat_range : tuple of float, optional
        ``(min, max)`` longitude and latitude, in degrees. Only soundings
        inside the box are read.

    Returns
    -------
    pandas.DataFrame
        One row per sounding. ``value`` is XCO2 in ppm, with the product's
        fill value as NaN. ``good`` is ``xco2_quality_flag == 0``.
        ``pressure_levels`` are the 20 retrieval levels, and ``ak_pressure``
        and ``ak`` the kernel on them, reordered to run from the surface up.
        ``apriori`` is the CO2 prior profile on the same levels,
        ``pressure_weight`` the retrieval's pressure weighting function, and
        ``apriori_column`` the prior XCO2. ``zenith`` and ``azimuth`` are the
        sensor angles toward the satellite, and ``solar_zenith`` and
        ``solar_azimuth`` the solar angles.
    """
    path = Path(path)
    with Dataset(path) as ds:
        lat = _float(ds["latitude"])
        lon = _float(ds["longitude"])
        keep = _in_ranges(lon, lat, lon_range, lat_range)
        (ii,) = np.nonzero(keep)
        block, ri = _span(ii)

        def pick(var: Any) -> np.ndarray:
            """Return one variable for the selected soundings."""
            return _float(var, block)[ri]

        sounding = ds["Sounding"]
        retrieval = ds["Retrieval"]
        ids = np.asarray(ds["sounding_id"][block])[ri]
        times = _seconds_since(ds["time"], block)[ri]
        flag = pick(ds["xco2_quality_flag"])
        columns = {
            "sounding_id": [str(int(s)) for s in ids.tolist()],
            "time": times,
            "longitude": lon[ii],
            "latitude": lat[ii],
            "surface_altitude": pick(sounding["altitude"]),
            "surface_pressure": pick(retrieval["psurf"]),
            "zenith": pick(ds["sensor_zenith_angle"]),
            "azimuth": _wrap_azimuth(pick(sounding["sensor_azimuth_angle"])),
            "solar_zenith": pick(ds["solar_zenith_angle"]),
            "solar_azimuth": _wrap_azimuth(pick(sounding["solar_azimuth_angle"])),
            "value": pick(ds["xco2"]),
            "uncertainty": pick(ds["xco2_uncertainty"]),
            "good": flag == 0,
            "ak_pressure": _rows(pick(ds["pressure_levels"])[:, ::-1]),
            "ak": _rows(pick(ds["xco2_averaging_kernel"])[:, ::-1]),
            "pressure_levels": _rows(pick(ds["pressure_levels"])[:, ::-1]),
            "apriori": _rows(pick(ds["co2_profile_apriori"])[:, ::-1]),
            "pressure_weight": _rows(pick(ds["pressure_weight"])[:, ::-1]),
            "apriori_column": pick(ds["xco2_apriori"]),
            "longitude_bounds": _rows(pick(ds["vertex_longitude"])),
            "latitude_bounds": _rows(pick(ds["vertex_latitude"])),
            "quality_flag": flag,
            "operation_mode": pick(sounding["operation_mode"]),
            "land_fraction": pick(sounding["land_fraction"]),
            "orbit": pick(sounding["orbit"]),
        }
    df = pd.DataFrame(columns, index=pd.RangeIndex(len(ii)))
    df["species"] = "xco2"
    df["units"] = "ppm"
    return df
