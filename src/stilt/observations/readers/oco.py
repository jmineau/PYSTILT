"""OCO-2 and OCO-3 Lite XCO2 files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from ._common import _float, _in_ranges, _rows, _seconds_since, _wrap_azimuth


def read_oco2(
    path: str | Path,
    *,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Read an OCO-2 or OCO-3 L2 Lite XCO2 file into a table of soundings.

    Follows the Lite FP v10/v11 layout (``oco2_LtCO2_*.nc4``,
    ``oco3_LtCO2_*.nc4``): the retrieval and its levels at the root, the
    viewing geometry and surface altitude under ``Sounding``, the surface
    pressure under ``Retrieval``. ``lon_range`` and ``lat_range`` keep only
    the soundings inside a box.

    ``value`` is ``xco2`` in ppm with the product's fill as NaN; ``good`` is
    ``xco2_quality_flag == 0``. The twenty ``pressure_levels`` and the
    kernel on them (``ak_pressure``, ``ak``) are reordered from the surface
    up (the file lists them from space down). ``apriori`` is the CO2 prior
    profile on the same levels, ``pressure_weight`` the retrieval's pressure
    weighting function, and ``apriori_column`` the prior XCO2. ``zenith`` and
    ``azimuth`` are the sensor angles toward the satellite; the solar angles
    are alongside.
    """
    path = Path(path)
    with Dataset(path) as ds:
        lat = _float(ds["latitude"])
        lon = _float(ds["longitude"])
        keep = _in_ranges(lon, lat, lon_range, lat_range)
        (ii,) = np.nonzero(keep)
        i0, i1 = (int(ii.min()), int(ii.max()) + 1) if ii.size else (0, 0)
        ri = ii - i0

        def pick(var: Any) -> np.ndarray:
            """Read one variable over the selected soundings."""
            return _float(var, slice(i0, i1))[ri]

        sounding = ds["Sounding"]
        retrieval = ds["Retrieval"]
        ids = np.asarray(ds["sounding_id"][i0:i1])[ri]
        times = _seconds_since(ds["time"], slice(i0, i1))[ri]
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
