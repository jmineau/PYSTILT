"""TROPOMI (Sentinel-5P) methane: operational L2 orbit files and the TROPOMI+GOSAT blended files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from ._common import _float, _in_ranges, _orbit_from_name, _rows, _wrap_azimuth


def read_tropomi_ch4(
    path: str | Path,
    *,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Read a TROPOMI (Sentinel-5P) L2 methane file into a table of soundings.

    Handles the operational ``S5P_*_L2__CH4___`` orbit files (``PRODUCT``
    group, ``scanline`` × ``ground_pixel``) and the flat TROPOMI+GOSAT
    blended files (``S5P_BLND_L2__CH4___``). ``lon_range`` and ``lat_range``
    keep only the pixels inside a box, which is worth doing on a whole orbit.

    ``value`` is the bias-corrected XCH4 (or the blended XCH4 in a blended
    file) in ppb; ``good`` is ``qa_value >= 0.5``, the product's own
    recommendation. The pressure grid is rebuilt from ``surface_pressure``
    and ``pressure_interval``: ``pressure_levels`` are the thirteen layer
    boundaries from the surface up (the top one is 0 hPa) and ``ak_pressure``
    the twelve layer midpoints that go with ``ak``. ``altitude_levels`` are
    the product's own heights of those boundaries, which is the argument to
    use for a slant path. ``apriori`` is the prior profile as a mole fraction
    per layer, ppb. ``zenith`` and ``azimuth`` are the viewing angles toward
    the satellite; the blended files carry none.
    """
    path = Path(path)
    with Dataset(path) as ds:
        if "PRODUCT" in ds.groups:
            return _tropomi_operational(ds, path, lon_range, lat_range)
        return _tropomi_blended(ds, path, lon_range, lat_range)


def _tropomi_layers(
    psfc: np.ndarray, dp: np.ndarray, nlayer: int
) -> tuple[np.ndarray, np.ndarray]:
    """Layer boundaries and midpoints in hPa, surface first, from psfc and dp."""
    k = np.arange(nlayer + 1)
    levels = psfc[:, None] - k[None, :] * dp[:, None]
    mids = psfc[:, None] - (np.arange(nlayer) + 0.5)[None, :] * dp[:, None]
    return levels, mids


def _tropomi_operational(
    ds: Dataset,
    path: Path,
    lon_range: tuple[float, float] | None,
    lat_range: tuple[float, float] | None,
) -> pd.DataFrame:
    p = ds["PRODUCT"]
    geo = p["SUPPORT_DATA/GEOLOCATIONS"]
    det = p["SUPPORT_DATA/DETAILED_RESULTS"]
    inp = p["SUPPORT_DATA/INPUT_DATA"]

    lat = _float(p["latitude"])[0]
    lon = _float(p["longitude"])[0]
    keep = _in_ranges(lon, lat, lon_range, lat_range)
    sl, gp = np.nonzero(keep)

    # read the bounding slab of each variable once, then pick the kept pixels
    s0, s1 = (int(sl.min()), int(sl.max()) + 1) if sl.size else (0, 0)
    g0, g1 = (int(gp.min()), int(gp.max()) + 1) if gp.size else (0, 0)
    si, gi = sl - s0, gp - g0

    def pick(var: Any) -> np.ndarray:
        return _float(var, (0, slice(s0, s1), slice(g0, g1)))[si, gi]

    time_utc = np.asarray(p["time_utc"][0, s0:s1]).astype(str)
    times = pd.to_datetime(time_utc[si]).tz_localize(None)
    orbit = getattr(ds, "orbit", None)
    orbit_str = f"{int(orbit):05d}" if orbit is not None else _orbit_from_name(path)
    scanline = np.asarray(p["scanline"][s0:s1])[si]
    ground_pixel = np.asarray(p["ground_pixel"][g0:g1])[gi]

    psfc = pick(inp["surface_pressure"]) / 100.0
    dp = pick(inp["pressure_interval"]) / 100.0
    nlayer = len(p.dimensions["layer"])
    levels, mids = _tropomi_layers(psfc, dp, nlayer)
    ak = pick(det["column_averaging_kernel"])[:, ::-1]
    alt = pick(inp["altitude_levels"])[:, ::-1]
    apriori = (
        pick(inp["methane_profile_apriori"]) / pick(inp["dry_air_subcolumns"]) * 1e9
    )[:, ::-1]
    qa = pick(p["qa_value"])

    columns = {
        "sounding_id": [
            f"{orbit_str}_{s:04d}_{g:03d}"
            for s, g in zip(scanline.tolist(), ground_pixel.tolist(), strict=True)
        ],
        "time": times,
        "longitude": lon[sl, gp],
        "latitude": lat[sl, gp],
        "surface_altitude": pick(inp["surface_altitude"]),
        "surface_pressure": psfc,
        "zenith": pick(geo["viewing_zenith_angle"]),
        "azimuth": _wrap_azimuth(pick(geo["viewing_azimuth_angle"])),
        "solar_zenith": pick(geo["solar_zenith_angle"]),
        "solar_azimuth": _wrap_azimuth(pick(geo["solar_azimuth_angle"])),
        "value": pick(p["methane_mixing_ratio_bias_corrected"]),
        "uncertainty": pick(p["methane_mixing_ratio_precision"]),
        "good": qa >= 0.5,
        "ak_pressure": _rows(mids),
        "ak": _rows(ak),
        "pressure_levels": _rows(levels),
        "altitude_levels": _rows(alt),
        "apriori": _rows(apriori),
        "longitude_bounds": _rows(pick(geo["longitude_bounds"])),
        "latitude_bounds": _rows(pick(geo["latitude_bounds"])),
        "qa_value": qa,
        "xch4_uncorrected": pick(p["methane_mixing_ratio"]),
        "scanline": scanline,
        "ground_pixel": ground_pixel,
    }
    return _tropomi_frame(columns["sounding_id"], columns)


def _tropomi_blended(
    ds: Dataset,
    path: Path,
    lon_range: tuple[float, float] | None,
    lat_range: tuple[float, float] | None,
) -> pd.DataFrame:
    lat = _float(ds["latitude"])
    lon = _float(ds["longitude"])
    keep = _in_ranges(lon, lat, lon_range, lat_range)
    (ii,) = np.nonzero(keep)
    i0, i1 = (int(ii.min()), int(ii.max()) + 1) if ii.size else (0, 0)
    ri = ii - i0

    def pick(var: Any) -> np.ndarray:
        return _float(var, slice(i0, i1))[ri]

    times = pd.to_datetime(np.asarray(ds["time_utc"][i0:i1]).astype(str)[ri])
    times = times.tz_localize(None)
    orbit_str = _orbit_from_name(path)
    psfc = pick(ds["surface_pressure"]) / 100.0
    dp = pick(ds["pressure_interval"]) / 100.0
    nlayer = len(ds.dimensions["layer"])
    levels, mids = _tropomi_layers(psfc, dp, nlayer)
    ak = pick(ds["column_averaging_kernel"])[:, ::-1]
    apriori = (
        pick(ds["methane_profile_apriori"]) / pick(ds["dry_air_subcolumns"]) * 1e9
    )[:, ::-1]
    qa = pick(ds["qa_value"])
    value_var = (
        "methane_mixing_ratio_blended"
        if "methane_mixing_ratio_blended" in ds.variables
        else "methane_mixing_ratio_bias_corrected"
    )
    columns = {
        "sounding_id": [f"{orbit_str}_{i}" for i in ii.tolist()],
        "time": times,
        "longitude": lon[ii],
        "latitude": lat[ii],
        "surface_altitude": pick(ds["surface_altitude"]),
        "surface_pressure": psfc,
        "value": pick(ds[value_var]),
        "uncertainty": pick(ds["methane_mixing_ratio_precision"]),
        "good": qa >= 0.5,
        "ak_pressure": _rows(mids),
        "ak": _rows(ak),
        "pressure_levels": _rows(levels),
        "apriori": _rows(apriori),
        "longitude_bounds": _rows(pick(ds["longitude_bounds"])),
        "latitude_bounds": _rows(pick(ds["latitude_bounds"])),
        "qa_value": qa,
        "xch4_bias_corrected": pick(ds["methane_mixing_ratio_bias_corrected"]),
        "xch4_uncorrected": pick(ds["methane_mixing_ratio"]),
    }
    return _tropomi_frame(columns["sounding_id"], columns)


def _tropomi_frame(ids: list[str], columns: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(columns, index=pd.RangeIndex(len(ids)))
    df["species"] = "xch4"
    df["units"] = "ppb"
    return df
