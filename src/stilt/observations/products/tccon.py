"""TCCON GGG2020 public site files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from ._common import _float, _rows, _seconds_since, _wrap_azimuth

_STD_ATMOSPHERE_HPA = 1013.25


def read_tccon(
    path: str | Path,
    species: str = "xco2",
    *,
    time_range: tuple[Any, Any] | None = None,
) -> pd.DataFrame:
    """
    Read a TCCON GGG2020 public file into a table of soundings.

    Works on the ``*.public.nc`` and ``*.public.qc.nc`` site files from
    CaltechDATA. ``species`` is the column variable to read: ``xco2``,
    ``xch4``, ``xco``, ``xn2o`` or ``xh2o``. ``time_range`` keeps only the
    spectra between two times, which a multi-year site file needs.

    ``value`` and ``uncertainty`` are the species and its one-sigma error in
    the file's units; ``good`` is ``flag == 0`` where the file has a flag
    and true everywhere in a ``qc`` file, which holds only flagged-good
    data. The kernel ``ak`` sits on the site's fixed ``ak_pressure`` grid
    (hPa, median pressures) and ``apriori`` is the prior profile of the gas
    on the same altitude grid; ``pressure_levels`` are the prior's pressures
    in hPa per spectrum and ``altitude_levels`` that grid in metres above
    sea level, both from the surface up, so
    :func:`~stilt.observations.pressure_altitudes` and
    :func:`~stilt.observations.slant_points` take them directly.
    ``surface_altitude`` is the instrument's geometric altitude and
    ``surface_pressure`` the pressure it measured. ``zenith`` and ``azimuth``
    are the solar angles, which is the direction the instrument looks.
    """
    path = Path(path)
    species = species.lower()
    if not species.startswith("x"):
        raise ValueError(
            f"read_tccon species must be a column variable like 'xco2', got {species!r}."
        )
    gas = species[1:]
    with Dataset(path) as ds:
        for name in (species, f"{species}_error", f"ak_{species}"):
            if name not in ds.variables:
                raise ValueError(f"read_tccon: {path.name} has no {name!r} variable.")
        times_all = _seconds_since(ds["time"])
        keep = np.ones(len(times_all), dtype=bool)
        if time_range is not None:
            start, stop = (pd.Timestamp(t) for t in time_range)
            keep &= (times_all >= start) & (times_all <= stop)
        (ii,) = np.nonzero(keep)
        i0, i1 = (int(ii.min()), int(ii.max()) + 1) if ii.size else (0, 0)
        ri = ii - i0

        def pick(var: Any) -> np.ndarray:
            return _float(var, slice(i0, i1))[ri]

        site = str(getattr(ds, "long_name", path.stem[:2]))
        times = times_all[ii]
        ak_pressure = _float(ds["ak_pressure"])
        ak = pick(ds[f"ak_{species}"])
        prior_alt_m = _float(ds["prior_altitude"]) * 1000.0
        prior_pres = pick(ds["prior_pressure"])
        if str(getattr(ds["prior_pressure"], "units", "atm")).lower() == "atm":
            prior_pres = prior_pres * _STD_ATMOSPHERE_HPA
        apriori = pick(ds[f"prior_{gas}"]) if f"prior_{gas}" in ds.variables else None
        good = (
            pick(ds["flag"]) == 0 if "flag" in ds.variables else np.ones(len(ii), bool)
        )
        columns: dict[str, Any] = {
            "sounding_id": [f"{site}_{t.strftime('%Y%m%dT%H%M%S')}" for t in times],
            "time": times,
            "longitude": pick(ds["long"]),
            "latitude": pick(ds["lat"]),
            "surface_altitude": pick(ds["zobs"]) * 1000.0,
            "surface_pressure": pick(ds["pout"]),
            "zenith": pick(ds["solzen"]),
            "azimuth": _wrap_azimuth(pick(ds["azim"])),
            "value": pick(ds[species]),
            "uncertainty": pick(ds[f"{species}_error"]),
            "good": good,
            "ak_pressure": [ak_pressure] * len(ii),
            "ak": _rows(ak),
            "pressure_levels": _rows(prior_pres),
            "altitude_levels": [prior_alt_m] * len(ii),
        }
        if apriori is not None:
            columns["apriori"] = _rows(apriori)
        units = str(getattr(ds[species], "units", ""))
    df = pd.DataFrame(columns, index=pd.RangeIndex(len(ii)))
    df["solar_zenith"] = df["zenith"]
    df["solar_azimuth"] = df["azimuth"]
    df["species"] = species
    df["units"] = units
    return df
