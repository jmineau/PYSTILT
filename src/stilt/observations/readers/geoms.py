"""
COCCON GEOMS files: EM27/SUN retrievals from PROFFAST.

COCCON, the network of EM27/SUN spectrometers, processes its spectra with
PROFFAST and releases the results as GEOMS HDF5 files, one per site and day
(``groundbased_ftir.coccon_*.h5``), through ESA's EVDC. PROFFASTpylot writes
the same files for any EM27/SUN its user runs. They follow the GEOMS template
``GEOMS-TE-FTIR-COCCON-001``: variables named like
``CH4.COLUMN.MIXING.RATIO.VOLUME.DRY_ABSORPTION.SOLAR``, with each
measurement's column sensitivity already interpolated to its solar zenith
angle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from netCDF4 import Dataset

from ._common import _float, _rows, _wrap_azimuth

_MJD2K_ORIGIN = pd.Timestamp("2000-01-01")
# mole-fraction scale of a GEOMS units string
_UNIT_SCALE = {"1": 1.0, "ppmv": 1e-6, "ppbv": 1e-9, "pptv": 1e-12}


def _units(var: Any) -> str:
    """Return a GEOMS variable's units (``VAR_UNITS``)."""
    return str(getattr(var, "VAR_UNITS", getattr(var, "units", ""))).strip()


def _scale(from_units: str, to_units: str) -> float:
    """Return the factor that converts mole fractions in ``from_units`` to ``to_units``."""
    try:
        return _UNIT_SCALE[from_units.lower()] / _UNIT_SCALE[to_units.lower()]
    except KeyError as e:
        raise ValueError(f"unknown mole-fraction units {e.args[0]!r}") from None


def read_coccon(path: str | Path, species: str) -> pd.DataFrame:
    """
    Read a COCCON GEOMS file of EM27/SUN retrievals into a table of soundings.

    Reads the GEOMS HDF5 files (``groundbased_ftir.coccon_*.h5``) that
    COCCON releases and that PROFFASTpylot writes, one site-day each.

    Parameters
    ----------
    path : str or Path
        The GEOMS file.
    species : str
        Column to read: ``xco2``, ``xch4``, ``xco``, or ``xh2o``.

    Returns
    -------
    pandas.DataFrame
        One row per measurement. ``value`` and ``uncertainty`` are the
        column and its random error, in ppm (``xco2``, ``xch4``, ``xh2o``) or
        ppb (``xco``), as the file gives them. ``good`` is True for every
        row: the file holds only the measurements that passed COCCON's
        quality filter. ``ak`` is the column sensitivity at the
        measurement's solar zenith angle, on ``ak_pressure`` (hPa), which is
        also ``pressure_levels``, with ``altitude_levels`` their heights (m
        above sea level), all from the surface up. ``apriori`` is the prior
        profile, converted to the units of ``value``. ``surface_altitude`` is
        the instrument's altitude and ``surface_pressure`` the pressure at
        the instrument. ``zenith`` and ``azimuth`` are the solar angles, the
        direction the instrument looks, with the azimuth clockwise from
        north. ``sounding_id`` is the site and time, since the file names
        no spectra.

    Notes
    -----
    Tested on files written by PROFFASTpylot's GEOMS writer (PROFFAST 2.4)
    for KIT's example data. A file's times are ``DATETIME``, days since
    2000-01-01 UTC (MJD2K) to five decimals; ``time`` rounds them to the
    second.
    """
    path = Path(path)
    species = species.lower()
    if not species.startswith("x"):
        raise ValueError(
            f"read_coccon species must be a column variable like 'xch4', got {species!r}."
        )
    gas = species[1:].upper()
    column_name = f"{gas}.COLUMN.MIXING.RATIO.VOLUME.DRY_ABSORPTION.SOLAR"
    names = {
        "value": column_name,
        "uncertainty": f"{column_name}_UNCERTAINTY.RANDOM.STANDARD",
        "ak": f"{gas}.COLUMN_ABSORPTION.SOLAR_AVK",
        "apriori": f"{gas}.MIXING.RATIO.VOLUME.DRY_APRIORI",
    }
    with Dataset(path) as ds:
        for name in names.values():
            if name not in ds.variables:
                raise ValueError(f"read_coccon: {path.name} has no {name!r} variable.")
        units = {"ppmv": "ppm", "ppbv": "ppb"}.get(
            _units(ds[column_name]), _units(ds[column_name])
        )
        # stored to 1e-5 day, so round to the second the spectrum was taken
        days = pd.to_timedelta(_float(ds["DATETIME"]), unit="D")
        times = (_MJD2K_ORIGIN + days).round("s")
        site = str(getattr(ds, "DATA_LOCATION", "") or path.stem).strip().lower()
        apriori = _float(ds[names["apriori"]]) * _scale(
            _units(ds[names["apriori"]]), _units(ds[column_name])
        )
        pressure = _float(ds["PRESSURE_INDEPENDENT"])
        columns: dict[str, Any] = {
            "sounding_id": [f"{site}_{t.strftime('%Y%m%dT%H%M%S')}" for t in times],
            "time": times,
            "longitude": _float(ds["LONGITUDE.INSTRUMENT"]),
            "latitude": _float(ds["LATITUDE.INSTRUMENT"]),
            "surface_altitude": _float(ds["ALTITUDE.INSTRUMENT"]) * 1000.0,
            "surface_pressure": _float(ds["SURFACE.PRESSURE_INDEPENDENT"]),
            "zenith": _float(ds["ANGLE.SOLAR_ZENITH.ASTRONOMICAL"]),
            "azimuth": _wrap_azimuth(_float(ds["ANGLE.SOLAR_AZIMUTH"])),
            "value": _float(ds[names["value"]]),
            "uncertainty": _float(ds[names["uncertainty"]]),
            "ak_pressure": _rows(pressure),
            "ak": _rows(_float(ds[names["ak"]])),
            "pressure_levels": _rows(pressure),
            "altitude_levels": _rows(_float(ds["ALTITUDE"]) * 1000.0),
            "apriori": _rows(apriori),
        }
    df = pd.DataFrame(columns, index=pd.RangeIndex(len(times)))
    df["good"] = True
    df["solar_zenith"] = df["zenith"]
    df["solar_azimuth"] = df["azimuth"]
    df["species"] = species
    df["units"] = units
    return df


__all__ = ["read_coccon"]
