"""
PROFFAST output: EM27/SUN retrievals as PROFFASTpylot writes them.

A PROFFASTpylot run writes one results table for the whole run,
``comb_invparms_<site>_<instrument>_<start>-<end>.csv``, and, for each day,
the files the kernels and priors come from in ``raw_output_proffast/``:
``<site><YYMMDD>-colsens.dat`` (each gas's column sensitivity, tabulated
against solar zenith angle), ``-pT_fast_out.dat`` (the prior pressure and
temperature) and ``-VMR_fast_out.dat`` (the prior mole fractions).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import _wrap_azimuth

#: Gases a PROFFAST run retrieves, by the column name in the results table,
#: with the gas whose prior they share. ``_STR`` and ``_S5P`` are COCCON's
#: alternative CO2 and CH4 retrievals.
_GASES = {
    "XCO2": "CO2",
    "XCO2_STR": "CO2",
    "XCH4": "CH4",
    "XCH4_S5P": "CH4",
    "XCO": "CO",
    "XH2O": "H2O",
}
#: The VMR file's columns, as PROFFASTpylot reads them.
_VMR_COLUMNS = [
    "Index",
    "Altitude",
    "H2O",
    "HDO",
    "CO2",
    "CH4",
    "N2O",
    "CO",
    "O2",
    "HF",
]
#: The pT file's columns; its header line has spaces inside the names.
_PT_COLUMNS = [
    "Index",
    "Altitude",
    "Temperature",
    "Pressure",
    "DryAirColumn",
    "H2O",
    "HDO",
]
#: COCCON's quality screen, as its GEOMS files apply it.
_MAX_SZA = 75.0
_XAIR_RANGE = (0.98, 1.02)


def _read_colsens(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Return each gas's column sensitivity table from a ``colsens.dat`` file.

    The file has one block per gas: the gas name in quotes, a line of solar
    zenith angles in radians, then one line per level of altitude (km),
    pressure (hPa) and the sensitivity at each angle. Names longer than the
    field (``CO2_STR``, ``CH4_S5P``) lose their closing quote.
    """
    tables: dict[str, tuple[np.ndarray, list[list[float]]]] = {}
    gas = ""
    for line in path.read_text().splitlines():
        text = line.strip()
        if text.startswith("'"):
            gas = text.strip("' ")
            tables[gas] = (np.empty(0), [])
        elif "SZA[rad]" in text:
            tables[gas] = (np.array(text.split("->")[1].split(), dtype=float), [])
        elif text and text[0].isdigit() and gas:
            tables[gas][1].append([float(v) for v in text.split()])
    return {gas: (sza, np.array(rows)[:, 2:]) for gas, (sza, rows) in tables.items()}


def _day_files(raw_dir: Path, day: str, suffix: str) -> Path:
    """Return the one ``<site><day>-<suffix>`` file in a run's raw output."""
    found = sorted(raw_dir.glob(f"*{day}-{suffix}"))
    if len(found) != 1:
        what = "no" if not found else f"{len(found)}"
        raise FileNotFoundError(
            f"read_proffast: {what} '*{day}-{suffix}' file in {raw_dir}; it needs "
            "exactly one per measurement day."
        )
    return found[0]


def read_proffast(
    path: str | Path,
    species: str,
    *,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read a PROFFASTpylot run's results into a table of soundings.

    Reads the run's ``comb_invparms_*.csv`` for the columns, and each day's
    ``colsens.dat``, ``pT_fast_out.dat`` and ``VMR_fast_out.dat`` for the
    averaging kernels and priors.

    Parameters
    ----------
    path : str or Path
        The run's ``comb_invparms_*.csv``.
    species : str
        Column to read: ``xco2``, ``xch4``, ``xco``, ``xh2o``, or COCCON's
        alternative retrievals ``xco2_str`` and ``xch4_s5p``.
    raw_dir : str or Path, optional
        The folder of the daily files. Defaults to ``raw_output_proffast``
        next to the CSV, where PROFFASTpylot puts them.

    Returns
    -------
    pandas.DataFrame
        One row per spectrum, all of them. ``value`` is the column in ppm,
        or ppb for ``xco``, before any instrument calibration. PROFFAST
        reports no error for a single spectrum, so ``uncertainty`` is NaN.
        ``good`` is COCCON's quality screen: solar zenith angle at most 75
        degrees and ``XAIR`` between 0.98 and 1.02. ``ak`` is the day's
        column sensitivity interpolated linearly to the spectrum's apparent
        solar zenith angle, and ``ak_extrapolated`` marks spectra beyond the
        table's last angle, which take its last column. ``ak_pressure`` and
        ``pressure_levels`` (hPa) and ``altitude_levels`` (m above sea level)
        are the day's prior grid, and ``apriori`` the prior in the units of
        ``value``, all from the surface up. ``zenith`` and ``azimuth`` are
        the solar angles, the direction the instrument looks, with the
        azimuth clockwise from north. ``sounding_id`` is the spectrum's file
        name. The other columns of the results table, such as ``XAIR`` and
        each window's fit statistics, are left out.

    Notes
    -----
    COCCON's GEOMS files (:func:`read_coccon`) hold the same retrieval with
    each instrument's calibration factor applied, only the spectra that
    pass the screen above, and an uncertainty from the scatter of eleven
    neighboring spectra. Tested on a run of PROFFASTpylot (PROFFAST 2.4)
    on KIT's example data, whose kernels and priors match its GEOMS files.
    """
    path = Path(path)
    raw_dir = path.parent / "raw_output_proffast" if raw_dir is None else Path(raw_dir)
    column = species.upper()
    if column not in _GASES:
        raise ValueError(
            f"read_proffast species must be one of {sorted(s.lower() for s in _GASES)}, "
            f"got {species!r}."
        )
    gas = _GASES[column]
    results = pd.read_csv(path, skipinitialspace=True, dtype={"YYMMDD": str})
    results.columns = results.columns.str.strip()

    sza = results["appSZA"].to_numpy(dtype=float)
    n = len(results)
    ak: list[np.ndarray] = [np.empty(0)] * n
    pressure: list[np.ndarray] = [np.empty(0)] * n
    altitude: list[np.ndarray] = [np.empty(0)] * n
    apriori: list[np.ndarray] = [np.empty(0)] * n
    extrapolated = np.zeros(n, dtype=bool)
    days = results["YYMMDD"].to_numpy(dtype=str)
    for day in np.unique(days):
        idx = np.flatnonzero(days == day)
        table_sza, table = _read_colsens(_day_files(raw_dir, day, "colsens.dat"))[
            column[1:]
        ]
        pt = pd.read_csv(
            _day_files(raw_dir, day, "pT_fast_out.dat"),
            sep=r"\s+",
            header=None,
            skiprows=1,
            names=_PT_COLUMNS,
        )
        vmr = pd.read_csv(
            _day_files(raw_dir, day, "VMR_fast_out.dat"),
            sep=r"\s+",
            header=None,
            names=_VMR_COLUMNS,
            usecols=range(len(_VMR_COLUMNS)),
        )
        if not len(table) == len(pt) == len(vmr):
            raise ValueError(
                f"read_proffast: the {day} colsens, pT and VMR files have "
                f"{len(table)}, {len(pt)} and {len(vmr)} levels."
            )
        day_pressure = pt["Pressure"].to_numpy(dtype=float) / 100.0
        day_altitude = pt["Altitude"].to_numpy(dtype=float)
        day_apriori = vmr[gas].to_numpy(dtype=float)
        for i in idx:
            angle = np.radians(sza[i])
            ak[i] = np.array([np.interp(angle, table_sza, level) for level in table])
            extrapolated[i] = angle > table_sza[-1]
            pressure[i], altitude[i], apriori[i] = (
                day_pressure,
                day_altitude,
                day_apriori,
            )

    units = "ppb" if gas == "CO" else "ppm"
    scale = 1000.0 if gas == "CO" else 1.0
    xair = results["XAIR"].to_numpy(dtype=float)
    df = pd.DataFrame(
        {
            "sounding_id": results["spectrum"].astype(str).str.strip(),
            "time": pd.to_datetime(results["UTC"]),
            "longitude": results["londeg"].to_numpy(dtype=float),
            "latitude": results["latdeg"].to_numpy(dtype=float),
            "surface_altitude": results["altim"].to_numpy(dtype=float),
            "surface_pressure": results["gndP"].to_numpy(dtype=float),
            "zenith": sza,
            "azimuth": _wrap_azimuth(results["azimuth"].to_numpy(dtype=float) + 180.0),
            "value": results[column].to_numpy(dtype=float) * scale,
            "uncertainty": np.nan,
            "good": (sza <= _MAX_SZA)
            & (xair >= _XAIR_RANGE[0])
            & (xair <= _XAIR_RANGE[1]),
            "ak_pressure": pressure,
            "ak": ak,
            "ak_extrapolated": extrapolated,
            "pressure_levels": pressure,
            "altitude_levels": altitude,
            "apriori": [a * scale for a in apriori],
        },
        index=pd.RangeIndex(len(results)),
    )
    df["solar_zenith"] = df["zenith"]
    df["solar_azimuth"] = df["azimuth"]
    df["species"] = species.lower()
    df["units"] = units
    return df


__all__ = ["read_proffast"]
