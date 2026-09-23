"""
GGG2020 output: ``.oof`` text files and the private / public netCDF files.

GGG is the retrieval behind TCCON and, through EGI, behind many EM27/SUN
operators. It writes a spectrum-per-row ``.oof`` "official output file" and,
from the same run, a ``*.private.nc`` with the priors and the averaging
kernel tables; the ``*.public.nc`` files TCCON distributes are the private
files with the kernels already expanded per spectrum.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset, chartostring

from ._common import _float, _rows, _seconds_since, _wrap_azimuth

_STD_ATMOSPHERE_HPA = 1013.25
_AIRMASS_VAR = "o2_7885_am_o2"
# mole-fraction scale of a units string; "" and "parts" are plain mole fraction
_UNIT_SCALE = {"": 1.0, "parts": 1.0, "ppm": 1e-6, "ppb": 1e-9, "ppt": 1e-12}
_COLUMN = re.compile(r"^(?P<name>\w+?)(?:\((?P<units>[^)]*)\))?(?P<error>_error)?$")


def _check_species(species: str, who: str) -> str:
    """Validate a column-variable name like ``xch4``; return it lower-cased."""
    species = species.lower()
    if not species.startswith("x"):
        raise ValueError(
            f"{who} species must be a column variable like 'xco2', got {species!r}."
        )
    return species


def _scale(from_units: str, to_units: str) -> float:
    """Factor that converts mole fractions in ``from_units`` to ``to_units``."""
    try:
        return _UNIT_SCALE[from_units.strip().lower()] / _UNIT_SCALE[to_units.lower()]
    except KeyError as e:
        raise ValueError(f"unknown mole-fraction units {e.args[0]!r}") from None


# -- .oof ----------------------------------------------------------------------


def _oof_header(lines: list[str], path: Path) -> tuple[int, float | None, list[str]]:
    """
    Header length, the missing value, and the raw column names of a .oof.

    The first line gives the header length and a variable count; the count
    includes variables the flag table marks as not output, so the column
    line itself says how many columns there are.
    """
    try:
        nhead = int(re.split(r"[\s,]+", lines[0].strip())[0])
    except (ValueError, IndexError):
        raise ValueError(
            f"read_ggg_oof: {path.name} does not start with the header line count."
        ) from None
    missing = None
    for line in lines[1:nhead]:
        if line.lower().startswith("missing:"):
            missing = float(line.split(":", 1)[1])
    names = re.split(r"[\s,]+", lines[nhead - 1].strip()) if nhead <= len(lines) else []
    if names[:2] != ["flag", "spectrum"]:
        raise ValueError(
            f"read_ggg_oof: {path.name} has no 'flag spectrum ...' column line "
            f"at line {nhead}."
        )
    return nhead, missing, names


def _clean_column(raw: str) -> tuple[str, str]:
    """``xch4(ppm)_error`` -> ``('xch4_error', 'ppm')``; ``lat(deg)`` -> ``('lat', 'deg')``."""
    m = _COLUMN.match(raw)
    if m is None:
        return raw, ""
    return m["name"] + (m["error"] or ""), m["units"] or ""


def read_ggg_oof(path: str | Path, species: str = "xch4") -> pd.DataFrame:
    """
    Read a GGG2020 ``.oof`` file into a table of soundings.

    A ``.oof`` is one day of retrievals from one instrument (``*.vav.ada.aia.oof``),
    which is how EGI delivers EM27/SUN results. ``species`` is the column
    variable to read: ``xch4``, ``xco2``, ``xco``, ``xh2o``, ``xn2o``, ...

    ``value`` and ``uncertainty`` are the species and its one-sigma error in
    the units the column header gives (``xch4(ppm)`` -> ``ppm``); ``good`` is
    ``flag == 0``. ``sounding_id`` is the spectrum name. ``surface_altitude``
    is the instrument's geometric altitude and ``surface_pressure`` the
    pressure it measured; ``zenith`` and ``azimuth`` are the solar angles.
    ``time`` is built from the ``year``, ``day`` and fractional UT ``hour``
    columns.

    A ``.oof`` carries no averaging kernel and no prior profile, so the
    ``ak``, ``ak_pressure`` and ``pressure_levels`` columns are absent. The
    kernels are in the run's ``*.private.nc`` (:func:`read_ggg_netcdf`), or
    come from a site kernel table keyed by solar zenith angle. The other
    ``x<gas>`` columns, their errors, ``flag``, ``zmin`` (km) and ``xluft`` are
    kept under their own names.
    """
    path = Path(path)
    species = _check_species(species, "read_ggg_oof")
    lines = path.read_text().splitlines()
    nhead, missing, raw_names = _oof_header(lines, path)
    cleaned = [_clean_column(n) for n in raw_names]
    names = [n for n, _ in cleaned]
    units = dict(cleaned)

    rows = [re.split(r"[\s,]+", line.strip()) for line in lines[nhead:] if line.strip()]
    bad = [i for i, r in enumerate(rows) if len(r) != len(names)]
    if bad:
        raise ValueError(
            f"read_ggg_oof: {path.name} data line {nhead + 1 + bad[0]} has "
            f"{len(rows[bad[0]])} fields, expected {len(names)}."
        )
    # pd.Index, not the bare list: pandas 2.x stubs reject a list[str] here
    raw = pd.DataFrame(rows, columns=pd.Index(names))
    numeric = [n for n in names if n != "spectrum"]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
    if missing is not None:
        raw[numeric] = raw[numeric].mask(np.isclose(raw[numeric].to_numpy(), missing))

    for name in (species, f"{species}_error"):
        if name not in raw.columns:
            raise ValueError(f"read_ggg_oof: {path.name} has no {name!r} column.")

    year = raw["year"].to_numpy(dtype=float).round().astype(int)
    day = raw["day"].to_numpy(dtype=float)
    hour = raw["hour"].to_numpy(dtype=float)
    times = (
        pd.to_datetime(pd.Series(year).astype(str), format="%Y")
        + pd.to_timedelta(day - 1, unit="D")
        + pd.to_timedelta(hour, unit="h")
    ).dt.round("s")

    df = pd.DataFrame(
        {
            "sounding_id": raw["spectrum"].astype(str),
            "time": times,
            "longitude": raw["long"],
            "latitude": raw["lat"],
            "surface_altitude": raw["zobs"] * 1000.0,
            "surface_pressure": raw["pout"],
            "zenith": raw["solzen"],
            "azimuth": _wrap_azimuth(raw["azim"].to_numpy()),
            "value": raw[species],
            "uncertainty": raw[f"{species}_error"],
            "good": raw["flag"] == 0,
        }
    )
    df["solar_zenith"] = df["zenith"]
    df["solar_azimuth"] = df["azimuth"]
    df["species"] = species
    df["units"] = units[species]
    df["flag"] = raw["flag"].astype(int)
    df["zmin"] = raw["zmin"]
    taken = {"flag", "spectrum", "year", "day", "hour", "lat", "long", "zobs", "zmin"}
    taken |= {"solzen", "azim", "pout", species, f"{species}_error"}
    for name in names:
        if name.startswith("x") and name not in taken:
            df[name] = raw[name]
    return df


# -- private / public netCDF ---------------------------------------------------


def _expand_ak_table(
    ds: Any, species: str, xgas: np.ndarray, airmass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-spectrum kernels from a private file's ``(ak_altitude, slant bin)`` table.

    GGG tabulates the column averaging kernel against slant xgas, the
    retrieved xgas times the O2-window airmass, and its public-file writer
    interpolates each altitude row linearly along that axis. The second
    array flags spectra whose slant xgas fell outside the table, where the
    end column is used (the public files call these extrapolation flags).
    """
    bins = _float(ds[f"ak_slant_{species}_bin"])
    table = _float(ds[f"ak_{species}"])  # (ak_altitude, bin)
    slant = xgas * airmass
    ak = np.empty((len(slant), table.shape[0]))
    for k in range(table.shape[0]):
        ak[:, k] = np.interp(slant, bins, table[k, :])
    with np.errstate(invalid="ignore"):
        extrapolated = (slant < bins[0]) | (slant > bins[-1])
    return ak, extrapolated


def read_ggg_netcdf(
    path: str | Path,
    species: str = "xco2",
    *,
    time_range: tuple[Any, Any] | None = None,
) -> pd.DataFrame:
    """
    Read a GGG2020 netCDF file, private or public, into a table of soundings.

    A ``*.private.nc`` is what GGG writes for a run; a ``*.public.nc`` (or
    ``*.public.qc.nc``) is the TCCON release of the same layout. ``species``
    is the column variable to read: ``xco2``, ``xch4``, ``xco``, ``xn2o`` or
    ``xh2o``. ``time_range`` keeps only the spectra between two times, which a
    multi-year site file needs.

    ``value`` and ``uncertainty`` are the species and its one-sigma error in
    the file's units; ``good`` is ``flag == 0`` where the file has a flag and
    true everywhere in a ``qc`` file, which holds only flagged-good data.
    The kernel ``ak`` sits on the site's fixed ``ak_pressure`` grid (hPa,
    median pressures). A public file stores it per spectrum; a private file
    stores a table against slant xgas, which is interpolated per spectrum the
    way GGG's public writer does (needs the O2-window airmass,
    ``o2_7885_am_o2``), with ``ak_extrapolated`` marking spectra beyond the
    table. ``apriori`` is the prior profile of the gas in the species' units
    on the prior altitude grid; ``pressure_levels`` are the prior's pressures
    in hPa per spectrum and ``altitude_levels`` that grid in metres above sea
    level, both from the surface up, so
    :func:`~stilt.observations.pressure_altitudes` and
    :func:`~stilt.observations.slant_points` take them directly. A private
    file shares each prior between the spectra ``prior_index`` points at it.
    ``surface_altitude`` is the instrument's geometric altitude and
    ``surface_pressure`` the pressure it measured. ``zenith`` and ``azimuth``
    are the solar angles, which is the direction the instrument looks.
    ``sounding_id`` is the spectrum name when the file keeps it (private),
    else the site and time.
    """
    path = Path(path)
    species = _check_species(species, "read_ggg_netcdf")
    gas = species[1:]
    with Dataset(path) as ds:
        for name in (species, f"{species}_error", f"ak_{species}"):
            if name not in ds.variables:
                raise ValueError(
                    f"read_ggg_netcdf: {path.name} has no {name!r} variable."
                )
        times_all = _seconds_since(ds["time"])
        keep = np.ones(len(times_all), dtype=bool)
        if time_range is not None:
            start, stop = (pd.Timestamp(t) for t in time_range)
            keep &= (times_all >= start) & (times_all <= stop)
        (ii,) = np.nonzero(keep)
        i0, i1 = (int(ii.min()), int(ii.max()) + 1) if ii.size else (0, 0)
        ri = ii - i0

        def pick(var: Any) -> np.ndarray:
            """Read one time-indexed variable over the selected spectra."""
            return _float(var, slice(i0, i1))[ri]

        site = str(getattr(ds, "long_name", "") or "").strip() or path.stem[:2]
        times = times_all[ii]
        units = str(getattr(ds[species], "units", ""))
        value = pick(ds[species])

        # averaging kernel: per spectrum (public) or a slant-xgas table (private)
        ak_pressure = _float(ds["ak_pressure"])
        ak_var = ds[f"ak_{species}"]
        if ak_var.dimensions[0] == "time":
            ak = pick(ak_var)
            ak_extrapolated = np.zeros(len(ii), dtype=bool)
        else:
            if _AIRMASS_VAR not in ds.variables:
                raise ValueError(
                    f"read_ggg_netcdf: {path.name} stores ak_{species} as a slant-xgas "
                    f"table but has no {_AIRMASS_VAR!r} airmass to place spectra on it."
                )
            ak, ak_extrapolated = _expand_ak_table(
                ds, species, value, pick(ds[_AIRMASS_VAR])
            )

        # priors: per spectrum (public) or shared through prior_index (private)
        prior_pressure_var = ds["prior_pressure"]
        if prior_pressure_var.dimensions[0] == "time":
            prior_rows = ri

            def prior(var: Any) -> np.ndarray:
                """One prior row per selected spectrum."""
                return _float(var, slice(i0, i1))[prior_rows]
        else:
            prior_rows = np.asarray(ds["prior_index"][i0:i1], dtype=int)[ri]

            def prior(var: Any) -> np.ndarray:
                """The prior each selected spectrum points at."""
                return _float(var)[prior_rows]

        prior_alt_m = _float(ds["prior_altitude"]) * 1000.0
        prior_pres = prior(prior_pressure_var)
        if str(getattr(prior_pressure_var, "units", "atm")).lower() == "atm":
            prior_pres = prior_pres * _STD_ATMOSPHERE_HPA
        apriori = None
        for name in (f"prior_{gas}", f"prior_1{gas}"):
            if name in ds.variables:
                prior_units = str(getattr(ds[name], "units", ""))
                apriori = prior(ds[name]) * _scale(prior_units, units)
                break

        good = (
            pick(ds["flag"]) == 0 if "flag" in ds.variables else np.ones(len(ii), bool)
        )
        if "spectrum" in ds.variables:
            spectra = ds["spectrum"][i0:i1]
            if (
                np.ndim(spectra) == 2
            ):  # raw chars; netCDF4 decodes when _Encoding is set
                spectra = chartostring(np.asarray(spectra))
            sounding_id = [str(s).strip() for s in np.asarray(spectra)[ri]]
        else:
            sounding_id = [f"{site}_{t.strftime('%Y%m%dT%H%M%S')}" for t in times]

        columns: dict[str, Any] = {
            "sounding_id": sounding_id,
            "time": times,
            "longitude": pick(ds["long"]),
            "latitude": pick(ds["lat"]),
            "surface_altitude": pick(ds["zobs"]) * 1000.0,
            "surface_pressure": pick(ds["pout"]),
            "zenith": pick(ds["solzen"]),
            "azimuth": _wrap_azimuth(pick(ds["azim"])),
            "value": value,
            "uncertainty": pick(ds[f"{species}_error"]),
            "good": good,
            "ak_pressure": [ak_pressure] * len(ii),
            "ak": _rows(ak),
            "ak_extrapolated": ak_extrapolated,
            "pressure_levels": _rows(prior_pres),
            "altitude_levels": [prior_alt_m] * len(ii),
        }
        if apriori is not None:
            columns["apriori"] = _rows(apriori)
    df = pd.DataFrame(columns, index=pd.RangeIndex(len(ii)))
    df["solar_zenith"] = df["zenith"]
    df["solar_azimuth"] = df["azimuth"]
    df["species"] = species
    df["units"] = units
    return df


__all__ = ["read_ggg_netcdf", "read_ggg_oof"]
