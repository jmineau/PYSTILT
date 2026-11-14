"""
Build the small product files under ``tests/observations/data/products``.

Each sample is a slice of a real granule with only the variables the readers
use, kept under the granule's own file name, so the tests run against the
products' own layouts. Re-run this with the
source files at hand when a product changes::

    uv run python tests/observations/data/products/make_samples.py \
        --tropomi S5P_OFFL_L2__CH4____20231019T192749_....nc \
        --blended S5P_BLND_L2__CH4____20180430T001950_....nc \
        --tccon if20120823_20121201.public.qc.nc \
        --oco2 oco2_LtCO2_260208_B11300Ar_260529230202s.nc4 \
        --oco3 oco3_LtCO2_260208_B11077Ar_260327001412s.nc4

The OCO-2 and OCO-3 Lite files come from NASA GES DISC (an Earthdata login,
free) and are open data: OCO-2 L2 Lite FP v11.2r, doi:10.5067/C4OENLFCYVSA;
OCO-3 L2 Lite FP v11r, doi:10.5067/8U0VGVQC7HZG. Each slice keeps the
granule's global attributes, the DOI among them.

The GGG samples are synthetic too, on the GGG2020 ``.oof`` and
``*.private.nc`` layouts, and so are the COCCON GEOMS sample (the layout of
PROFFASTpylot's GEOMS writer) and the PROFFASTpylot run folder, so no instrument team's retrievals are
redistributed here. They keep the format's quirks, including the .oof header
count that exceeds the number of columns written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

HERE = Path(__file__).parent


def copy_subset(src: Path, dst: Path, keep: set[str], slices: dict[str, slice]) -> None:
    """Copy the variables in ``keep`` (paths like ``PRODUCT/latitude``) with dims sliced."""
    with Dataset(src) as s, Dataset(dst, "w", format="NETCDF4") as d:
        for name in s.ncattrs():
            d.setncattr(name, s.getncattr(name))
        d.setncattr("stilt_sample_source", src.name)
        d.setncattr(
            "stilt_sample_note",
            "Subset of the source granule, made by tests/observations/data/products/make_samples.py "
            "for the PYSTILT test suite. Not for science use.",
        )

        def walk(sg, dg, prefix: str) -> None:
            for dname, dim in sg.dimensions.items():
                sl = slices.get(dname)
                size = len(range(*sl.indices(len(dim)))) if sl else len(dim)
                dg.createDimension(dname, size)
            for vname, var in sg.variables.items():
                if prefix + vname not in keep:
                    continue
                var.set_auto_maskandscale(False)
                var.set_auto_chartostring(False)
                fill = getattr(var, "_FillValue", None)
                is_str = var.dtype is str or getattr(var.dtype, "kind", "") in "US"
                out = dg.createVariable(
                    vname, var.dtype, var.dimensions, fill_value=fill, zlib=not is_str
                )
                out.set_auto_maskandscale(False)  # raw bytes in, raw bytes out
                out.set_auto_chartostring(False)
                for a in var.ncattrs():
                    if a != "_FillValue":
                        out.setncattr(a, var.getncattr(a))
                idx = tuple(slices.get(dn, slice(None)) for dn in var.dimensions)
                out[:] = var[idx] if var.dimensions else var[...]
            for gname, group in sg.groups.items():
                walk(group, dg.createGroup(gname), prefix + gname + "/")

        walk(s, d, "")


def tropomi(src: Path) -> None:
    """Ten by ten pixels of an operational S5P L2 CH4 orbit, around a good pixel over Utah."""
    with Dataset(src) as ds:
        p = ds["PRODUCT"]
        lat = p["latitude"][0]
        lon = p["longitude"][0]
        qa = p["qa_value"][0]
        box = (lon > -113.5) & (lon < -110.5) & (lat > 39.5) & (lat < 42.0) & (qa > 0.5)
        sl0, gp0 = np.argwhere(box)[0]
    keep = {
        "PRODUCT/" + v
        for v in (
            "scanline",
            "ground_pixel",
            "time",
            "corner",
            "layer",
            "level",
            "delta_time",
            "time_utc",
            "qa_value",
            "latitude",
            "longitude",
            "methane_mixing_ratio",
            "methane_mixing_ratio_precision",
            "methane_mixing_ratio_bias_corrected",
        )
    }
    keep |= {
        "PRODUCT/SUPPORT_DATA/GEOLOCATIONS/" + v
        for v in (
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "latitude_bounds",
            "longitude_bounds",
        )
    }
    keep |= {"PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/column_averaging_kernel"}
    keep |= {
        "PRODUCT/SUPPORT_DATA/INPUT_DATA/" + v
        for v in (
            "surface_altitude",
            "surface_pressure",
            "pressure_interval",
            "methane_profile_apriori",
            "altitude_levels",
            "dry_air_subcolumns",
            "surface_classification",
        )
    }
    copy_subset(
        src,
        HERE / src.name,
        keep,
        {"scanline": slice(sl0 - 3, sl0 + 7), "ground_pixel": slice(gp0 - 3, gp0 + 7)},
    )


def blended(src: Path) -> None:
    """The first forty observations of a TROPOMI+GOSAT blended CH4 file (flat layout)."""
    with Dataset(src) as ds:
        keep = set(ds.variables)
    copy_subset(src, HERE / src.name, keep, {"nobs": slice(0, 40)})


def tccon(src: Path) -> None:
    """The first day of a GGG2020 public TCCON file."""
    keep = {
        "time",
        "prior_time",
        "year",
        "day",
        "hour",
        "lat",
        "long",
        "zobs",
        "zmin",
        "pout",
        "tout",
        "solzen",
        "azim",
        "xco2",
        "xco2_error",
        "xch4",
        "xch4_error",
        "xco",
        "xco_error",
        "xh2o",
        "xh2o_error",
        "ak_altitude",
        "ak_pressure",
        "ak_xco2",
        "ak_xch4",
        "ak_xco",
        "extrapolation_flags_ak_xco2",
        "extrapolation_flags_ak_xch4",
        "extrapolation_flags_ak_xco",
        "prior_altitude",
        "prior_pressure",
        "prior_co2",
        "prior_ch4",
        "prior_co",
        "prior_h2o",
        "prior_temperature",
    }
    copy_subset(src, HERE / src.name, keep, {"time": slice(0, 24)})


OOF_COLUMNS = [
    "flag",
    "spectrum",
    "year",
    "day",
    "hour",
    "lat(deg)",
    "long(deg)",
    "zobs(km)",
    "zmin(km)",
    "solzen(deg)",
    "azim(deg)",
    "osds(ppm)",
    "opd(cm)",
    "fovi(rad)",
    "graw(cm-1)",
    "tins(C)",
    "pins(mbar)",
    "tout(C)",
    "pout(hPa)",
    "hout(%RH)",
    "sia(AU)",
    "fvsi(%)",
    "wspd(m/s)",
    "wdir(deg)",
    "xluft",
    "xluft_error",
    "xh2o(ppm)",
    "xh2o(ppm)_error",
    "xth2o(ppm)",
    "xth2o(ppm)_error",
    "xhdo(ppm)",
    "xhdo(ppm)_error",
    "xco(ppb)",
    "xco(ppb)_error",
    "xn2o(ppb)",
    "xn2o(ppb)_error",
    "xch4(ppm)",
    "xch4(ppm)_error",
    "xlco2(ppm)",
    "xlco2(ppm)_error",
    "xwco2(ppm)",
    "xwco2(ppm)_error",
    "xco2(ppm)",
    "xco2(ppm)_error",
    "xo2",
    "xo2_error",
]
#: GGG writes the flag table's variable count here, which is larger than the
#: number of columns actually written (variables with Output=0 are skipped).
#: The readers must take the column line as authoritative, so the sample keeps
#: the discrepancy.
OOF_DECLARED_NVAR = 54
OOF_LON, OOF_LAT, OOF_ZOBS_KM = -111.85, 40.77, 1.45


def oof_synthetic(n: int = 12) -> None:
    """
    A made-up GGG2020 .oof: the real layout and header quirks, invented values.

    One instrument-day of an EM27/SUN as EGI would deliver it. Twelve spectra
    three minutes apart on 2023-07-15, one of them flagged, with the solar
    zenith falling and the azimuth rising through the morning.
    """
    header = [
        "  written by tests/observations/data/products/make_samples.py",
        "  SYNTHETIC FILE - invented values on the GGG2020 official-output layout.",
        "  Not measurements, not for science use.",
        "missing:  9.8765E+35",
        "format:(a57,1x,f13.8,23f13.5,022(1pe13.5))",
        "",
        " # Variable  Output  Scale  Format   Unit     Vmin     Vmax     Description",
        '  1 "year"        1  1.0E+00 "f7.0" "      "   2014.0   2035.0   Year',
        '  2 "day"         1  1.0E+00 "f6.0" "      "   0        367      Day of the year',
        '  3 "hour"        1  1.0E+00 "f8.3" "      "  -12.0     36.0     Fractional UT Hour',
        '  4 "run"         0  1.0E+00 "f6.0" "      "   0        999999   Not output',
        "",
    ]
    rows = []
    for i in range(n):
        hour = 16.0 + i * 0.05  # every three minutes
        flag = 2 if i == 5 else 0  # one flagged spectrum so `good` is exercised
        rows.append(
            f"{flag:3d} zz20230715s0e00a.{i + 1:04d}".ljust(61)
            + f"{2023.0:7.0f}{196.0:6.0f}{hour:8.3f}"
            + f"{OOF_LAT:10.3f}{OOF_LON:10.3f}{OOF_ZOBS_KM:8.2f}{1.44:8.2f}"
            + f"{40.0 - i * 0.5:8.2f}{110.0 + i * 0.8:8.2f}"
            + f"{-0.147:8.3f}{1.78:6.2f}{0.005:7.3f}{0.2411:8.4f}"
            + f"{27.0:7.1f}{853.3:8.1f}{19.0:7.1f}{853.3:8.1f}{33.0:7.1f}"
            + f"{67.3:8.1f}{0.160:8.3f}{0.0:6.1f}{329.0:6.0f}"
            + f"{0.9988 + i * 1e-4:9.4f}{0.0011:8.4f}"
            + f"{1920.0 + i:10.2f}{3.11:9.2f}{1990.0 + i:10.2f}{5.46:9.2f}"
            + f"{1444.0 + i:10.2f}{4.33:9.2f}"
            + f"{87.8 + i * 0.1:8.1f}{0.6:8.1f}{325.52:9.2f}{0.99:8.2f}"
            + f"{1.8698 + i * 1e-4:10.4f}{0.0020:9.4f}"
            + f"{419.38:10.2f}{0.53:9.2f}{425.68:10.2f}{0.84:9.2f}"
            + f"{419.42:10.2f}{0.43:9.2f}{0.2095:8.4f}{0.0:8.4f}"
        )
    nhead = len(header) + 2  # count line + header + the column line
    body = [
        f"{nhead:4d} {OOF_DECLARED_NVAR:11d}",
        *header,
        "  " + "  ".join(OOF_COLUMNS),
        *rows,
    ]
    (HERE / "zz20230715.vav.ada.aia.oof").write_text("\n".join(body) + "\n")


def ggg_private_synthetic() -> None:
    """
    A made-up GGG2020 *.private.nc: the real layout, invented values.

    Four spectra sharing two prior profiles through ``prior_index``, with the
    averaging kernels stored the way a private file does - a table against
    slant xgas that the reader interpolates per spectrum.
    """
    nt, nlev, nbin = 4, 51, 15
    dst = HERE / "zz20230715_20230715.private.nc"
    alt_km = np.linspace(0.0, 70.0, nlev)
    pres_atm = 1.0025 * np.exp(-alt_km / 8.5)  # surface-first, monotonically falling
    with Dataset(dst, "w", format="NETCDF4") as d:
        d.setncattr("source", "SYNTHETIC GGG2020 private layout for PYSTILT tests")
        d.setncattr("long_name", "synthetic_site")
        d.setncattr(
            "stilt_sample_note",
            "Invented values on the GGG2020 *.private.nc layout, made by "
            "tests/observations/data/products/make_samples.py. Not measurements.",
        )
        d.createDimension("time", nt)
        d.createDimension("prior_time", 2)
        d.createDimension("prior_altitude", nlev)
        d.createDimension("ak_altitude", nlev)
        d.createDimension("ak_slant_xgas_bin", nbin)
        d.createDimension("specname", 21)

        def var(name, dims, values, **attrs):
            v = d.createVariable(name, "f4" if values.dtype.kind == "f" else "i2", dims)
            for k, a in attrs.items():
                v.setncattr(k, a)
            v[:] = values
            return v

        t0 = np.datetime64("2023-07-15T16:00:00")
        secs = (
            (t0 - np.datetime64("1970-01-01T00:00:00"))
            .astype("timedelta64[s]")
            .astype(float)
        )
        tv = d.createVariable("time", "f8", ("time",))
        tv.setncattr("units", "seconds since 1970-01-01 00:00:00")
        tv[:] = secs + np.arange(nt) * 180.0
        pv = d.createVariable("prior_time", "f8", ("prior_time",))
        pv.setncattr("units", "seconds since 1970-01-01 00:00:00")
        pv[:] = secs + np.array([0.0, 600.0])

        spec = d.createVariable("spectrum", "S1", ("time", "specname"))
        spec.setncattr("_Encoding", "ascii")
        spec[:] = np.array(
            [list(f"zz20230715s0e00a.{i + 1:04d}".ljust(21)) for i in range(nt)],
            dtype="S1",
        )

        var("lat", ("time",), np.full(nt, OOF_LAT, "f4"), units="degrees_north")
        var("long", ("time",), np.full(nt, OOF_LON, "f4"), units="degrees_east")
        var("zobs", ("time",), np.full(nt, 0.24, "f4"), units="km")
        var("pout", ("time",), np.full(nt, 985.0, "f4"), units="hPa")
        var(
            "solzen",
            ("time",),
            np.linspace(79.9, 80.6, nt).astype("f4"),
            units="degrees",
        )
        var(
            "azim",
            ("time",),
            np.linspace(100.0, 103.0, nt).astype("f4"),
            units="degrees",
        )
        var("zmin", ("time",), np.full(nt, 0.23, "f4"), units="km")
        var(
            "flag",
            ("time",),
            np.zeros(nt, "i2"),
            description="data quality flag, 0 = good",
        )
        var(
            "prior_index",
            ("time",),
            np.array([0, 0, 1, 1], "i2"),
            description="Index of the prior profile associated with each measurement",
        )

        for gas, val in (("ch4", 1.87e-6), ("co2", 415.0e-6), ("co", 90e-9)):
            v = f"x{gas}"
            var(
                v,
                ("time",),
                np.full(nt, val * 1e6 if gas != "co" else val * 1e9, "f4"),
                units="ppm" if gas != "co" else "ppb",
            )
            var(
                f"{v}_error",
                ("time",),
                np.full(nt, 0.002 if gas == "ch4" else 0.5, "f4"),
                units="ppm" if gas != "co" else "ppb",
            )
            # prior profiles in "parts" (plain mole fraction), as a private file writes them
            prof = np.outer(np.ones(2), val * np.linspace(1.0, 0.75, nlev)).astype("f4")
            var(
                f"prior_1{gas}",
                ("prior_time", "prior_altitude"),
                prof,
                units="",
                description=f"a priori concentration profile of 1{gas}, in parts",
            )
            # kernel table against slant xgas
            xg = val * 1e6 if gas != "co" else val * 1e9
            bins = np.linspace(xg * 1.05, xg * 18.0, nbin).astype("f4")
            var(f"ak_slant_{v}_bin", ("ak_slant_xgas_bin",), bins)
            table = (
                0.95
                + 0.05 * np.linspace(0, 1, nlev)[:, None]
                + 0.02 * np.linspace(0, 1, nbin)[None, :]
            ).astype("f4")
            var(f"ak_{v}", ("ak_altitude", "ak_slant_xgas_bin"), table)

        var("prior_altitude", ("prior_altitude",), alt_km.astype("f4"), units="km")
        var(
            "prior_pressure",
            ("prior_time", "prior_altitude"),
            np.outer(np.ones(2), pres_atm).astype("f4"),
            units="atm",
        )
        var("ak_altitude", ("ak_altitude",), alt_km.astype("f4"), units="km")
        var(
            "ak_pressure",
            ("ak_altitude",),
            (pres_atm * 1013.25).astype("f4"),
            units="hPa",
        )
        # O2-window airmass: what the per-spectrum kernel is interpolated at
        var("o2_7885_am_o2", ("time",), np.full(nt, 5.53, "f4"))


#: Every variable read_oco2 uses, by its path in a Lite file.
OCO_KEEP = {
    "sounding_id",
    "levels",
    "time",
    "latitude",
    "longitude",
    "xco2",
    "xco2_uncertainty",
    "xco2_apriori",
    "xco2_quality_flag",
    "solar_zenith_angle",
    "sensor_zenith_angle",
    "pressure_levels",
    "pressure_weight",
    "xco2_averaging_kernel",
    "co2_profile_apriori",
    "vertex_latitude",
    "vertex_longitude",
    "Sounding/solar_azimuth_angle",
    "Sounding/sensor_azimuth_angle",
    "Sounding/altitude",
    "Sounding/land_fraction",
    "Sounding/operation_mode",
    "Sounding/orbit",
    "Retrieval/psurf",
}


def oco(
    src: Path, box: tuple[float, float, float, float], mode: int, n: int = 12
) -> None:
    """
    The first ``n`` consecutive soundings of an OCO-2 or OCO-3 Lite file in a box.

    ``box`` is ``(lon_min, lon_max, lat_min, lat_max)``, and every sounding in
    the slice must be in operation mode ``mode`` (0 nadir, 1 glint, 2 target,
    3 transition, 4 SAM).
    """
    lon0, lon1, lat0, lat1 = box
    with Dataset(src) as ds:
        lon = ds["longitude"][:]
        lat = ds["latitude"][:]
        ok = (
            (lon > lon0)
            & (lon < lon1)
            & (lat > lat0)
            & (lat < lat1)
            & (ds["Sounding/operation_mode"][:] == mode)
        )
    start = next(i for i in np.flatnonzero(ok) if ok[i : i + n].all())
    copy_subset(
        src, HERE / src.name, OCO_KEEP, {"sounding_id": slice(start, start + n)}
    )


#: The synthetic COCCON GEOMS file, named the way COCCON names its files.
GEOMS_NAME = (
    "groundbased_ftir.coccon_zz000_synthetic_20230715t160000z_20230715t161200z_001.h5"
)


def geoms_synthetic() -> None:
    """
    A made-up COCCON GEOMS file: the real layout, invented values.

    Five EM27/SUN measurements three minutes apart on 2023-07-15 on the
    GEOMS-TE-FTIR-COCCON-001 template that PROFFASTpylot writes: dotted
    variable names, ``VAR_*`` attributes, unnamed dimensions, and a CH4
    prior in ppbv against a column in ppmv. The third XCH4 is the fill
    value. Ten levels stand in for the real 49.
    """
    nt, nlev = 5, 10
    fill = np.float32(-900000.0)
    alt_km = np.linspace(OOF_ZOBS_KM, 30.0, nlev)
    pres = 853.3 * np.exp(-(alt_km - OOF_ZOBS_KM) / 8.0)
    sza = 40.0 - 0.5 * np.arange(nt)
    with Dataset(HERE / GEOMS_NAME, "w", format="NETCDF4") as d:
        d.setncattr("DATA_LOCATION", "SYNTHETIC")
        d.setncattr("DATA_SOURCE", "FTIR.COCCON_ZZ000")
        d.setncattr("DATA_TEMPLATE", "GEOMS-TE-FTIR-COCCON-001")
        d.setncattr("DATA_PROCESSOR", "PROFFAST Version 2.4")
        d.setncattr(
            "stilt_sample_note",
            "Invented values on the COCCON GEOMS layout, made by "
            "tests/observations/data/products/make_samples.py. Not measurements.",
        )
        d.createDimension("phony_dim_0", nt)
        d.createDimension("phony_dim_1", nlev)
        dims1, dims2 = ("phony_dim_0",), ("phony_dim_0", "phony_dim_1")

        def var(name, values, units, dtype="f4"):
            v = d.createVariable(
                name,
                dtype,
                dims2 if np.ndim(values) == 2 else dims1,
                fill_value=np.array(fill, dtype=dtype),
            )
            v.setncattr("VAR_NAME", name)
            v.setncattr("VAR_UNITS", units)
            v.setncattr("units", units)
            v.setncattr("VAR_FILL_VALUE", np.array(fill, dtype=dtype))
            v.setncattr(
                "VAR_DEPEND",
                "DATETIME;ALTITUDE" if np.ndim(values) == 2 else "DATETIME",
            )
            v[:] = values
            return v

        def per_level(profile):
            return np.tile(np.asarray(profile, dtype="f4"), (nt, 1))

        mjd2k = (
            np.datetime64("2023-07-15T16:00:00") - np.datetime64("2000-01-01")
        ) / np.timedelta64(1, "D")
        var(
            "DATETIME",
            np.round(mjd2k + np.arange(nt) * 180.0 / 86400.0, 5),
            "MJD2K",
            "f8",
        )
        var("LATITUDE.INSTRUMENT", np.full(nt, OOF_LAT), "deg")
        var("LONGITUDE.INSTRUMENT", np.full(nt, OOF_LON), "deg")
        var("ALTITUDE.INSTRUMENT", np.full(nt, OOF_ZOBS_KM), "km")
        var("ANGLE.SOLAR_ZENITH.ASTRONOMICAL", sza, "deg")
        var("ANGLE.SOLAR_AZIMUTH", 110.0 + 0.8 * np.arange(nt), "deg")
        var("SURFACE.PRESSURE_INDEPENDENT", np.full(nt, 853.3), "hPa")
        var("ALTITUDE", per_level(alt_km), "km")
        var("PRESSURE_INDEPENDENT", per_level(pres), "hPa")
        var("TEMPERATURE_INDEPENDENT", per_level(np.linspace(300.0, 220.0, nlev)), "K")
        for gas, column, col_units, prior_units, prior_surface in (
            ("CH4", 1.8698, "ppmv", "ppbv", 1990.0),
            ("CO2", 419.38, "ppmv", "ppmv", 421.0),
            ("CO", 87.8, "ppbv", "ppbv", 110.0),
            ("H2O", 1920.0, "ppmv", "ppmv", 8000.0),
        ):
            name = f"{gas}.COLUMN.MIXING.RATIO.VOLUME.DRY_ABSORPTION.SOLAR"
            values = column + 1e-3 * column * np.arange(nt)
            if gas == "CH4":
                values[2] = fill
            var(name, values, col_units)
            var(
                f"{name}_UNCERTAINTY.RANDOM.STANDARD",
                np.full(nt, 0.001 * column),
                col_units,
            )
            # the kernel falls with height and grows near the surface at high sun
            ak = np.outer(1.0 + 0.002 * sza, np.linspace(1.05, 0.6, nlev))
            var(f"{gas}.COLUMN_ABSORPTION.SOLAR_AVK", ak, "1")
            var(
                f"{gas}.MIXING.RATIO.VOLUME.DRY_APRIORI",
                per_level(prior_surface * np.linspace(1.0, 0.8, nlev)),
                prior_units,
            )


#: The synthetic PROFFASTpylot results folder, named the way PROFFASTpylot names it.
PROFFAST_RUN = "ZZsynthetic_SN000_230715-230716"
#: The angles (radians) of a real colsens.dat table.
COLSENS_SZA = [
    0.0,
    0.3965,
    0.5607,
    0.6867,
    0.7930,
    0.8866,
    0.9712,
    1.049,
    1.121,
    1.189,
    1.254,
    1.315,
    1.373,
    1.430,
    1.484,
]


def proffast_synthetic() -> None:
    """
    A made-up PROFFASTpylot run: the real file layouts, invented values.

    Two days, three spectra each, on ten levels (a real run has 49): the
    ``comb_invparms`` CSV with its comma-and-space columns, and each day's
    ``colsens.dat`` (gas names in quotes, the long ones without the closing
    quote), ``pT_fast_out.dat`` (spaces inside the header's names) and
    ``VMR_fast_out.dat`` (no header). One spectrum is past the table's last
    angle, and one has an XAIR outside COCCON's screen.
    """
    run = HERE / PROFFAST_RUN
    raw = run / "raw_output_proffast"
    raw.mkdir(parents=True, exist_ok=True)
    nlev = 10
    alt_m = np.linspace(OOF_ZOBS_KM * 1000.0, 30000.0, nlev)
    pres_pa = 85330.0 * np.exp(-(alt_m - alt_m[0]) / 8000.0)
    gases = ["H2O", "HDO", "CO2", "CO2_STR", "CH4", "CH4_S5P", "N2O", "CO", "O2", "HF"]
    for day in ("230715", "230716"):
        lines = []
        for k, gas in enumerate(gases):
            name = f"'{gas}'" if len(gas) < 7 else f"'{gas}"
            lines += [
                " ",
                f"{name:<8s}",
                "$",
                "          SZA[rad] ->   " + "".join(f"{a:13.3E}" for a in COLSENS_SZA),
                " ",
                " alt [km]     p [mbar]",
            ]
            for z, p in zip(alt_m, pres_pa, strict=True):
                # falls with height, grows with the sun's angle; gas k offsets it
                sens = (1.05 - 0.45 * (z - alt_m[0]) / (alt_m[-1] - alt_m[0])) * (
                    1.0 + 0.2 * np.asarray(COLSENS_SZA)
                ) + 0.01 * k
                lines.append(
                    f"{z / 1000:11.3E}{p / 100:13.3E}"
                    + "".join(f"{v:13.3E}" for v in sens)
                )
        (raw / f"ZZsynthetic{day}-colsens.dat").write_text("\n".join(lines) + "\n")
        pt = [
            "Level index    Altitude(m)       Temperature(K)    Pressure(Pa)      dry air column    H2O[ppmv]         HDO[ppmv]"
        ]
        vmr = []
        for i, (z, p) in enumerate(zip(alt_m, pres_pa, strict=True)):
            t = 300.0 - 6.5 * (z - alt_m[0]) / 1000.0
            h2o = 8000.0 * np.exp(-(z - alt_m[0]) / 2000.0)
            pt.append(
                f"{i + 1:8d}{z:18.5E}{t:18.5E}{p:18.5E}{3e27:18.5E}{h2o:18.5E}{0.8 * h2o:18.5E}"
            )
            frac = i / (nlev - 1)
            vmr.append(
                f"{i + 1:4d}{z:18.5E}{h2o:18.5E}{0.8 * h2o:18.5E}{421.0 - 5 * frac:18.5E}"
                f"{1.99 - 0.4 * frac:18.5E}{0.33:18.5E}{0.11 - 0.05 * frac:18.5E}{2.0946e5:18.5E}{1e-7:18.5E}"
            )
        (raw / f"ZZsynthetic{day}-pT_fast_out.dat").write_text("\n".join(pt) + "\n")
        (raw / f"ZZsynthetic{day}-VMR_fast_out.dat").write_text("\n".join(vmr) + "\n")

    columns = [
        "UTC",
        "LocalTime",
        "JulianDate",
        "YYMMDD",
        "UTtimeh",
        "spectrum",
        "gndPmap",
        "gndP",
        "gndTmap",
        "gndT",
        "latdeg",
        "londeg",
        "altim",
        "appSZA",
        "azimuth",
        "XH2O",
        "XAIR",
        "XCO2",
        "XCH4",
        "XCO2_STR",
        "XCH4_S5P",
        "XCO",
        "job01_rms",
    ]
    rows = []
    #: (day, hour, minute, apparent SZA, PROFFAST azimuth from south, XAIR)
    spectra = [
        ("230715", 16, 0, 40.0, -70.0, 1.0001),
        ("230715", 16, 3, 39.5, -69.2, 0.9700),  # fails the XAIR screen
        ("230715", 16, 6, 39.0, -68.4, 1.0003),
        ("230716", 1, 30, 86.0, 115.0, 1.0002),  # past the table's last angle
        ("230716", 16, 0, 40.2, -70.1, 0.9999),
        ("230716", 16, 3, 39.7, -69.3, 1.0000),
    ]
    for i, (day, hh, mm, sza, azi, xair) in enumerate(spectra):
        t = pd.Timestamp(f"20{day[:2]}-{day[2:4]}-{day[4:]} {hh:02d}:{mm:02d}:00")
        rows.append(
            [
                str(t),
                str(t - pd.Timedelta(hours=6)),
                f"{t.to_julian_date():.5f}",
                day,
                f"{hh + mm / 60:6.1f}",
                f"{day}_{hh:02d}{mm:02d}00SN.BIN",
                "853.30",
                "853.30",
                "300.00",
                "300.00",
                f"{OOF_LAT:.5f}",
                f"{OOF_LON:.5f}",
                f"{OOF_ZOBS_KM * 1000:.3f}",
                f"{sza:.2f}",
                f"{azi:.2f}",
                f"{1920.0 + i:.5e}",
                f"{xair:.5e}",
                f"{419.38 + 0.1 * i:.5e}",
                f"{1.8698 + 1e-4 * i:.5e}",
                f"{419.10 + 0.1 * i:.5e}",
                f"{1.8710 + 1e-4 * i:.5e}",
                f"{0.0878 + 1e-4 * i:.5e}",
                f"{1.8e-3:.5e}",
            ]
        )
    text = ", ".join(columns) + "\n" + "\n".join(", ".join(r) for r in rows) + "\n"
    (run / f"comb_invparms_{PROFFAST_RUN}.csv").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tropomi", type=Path)
    ap.add_argument("--blended", type=Path)
    ap.add_argument("--tccon", type=Path)
    ap.add_argument("--oco2", type=Path)
    ap.add_argument("--oco3", type=Path)
    a = ap.parse_args()
    if a.tropomi:
        tropomi(a.tropomi)
    if a.blended:
        blended(a.blended)
    if a.tccon:
        tccon(a.tccon)
    if a.oco2:
        oco(a.oco2, box=(-95.2, -94.7, 29.2, 29.6), mode=0)  # nadir, near Houston
    if a.oco3:
        oco(a.oco3, box=(-97.6, -97.0, 36.3, 36.9), mode=2)  # target, Lamont (SGP)
    # The GGG samples need no source file: they are synthetic by construction.
    oof_synthetic()
    ggg_private_synthetic()
    geoms_synthetic()
    proffast_synthetic()


if __name__ == "__main__":
    main()
