"""
Build the small product files under ``tests/data/products``.

Each sample is a slice of a real granule with only the variables the readers
use, kept under the granule's own file name, so the tests run against the
products' own layouts. Re-run this with the
source files at hand when a product changes::

    uv run python tests/data/products/make_samples.py \
        --tropomi S5P_OFFL_L2__CH4____20231019T192749_....nc \
        --blended S5P_BLND_L2__CH4____20180430T001950_....nc \
        --tccon if20120823_20121201.public.qc.nc

The OCO-2 sample is synthetic: it follows the OCO-2 Lite v11 layout (variable
names, groups, dimensions, units) with made-up values, because Lite files sit
behind an Earthdata login. Replace it with a real slice when one is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
            "Subset of the source granule, made by tests/data/products/make_samples.py "
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
                fill = getattr(var, "_FillValue", None)
                is_str = var.dtype is str or getattr(var.dtype, "kind", "") in "US"
                out = dg.createVariable(
                    vname, var.dtype, var.dimensions, fill_value=fill, zlib=not is_str
                )
                out.set_auto_maskandscale(False)  # raw bytes in, raw bytes out
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


def oco2_synthetic() -> None:
    """A made-up OCO-2 Lite v11 file: the real layout, invented values."""
    rng = np.random.default_rng(0)
    n, nlev = 6, 20
    dst = HERE / "oco2_LtCO2_231019_B11210Ar_synthetic.nc4"
    with Dataset(dst, "w", format="NETCDF4") as d:
        d.setncattr("title", "Synthetic OCO-2 Lite layout for PYSTILT tests")
        d.setncattr(
            "stilt_sample_note",
            "Synthetic file following the OCO-2 L2 Lite FP v11 layout with invented "
            "values, made by tests/data/products/make_samples.py. Not a real granule.",
        )
        d.createDimension("sounding_id", n)
        d.createDimension("levels", nlev)
        d.createDimension("vertices", 4)
        d.createDimension("epoch_dimension", 7)

        def var(group, name, dtype, dims, values, **attrs):
            fill = attrs.pop("_FillValue", None)
            v = group.createVariable(name, dtype, dims, fill_value=fill)
            for k, a in attrs.items():
                v.setncattr(k, a)
            v[:] = values
            return v

        base = np.datetime64("2023-10-19T19:48:00")
        secs = (base - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        var(
            d,
            "sounding_id",
            "i8",
            ("sounding_id",),
            2023101919480000 + np.arange(n) * 10,
            units="1",
            long_name="sounding ID",
        )
        var(d, "levels", "f4", ("levels",), np.arange(1, nlev + 1), units="1")
        var(
            d,
            "time",
            "f8",
            ("sounding_id",),
            secs + np.arange(n) * 0.333,
            units="seconds since 1970-01-01 00:00:00",
            long_name="time",
        )
        var(
            d,
            "latitude",
            "f4",
            ("sounding_id",),
            40.70 + 0.02 * np.arange(n),
            units="degrees_north",
        )
        var(
            d,
            "longitude",
            "f4",
            ("sounding_id",),
            -111.90 + 0.01 * np.arange(n),
            units="degrees_east",
        )
        xco2 = np.array([418.1, 418.4, -999999.0, 419.0, 418.8, 418.2], dtype="f4")
        var(
            d,
            "xco2",
            "f4",
            ("sounding_id",),
            xco2,
            units="ppm",
            _FillValue=np.float32(-999999.0),
        )
        var(
            d,
            "xco2_uncertainty",
            "f4",
            ("sounding_id",),
            np.full(n, 0.5, "f4"),
            units="ppm",
        )
        var(
            d,
            "xco2_apriori",
            "f4",
            ("sounding_id",),
            np.full(n, 417.0, "f4"),
            units="ppm",
        )
        var(
            d,
            "xco2_quality_flag",
            "i1",
            ("sounding_id",),
            np.array([0, 0, 1, 0, 1, 0], "i1"),
            comment="0=good, 1=bad",
        )
        var(
            d,
            "solar_zenith_angle",
            "f4",
            ("sounding_id",),
            np.full(n, 52.0, "f4"),
            units="degrees",
        )
        var(
            d,
            "sensor_zenith_angle",
            "f4",
            ("sounding_id",),
            np.full(n, 20.0, "f4"),
            units="degrees",
        )
        # levels 1..20 from space to the surface; level 20 pressure = psurf
        psurf = np.array([868.0, 866.0, 870.0, 865.0, 869.0, 867.0], "f4")
        sigma = np.linspace(0.0001, 1.0, nlev)
        plev = psurf[:, None] * sigma[None, :]
        var(
            d,
            "pressure_levels",
            "f4",
            ("sounding_id", "levels"),
            plev,
            units="hPa",
            comment="level 20 is the surface",
        )
        ak = np.tile(np.linspace(0.3, 1.1, nlev), (n, 1)).astype("f4")
        var(d, "xco2_averaging_kernel", "f4", ("sounding_id", "levels"), ak, units="1")
        pw = np.gradient(sigma) / np.gradient(sigma).sum()
        var(
            d,
            "pressure_weight",
            "f4",
            ("sounding_id", "levels"),
            np.tile(pw, (n, 1)),
            units="1",
        )
        var(
            d,
            "co2_profile_apriori",
            "f4",
            ("sounding_id", "levels"),
            np.tile(np.linspace(405.0, 420.0, nlev), (n, 1)),
            units="ppm",
        )
        var(
            d,
            "vertex_latitude",
            "f4",
            ("sounding_id", "vertices"),
            40.70 + 0.02 * np.arange(n)[:, None] + np.array([-0.01, -0.01, 0.01, 0.01]),
        )
        var(
            d,
            "vertex_longitude",
            "f4",
            ("sounding_id", "vertices"),
            -111.90
            + 0.01 * np.arange(n)[:, None]
            + np.array([-0.01, 0.01, 0.01, -0.01]),
        )
        s = d.createGroup("Sounding")
        var(
            s,
            "solar_azimuth_angle",
            "f4",
            ("sounding_id",),
            np.full(n, 205.0, "f4"),
            units="degrees",
        )
        var(
            s,
            "sensor_azimuth_angle",
            "f4",
            ("sounding_id",),
            np.full(n, 98.0, "f4"),
            units="degrees",
            comment="degrees East of North",
        )
        var(
            s,
            "altitude",
            "f4",
            ("sounding_id",),
            np.full(n, 1300.0, "f4") + rng.normal(0, 5, n).astype("f4"),
            units="m",
            long_name="surface altitude",
        )
        var(s, "land_fraction", "f4", ("sounding_id",), np.full(n, 100.0, "f4"))
        var(
            s,
            "operation_mode",
            "i1",
            ("sounding_id",),
            np.zeros(n, "i1"),
            comment="0=Nadir,1=Glint,2=Target,3=Transition,4=SAM",
        )
        var(s, "orbit", "i4", ("sounding_id",), np.full(n, 49123, "i4"))
        var(s, "footprint", "i1", ("sounding_id",), np.arange(1, n + 1, dtype="i1"))
        r = d.createGroup("Retrieval")
        var(r, "psurf", "f4", ("sounding_id",), psurf, units="hPa")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tropomi", type=Path)
    ap.add_argument("--blended", type=Path)
    ap.add_argument("--tccon", type=Path)
    ap.add_argument("--no-oco2", action="store_true")
    a = ap.parse_args()
    if a.tropomi:
        tropomi(a.tropomi)
    if a.blended:
        blended(a.blended)
    if a.tccon:
        tccon(a.tccon)
    if not a.no_oco2:
        oco2_synthetic()


if __name__ == "__main__":
    main()
