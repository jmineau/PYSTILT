"""
Cut the committed sample from a FLEXPART 11 run: the first 5 particles, every time.

The run: FLEXPART 11 (git bb09a19), backward 24 h from 2023-06-01 18 UTC, 1000
particles released 25 m above ground at 40.766 N, 111.848 W, on GFS 0.5 degree
analyses and 3 h forecasts, PARTOPTIONS LONGITUDE, LATITUDE, HEIGHT, PV, QV,
DENSITY, TEMPERATURE, PRESSURE, MIXINGHEIGHT, TOPOGRAPHY.

    python make_sample.py <the run's partoutput_20230601180000_init.nc>
"""

import sys
from pathlib import Path

import netCDF4

N_PARTICLES = 5


def main() -> None:
    """Write the sample beside this script."""
    source = sys.argv[1]
    target = Path(__file__).with_name(Path(source).name)
    with netCDF4.Dataset(source) as src, netCDF4.Dataset(target, "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        for name in src.dimensions:
            dst.createDimension(name, None)
        for name, var in src.variables.items():
            fill = (
                var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
            )
            sizes = [
                N_PARTICLES if d == "particle" else len(src.dimensions[d])
                for d in var.dimensions
            ]
            out = dst.createVariable(
                name,
                var.dtype,
                var.dimensions,
                fill_value=fill,
                zlib=True,
                complevel=1,
                chunksizes=sizes,
            )
            out.setncatts(
                {k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"}
            )
            out.set_auto_maskandscale(False)
            var.set_auto_maskandscale(False)
            keep = tuple(
                slice(0, N_PARTICLES) if d == "particle" else slice(None)
                for d in var.dimensions
            )
            out[:] = var[keep]


if __name__ == "__main__":
    main()
