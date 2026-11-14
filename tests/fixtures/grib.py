"""Small GFS-like GRIB2 files for tests, written with eccodes."""

import numpy as np
import pandas as pd

#: Every column: ground at 500 m with 950 hPa, the 900 hPa level at 1000 m,
#: 800 hPa at 2000 m, and 1000 hPa below the ground at 100 m.
LEVELS = {1000: 100.0, 900: 1000.0, 800: 2000.0}


def write_gfs(path, time, sp=95000.0, orog=500.0, levels=None, global_grid=False):
    """Write a GFS-like GRIB2 file: sp and orog at the surface, gh on pressure levels."""
    levels = LEVELS if levels is None else levels
    if global_grid:
        grid = {
            "Ni": 4,
            "Nj": 3,
            "lat": (45.0, 35.0),
            "lon": (0.0, 270.0),
            "step": (90.0, 5.0),
        }
    else:
        grid = {
            "Ni": 3,
            "Nj": 3,
            "lat": (42.0, 40.0),
            "lon": (248.0, 250.0),
            "step": (1.0, 1.0),
        }
    fields = [("sp", "surface", 0, sp), ("orog", "surface", 0, orog)]
    fields += [("gh", "isobaricInhPa", p, h) for p, h in levels.items()]
    import eccodes

    stamp = pd.Timestamp(time)
    with open(path, "wb") as f:
        for name, level_type, level, value in fields:
            h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            keys = {
                "Ni": grid["Ni"],
                "Nj": grid["Nj"],
                "latitudeOfFirstGridPointInDegrees": grid["lat"][0],
                "latitudeOfLastGridPointInDegrees": grid["lat"][1],
                "longitudeOfFirstGridPointInDegrees": grid["lon"][0],
                "longitudeOfLastGridPointInDegrees": grid["lon"][1],
                "iDirectionIncrementInDegrees": grid["step"][0],
                "jDirectionIncrementInDegrees": grid["step"][1],
                "dataDate": int(stamp.strftime("%Y%m%d")),
                "dataTime": int(stamp.strftime("%H%M")),
                "centre": "kwbc",
                "typeOfLevel": level_type,
            }
            for key, val in keys.items():
                eccodes.codes_set(h, key, val)
            if level:
                eccodes.codes_set(h, "level", level)
            eccodes.codes_set(h, "shortName", name)
            values = np.broadcast_to(
                np.asarray(value, dtype=float), grid["Ni"] * grid["Nj"]
            )
            eccodes.codes_set_values(h, np.ascontiguousarray(values))
            eccodes.codes_write(h, f)
            eccodes.codes_release(h)
    return path
