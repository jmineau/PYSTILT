"""
CF-1.10 trajectories from a particle table, behind ``particles.stilt.to_cf_trajectories``.

The particle table has one row per particle per output step. This module
turns it into an :class:`xarray.Dataset` that follows the CF conventions for
trajectories (``featureType: trajectory``), which trajan, cf-xarray and
other CF tools read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# Columns every particle table has (``stilt.particles.PARTICLE_SCHEMA``).
_REQUIRED_COLUMNS = ("particle", "age", "time", "lon", "lat", "zagl")

# Units and long names of the columns PYSTILT and HYSPLIT write, for the
# variables of the trajectory dataset. A column not listed here is exported
# without units.
_COLUMN_ATTRS: dict[str, dict[str, str]] = {
    "foot": {
        "units": "ppm m2 s umol-1",
        "long_name": "sensitivity to surface fluxes over the output step",
    },
    "foot_no_hnf_dilution": {
        "units": "ppm m2 s umol-1",
        "long_name": "foot before the near-field correction",
    },
    "mlht": {"units": "m", "long_name": "mixed-layer height above ground"},
    "dens": {"units": "kg m-3", "long_name": "air density"},
    "samt": {"units": "minutes", "long_name": "time spent in the surface layer"},
    "sigw": {"units": "m s-1", "long_name": "standard deviation of vertical wind"},
    "tlgr": {"units": "s", "long_name": "Lagrangian time scale"},
    "pres": {"units": "hPa", "long_name": "air pressure"},
    "zsfc": {"units": "m", "long_name": "terrain height above sea level"},
    "zcfx": {"units": "m", "long_name": "vertical displacement by convective flux"},
    "release_height": {"units": "m", "long_name": "release height"},
}


def to_cf_trajectories(particles: pd.DataFrame) -> xr.Dataset:
    """Build the trajectory dataset; see ``ParticlesAccessor.to_cf_trajectories``."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in particles.columns]
    if missing:
        raise ValueError(f"The particle table has no column {', '.join(missing)}.")
    if "realization" in particles.columns and particles["realization"].nunique() > 1:
        raise ValueError(
            "The particle table holds several realizations. Export one at a time: "
            "particles[particles.realization == k]."
        )

    table = particles.copy()
    table["time"] = _naive_utc(table["time"])
    if table.duplicated(["particle", "age"]).any():
        raise ValueError("A particle has two rows at the same age.")

    # Rows from the release onward: |age| grows along obs, backward or forward.
    table = table.assign(_step=table["age"].abs()).sort_values(["particle", "_step"])
    ids, row = np.unique(table["particle"].to_numpy(), return_inverse=True)
    obs = table.groupby("particle", sort=False).cumcount().to_numpy()
    shape = (len(ids), int(obs.max()) + 1 if len(obs) else 0)

    def grid(values: np.ndarray) -> np.ndarray:
        """Place one column's values at (trajectory, obs), missing elsewhere."""
        fill = np.datetime64("NaT", "ns") if values.dtype.kind == "M" else np.nan
        out = np.full(shape, fill, dtype=values.dtype if fill is not np.nan else float)
        out[row, obs] = values
        return out

    dims = ("trajectory", "obs")
    coords = {
        "trajectory": (
            "trajectory",
            ids.astype(np.int32),
            {"cf_role": "trajectory_id", "long_name": "particle number"},
        ),
        "time": (
            dims,
            grid(table["time"].to_numpy(dtype="datetime64[ns]")),
            {"standard_name": "time", "long_name": "time"},
        ),
        "lon": (
            dims,
            grid(table["lon"].to_numpy(dtype=float)),
            {
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
            },
        ),
        "lat": (
            dims,
            grid(table["lat"].to_numpy(dtype=float)),
            {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
            },
        ),
        "zagl": (
            dims,
            grid(table["zagl"].to_numpy(dtype=float)),
            {
                "standard_name": "height",
                "long_name": "height above ground",
                "units": "m",
                "positive": "up",
            },
        ),
    }
    data_vars = {
        "age": (
            dims,
            grid(table["age"].to_numpy(dtype=float)),
            {"long_name": "time since release", "units": "minutes"},
        )
    }
    skip = {*_REQUIRED_COLUMNS, "time", "_step", "realization"}
    for name in table.columns:
        if name in skip or not pd.api.types.is_numeric_dtype(table[name]):
            continue
        attrs = _COLUMN_ATTRS.get(str(name), {})
        values = table[name].to_numpy(dtype=float)
        data_vars[str(name)] = (dims, grid(values), dict(attrs))

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs = {
        "Conventions": "CF-1.10",
        "featureType": "trajectory",
        "title": "PYSTILT particle trajectories",
        "source": "PYSTILT particle table",
        "history": "Converted from a PYSTILT particle table",
    }
    if "realization" in table.columns and len(table):
        ds.attrs["realization"] = int(table["realization"].iloc[0])
    # Whole minutes since the first time, as the particle table counts them.
    first = table["time"].min()
    if not pd.isna(first):
        stamp = first.strftime("%Y-%m-%d %H:%M:%S")
        ds.variables["time"].encoding.update(
            {
                "units": f"minutes since {stamp}",
                "calendar": "standard",
                "dtype": "float64",
                "_FillValue": np.nan,
            }
        )
    return ds


def _naive_utc(values: pd.Series) -> pd.Series:
    """Return times as naive UTC; naive input is taken as UTC already."""
    times = pd.to_datetime(values)
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    return times
