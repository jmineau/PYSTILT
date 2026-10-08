"""
Particles, footprints, and variants for tests of the output directory.

They are small and random but fixed by a seed, on :data:`GRID`, for a
receptor at the WBB site.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from stilt.config import Variant
from stilt.footprint.config import FootprintConfig
from stilt.output import Output
from stilt.receptors import PointReceptor
from stilt.spatial import Grid

from .factories import MET, make_receptor, make_variant
from .footprints import as_footprint

GRID = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.1, yres=0.1)


def output_variant(
    name: str = "hrrr", footprint: FootprintConfig | None = None, **overrides
) -> Variant:
    """Return a resolved variant on :data:`MET`; *overrides* change the transport fields."""
    return make_variant(name, met_config=MET, footprint=footprint, **overrides)


VARIANT = output_variant()
#: VARIANT with footprints on GRID.
FEET = output_variant(footprint=FootprintConfig(grid=GRID))


def receptor_at(hour: int = 12, day: int = 15) -> PointReceptor:
    """Return the WBB receptor at *hour* on 2024-07-*day*."""
    return make_receptor(dt.datetime(2024, 7, day, hour))


def fake_particles(receptor: PointReceptor, n: int = 50) -> pd.DataFrame:
    """Return *n* particles of ten one-minute steps, random but fixed by the receptor's time."""
    rng = np.random.default_rng(int(receptor.time.timestamp()) % 1000)
    steps = np.arange(-1, -11, -1, dtype=float)
    particle = np.repeat(np.arange(1, n + 1, dtype=float), len(steps))
    time = np.tile(steps, n)
    data = pd.DataFrame(
        {
            "age": time,
            "particle": particle,
            "lon": -111.85 + rng.normal(0, 0.1, len(time)),
            "lat": 40.77 + rng.normal(0, 0.1, len(time)),
            "zagl": rng.uniform(0, 500, len(time)),
            "foot": rng.uniform(0, 0.1, len(time)),
        }
    )
    data["time"] = pd.Timestamp(receptor.time) + pd.to_timedelta(
        data["age"], unit="min"
    )
    return data


def fake_footprint(
    receptor: PointReceptor, hours=(-2, -1, 0), seed: int = 0, name: str = "hrrr"
) -> xr.DataArray:
    """A footprint on GRID with a few non-zero cells per layer."""
    x_axis, y_axis = GRID.axes
    rng = np.random.default_rng(seed)
    values = np.zeros((len(hours), len(y_axis), len(x_axis)))
    for t in range(len(hours)):
        for _ in range(4):
            values[t, rng.integers(len(y_axis)), rng.integers(len(x_axis))] = (
                rng.uniform(0.01, 1.0)
            )
    times = [pd.Timestamp(receptor.time) + pd.Timedelta(hours=h) for h in hours]
    data = xr.DataArray(
        values,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": y_axis, "lon": x_axis},
    )
    return as_footprint(data, receptor, FootprintConfig(grid=GRID), name)


def write_one(out: Output, variant: Variant = VARIANT, receptor=None) -> Path:
    """Write one receptor's particles for *variant*, making its folder."""
    receptor = receptor if receptor is not None else receptor_at()
    return out.write_particles(variant, receptor, fake_particles(receptor), [])
