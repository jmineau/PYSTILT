"""
Particles, footprints, and variants for tests of the output directory.

They are small and random but fixed by a seed, on :data:`GRID`, for a
release at the WBB site.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from runtraj.config import Variant
from runtraj.output import Output
from runtraj.releases import PointRelease

from stilt.footprint.config import FootprintConfig
from stilt.footprint.grid import Grid

from .factories import MET, make_release, make_variant
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


def release_at(hour: int = 12, day: int = 15) -> PointRelease:
    """Return the WBB release at *hour* on 2024-07-*day*."""
    return make_release(dt.datetime(2024, 7, day, hour))


def fake_particles(release: PointRelease, n: int = 50) -> pd.DataFrame:
    """Return *n* particles of ten one-minute steps, random but fixed by the release's time."""
    rng = np.random.default_rng(int(release.time.timestamp()) % 1000)
    steps = np.arange(-60, -660, -60, dtype=float)
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
    data["time"] = pd.Timestamp(release.time) + pd.to_timedelta(data["age"], unit="s")
    return data


def fake_footprint(
    release: PointRelease,
    hours=(-2, -1, 0),
    seed: int = 0,
    name: str = "hrrr",
    tres: str = "1h",
) -> xr.DataArray:
    """A footprint on GRID with a few non-zero cells per layer, *hours* layers of *tres* from the release time."""
    x_axis, y_axis = GRID.axes
    rng = np.random.default_rng(seed)
    values = np.zeros((len(hours), len(y_axis), len(x_axis)))
    for t in range(len(hours)):
        for _ in range(4):
            values[t, rng.integers(len(y_axis)), rng.integers(len(x_axis))] = (
                rng.uniform(0.01, 1.0)
            )
    step = pd.Timedelta(tres)
    times = [pd.Timestamp(release.time) + h * step for h in hours]
    data = xr.DataArray(
        values,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": y_axis, "lon": x_axis},
    )
    config = FootprintConfig(grid=GRID, tres=tres)
    return as_footprint(data, release, config, name)


def write_one(out: Output, variant: Variant = VARIANT, release=None) -> Path:
    """Write one release's particles for *variant*, making its folder."""
    release = release if release is not None else release_at()
    return out.write_particles(variant, release, fake_particles(release), [])


def as_written_before_the_rename(path: Path) -> Path:
    """Rewrite a particle or footprint file as PYSTILT wrote it when a release was a receptor."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = ["receptor" if n == "release" else n for n in table.column_names]
    meta = {
        (b"stilt:receptor" if k == b"runtraj:release" else k): v
        for k, v in (table.schema.metadata or {}).items()
    }
    pq.write_table(table.rename_columns(names).replace_schema_metadata(meta), path)
    return path


def as_written_with_hours(path: Path) -> Path:
    """Rewrite a footprint file as PYSTILT wrote it when its layers were named ``hour``."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = ["hour" if n == "layer" else n for n in table.column_names]
    meta = {
        (b"stilt:hours" if k == b"stilt:layers" else k): v
        for k, v in (table.schema.metadata or {}).items()
    }
    pq.write_table(table.rename_columns(names).replace_schema_metadata(meta), path)
    return path
