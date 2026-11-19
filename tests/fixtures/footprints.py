"""Footprint arrays made from given values, for tests that need one without particles."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
from runtraj.execution.worker import (
    run_stage as _run_stage,  # the worker's own, whatever a test puts in its place
)
from runtraj.releases import PointRelease, Release
from runtraj.spatial import Grid

from stilt.footprint.config import FootprintConfig
from stilt.footprint.io import _describe, read_footprint
from stilt.footprint.targets import Mesh


def as_footprint(
    data: xr.DataArray,
    release: Release,
    config: FootprintConfig,
    name: str = "",
) -> xr.DataArray:
    """
    Return *data* as the footprint of *release*, with the attributes ``calc_footprint`` gives one.

    PYSTILT has no public footprint constructor: a footprint is made from
    particles. Tests that need one with chosen values build it here, the one
    test module that reaches the private helper.
    """
    return _describe(data, release, config, name)


def run_footprint(sim, particles: pd.DataFrame) -> xr.DataArray | None:
    """
    Make the footprint of *sim* from *particles* as the worker does, and return it as stored.

    The footprint stage runs through the worker's own ``run_stage``, with
    *particles* in memory in place of a particle file. ``None`` for an empty
    footprint, whose file is written all the same.
    """
    sim.__dict__["particles"] = particles
    _run_stage(sim, "footprint")
    return read_footprint(sim.path("footprint"))


def make_footprint(
    xres: float = 0.1, yres: float = 0.1, n_times: int = 1
) -> xr.DataArray:
    """Return a footprint of zeros at two cells of a small grid, from 2023-01-01 12 UTC, *n_times* hours long."""
    release_time = dt.datetime(2023, 1, 1, 12)
    release = PointRelease(
        time=release_time,
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    grid = Grid(
        xmin=-114.0,
        xmax=-113.8,
        ymin=39.0,
        ymax=39.2,
        xres=xres,
        yres=yres,
    )
    config = FootprintConfig(grid=grid)

    lons = np.array([-113.95, -113.85])
    lats = np.array([39.05, 39.15])
    times = [release_time + pd.Timedelta(hours=i) for i in range(n_times)]

    data = xr.DataArray(
        np.zeros((n_times, len(lats), len(lons))),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"units": "ppm (umol-1 m2 s)"},
    )
    return as_footprint(data, release, config, "slv")


def windows_spec(shift: float = 0.0) -> dict:
    """Return a geometry of two 0.25 degree windows on :func:`make_footprint`'s grid, moved east by *shift*."""
    return {
        "kind": "windows",
        "coords": [(-113.95 + shift, 39.05), (-113.85 + shift, 39.15)],
        "size": 0.25,  # >= 2 native cells so no under-resolution warning
        "ids": ["a", "b"],
    }


def geometry_footprint() -> tuple[xr.DataArray, FootprintConfig, Mesh]:
    """Return a footprint whose grid was derived for :func:`windows_spec`, its config, and that geometry."""
    base = make_footprint()
    fc = FootprintConfig(grid=base.stilt.grid, geometry=windows_spec())
    assert fc.geometry is not None
    mesh = Mesh.from_spec(fc.geometry)
    fc = fc.model_copy(update={"geometry_hash": mesh.hash})
    return as_footprint(base, base.stilt.release, fc, "geo"), fc, mesh
