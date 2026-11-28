"""
PYSTILT's footprint of committed particles against STILT-R's, saved once.

The kernel tests in ``test_gridding.py`` check its shape, symmetry and sum,
which a wrong kernel width keeps. This compares every cell with STILT-R's
footprint of the same particles (``tests/fixtures/footprint_reference``), so a
change to the smoothing fails the unit suite without R. The ``r_only`` test
``test_committed_reference_matches_stilt_r`` checks the saved file against
STILT-R itself.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
from runtraj.releases import PointRelease

from stilt.footprint import calc_footprint
from stilt.footprint.grid import Grid

from ..fixtures.footprint_reference.make_reference import (
    GRID,
    HERE,
    LATITUDE,
    LONGITUDE,
)

#: The run time the STILT-R helper gives ``calc_footprint``.
RELEASE_TIME = dt.datetime(2015, 12, 10)


def _pystilt_footprint() -> xr.DataArray:
    """Grid the reference particles with PYSTILT as STILT-R's helper did."""
    particles = pd.read_csv(HERE / "particles.csv").rename(
        columns={"indx": "particle", "time": "age", "long": "lon", "lati": "lat"}
    )
    particles["age"] *= 60  # STILT-R's minutes; PYSTILT's age is in seconds
    release = PointRelease(
        time=RELEASE_TIME, longitude=LONGITUDE, latitude=LATITUDE, altitude=5.0
    )
    return calc_footprint(
        particles,
        release,
        Grid(**GRID),
        smooth_factor=1.0,
        tres="1h",
        hnf_plume=False,  # STILT-R's calc_footprint grids foot as given
        name="reference",
    )


def _stilt_r_footprint() -> xr.Dataset:
    with xr.open_dataset(HERE / "stilt_r_foot.nc") as ds:
        return ds.load()


def _assert_matches_stilt_r(foot: xr.DataArray, r: xr.Dataset) -> None:
    np.testing.assert_allclose(foot["lat"].values, r["lat"].values, rtol=1e-13)
    np.testing.assert_allclose(foot["lon"].values, r["lon"].values, rtol=1e-13)
    np.testing.assert_allclose(
        foot.transpose("time", "lat", "lon").values.astype(np.float64),
        r["foot"].values.astype(np.float64),
        rtol=1e-7,
        atol=1e-15,
    )


def test_footprint_matches_the_saved_stilt_r_footprint():
    foot = _pystilt_footprint()
    r = _stilt_r_footprint()
    assert foot.sizes["time"] == r.sizes["time"] == 6
    assert float(r["foot"].sum()) > 0
    _assert_matches_stilt_r(foot, r)
