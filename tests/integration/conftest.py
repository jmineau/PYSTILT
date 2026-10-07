"""
Fixtures for the integration tests: receptors and configs at the WBB site.

The tests run HYSPLIT on real meteorology, the test met cache that the
``met_dir`` fixture finds, and are marked ``integration``.
"""

from pathlib import Path

import pytest

from stilt.config import ProjectConfig
from stilt.receptors import ColumnReceptor, MultiPointReceptor
from stilt.spatial import Grid
from stilt.transport.hysplit import MetConfig

from ..fixtures.factories import make_met_config
from ..fixtures.r_stilt_reference import (
    REFERENCE_MET_FILE_FORMAT,
    REFERENCE_TIME,
    reference_grid,
    reference_receptor,
)


def reference_met(met_dir: Path) -> MetConfig:
    """Return the met config of the test met cache: 6-hour HRRR files."""
    return make_met_config(
        met_dir, file_format=REFERENCE_MET_FILE_FORMAT, file_tres="6h"
    )


@pytest.fixture(scope="session")
def wbb_receptor():
    """Single WBB-like receptor matching STILT-R tutorial parameters."""
    return reference_receptor()


@pytest.fixture(scope="session")
def wbb_column_receptor():
    """Column receptor at WBB - same lat/lon, two heights."""
    return ColumnReceptor(
        time=REFERENCE_TIME,
        longitude=-112.0,
        latitude=40.5,
        bottom=5.0,
        top=1000.0,
    )


@pytest.fixture(scope="session")
def wbb_multipoint_receptor():
    """Three-location multipoint receptor at WBB area."""
    return MultiPointReceptor(
        time=REFERENCE_TIME,
        longitudes=[-112.0, -111.5, -111.0],
        latitudes=[40.5, 41.0, 41.5],
        altitudes=[5.0, 500.0, 1000.0],
    )


@pytest.fixture(scope="session")
def wbb_grid() -> Grid:
    """Domain grid covering the WBB area at 0.01° resolution."""
    return reference_grid()


@pytest.fixture(scope="session")
def wbb_config(met_dir, wbb_grid) -> ProjectConfig:
    """Minimal ProjectConfig for integration tests (n_hours=-6, numpar=100)."""
    return ProjectConfig(
        mets={"hrrr": reference_met(met_dir)},
        n_hours=-6,
        numpar=100,
        grid=wbb_grid,
        variants={"hrrr": {}},
    )


@pytest.fixture(scope="session")
def traj_only_config(met_dir) -> ProjectConfig:
    """ProjectConfig without footprints for trajectory-only tests."""
    return ProjectConfig(
        mets={"hrrr": reference_met(met_dir)},
        n_hours=-6,
        numpar=100,
        variants={"hrrr": {}},
    )


@pytest.fixture(scope="session")
def multifoot_config(met_dir, wbb_grid) -> ProjectConfig:
    """Config with a second, coarser footprint on the same particles."""
    coarse_grid = Grid(
        xmin=-113.0, xmax=-111.0, ymin=39.5, ymax=41.5, xres=0.05, yres=0.05
    )
    return ProjectConfig(
        mets={"hrrr": reference_met(met_dir)},
        n_hours=-6,
        numpar=100,
        grid=wbb_grid,
        variants={
            "hrrr": {},
            "coarse": {"grid": coarse_grid.model_dump()},
        },
    )


@pytest.fixture(scope="session")
def multipoint_config(met_dir) -> ProjectConfig:
    """Config with a wider domain covering all three multipoint receptor locations."""
    grid = Grid(xmin=-113.0, xmax=-110.5, ymin=39.5, ymax=42.0, xres=0.01, yres=0.01)
    return ProjectConfig(
        mets={"hrrr": reference_met(met_dir)},
        n_hours=-6,
        numpar=100,
        grid=grid,
        variants={"hrrr": {}},
    )
