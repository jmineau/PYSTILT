"""Shared fixtures for the PYSTILT test suite."""

import datetime as dt

import pytest

from stilt.config import (
    FootprintConfig,
    Grid,
    MetConfig,
    ModelConfig,
    STILTParams,
)
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor

# ---------------------------------------------------------------------------
# Receptor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def point_receptor():
    """A simple single-point receptor."""
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, 12, 0),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


@pytest.fixture
def column_receptor():
    """A vertical-column receptor (two heights, same lon/lat)."""
    return ColumnReceptor(
        time=dt.datetime(2023, 1, 1, 12, 0),
        longitude=-111.85,
        latitude=40.77,
        bottom=5.0,
        top=50.0,
    )


@pytest.fixture
def multipoint_receptor():
    """A multi-point receptor (three different locations)."""
    return MultiPointReceptor(
        time=dt.datetime(2023, 1, 1, 12, 0),
        longitudes=[-111.85, -111.86, -111.84],
        latitudes=[40.77, 40.78, 40.76],
        altitudes=[5.0, 5.0, 5.0],
    )


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def met_config(tmp_path):
    """MetConfig pointing at a temporary directory."""
    return MetConfig(
        directory=tmp_path / "met",
        file_format="%Y%m%d_%H",
        file_tres="1h",
    )


@pytest.fixture
def stilt_params(tmp_path):
    """Minimal STILTParams for use in simulation tests."""
    return STILTParams(
        n_hours=-24,
        numpar=100,
    )


@pytest.fixture
def model_config(tmp_path, met_config):
    """Minimal ModelConfig with one met entry."""
    return ModelConfig(
        n_hours=-24,
        numpar=100,
        mets={"hrrr": met_config},
    )


@pytest.fixture
def grid():
    return Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.01, yres=0.01)


@pytest.fixture
def footprint_config(grid):
    return FootprintConfig(grid=grid)
