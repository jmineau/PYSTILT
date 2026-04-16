"""Shared fixtures for PYSTILT integration tests.

Integration tests exercise real met files and real HYSPLIT execution.
They are collected and run by default; the ``met_dir`` fixture skips
automatically when the stilt-tutorials data is not present.

Met directory resolution order:
  1. STILT_TEST_MET_DIR environment variable
  2. tests/stilt-tutorials/01-wbb/met within the repo (gitignored; clone with
     ``git clone --depth 1 https://github.com/uataq/stilt-tutorials tests/stilt-tutorials``)

R-STILT reference footprint resolution order (for footprint fidelity tests):
  1. STILT_R_REF_SIM_DIR environment variable
  2. <repo>/../R-STILT_fork/out/by-id/201512100000_-112_40.5_5 (dev workspace)
"""

import datetime as dt
import os
from pathlib import Path

import pytest

from stilt.config import (
    FootprintConfig,
    Grid,
    ModelConfig,
)
from stilt.receptor import Receptor

# ---------------------------------------------------------------------------
# Marker - apply to every test that needs real met + HYSPLIT
# ---------------------------------------------------------------------------

integration = pytest.mark.integration

# ---------------------------------------------------------------------------
# Met directory
# ---------------------------------------------------------------------------

# tests/ is one level up from this file:
#   tests/integration/conftest.py → tests/
_TESTS_DIR = Path(__file__).parents[1]
_DEFAULT_MET_DIR = _TESTS_DIR / "stilt-tutorials" / "01-wbb" / "met"


@pytest.fixture(scope="session")
def met_dir() -> Path:
    """Path to 01-wbb HRRR ARL met files.

    Override with STILT_TEST_MET_DIR if stilt-tutorials is elsewhere.
    Default: tests/stilt-tutorials/01-wbb/met (gitignored).
    """
    path = Path(os.environ.get("STILT_TEST_MET_DIR", _DEFAULT_MET_DIR))
    if not path.exists():
        pytest.skip(
            f"Met directory not found: {path}\n"
            "Clone uataq/stilt-tutorials or set STILT_TEST_MET_DIR."
        )
    return path


# ---------------------------------------------------------------------------
# Reference receptor - mirrors R-STILT integration test suite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def wbb_receptor() -> Receptor:
    """Single WBB-like receptor matching R-STILT tutorial parameters."""
    return Receptor(
        time=dt.datetime(2015, 12, 10, 0, 0),
        longitude=-112.0,
        latitude=40.5,
        altitude=5.0,
    )


@pytest.fixture(scope="session")
def column_receptor() -> Receptor:
    """Column receptor - same lat/lon, two heights - matching R-STILT test_run_stilt_column."""
    return Receptor(
        time=dt.datetime(2015, 12, 10, 0, 0),
        longitude=-112.0,
        latitude=40.5,
        altitude=[5.0, 1000.0],
    )


@pytest.fixture(scope="session")
def multipoint_receptor() -> Receptor:
    """Three-location multipoint receptor - matching R-STILT test_run_stilt_multipoint."""
    return Receptor(
        time=dt.datetime(2015, 12, 10, 0, 0),
        longitude=[-112.0, -111.5, -111.0],
        latitude=[40.5, 41.0, 41.5],
        altitude=[5.0, 500.0, 1000.0],
    )


# ---------------------------------------------------------------------------
# ModelConfig fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def wbb_grid() -> Grid:
    """Domain grid covering the WBB area at 0.01° resolution."""
    return Grid(xmin=-113.0, xmax=-111.0, ymin=39.5, ymax=41.5, xres=0.01, yres=0.01)


@pytest.fixture(scope="session")
def wbb_config(met_dir, wbb_grid) -> ModelConfig:
    """Minimal ModelConfig for integration tests.

    Uses n_hours=-6 and numpar=100 to keep runtime under ~2 minutes.
    """
    return ModelConfig(
        mets={
            "hrrr": {
                "directory": met_dir,
                "file_format": "%Y%m%d.%Hz.hrrra",
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
        footprints={
            "default": FootprintConfig(grid=wbb_grid),
        },
    )


@pytest.fixture(scope="session")
def traj_only_config(met_dir) -> ModelConfig:
    """ModelConfig without footprints for trajectory-only tests."""
    return ModelConfig(
        mets={
            "hrrr": {
                "directory": met_dir,
                "file_format": "%Y%m%d.%Hz.hrrra",
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
    )


@pytest.fixture(scope="session")
def multifoot_config(met_dir, wbb_grid) -> ModelConfig:
    """Config with two named footprints at different resolutions."""
    coarse_grid = Grid(
        xmin=-113.0, xmax=-111.0, ymin=39.5, ymax=41.5, xres=0.05, yres=0.05
    )
    return ModelConfig(
        mets={
            "hrrr": {
                "directory": met_dir,
                "file_format": "%Y%m%d.%Hz.hrrra",
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
        footprints={
            "fine": FootprintConfig(grid=wbb_grid),
            "coarse": FootprintConfig(grid=coarse_grid),
        },
    )


@pytest.fixture(scope="session")
def multipoint_config(met_dir) -> ModelConfig:
    """Config with a wider domain covering all three multipoint receptor locations."""
    grid = Grid(xmin=-113.0, xmax=-110.5, ymin=39.5, ymax=42.0, xres=0.01, yres=0.01)
    return ModelConfig(
        mets={
            "hrrr": {
                "directory": met_dir,
                "file_format": "%Y%m%d.%Hz.hrrra",
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
        footprints={"default": FootprintConfig(grid=grid)},
    )


# ---------------------------------------------------------------------------
# R-STILT reference simulation (footprint fidelity tests)
# ---------------------------------------------------------------------------

_R_SIM_ID = "201512100000_-112_40.5_5"
# Default: sibling R-STILT_fork dev workspace relative to the PYSTILT repo root
_DEFAULT_R_REF_DIR = (
    _TESTS_DIR.parent / ".." / "R-STILT_fork" / "out" / "by-id" / _R_SIM_ID
)


@pytest.fixture(scope="session")
def r_ref_sim_dir() -> Path:
    """Directory of the R-STILT reference simulation used for footprint fidelity tests.

    Override with STILT_R_REF_SIM_DIR if the R-STILT_fork workspace is elsewhere.
    Default: <repo>/../R-STILT_fork/out/by-id/201512100000_-112_40.5_5 (dev workspace).
    """
    path = Path(os.environ.get("STILT_R_REF_SIM_DIR", _DEFAULT_R_REF_DIR)).resolve()
    if not path.exists():
        pytest.skip(
            f"R-STILT reference sim dir not found: {path}\n"
            "Set STILT_R_REF_SIM_DIR or run R-STILT_fork to generate the reference."
        )
    return path
