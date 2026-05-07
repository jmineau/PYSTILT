"""Shared fixtures for the PYSTILT test suite."""

import datetime as dt
import os
import shutil
from pathlib import Path

import pytest

from stilt.config import (
    FootprintConfig,
    Grid,
    MetConfig,
    ModelConfig,
    STILTParams,
)
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor

from .fixtures.r_stilt_reference import reference_grid, reference_receptor

# ---------------------------------------------------------------------------
# Marker - apply to every test that needs real met + HYSPLIT
# ---------------------------------------------------------------------------

integration = pytest.mark.integration

# ---------------------------------------------------------------------------
# Met directory
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).parent
_DEFAULT_MET_DIR = _TESTS_DIR / "stilt-tutorials" / "01-wbb" / "met"


@pytest.fixture(scope="session")
def met_dir() -> Path:
    """
    Path to 01-wbb HRRR ARL met files.

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


# ---------------------------------------------------------------------------
# R-STILT fixtures (skip gracefully when R or STILT_R_DIR not available)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def r_stilt_dir() -> Path:
    """Path to a uataq/stilt R checkout. Set STILT_R_DIR env var to enable."""
    env = os.environ.get("STILT_R_DIR")
    if env:
        path = Path(env)
        if path.exists():
            return path
    pytest.skip(
        "STILT_R_DIR not set or path not found. "
        "Set STILT_R_DIR=/path/to/uataq-stilt to run R-STILT comparison tests."
    )


@pytest.fixture(scope="session")
def rscript(r_stilt_dir) -> str:  # noqa: ARG001
    """Path to the Rscript executable. Skips if not on PATH."""
    exe = shutil.which("Rscript")
    if not exe:
        pytest.skip("Rscript not found on PATH.")
    return exe


# ---------------------------------------------------------------------------
# WBB integration fixtures (real met + HYSPLIT; session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def wbb_receptor():
    """Single WBB-like receptor matching R-STILT tutorial parameters."""
    return reference_receptor()


@pytest.fixture(scope="session")
def wbb_column_receptor():
    """Column receptor at WBB - same lat/lon, two heights."""
    return ColumnReceptor(
        time=dt.datetime(2015, 12, 10, 0, 0),
        longitude=-112.0,
        latitude=40.5,
        bottom=5.0,
        top=1000.0,
    )


@pytest.fixture(scope="session")
def wbb_multipoint_receptor():
    """Three-location multipoint receptor matching R-STILT test_run_stilt_multipoint."""
    return MultiPointReceptor(
        time=dt.datetime(2015, 12, 10, 0, 0),
        longitudes=[-112.0, -111.5, -111.0],
        latitudes=[40.5, 41.0, 41.5],
        altitudes=[5.0, 500.0, 1000.0],
    )


@pytest.fixture(scope="session")
def wbb_grid() -> Grid:
    """Domain grid covering the WBB area at 0.01° resolution."""
    return reference_grid()


@pytest.fixture(scope="session")
def wbb_config(met_dir, wbb_grid) -> ModelConfig:
    """Minimal ModelConfig for integration tests (n_hours=-6, numpar=100)."""
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
