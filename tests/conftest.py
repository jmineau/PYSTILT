"""Shared fixtures for the PYSTILT test suite."""

import datetime as dt
import os
import shutil
from pathlib import Path

import pytest

from stilt.config import ProjectConfig
from stilt.releases import ColumnRelease, MultiPointRelease, PointRelease
from stilt.spatial import Grid
from tests.fixtures.factories import blocks

from .fixtures.factories import make_hysplit_dir, make_met_config

# ---------------------------------------------------------------------------
# Marker - apply to every test that needs real met + HYSPLIT
# ---------------------------------------------------------------------------

integration = pytest.mark.integration

# Parametrized fixtures that run HYSPLIT (and STILT-R) once per parameter for
# several tests. CI runs the integration and STILT-R tests on several workers
# (pytest-xdist, ``--dist loadgroup``); grouping keeps the tests of one parameter
# on one worker, so each run still happens once.
_SHARED_RUNS = ("scenario_outputs", "hysplit_run")


# tryfirst: xdist reads the groups in its own pytest_collection_modifyitems.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Put the tests that share one run of a ``_SHARED_RUNS`` fixture in one xdist group."""
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        for name in _SHARED_RUNS:
            if name in callspec.params:
                value = callspec.params[name]
                group = f"{name}-{getattr(value, 'name', value)}"
                item.add_marker(pytest.mark.xdist_group(group))


# ---------------------------------------------------------------------------
# HYSPLIT's hycs_std
# ---------------------------------------------------------------------------

#: The markers of the tests that run a real ``hycs_std``.
_REAL_HYSPLIT = ("integration", "fidelity")

#: What the stand-in prints when it starts, as ``hycs_std`` does.
STAND_IN_VERSION = "hysplit.v5.1.0-test"


@pytest.fixture(scope="session")
def stand_in_hycs_std(tmp_path_factory) -> Path:
    """
    The ``exec/`` folder of a stand-in HYSPLIT, whose ``hycs_std`` prints a version and stops.

    It stops as ``hycs_std`` does without ``CONTROL``. PYSTILT does not
    include HYSPLIT, and a run's settings hold the SHA-256 of the
    ``hycs_std`` it runs and of its data tables, so the unit tests use these
    instead. The tables are in ``bdyfiles/`` beside ``exec/``.
    """
    folder = make_hysplit_dir(
        tmp_path_factory.mktemp("hysplit"),
        STAND_IN_VERSION,
        body="echo ' Last Changed Date: 2021-05-13'\nexit 2\n",
    )
    return folder / "exec"


@pytest.fixture(autouse=True)
def _hycs_std_on_path(request, monkeypatch):
    """
    Put the stand-in ``hycs_std`` first on ``PATH`` for a unit test; skip a real-HYSPLIT test without one.

    A test marked ``integration`` or ``fidelity`` runs the first ``hycs_std``
    on ``PATH``, such as STILT-R's ``bin/linux_x64/hycs_std``, and is
    skipped when there is none.
    """
    if any(request.node.get_closest_marker(m) for m in _REAL_HYSPLIT):
        if shutil.which("hycs_std") is None:
            pytest.skip(
                "No hycs_std on PATH. Put the directory of STILT-R's "
                "bin/linux_x64/hycs_std (or another HYSPLIT build) on PATH."
            )
        return
    stand_in = request.getfixturevalue("stand_in_hycs_std")
    monkeypatch.setenv("PATH", f"{stand_in}{os.pathsep}{os.environ.get('PATH', '')}")


# ---------------------------------------------------------------------------
# Met directory
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _met_catalogs_stay_out_of_the_home_folder(tmp_path_factory, monkeypatch):
    """A met without a ``subgrid_dir`` keeps its catalog in the user's cache: not a real one."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.getbasetemp() / "cache"))


_TESTS_DIR = Path(__file__).parent
_MET_CACHE = _TESTS_DIR / "met_cache"

# SLV bounding box for bbox-cropped HRRRArchive downloads. Each 6h block is a
# 3.42 GB full-domain file, cropped to 130 x 129 cells (30.2 MB) once it is down.
_SLV_BBOX = (-114.0, 39.0, -110.0, 42.0)  # (west, south, east, north)

# Individual 6h HRRR blocks needed by all 12 fidelity scenarios.
# Each entry is a (block_start, block_end) pair that maps to exactly one file.
# Winter blocks cover day_backward (24h), which needs blocks from the previous day.
# Bump the GHA cache key in tests.yml, stilt-r.yml and docs.yml when you add blocks here.
_MET_BLOCKS: list[tuple[str, str]] = [
    # Winter 2021-01-15 (includes day_backward 24h)
    ("2021-01-14 06:00", "2021-01-14 11:59"),
    ("2021-01-14 12:00", "2021-01-14 17:59"),
    ("2021-01-14 18:00", "2021-01-14 23:59"),
    ("2021-01-15 00:00", "2021-01-15 05:59"),
    ("2021-01-15 06:00", "2021-01-15 11:59"),
    # Summer 2021-07-15
    ("2021-07-15 00:00", "2021-07-15 05:59"),
    ("2021-07-15 06:00", "2021-07-15 11:59"),
]


@pytest.fixture(scope="session")
def met_dir() -> Path:
    """
    Path to HRRR ARL met files for fidelity tests.

    Resolution order:
    1. STILT_TEST_MET_DIR env var — point at a pre-downloaded cache directory.
    2. STILT_TEST_FETCH_MET=1 — download all 7 bbox-cropped 6h blocks in parallel
       via HRRRArchive into tests/met_cache/ (gitignored; cached in CI).
       Already-present files are skipped automatically. Each block is
       downloaded whole (3.42 GB) and then cropped, so the 7 blocks download
       23.9 GB, all staged in tests/met_cache/ at once, and keep 211 MB.
    3. Otherwise skip.
    """
    env = os.environ.get("STILT_TEST_MET_DIR")
    if env and Path(env).exists():
        # Absolute, so STILT-R (which runs in its own working directory)
        # finds the files too when the variable is a relative path.
        return Path(env).resolve()

    if os.environ.get("STILT_TEST_FETCH_MET") == "1":
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from arlmet.archives import HRRRArchive

        _MET_CACHE.mkdir(exist_ok=True)
        archive = HRRRArchive()

        with ThreadPoolExecutor(max_workers=len(_MET_BLOCKS)) as pool:
            futures = {
                pool.submit(archive.fetch, s, e, dest_dir=_MET_CACHE, bbox=_SLV_BBOX): (
                    s,
                    e,
                )
                for s, e in _MET_BLOCKS
            }
            for fut in as_completed(futures):
                fut.result()  # surface any download errors immediately

        return _MET_CACHE

    pytest.skip(
        "Met files not available. Set STILT_TEST_MET_DIR to a directory with "
        "HRRR ARL files, or set STILT_TEST_FETCH_MET=1 to download automatically."
    )


# ---------------------------------------------------------------------------
# Release fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def point_release():
    """A simple single-point release."""
    return PointRelease(
        time=dt.datetime(2023, 1, 1, 12, 0),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


@pytest.fixture
def column_release():
    """A vertical-column release (two heights, same lon/lat)."""
    return ColumnRelease(
        time=dt.datetime(2023, 1, 1, 12, 0),
        longitude=-111.85,
        latitude=40.77,
        bottom=5.0,
        top=50.0,
    )


@pytest.fixture
def multipoint_release():
    """A multi-point release (three different locations)."""
    return MultiPointRelease(
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
    return make_met_config(tmp_path / "met")


@pytest.fixture
def model_config(tmp_path, met_config):
    """Minimal ProjectConfig with one met entry."""
    return ProjectConfig(
        **blocks(
            n_hours=-24, numpar=100, mets={"hrrr": met_config}, variants={"hrrr": {}}
        )
    )


@pytest.fixture
def grid():
    return Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.01, yres=0.01)


# ---------------------------------------------------------------------------
# STILT-R fixtures (skip gracefully when R or STILT_R_DIR not available)
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
        "Set STILT_R_DIR=/path/to/uataq-stilt to run STILT-R comparison tests."
    )


@pytest.fixture(scope="session")
def rscript(r_stilt_dir) -> str:
    """Path to the Rscript executable. Skips if not on PATH."""
    exe = shutil.which("Rscript")
    if not exe:
        pytest.skip("Rscript not found on PATH.")
    return exe
