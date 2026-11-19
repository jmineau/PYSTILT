"""
Fixtures for the observation tests.

The observation tests use only these fixtures and the files under ``data/``,
so this directory does not depend on the rest of the test suite.
"""

import datetime as dt

import pytest
from runtraj.releases import PointRelease


@pytest.fixture
def point_release():
    """A simple single-point release."""
    return PointRelease(
        time=dt.datetime(2023, 1, 1, 12, 0),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
