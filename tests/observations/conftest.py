"""
Fixtures for the observation tests.

The observation tests use only these fixtures and the files under ``data/``,
so this directory does not depend on the rest of the test suite.
"""

import datetime as dt

import pytest

from stilt.receptors import PointReceptor


@pytest.fixture
def point_receptor():
    """A simple single-point receptor."""
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, 12, 0),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
