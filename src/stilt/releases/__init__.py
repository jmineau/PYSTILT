"""
Releases, the places and times particles are released from.

A :class:`PointRelease` releases from one point, a :class:`ColumnRelease`
from a vertical line, and a :class:`MultiPointRelease` from several points
at once (for example a slanted satellite sounding). :func:`read_releases`
reads them from a CSV file, and :func:`releases_to_frame` gives them as one
table with a row per release point.

``models``
    The release models, their times, and their ids.
``table``
    The release table and its CSV file.
``validation``
    The checks a release must pass, shared by the models and the table.
"""

from .models import (
    ColumnRelease,
    MultiPointRelease,
    PointRelease,
    Release,
    VerticalReference,
    parse_release_id,
)
from .table import (
    read_releases,
    releases_from_frame,
    releases_to_frame,
    write_releases,
)

__all__ = [
    "ColumnRelease",
    "MultiPointRelease",
    "PointRelease",
    "Release",
    "VerticalReference",
    "parse_release_id",
    "read_releases",
    "releases_from_frame",
    "releases_to_frame",
    "write_releases",
]
