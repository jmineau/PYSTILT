"""
Receptors, the places and times particles are released from.

A :class:`PointReceptor` releases from one point, a :class:`ColumnReceptor`
from a vertical line, and a :class:`MultiPointReceptor` from several points
at once (for example a slanted satellite sounding). :func:`read_receptors`
reads them from a CSV file, and :func:`receptors_to_frame` gives them as one
table with a row per release point.

``models``
    The receptor models, their times, and their ids.
``table``
    The receptor table and its CSV file.
``validation``
    The checks a receptor must pass, shared by the models and the table.
"""

from .models import (
    ColumnReceptor,
    MultiPointReceptor,
    PointReceptor,
    Receptor,
    VerticalReference,
    parse_receptor_id,
    parse_time,
)
from .table import (
    append_receptors_csv,
    check_distinct_ids,
    read_receptor_frame,
    read_receptors,
    receptor_rows,
    receptors_from_frame,
    receptors_from_rows,
    receptors_to_csv,
    receptors_to_frame,
    write_receptors,
)

__all__ = [
    "ColumnReceptor",
    "MultiPointReceptor",
    "PointReceptor",
    "Receptor",
    "VerticalReference",
    "append_receptors_csv",
    "check_distinct_ids",
    "parse_receptor_id",
    "parse_time",
    "read_receptor_frame",
    "read_receptors",
    "receptor_rows",
    "receptors_from_frame",
    "receptors_from_rows",
    "receptors_to_csv",
    "receptors_to_frame",
    "write_receptors",
]
