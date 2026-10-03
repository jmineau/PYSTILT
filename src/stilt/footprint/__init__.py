"""
Footprints of receptors, calculated from particles and applied to surface fluxes.

``gridding``
    :func:`calculate`, the footprint from particles, as in STILT-R.
``aggregation``
    Summing footprints onto other grids or polygons (:func:`jacobian`).
``io``
    The footprint array and its attributes, and footprint files.
``accessor``
    ``foot.stilt``, registered when this package is imported.
"""

from .accessor import FootprintAccessor
from .aggregation import Jacobian, jacobian
from .gridding import calculate
from .io import (
    FOOTPRINT_SCHEMA,
    UNITS,
    read_footprint,
    write_empty_footprint,
    write_footprint,
)

__all__ = [
    "FOOTPRINT_SCHEMA",
    "UNITS",
    "FootprintAccessor",
    "Jacobian",
    "calculate",
    "jacobian",
    "read_footprint",
    "write_empty_footprint",
    "write_footprint",
]
