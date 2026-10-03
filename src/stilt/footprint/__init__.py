"""
Footprints of receptors, calculated from particles and applied to surface fluxes.

``gridding``
    :func:`calculate`, the footprint from particles, as in STILT-R.
``aggregation``
    Summing footprints onto other grids or polygons (:func:`jacobian`).
``io``
    The footprint array and its attributes, and footprint files.
``targets``
    The geometries a footprint is summed onto (:class:`Mesh`, :class:`Zones`)
    and the overlap weights.
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
from .targets import Geometry, Mesh, Zones, check_resolution, overlap_weights

__all__ = [
    "FOOTPRINT_SCHEMA",
    "UNITS",
    "FootprintAccessor",
    "Geometry",
    "Jacobian",
    "Mesh",
    "Zones",
    "calculate",
    "check_resolution",
    "jacobian",
    "overlap_weights",
    "read_footprint",
    "write_empty_footprint",
    "write_footprint",
]
