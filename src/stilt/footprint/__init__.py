"""
Footprints of receptors, calculated from particles and applied to surface fluxes.

``config``
    :class:`FootprintConfig`, the footprint settings.
``gridding``
    :func:`calc_footprint`, the footprint from particles, as in STILT-R.
``aggregation``
    Summing footprints onto other grids or polygons (:func:`jacobian`).
``stage``
    :class:`FootprintStage`, what a ``stages`` block of ``config.yaml`` names.
``foot``
    What the stage computes from particles before gridding them:
    :func:`add_foot` and the near-field correction (:func:`correct_near_field`).
``io``
    The footprint array and its attributes, footprint files, and
    :func:`open_footprints`, many stored footprints as one dataset.
``targets``
    The geometries a footprint is summed onto (:class:`Mesh`, :class:`Zones`)
    and the overlap weights.
``accessor``
    ``foot.stilt``, registered when this package is imported.
"""

from .accessor import FootprintAccessor
from .aggregation import Jacobian, jacobian
from .config import FootprintConfig
from .foot import HNF_PLUME_COLUMNS, add_foot, correct_near_field
from .gridding import calc_footprint
from .io import (
    FOOTPRINT_SCHEMA,
    UNITS,
    open_footprints,
    read_footprint,
    write_empty_footprint,
    write_footprint,
)
from .stage import FootprintStage, footprint_particles
from .targets import Geometry, Mesh, Zones, check_resolution, overlap_weights

__all__ = [
    "FOOTPRINT_SCHEMA",
    "HNF_PLUME_COLUMNS",
    "UNITS",
    "FootprintAccessor",
    "FootprintConfig",
    "FootprintStage",
    "Geometry",
    "Jacobian",
    "Mesh",
    "Zones",
    "add_foot",
    "calc_footprint",
    "check_resolution",
    "correct_near_field",
    "footprint_particles",
    "jacobian",
    "open_footprints",
    "overlap_weights",
    "read_footprint",
    "write_empty_footprint",
    "write_footprint",
]
