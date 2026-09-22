"""
Column and satellite observation helpers: the X-STILT port.

Everything here happens before a simulation. Soundings are grouped into
overpasses, a subset is chosen to run, extra receptors are spread across a
pixel, and a slant line of sight is laid out as points. The functions take
and return plain arrays and tuples so they work on whatever table your
product reader produces; the results become :class:`~stilt.Receptor`
objects. Particle weighting (averaging kernel, pressure weighting, lifetime
decay) happens after the run and lives in :mod:`stilt.transforms`.
"""

from .selection import group_by_overpass, jitter_points, select_observations_spatial
from .slant import slant_points

__all__ = [
    "group_by_overpass",
    "jitter_points",
    "select_observations_spatial",
    "slant_points",
]
