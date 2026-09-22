"""
Column and satellite observation helpers: the X-STILT port.

Everything here happens before a simulation. Soundings are grouped into
overpasses, a subset is chosen to run, extra receptors are spread across a
pixel, and a slant line of sight is laid out as points. The functions take
and return plain arrays and tuples so they work on whatever table your
product reader produces; the results become :class:`~stilt.Receptor`
objects. Particle weighting (averaging kernel, pressure weighting, lifetime
decay) happens after the run and lives in :mod:`stilt.transforms`. After
the run, :func:`transport_error` turns a simulation's main and
wind-perturbed particles plus a flux field into the transport error on the
modelled enhancement, :func:`background` samples a mole-fraction field at
the trajectory endpoints for the background that enhancement adds to, and
:func:`variogram` and :func:`fit_variogram` turn analysis-minus-observation
winds into the wind-error settings that run needs.
"""

from .backgrounds import Background, background, particle_background
from .selection import group_by_overpass, jitter_points, select_observations_spatial
from .slant import slant_points
from .uncertainty import TransportError, transport_error
from .winds import VariogramFit, fit_variogram, variogram

__all__ = [
    "Background",
    "TransportError",
    "background",
    "VariogramFit",
    "fit_variogram",
    "group_by_overpass",
    "jitter_points",
    "particle_background",
    "select_observations_spatial",
    "slant_points",
    "transport_error",
    "variogram",
]
