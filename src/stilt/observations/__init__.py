"""
Column and satellite observation helpers: the X-STILT port.

Everything here happens before a simulation. A product file is read into a
table of soundings (:func:`read_tropomi_ch4`, :func:`read_oco2`,
:func:`read_tccon`, or your own reader of the same shape), soundings are
grouped into overpasses, a subset is chosen to run, extra receptors are spread across a
pixel, and a slant line of sight is laid out as points, at altitudes
taken from the retrieval's pressure levels if you like. The functions take
and return plain arrays and tuples so they work on whatever table your
product reader produces; the results become :class:`~stilt.Receptor`
objects. Particle weighting (averaging kernel, pressure weighting, lifetime
decay) happens after the run and lives in :mod:`stilt.transforms`. After
the run, :func:`transport_error` turns a simulation's main and
wind-perturbed particles plus a flux field into the transport error on the
modelled enhancement, :func:`background` samples a mole-fraction field at
the trajectory endpoints for the background that enhancement adds to,
:func:`plume_polygon` and :func:`plume_background` outline a forward-run
plume at overpass time and take the background from the soundings beside
it, and :func:`variogram` and :func:`fit_variogram` turn analysis-minus-observation
winds into the wind-error settings that run needs.
"""

from .backgrounds import Background, background, particle_background
from .plumes import Plume, PlumeBackground, plume_background, plume_polygon
from .readers import read_oco2, read_tccon, read_tropomi_ch4
from .selection import group_by_overpass, jitter_points, select_observations_spatial
from .slant import pressure_altitudes, slant_points
from .uncertainty import TransportError, transport_error
from .winds import VariogramFit, fit_variogram, variogram

__all__ = [
    "Background",
    "Plume",
    "PlumeBackground",
    "TransportError",
    "background",
    "VariogramFit",
    "fit_variogram",
    "group_by_overpass",
    "jitter_points",
    "particle_background",
    "plume_background",
    "plume_polygon",
    "pressure_altitudes",
    "read_oco2",
    "read_tccon",
    "read_tropomi_ch4",
    "select_observations_spatial",
    "slant_points",
    "transport_error",
    "variogram",
]
