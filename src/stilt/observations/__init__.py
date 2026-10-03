"""
Tools for column and satellite observations, ported from X-STILT.

The functions take and return plain arrays and tables, so they work with
any product reader.

Before a run:

- Read a product into a table of soundings with :func:`read_tropomi_ch4`,
  :func:`read_oco2`, :func:`read_tccon`, :func:`read_ggg_oof`, or
  :func:`read_ggg_netcdf`.
- Group soundings into overpasses (:func:`group_by_overpass`) and choose
  which to run (:func:`select_observations_spatial`).
- Spread several receptors over a pixel (:func:`jitter_points`), or lay out
  a slant line of sight (:func:`slant_points`) at the retrieval's pressure
  levels (:func:`pressure_altitudes`). The points become
  :class:`~stilt.Receptor` objects.
- Derive wind-error settings from analysis-minus-observation winds
  (:func:`variogram`, :func:`fit_variogram`).

After a run:

- :func:`transport_error` gives the transport error of the modeled
  enhancement from the unperturbed and wind-perturbed particles.
- :func:`background` samples a mole-fraction field where the particles end.
- :func:`plume_polygon` and :func:`plume_background` outline a plume from a
  forward run and take the background from soundings outside it.

Particle weighting (averaging kernel, pressure weighting, lifetime decay)
is in :mod:`stilt.transforms`.
"""

from .backgrounds import Background, background
from .plumes import Plume, PlumeBackground, plume_background, plume_polygon
from .readers import (
    read_ggg_netcdf,
    read_ggg_oof,
    read_oco2,
    read_tccon,
    read_tropomi_ch4,
)
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
    "plume_background",
    "plume_polygon",
    "pressure_altitudes",
    "read_ggg_netcdf",
    "read_ggg_oof",
    "read_oco2",
    "read_tccon",
    "read_tropomi_ch4",
    "select_observations_spatial",
    "slant_points",
    "transport_error",
    "variogram",
]
