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
- Make a receptor for each sounding, and their averaging-kernel table, with
  :func:`receptors_from_soundings`. For other layouts, spread several
  receptors over a pixel (:func:`jitter_points`), or lay out a slant line
  of sight (:func:`slant_points`) at the retrieval's pressure levels
  (:func:`pressure_altitudes`). Every reader's table has the columns in
  :data:`SOUNDING_SCHEMA` (:func:`check_soundings`).

After a run:

- :func:`modelled_column` gives the column a retrieval would report for the
  modelled air: enhancement, background, and the prior term.
- :func:`plume_polygon` and :func:`plume_background` outline a plume from a
  forward run and take the background from soundings outside it.

What a simulation's particles give is on the simulation:
``sim.background(field)`` and ``sim.transport_error(error_sim, flux)``
(:mod:`stilt.particles`). The wind-error statistics are in
:mod:`stilt.meteorology` (``variogram``, ``fit_variogram``), and particle
weighting (averaging kernel, pressure weighting, lifetime decay) is in
:mod:`stilt.transforms`.
"""

from .columns import modelled_column
from .placement import (
    jitter_points,
    pressure_altitudes,
    receptors_from_soundings,
    slant_points,
)
from .plumes import Plume, PlumeBackground, plume_background, plume_polygon
from .readers import (
    read_ggg_netcdf,
    read_ggg_oof,
    read_oco2,
    read_tccon,
    read_tropomi_ch4,
)
from .readers.schema import SOUNDING_SCHEMA, check_soundings
from .selection import group_by_overpass, select_observations_spatial

__all__ = [
    "SOUNDING_SCHEMA",
    "check_soundings",
    "modelled_column",
    "receptors_from_soundings",
    "Plume",
    "PlumeBackground",
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
]
