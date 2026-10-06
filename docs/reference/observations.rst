Observations
============

Functions for column and satellite workflows. They group soundings into
overpasses, choose which soundings to run, spread receptors across a pixel,
lay out slant lines of sight, and turn a retrieval's pressure levels into
altitudes (:doc:`/tutorials/satellite_column`,
:doc:`/guides/slant_columns`). They take plain arrays and tables and return
the inputs for :class:`~stilt.Receptor` objects. Particle weighting,
including per-receptor averaging-kernel tables, is in :doc:`transforms`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.receptors_from_soundings
   stilt.observations.group_by_overpass
   stilt.observations.select_observations_spatial
   stilt.observations.jitter_points
   stilt.observations.slant_points
   stilt.observations.pressure_altitudes

Product readers
---------------

Each reader returns a table of soundings with the columns in
``SOUNDING_SCHEMA``, which :doc:`/guides/readers` describes.

.. autodata:: stilt.observations.readers.schema.SOUNDING_SCHEMA
   :no-value:

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.check_soundings

   stilt.observations.read_tropomi_ch4
   stilt.observations.read_oco2
   stilt.observations.read_tccon
   stilt.observations.read_ggg_oof
   stilt.observations.read_ggg_netcdf

Plume background
----------------

The outline of a forward-run plume over a satellite swath, and the
background from the soundings beside it (:doc:`/guides/plume_background`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.plume_polygon
   stilt.observations.Plume
   stilt.observations.plume_background
   stilt.observations.PlumeBackground
   stilt.observations.plumes.kernel_density
   stilt.observations.plumes.density_polygon

The modelled column
-------------------

The column a retrieval would report for the modelled air: the enhancement,
the background, and the retrieval's prior where it is not sensitive.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.modelled_column
