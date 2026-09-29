Observations
============

Functions for column and satellite workflows. They group soundings into
overpasses, choose which soundings to run, spread receptors across a pixel,
lay out slant lines of sight, and turn a retrieval's pressure levels into
altitudes (:doc:`/advanced/observations`,
:doc:`/guides/slant_columns`). They take plain arrays and tables and return
the inputs for :class:`~stilt.Receptor` objects. Particle weighting,
including per-receptor averaging-kernel tables, is in :doc:`transforms`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.group_by_overpass
   stilt.observations.select_observations_spatial
   stilt.observations.jitter_points
   stilt.observations.slant_points
   stilt.observations.pressure_altitudes

Product readers
---------------

Each reader returns a table of soundings with the columns listed in
:doc:`/guides/readers`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.read_tropomi_ch4
   stilt.observations.read_oco2
   stilt.observations.read_tccon
   stilt.observations.read_ggg_oof
   stilt.observations.read_ggg_netcdf

Transport error
---------------

The transport error of the modelled enhancement, from wind-perturbed
trajectories (:doc:`/guides/transport_error`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.transport_error
   stilt.observations.TransportError

Background
----------

The background mole fraction at a receptor. A field is sampled at the
trajectory endpoints and weighted like the footprint
(:doc:`/guides/background`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.background
   stilt.observations.Background
   stilt.observations.particle_background
   stilt.observations.backgrounds.sample_field
   stilt.observations.backgrounds.endpoint_weights
   stilt.observations.backgrounds.vertical_dim

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

Wind-error statistics
---------------------

Variograms of analysis-minus-observation winds. Fit them to get the
wind-error settings for a transport-error run (:doc:`/guides/wind_errors`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.variogram
   stilt.observations.fit_variogram
   stilt.observations.VariogramFit
