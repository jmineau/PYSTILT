Observations
============

Helpers for column and satellite workflows: overpass grouping, sounding
selection, pixel jitter, and slant line-of-sight geometry with altitudes
from a retrieval's pressure levels (:doc:`/advanced/observations`,
:doc:`/guides/slant_columns`). They work on plain arrays and tables and
produce the inputs to :class:`~stilt.Receptor` objects. Particle weighting,
including per-receptor averaging-kernel tables, lives in :doc:`transforms`.

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

One module per instrument, each returning a table of soundings with the
columns in :doc:`/guides/readers`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.read_tropomi_ch4
   stilt.observations.read_oco2
   stilt.observations.read_tccon

Transport error
---------------

The transport error on the modelled enhancement from wind-perturbed
trajectories (:doc:`/guides/transport_error`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.transport_error
   stilt.observations.TransportError

Background
----------

The background mole fraction at a receptor, from a field sampled at the
trajectory endpoints and weighted like the footprint (:doc:`/guides/background`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.background
   stilt.observations.Background
   stilt.observations.particle_background
   stilt.observations.backgrounds.sample_field
   stilt.observations.backgrounds.endpoint_weights
   stilt.observations.backgrounds.vertical_dim

Wind-error statistics
---------------------

Variograms of analysis-minus-observation winds, from which the wind-error
settings of a transport-error run are derived (:doc:`/guides/wind_errors`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.variogram
   stilt.observations.fit_variogram
   stilt.observations.VariogramFit
