Observations
============

Helpers for column and satellite workflows: overpass grouping, sounding
selection, pixel jitter, and slant line-of-sight geometry
(:doc:`/advanced/observations`). They work on plain arrays and tables and
produce the inputs to :class:`~stilt.Receptor` objects. Particle weighting,
including per-receptor averaging-kernel tables, lives in :doc:`transforms`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.group_by_overpass
   stilt.observations.select_observations_spatial
   stilt.observations.jitter_points
   stilt.observations.slant_points

Transport error
---------------

The transport error on the modelled enhancement from wind-perturbed
trajectories (:doc:`/guides/transport_error`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.transport_error
   stilt.observations.TransportError

Wind-error statistics
---------------------

The wind-error settings a transport-error run needs, from
analysis-minus-observation winds (:doc:`/guides/wind_errors`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.wind_error_scales
   stilt.observations.WindErrorScales
   stilt.observations.variogram
   stilt.observations.fit_variogram
   stilt.observations.VariogramFit
