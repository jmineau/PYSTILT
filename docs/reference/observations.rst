Observations
============

The observation layer is still alpha. It defines the normalized observation,
scene grouping, receptor builders, and selection helpers used for column and
satellite-style workflows (:doc:`/advanced/observations`). Particle weighting
lives in :doc:`transforms`.

Observation models
------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.Observation
   stilt.observations.Scene

Geometry
--------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.HorizontalGeometry
   stilt.observations.ViewingGeometry
   stilt.observations.LineOfSight

Builders, grouping, and selection
---------------------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.build_point_receptor
   stilt.observations.build_column_receptor
   stilt.observations.build_slant_receptor
   stilt.observations.build_multipoint_receptor
   stilt.observations.group_by_overpass
   stilt.observations.group_observations
   stilt.observations.filter_observations
   stilt.observations.select_observations_spatial
   stilt.observations.jitter_observation
