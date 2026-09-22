Observations
============

The observation layer is still alpha, but it already defines the core scene,
sensor, and receptor-builder objects used for column and satellite-style
workflows. Particle weighting lives in :doc:`transforms`.

Observation models
------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.Observation
   stilt.observations.Scene
   stilt.observations.PointSensor
   stilt.observations.ColumnSensor

Geometry
--------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.HorizontalGeometry
   stilt.observations.ViewingGeometry
   stilt.observations.LineOfSight

Builders and grouping
---------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.build_point_receptor
   stilt.observations.build_column_receptor
   stilt.observations.build_slant_receptor
   stilt.observations.group_scenes_by_time_gap
   stilt.observations.group_scenes_by_swath
   stilt.observations.group_scenes_by_metadata
