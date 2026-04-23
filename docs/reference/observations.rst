Observations
============

The observation layer is still alpha, but it already defines the core scene,
sensor, and transform objects used for column and satellite-style workflows.

Observation models
------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.Observation
   stilt.observations.Scene
   stilt.observations.PointSensor
   stilt.observations.ColumnSensor
   stilt.observations.VerticalOperator
   stilt.observations.FirstOrderLifetimeChemistry

Geometry and weighting
----------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.observations.HorizontalGeometry
   stilt.observations.ViewingGeometry
   stilt.observations.LineOfSight
   stilt.observations.VerticalOperatorWeighting

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
