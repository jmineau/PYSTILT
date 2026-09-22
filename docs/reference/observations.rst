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
