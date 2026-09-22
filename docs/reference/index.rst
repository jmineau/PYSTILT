API Reference
=============

Every public class and function, grouped by what it's for. If you're
looking for how to do something rather than what a function takes, the
:doc:`../guides/index` is a better starting point.

- :doc:`core`: ``Model``, receptors, simulations, trajectories, footprints,
  and spatial geometries
- :doc:`configuration`: every ``config.yaml`` option
- :doc:`meteorology`: finding and staging meteorology files
- :doc:`execution`: running simulations locally, on Slurm, or on Kubernetes
- :doc:`transforms`: particle weighting (averaging kernel, pressure
  weighting, lifetime)
- :doc:`observations`: observations, scenes, and receptor builders for
  column and satellite work
- :doc:`project`: project folders and storage backends
- :doc:`hysplit`: the low-level HYSPLIT driver

.. toctree::
   :maxdepth: 2

   core
   configuration
   meteorology
   execution
   transforms
   observations
   project
   hysplit
