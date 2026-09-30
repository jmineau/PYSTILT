API Reference
=============

Every public class and function, grouped by what it is for. To find out how
to do something, start with the :doc:`../guides/index` instead.

:doc:`core`
   ``Model``, receptors, simulations, trajectories, footprints, spatial
   geometries, and flux sampling.

:doc:`configuration`
   Every ``config.yaml`` option.

:doc:`meteorology`
   Finding and staging meteorology files.

:doc:`execution`
   Running simulations locally or on Slurm.

:doc:`transforms`
   Particle weighting: averaging kernels, pressure weighting, and lifetime
   decay.

:doc:`observations`
   Column and satellite work: product readers, sounding selection, slant
   geometry, transport error, and backgrounds.

:doc:`project`
   Project folders and storage backends.

:doc:`hysplit`
   The low-level HYSPLIT driver.

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
