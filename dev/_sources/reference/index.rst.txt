API Reference
=============

Every public class and function, grouped by what it is for. To find out how
to do something, start with the :doc:`../guides/index` instead.

:doc:`core`
   ``Project``, receptors, simulations, particles, footprints, spatial
   geometries, and flux sampling.

:doc:`particles`
   The particle table a transport model run gives: its columns and units.

:doc:`configuration`
   Every ``config.yaml`` option.

:doc:`meteorology`
   Finding, downloading, and cropping meteorology files.

:doc:`execution`
   Running simulations locally or on Slurm.

:doc:`transforms`
   Particle weighting: averaging kernels, pressure weighting, and lifetime
   decay.

:doc:`observations`
   Column and satellite work: product readers and their columns, sounding
   selection, receptors from soundings, slant geometry, the modelled
   column, and plume backgrounds.

:doc:`project`
   The project, its simulations, and the output directory.

:doc:`hysplit`
   The low-level HYSPLIT driver.

:doc:`exceptions`
   The exceptions PYSTILT raises.

.. toctree::
   :maxdepth: 2

   core
   particles
   configuration
   meteorology
   execution
   transforms
   observations
   project
   hysplit
   exceptions
