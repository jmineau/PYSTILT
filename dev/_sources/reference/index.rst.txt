API Reference
=============

Every public class and function, grouped by what it is for. To find out how
to do something, start with the :doc:`../guides/index` instead.

:doc:`core`
   ``Project``, releases, simulations, particles, footprints, the
   background and transport error, spatial geometries, and sampling.

:doc:`particles`
   The particle table a transport model run gives: its columns and units.

:doc:`layout`
   The files of an output directory, for reading them without PYSTILT.

:doc:`configuration`
   Every ``config.yaml`` option.

:doc:`meteorology`
   Finding, downloading, and cropping meteorology files.

:doc:`execution`
   Running simulations locally or on Slurm.

:doc:`transforms`
   Particle weighting: averaging kernels, pressure weighting, and the loss
   of a species to decay, deposition, or oxidation.

:doc:`observations`
   Column, satellite and mobile measurements: product readers and their
   columns, sounding selection, releases from soundings and from tracks,
   slant geometry, the modeled column, and plume backgrounds.

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
   layout
   configuration
   meteorology
   execution
   transforms
   observations
   project
   hysplit
   exceptions
