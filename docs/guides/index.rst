User Guide
==========

These pages show how to set up a project, run it, and use the results. If
you are new to PYSTILT, start with :doc:`../getting_started/quickstart`.

**Setting up a project**

- :doc:`releases`: releases (receptors), where and when you measured (towers, columns,
  satellites)
- :doc:`tracks`: releases along the track of a car, train, aircraft, or
  ship, and the link from each measurement to its release
- :doc:`meteorology`: use your own ARL files or download them from NOAA
- :doc:`projects`: what goes in ``config.yaml``, variants, and how reruns
  and changed settings work
- :doc:`choosing_settings`: how many particles, how long a run, which grid
  and which met, and how to check each choice
- :doc:`transforms`: weight the particles before the footprint is made:
  averaging kernels and pressure weighting for columns, and the loss of a
  species that decays, deposits, or reacts with OH on the way

**Running**

- :doc:`running`: run on your computer, and the commands
- :doc:`slurm`: run thousands of simulations on an HPC cluster
- :doc:`containers`: run in Docker, Kubernetes or another scheduler
- :doc:`checking`: which simulations finished, why some failed, and what
  to do when a footprint looks wrong

**Results**

- :doc:`outputs`: load and plot footprints and particles, and write them
  for other tools (CF NetCDF, CF trajectories, openghg)
- :doc:`aggregation`: sum footprints over counties, hexagons, or sources,
  and many at once into a Jacobian
- :doc:`transport_error`: the transport uncertainty of a modeled
  enhancement, and the wind-error statistics it needs
- :doc:`background`: the mole fraction at the trajectory endpoints, as a
  background for the release
- :doc:`trajectory_analyses`: residence time, PSCF, CWT, mean paths, and
  the transit time from a source, from particles and footprints

**Column and satellite measurements**

- :doc:`readers`: read TROPOMI, OCO-2/3, TCCON, or EM27/SUN (GGG, COCCON,
  PROFFAST) files into a table of soundings, or add your own instrument
- :doc:`slant_columns`: instruments that look along a tilted path
  (EM27/SUN, TCCON, off-nadir satellites)
- :doc:`transforms`: the averaging kernel and pressure weighting a column
  footprint needs
- :ref:`plume-background`: a background from the soundings a forward-run
  plume did not reach
- :doc:`../tutorials/satellite_column`: from satellite soundings to
  releases, kernels, and modeled enhancements

**HYSPLIT**

- :doc:`hysplit_users`: HYSPLIT's starting points, settings and particle
  output in PYSTILT, a single trajectory, and many runs as a project
- :doc:`hysplit_build`: run another HYSPLIT build, or your own land-use
  and terrain tables

.. _coming-from-another-stilt:

**Coming from another STILT**

If you have used another STILT tool, these pages show where its settings and
steps are in PYSTILT.

- :doc:`migration/stilt_r`: where each ``run_stilt.r`` setting lives, and what
  PYSTILT does differently
- :doc:`migration/x_stilt`: the column and satellite workflow (releases from
  soundings, averaging kernels, backgrounds, and transport error)
- :doc:`migration/stiltctl`: running STILT on cloud infrastructure, and what
  replaces the work queue and the long-lived workers
- :doc:`migration/jena_stilt`: the Jena ``stiltR`` code behind the ICOS
  footprint service, how its footprints differ, and the settings that come
  closest

.. toctree::
   :hidden:
   :caption: Setting up a project

   releases
   tracks
   meteorology
   projects
   choosing_settings
   transforms

.. toctree::
   :hidden:
   :caption: Running

   running
   checking

.. toctree::
   :hidden:
   :caption: Results

   outputs
   aggregation
   transport_error
   background
   trajectory_analyses

.. toctree::
   :hidden:
   :caption: Column and satellite measurements

   readers
   slant_columns

.. toctree::
   :hidden:
   :caption: HYSPLIT

   hysplit_users
   hysplit_build

.. toctree::
   :hidden:
   :caption: Coming from another STILT

   migration/stilt_r
   migration/x_stilt
   migration/stiltctl
   migration/jena_stilt
