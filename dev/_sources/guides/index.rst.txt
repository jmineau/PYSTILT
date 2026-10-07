User Guide
==========

These pages show how to set up a project, run it, and use the results. If
you are new to PYSTILT, start with :doc:`../getting_started/quickstart`.

**Setting up a project**

- :doc:`receptors`: where and when you measured (towers, columns,
  satellites)
- :doc:`meteorology`: use your own ARL files or download them from NOAA
- :doc:`projects`: what goes in ``config.yaml``, variants, and how reruns
  and changed settings work

**Running**

- :doc:`running`: run on your computer, and the commands
- :doc:`slurm`: run thousands of simulations on an HPC cluster
- :doc:`containers`: run with Kubernetes or another scheduler
- :doc:`checking`: which simulations finished, why some failed, and what
  to do when a footprint looks wrong

**Results**

- :doc:`outputs`: load and plot footprints and particles
- :doc:`aggregation`: sum footprints over counties, hexagons, or sources,
  and many at once into a Jacobian
- :doc:`transport_error`: the modelled enhancement from a flux field, and
  its transport uncertainty
- :doc:`wind_errors`: the wind-error statistics a transport-error run needs
- :doc:`background`: the mole fraction at the trajectory endpoints, as a
  background for the receptor

**Column and satellite measurements**

- :doc:`readers`: read TROPOMI, OCO-2, or TCCON files into a table of
  soundings, or add your own instrument
- :doc:`slant_columns`: instruments that look along a tilted path
  (EM27/SUN, TCCON, off-nadir satellites)
- :doc:`transforms`: averaging kernels, pressure weighting, and your own
  particle weights
- :doc:`plume_background`: a background from the soundings a forward-run
  plume did not reach
- :doc:`../tutorials/satellite_column`: from satellite soundings to
  receptors, kernels, and modelled enhancements

**HYSPLIT**

- :doc:`hysplit_build`: run another HYSPLIT build, or your own land-use
  and terrain tables

.. toctree::
   :hidden:
   :caption: Setting up a project

   receptors
   meteorology
   projects

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
   wind_errors
   background

.. toctree::
   :hidden:
   :caption: Column and satellite measurements

   readers
   slant_columns
   transforms
   plume_background

.. toctree::
   :hidden:
   :caption: HYSPLIT

   hysplit_build
