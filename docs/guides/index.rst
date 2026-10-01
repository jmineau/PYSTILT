User Guide
==========

These pages show how to set up a project, run it, and use the results. If
you are new to PYSTILT, start with :doc:`../getting_started/quickstart`.

**Setting up a project**

- :doc:`receptors`: where and when you measured (towers, columns,
  satellites)
- :doc:`meteorology`: use your own ARL files or download them from NOAA
- :doc:`configuration`: what goes in ``config.yaml``
- :doc:`project_layout`: what is in a project folder, and how reruns work

**Running simulations**

- :doc:`execution/index`: pick where to run
- :doc:`execution/local`: on your computer or in a notebook
- :doc:`execution/slurm`: on an HPC cluster

**Working with results**

- :doc:`outputs`: load, plot, and aggregate footprints and particles
- :doc:`transport_error`: the modelled enhancement from a flux field, and
  its transport uncertainty
- :doc:`wind_errors`: estimate the wind-error statistics a transport-error
  run needs
- :doc:`background`: the mole fraction at the trajectory endpoints, as a
  background for the receptor
- :doc:`plume_background`: a background from the soundings a forward-run
  plume did not reach

**Column and satellite measurements**

- :doc:`readers`: read TROPOMI, OCO-2 or TCCON files into a table of
  soundings, or add your own instrument
- :doc:`../advanced/observations`: turn satellite soundings or column
  retrievals into receptors
- :doc:`slant_columns`: instruments that look along a tilted path
  (EM27/SUN, TCCON, off-nadir satellites)
- :doc:`../advanced/transforms`: averaging kernels, pressure weighting, and
  your own particle weights

**Coming from other STILT tools**

- :doc:`../migration/stilt_r`
- :doc:`../migration/x_stilt`
- :doc:`../migration/stiltctl`

**Under the hood**

- :doc:`../advanced/output_state`: how PYSTILT tracks finished work

.. toctree::
   :hidden:
   :caption: Setting up a project

   receptors
   meteorology
   configuration
   project_layout

.. toctree::
   :hidden:
   :caption: Running simulations

   execution/index

.. toctree::
   :hidden:
   :caption: Working with results

   outputs
   transport_error
   wind_errors
   background
   plume_background

.. toctree::
   :hidden:
   :caption: Column and satellite measurements

   readers
   ../advanced/observations
   slant_columns
   ../advanced/transforms

.. toctree::
   :hidden:
   :caption: Coming from other STILT tools

   ../migration/stilt_r
   ../migration/x_stilt
   ../migration/stiltctl

.. toctree::
   :hidden:
   :caption: Under the hood

   ../advanced/output_state
