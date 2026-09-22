User Guide
==========

Each page answers one practical question. If you haven't yet, start with
:doc:`../getting_started/quickstart`.

**Setting up a project**

- :doc:`receptors`: describe where and when you measured (towers, columns,
  satellites)
- :doc:`meteorology`: use your own ARL files or download them from NOAA
- :doc:`configuration`: what goes in ``config.yaml``
- :doc:`project_layout`: what's in a project folder, and how reruns work

**Running simulations**

- :doc:`execution/index`: pick where to run
- :doc:`execution/local`: on your computer or in a notebook
- :doc:`execution/slurm`: on an HPC cluster

**Working with results**

- :doc:`outputs`: load, plot, and aggregate footprints and trajectories
- :doc:`transport_error`: the modelled enhancement for a flux field, and
  how uncertain the transport makes it
- :doc:`wind_errors`: the wind-error statistics a transport-error run
  needs, from your meteorology and observed winds
- :doc:`background`: what the receptor saw before the domain's fluxes,
  from a mole-fraction field at the trajectory endpoints

**Column and satellite measurements**

- :doc:`../advanced/observations`: from satellite soundings or column
  retrievals to receptors
- :doc:`slant_columns`: instruments that look along a tilted path
  (EM27/SUN, TCCON, off-nadir satellites)
- :doc:`../advanced/transforms`: averaging kernels, pressure weighting, and
  your own particle weights

**Coming from other STILT tools**

- :doc:`../migration/r_stilt`
- :doc:`../migration/x_stilt`
- :doc:`../migration/stiltctl`

**Under the hood**

- :doc:`../advanced/output_state`: how PYSTILT tracks finished work
- :doc:`execution/kubernetes`: cloud workers (experimental)

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

.. toctree::
   :hidden:
   :caption: Column and satellite measurements

   ../advanced/observations
   slant_columns
   ../advanced/transforms

.. toctree::
   :hidden:
   :caption: Coming from other STILT tools

   ../migration/r_stilt
   ../migration/x_stilt
   ../migration/stiltctl

.. toctree::
   :hidden:
   :caption: Under the hood

   ../advanced/output_state
