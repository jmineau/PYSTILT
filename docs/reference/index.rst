API Reference
=============

This page gives an overview of the documented PYSTILT API. The public surface
is grouped by workflow area, closer to how pandas organizes its reference:

- ``stilt`` for project models, receptors, simulations, and outputs
- ``stilt.config`` for output configuration and STILT parameter models
- ``stilt.execution`` for local, Slurm, and Kubernetes execution
- ``stilt.observations`` for scene, sensor, and transform objects
- ``stilt.project`` and ``stilt.store`` for project roots and output stores

.. warning::

   PYSTILT is still in alpha. Expect APIs and workflow changes.

.. toctree::
   :maxdepth: 2

   core
   configuration
   meteorology
   execution
   observations
   project
   hysplit
