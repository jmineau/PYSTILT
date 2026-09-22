Project And Store
=================

A project is one root — a local directory or an object-store URI — holding
``config.yaml``, ``receptors.csv``, and ``simulations/by-id/``. Every output is
addressed by a *store key* relative to that root, and whether a simulation is
complete is read from the store by key (see :meth:`stilt.Simulation.is_complete`).
There is no registry: the simulations a project defines are its receptors
crossed with its met streams.

Project
-------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.project.Project
   stilt.project.simulation_prefix
   stilt.project.project_slug

Store backends
--------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.store.Store
   stilt.store.LocalStore
   stilt.store.FsspecStore
   stilt.store.make_store
