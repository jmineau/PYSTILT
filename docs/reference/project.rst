Project And Store
=================

A project is one folder (or cloud bucket path) holding ``config.yaml``,
``receptors.csv``, and ``simulations/by-id/``. Every output's location is
given relative to that folder, and a simulation is finished when its outputs
exist there (see :doc:`../advanced/output_state`). Most users never use
these classes directly; :class:`stilt.Model` does.

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
