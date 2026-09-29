Project And Store
=================

A project is one folder or cloud bucket path. It holds ``config.yaml``,
``receptors.csv``, and ``simulations/by-id/``. Every output's location is
given relative to the project root, and a simulation is finished when its
outputs exist there (see :doc:`../advanced/output_state`). The store reads
and writes those files, on a local disk or in a bucket. Most users never use
these classes directly. :class:`stilt.Model` uses them for you.

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
