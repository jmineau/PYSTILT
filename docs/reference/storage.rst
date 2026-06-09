Storage, Completion, And Manifest
=================================

The output runtime has no database locally. A simulation's state is split
between three small pieces: **completion** is computed by key from the store,
the **manifest** is a parquet registry of what was registered, and **storage**
helpers locate project inputs, outputs, and backing stores.

Completion
----------

A simulation is complete iff every artifact it is configured to produce exists
in the store (see :mod:`stilt.completion`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.completion.StatusCounts
   stilt.completion.expected_artifacts
   stilt.completion.is_complete

Manifest
--------

The registry of registered simulations, persisted as ``.stilt/manifest.parquet``
through a :class:`~stilt.storage.Store`. Registration metadata only; completion
is never stored here.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.manifest.Manifest

Storage helpers
---------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.storage.ProjectLayout
   stilt.storage.ProjectFiles
   stilt.storage.SimulationFiles
   stilt.storage.Storage
   stilt.storage.FsspecStore
