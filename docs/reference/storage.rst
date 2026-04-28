Storage And Index
=================

The output runtime is split between an index for simulation state and storage
helpers for locating project inputs, outputs, and backing stores.

Index models
------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.index.IndexCounts
   stilt.index.OutputSummary
   stilt.index.SimulationIndex

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
