Meteorology
===========

Meteorology in PYSTILT has both a durable configuration layer and a runtime
file-discovery layer. ``MetConfig`` describes a stream in ``config.yaml``;
``MetStream`` and ``MetArchive`` handle runtime resolution and staging.

Runtime helpers
---------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.MetID
   stilt.MetArchive
   stilt.MetStream

Field-by-field configuration details for :class:`stilt.config.MetConfig` live
on :doc:`configuration`.
