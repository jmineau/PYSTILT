Meteorology
===========

Meterology in PYSTILT has both a configuration layer and a runtime
file-resolution layer. ``MetConfig`` describes a stream in ``config.yaml``;
``MetStream`` handles runtime file resolution and staging.

Runtime helpers
---------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.meteorology.MetID
   stilt.MetStream

Field-by-field configuration details for :class:`stilt.config.MetConfig` live
on :doc:`configuration`.
