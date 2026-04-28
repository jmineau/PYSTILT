Meteorology
===========

Meteorology in PYSTILT has both a output configuration layer and a runtime
file-discovery layer. ``MetConfig`` describes a source in ``config.yaml``;
``MetSource`` handles runtime file lookup and staging.

Runtime helpers
---------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.meteorology.MetID
   stilt.MetSource

Field-by-field configuration details for :class:`stilt.config.MetConfig` live
on :doc:`configuration`.
