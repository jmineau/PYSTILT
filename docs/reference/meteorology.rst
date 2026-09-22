Meteorology
===========

:class:`~stilt.config.MetConfig` holds a meteorology source's settings in
``config.yaml``. :class:`~stilt.MetStream` does the work at run time: finding
the files a simulation needs and linking them into its working folder. See
:doc:`../guides/meteorology` for how to set it up.

Runtime helpers
---------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.meteorology.MetID
   stilt.MetStream

Field-by-field configuration details for :class:`stilt.config.MetConfig` live
on :doc:`configuration`.
