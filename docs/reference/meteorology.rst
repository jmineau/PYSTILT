Meteorology
===========

:class:`~stilt.MetConfig` holds the settings for one met in
``config.yaml``. Its fields are listed on :doc:`configuration`.
:class:`~stilt.meteorology.Met` finds the files a simulation needs, downloads them
when ``download`` is set, and crops them when asked. See
:doc:`../guides/meteorology` for how to set it up.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.meteorology.Met
