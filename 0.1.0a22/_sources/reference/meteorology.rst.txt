Meteorology
===========

:class:`~stilt.config.MetConfig` holds the settings for one meteorology
source in ``config.yaml``. Its fields are listed on :doc:`configuration`.
:class:`~stilt.MetStream` finds the files a simulation needs, downloads them
when a ``source`` is set, and links or copies them into the simulation's
folder. See
:doc:`../guides/meteorology` for how to set it up.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.MetStream
