Meteorology
===========

Each transport model reads its meteorology from a met config of its own.
HYSPLIT's, :class:`~stilt.transport.hysplit.MetConfig`, holds the settings
for one met in ``config.yaml``: ARL files in a directory, or an archive to
download. :class:`~stilt.transport.hysplit.Met` finds the files a
simulation needs, downloads them when ``download`` is set, and crops them
when asked. See :doc:`../guides/meteorology` for how to set it up.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.MetConfig
   stilt.transport.hysplit.Met
   stilt.meteorology.run_window

Wind-error statistics
---------------------

Variograms of analysis-minus-observation winds. Fit them to get the
wind-error settings for a transport-error run (:doc:`/guides/wind_errors`).

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.meteorology.variogram
   stilt.meteorology.fit_variogram
   stilt.meteorology.VariogramFit
