HYSPLIT Integration
===================

:class:`stilt.hysplit.HYSPLITDriver` runs HYSPLIT once for one receptor in
one folder. It writes the input files such as ``CONTROL`` and ``SETUP.CFG``,
runs ``hycs_std``, and reads the particles it writes. :class:`stilt.Project`
uses it for every simulation. Use it directly only to run HYSPLIT outside a
project.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.hysplit.HYSPLITDriver
