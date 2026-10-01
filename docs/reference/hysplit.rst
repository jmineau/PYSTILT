HYSPLIT Integration
===================

HYSPLIT is the transport model behind every simulation. The worker gets it
as :class:`stilt.hysplit.HysplitModel` (see :doc:`execution`), which picks
the met files and hands the run to :class:`stilt.hysplit.HYSPLITDriver`.

:class:`stilt.hysplit.HYSPLITDriver` runs HYSPLIT once for one receptor in
one folder. It writes the input files such as ``CONTROL`` and ``SETUP.CFG``,
runs ``hycs_std``, and reads the particles it writes. Use it directly only
to run HYSPLIT outside a project.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.hysplit.HYSPLITDriver
