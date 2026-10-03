HYSPLIT Integration
===================

HYSPLIT is the transport model behind every simulation, unless a variant
names another. The worker gets it as
:class:`stilt.transport.hysplit.HysplitModel` (see :doc:`execution`), which
picks the met files and hands the run to
:class:`stilt.transport.hysplit.HYSPLITDriver`.

Config
------

:class:`stilt.transport.hysplit.HysplitConfig` holds every setting that
shapes the particles HYSPLIT makes. In ``config.yaml`` they are flat,
top-level keys.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.HysplitConfig
   stilt.transport.hysplit.model.finish_particles

Running HYSPLIT
---------------

:class:`stilt.transport.hysplit.HYSPLITDriver` runs HYSPLIT once for one
receptor in one folder. It writes the input files such as ``CONTROL`` and ``SETUP.CFG``,
runs ``hycs_std``, and reads the particles it writes. Use it directly only
to run HYSPLIT outside a project.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.HYSPLITDriver

Failure reasons
---------------

When a run fails, :func:`stilt.transport.hysplit.identify_failure_reason`
reads its log and returns the :class:`stilt.transport.hysplit.FailureReason`
for the first message it knows. :attr:`stilt.Simulation.outcome` reports it
as ``"failed:<reason>"``.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.FailureReason
   stilt.transport.hysplit.identify_failure_reason
