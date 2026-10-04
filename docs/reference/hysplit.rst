HYSPLIT Integration
===================

HYSPLIT is the transport model behind every simulation, unless a variant
names another. The worker gets it as
:class:`stilt.transport.hysplit.HysplitModel` (see :doc:`execution`), which
picks the met files, writes HYSPLIT's input files, runs ``hycs_std``, and
reads the particles it writes.

Config
------

:class:`stilt.transport.hysplit.HysplitConfig` holds every setting that
shapes the particles HYSPLIT makes. In ``config.yaml`` they are flat,
top-level keys.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.HysplitConfig
   stilt.transport.hysplit.finish_particles

Running HYSPLIT
---------------

:func:`~stilt.transport.hysplit.write_inputs` writes everything
``hycs_std`` reads for one receptor into a folder: ``CONTROL``,
``SETUP.CFG``, and the error and mixed-layer files when the settings call
for them. Use it to look at the input files of a run.
:func:`~stilt.transport.hysplit.read_particle_dat` reads a
``PARTICLE_STILT.DAT`` as a particle table, and
:func:`~stilt.transport.hysplit.finish_particles` adds the release heights
and the near-field correction, as a run does.

.. code-block:: python

   from stilt.transport.hysplit import finish_particles, read_particle_dat

   particles = read_particle_dat("PARTICLE_STILT.DAT", config.varsiwant)
   particles = finish_particles(particles, receptor, config)

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.write_inputs
   stilt.transport.hysplit.read_particle_dat

Failure reasons
---------------

When a run ends, PYSTILT reads HYSPLIT's log for the messages it knows.
A known failure, a timeout, or a run without particle output raises a
:class:`~stilt.exceptions.SimulationError` whose ``reason`` is a
:class:`stilt.transport.hysplit.FailureReason`. The worker records it, and
:attr:`stilt.Simulation.failure` and ``stilt status`` report it.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.FailureReason
