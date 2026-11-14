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

Running HYSPLIT
---------------

:func:`~stilt.transport.hysplit.write_inputs` writes everything
``hycs_std`` reads into a folder: ``CONTROL``, ``SETUP.CFG``, and the error
and mixed-layer files when the settings call for them. It takes what
HYSPLIT's ``CONTROL`` file holds a release time and release locations
(latitude, longitude and height, in ``CONTROL``'s order), with ``kmsl`` 0
for heights above ground and 1 above sea level, as ``SETUP.CFG``'s ``KMSL``.
:class:`~stilt.transport.hysplit.HysplitModel` gives it a receptor's time,
release points (``receptor.coords()``) and ``altitude_ref``. Use it to look
at the input files of a run, and
:func:`~stilt.transport.hysplit.run_hycs_std` to run ``hycs_std`` on them.
:func:`~stilt.transport.hysplit.read_particle_dat` reads a
``PARTICLE_STILT.DAT`` as a particle table. The release heights and the
near-field correction are PYSTILT's steps, the same for any model
(:doc:`particles`):

.. code-block:: python

   from pathlib import Path

   from stilt.particles import add_release_heights, correct_near_field
   from stilt.transport.hysplit import (
       HysplitConfig,
       read_particle_dat,
       run_hycs_std,
       write_inputs,
   )

   config = HysplitConfig(n_hours=-24, numpar=200)
   write_inputs(
       "run",
       release_time=receptor.time,
       locations=receptor.coords(),
       config=config,
       met_files=met_files,
       kmsl=1 if receptor.altitude_ref == "msl" else 0,
   )
   run_hycs_std(Path("run"))
   particles = read_particle_dat("run/PARTICLE_STILT.DAT", config.varsiwant)
   particles = add_release_heights(particles, receptor)
   particles = correct_near_field(particles, receptor, config.veght)

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.write_inputs
   stilt.transport.hysplit.run_hycs_std
   stilt.transport.hysplit.read_particle_dat

Failure reasons
---------------

When a run ends, PYSTILT reads HYSPLIT's log for the messages it knows.
When the particles stopped before ``n_hours``, it also checks where: a
particle more than 5 grid cells inside the met's domain means the met ran
out (``MET_COVERAGE``), since particles that leave the domain end at its
edge (:doc:`/development/validation`). A known failure, a timeout, or a run
without particle output raises a
:class:`~stilt.exceptions.SimulationError` whose ``reason`` is a
:class:`stilt.transport.hysplit.FailureReason`. The worker records it, and
:attr:`stilt.Simulation.failure` and ``stilt status`` report it.

Each reason, what happened, and what to do:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Reason
     - What happened, and what to do
   * - ``MET_COVERAGE``
     - The meteorology does not cover the run. The message says which:

       - ``No met file ... for <hours>``: a file the run needs is missing.
         Check the met's ``directory`` against the files on disk. A backward run needs files from ``n_hours`` before
         the receptor time to just after it, a forward run from the
         receptor time to just after ``n_hours`` later (:doc:`/guides/meteorology`).
         ``stilt check`` finds these before a run.
       - ``The met file ... is damaged`` or ``The met files hold no time
         step between ...``: a file is there but cut short, written
         partway, or has records lost to null bytes, so it lacks hours the
         run needs. Download or crop it again. PYSTILT checks each file
         before HYSPLIT runs (:meth:`arlmet.File.check`).
       - ``The met file ... holds one time period``: HYSPLIT cannot read a
         file with one time period. Use files that each hold at least two
         (``stilt check`` finds this before a run).
       - ``no more meteorology``: the particles reached a time that no met
         file holds. HYSPLIT says so in the ``WARNING`` file of the kept
         workdir.
       - ``The particles stopped ... h into the ... h run, up to ... grid
         cells inside the met's domain``: the particles stopped before
         ``n_hours`` and at least one was more than 5 grid cells from
         every side of the met grid.
         Particles that leave the domain end within 2 cells of its edge,
         so the met ran out, though HYSPLIT may have written no warning
         and ``Complete Hysplit``. Look for a met file with fewer hours
         than its name says, and download or crop it again. PYSTILT
         checks this only when ``outdt`` is 0, the default: with fewer
         rows, a particle's last row can be far from where it left.
       - A HYSPLIT message, such as ``start point not within (x,y,t) any
         data file``: the receptor is outside the files, in time or in
         space. ``source point off grid`` says the receptor is outside the
         met's grid (check the sign of its longitude), and ``Start height
         above mdl domain`` that it is above the met's top; the line gives the
         height and the top.
   * - ``VARYING_MET_INTERVAL``
     - The met files do not all have the same time step, such as hourly
       and 3-hourly files mixed in one directory. Keep one product per
       met.
   * - ``NO_PARTICLE_DATA``
     - The transport model ran but wrote no particles. Read the log
       (``sim.log``): it usually says why.
   * - ``FORTRAN_RUNTIME_ERROR``
     - HYSPLIT stopped with a Fortran error, which the message names. Read
       the log and the kept workdir's ``MESSAGE`` file.
   * - ``PARTICLE_LIMIT``
     - HYSPLIT reached ``maxpar`` before every location of a multipoint
       receptor got its particles, so the last ones would have fewer or
       none. Leave ``maxpar`` unset, or raise it (:doc:`/guides/receptors`).
   * - ``EXIT_STATUS``
     - ``hycs_std`` was killed by a signal, or exited with a non-zero
       status, and its log holds no other known message. The message names
       the signal or the status, and the log ends with the last lines of
       HYSPLIT's ``MESSAGE`` file. A ``SIGKILL`` that nothing in PYSTILT
       sent is often the kernel's out-of-memory killer: ask for more
       memory per task, or run fewer receptors at once. The particles
       HYSPLIT had written are not kept, and running again runs the
       simulation again.
   * - ``TIMEOUT``
     - The run took longer than ``timeout`` under ``execution:``. Raise it,
       or look for a slow node or a met file read over a slow filesystem.
   * - an error's name
     - Any other error is recorded by its class, such as ``ValueError``,
       with its traceback. That is usually a bug or a setup problem worth
       reporting.


.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.hysplit.FailureReason
