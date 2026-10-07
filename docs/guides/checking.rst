Checking A Run
==============

This page is for after a run: which simulations finished, why some did
not, and what to do when a footprint looks wrong.

Where each simulation stands
----------------------------

``stilt status`` counts the simulations of a project, per variant, and the
failed ones by reason:

.. code-block:: text

   $ stilt status ./my_project
   Project: /data/my_project  total=1200  completed=1150  pending=50
   failed: MISSING_MET_FILES 42, MET_COVERAGE 8  (why: the .failure.yaml beside each log, under /data/output/logs)

In Python, ``project.status()`` is the same, one row per simulation:

.. code-block:: python

   status = project.status()
   status.state.value_counts()
   status[status.state == "failed"][["receptor", "variant", "reason", "message"]]

Each simulation is in one of four states. The state comes from the files in
the output directory, so it is the same whoever asks and whenever.

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - State
     - What it means
     - What to do
   * - ``complete``
     - Its particles exist, and its footprint when the variant has a grid.
     - Nothing. A rerun skips it.
   * - ``failed``
     - The last run failed. A failure record says why (below).
     - Fix the cause and run again. Only the unfinished simulations run.
   * - ``interrupted``
     - A run started and was stopped before it finished: a time limit,
       preemption, Ctrl-C, or a killed process.
     - Run again. A Slurm task that gets its two-minute warning requeues
       itself; the rest are picked up by the next ``stilt run``.
   * - ``pending``
     - Nothing has run yet.
     - Run the project.

``stilt run`` says how its own simulations ended in its exit code: 0 when
all are complete, 1 when some failed, and 3 when some were interrupted. It
exits with 2 when its command line is wrong. A script or a scheduler can
rerun on 3, and stop to look on 1.

Why a simulation failed
-----------------------

A failed simulation does not stop the others. The worker writes a failure
record, ``<receptor id>.failure.yaml``, beside the simulation's log, and
goes on. The record is removed when the result is written, so a simulation
that later succeeds carries no stale note. ``sim.failure`` reads it:

.. code-block:: pycon

   >>> sim = project.simulation(receptor_id, "hrrr")
   >>> sim.failure
   {'step': 'particles', 'reason': 'MET_COVERAGE',
    'message': 'HYSPLIT: start point not within (x,y,t) any data file',
    'time': '2026-10-03T21:14:05+00:00'}

``step`` is ``particles`` when the transport model failed. That fails every
variant that shares those particles. It is ``footprint`` when only that
variant's footprint failed. ``message`` is the line of the model's log that
says what went wrong. An unexpected error, such as a bug, also has a
``traceback``.

The reasons
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Reason
     - What happened, and what to do
   * - ``MISSING_MET_FILES``
     - PYSTILT found fewer meteorology files than the run needs (``n_min``
       of the met). Check the met's ``directory`` and ``file_format``
       against the files on disk, and that the files cover the receptor
       time and ``n_hours`` before it (:doc:`meteorology`).
   * - ``MET_COVERAGE``
     - HYSPLIT found the files but the receptor is outside them: in time
       (after the last hour the files hold) or in space (off the met grid,
       or off a cropped met's area). Add the missing files, or widen the
       crop.
   * - ``MET_TRUNCATED``
     - A met file holds a single time step where it should hold more, and
       the particles stop before the end of the run. The file was cut
       short, as by a download that stopped. Download it again.
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
   * - ``TIMEOUT``
     - The run took longer than ``timeout`` under ``execution:``. Raise it,
       or look for a slow node or a met file read over a slow filesystem.
   * - an error's name
     - Any other error is recorded by its class, such as ``ValueError``,
       with its traceback. That is usually a bug or a setup problem worth
       reporting.

The log and the workdir
~~~~~~~~~~~~~~~~~~~~~~~

``sim.log`` is the transport model's log. HYSPLIT runs in a workdir, a
folder of its input and output files. A successful run's workdir is
removed. A failed run's is kept under ``scratch/`` in the output directory,
and ``sim.kept_workdir`` is where. For HYSPLIT it holds ``CONTROL``,
``SETUP.CFG``, and ``MESSAGE``, enough to rerun ``hycs_std`` by hand. Set
``keep_scratch: true`` under ``execution:`` to keep every run's workdir.

A run cut off partway leaves a log of one line, saying when and on which
node it started. That line is how ``status()`` tells ``interrupted`` from
``pending``.

Empty footprints
----------------

Sometimes a simulation runs fine but no particle reaches the footprint
grid. PYSTILT then writes a footprint file with no cells, marked empty. The
simulation is complete, so reruns skip it:

.. code-block:: python

   sim.has_footprint            # True
   sim.footprint is None        # True: no particle reached the grid

``project.footprints()`` gives an empty footprint no row and lists it in
``attrs["empty"]``, and a Jacobian lists them in ``H.empty``. An empty
footprint is not a footprint of zeros. The transport never connected the
receptor to your grid, so treating it as "the model says zero" in a
comparison or an inversion would be wrong. If you see many, the grid is
too small or not upwind: plot the particles (below) and widen it.

My footprint looks wrong
------------------------

Start from the particles. A map of them shows where the air came from,
whatever the footprint grid:

.. code-block:: python

   sim.particles.stilt.plot.map()
   sim.footprint.stilt.plot.map()

Then check, in order:

The time
   Receptor times are UTC. A time in local time puts the release hours
   off, and with it every trajectory.

The release height
   Altitudes are above ground unless the receptor says
   ``altitude_ref="msl"``. A tower inlet given above sea level as if above
   ground starts the particles high above the boundary layer, and the
   footprint is weak and far away.

The grid
   The footprint covers only the grid. Particles that leave it add
   nothing, so a footprint cut off at an edge needs a wider grid.

The length of the run
   ``n_hours`` sets how far back the particles go. A footprint that stops
   short of a source you expect may need a longer run. A run whose
   particles stop before ``n_hours`` ran out of meteorology: read the log.

The number of particles
   Few particles give a patchy footprint. Raise ``numpar`` (500 to 1000
   for a tower).

The near-field correction
   ``hnf_plume`` (on by default) corrects the footprint close to the
   receptor, where particles have not yet mixed through the boundary
   layer. Comparing a run with it off shows how much of the footprint is
   near the receptor.

The units
   Footprints are in ppm per (µmol m⁻² s⁻¹), per hour layer. Multiplied by
   a flux in µmol m⁻² s⁻¹ and summed, they give ppm. ``time_integrate``
   sums the hours into one layer.

PYSTILT's footprints match STILT-R's to a relative 10⁻⁷ per cell, which
the fidelity tests check (:doc:`../development`). If yours differ from a STILT-R
run of the same receptor, compare the settings first: STILT-R's names are
in :doc:`../migration/stilt_r`.

Why the output has separate trees
---------------------------------

The output directory keeps results, logs, and kept workdirs in separate
trees (``particles/``, ``footprints/``, ``logs/``, ``scratch/``). Each
results tree holds Parquet files only, so pyarrow, DuckDB, polars, and R
read it as one table, and a log or a failure record beside a result would
break that. Whether a simulation is complete is read from the results
trees alone; the logs tree only explains the ones that are not.
