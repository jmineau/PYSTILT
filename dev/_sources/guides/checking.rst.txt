Checking A Run
==============

This page is for after a run: which simulations finished, why some did
not, and what to do when a footprint looks wrong.

Where each simulation stands
----------------------------

``stilt status`` counts the simulations of a project, per variant, and the
failed ones by reason. Then it lists the output's settings folders
(:doc:`projects`):

.. code-block:: text

   $ stilt status ./my_project
   Project: /data/my_project  total=1200  complete=1150  failed=42  interrupted=0  pending=8
   failed: MET_COVERAGE 34, TIMEOUT 8  (why: each one's .failure.yaml, under /data/output/logs)
   Output: /data/output
     particles   settings=hrrr-5d01e7  1,150 files  hrrr
     footprints  settings=hrrr-93278c  1,150 files  hrrr

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
     - The result files its variant makes exist in the output directory.
       Today that is the particle file, and the footprint file when the
       variant has a grid.
     - Nothing. A rerun skips it.
   * - ``failed``
     - The last run failed. A failure record says why (below).
     - Fix the cause and run again. Only the unfinished simulations run.
   * - ``interrupted``
     - The transport model's run started and was stopped before it
       finished: a time limit, preemption, Ctrl-C, or a killed process.
     - Run again. A Slurm task that gets its two-minute warning requeues
       itself; the rest are picked up by the next ``stilt run``.
   * - ``pending``
     - The transport model has not run yet, or the particles exist and the
       footprint is not made yet.
     - Run the project.

A footprint that is not made yet is ``pending`` whether its step never
started or was stopped partway. The files cannot tell those apart: a
footprint variant added later, on particles that already exist, leaves
the same files.

``stilt run`` says how its own simulations ended in its exit code: 0 when
all are complete, 1 when some failed, and 3 when some were interrupted. It
exits with 2 when its command line is wrong. A script or a scheduler can
rerun on 3, and stop to look on 1.

Why a simulation failed
-----------------------

A failed simulation does not stop the others. The worker writes a failure
record, ``<receptor id>.failure.yaml``, in the output's ``logs/`` tree, and
goes on. A failed run's record is under its particles' settings folder,
beside the run's log. A failed footprint's record is under the footprint's
own settings folder (:doc:`../reference/layout`). The record is removed
when the result is written, so a simulation that later succeeds carries no
stale note. ``sim.failure`` reads it:

.. code-block:: pycon

   >>> sim = project.simulation(receptor_id, "hrrr")
   >>> sim.failure
   {'step': 'particles', 'reason': 'MET_COVERAGE',
    'message': 'HYSPLIT: start point not within (x,y,t) any data file',
    'time': '2026-10-03T21:14:05+00:00'}

``step`` is ``particles`` when the transport model failed. That fails every
variant that shares those particles. It is ``footprint`` when only that
variant's footprint failed. ``message`` is the line of the transport model's
log that says what went wrong. An unexpected error, such as a bug, also has a
``traceback``.

The reasons
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Reason
     - What happened, and what to do
   * - ``MET_COVERAGE``
     - The meteorology does not cover the run. The message says which:

       - ``No met file ... for <hours>``: a file the run needs is missing.
         Check the met's ``directory`` and ``file_format`` against the
         files on disk. A backward run needs files from ``n_hours`` before
         the receptor time to just after it (:doc:`meteorology`).
       - ``The met file ... is damaged`` or ``The met files hold no time
         step between ...``: a file is there but cut short, written
         partway, or has records lost to null bytes, so it lacks hours the
         run needs. Download or crop it again. PYSTILT checks each file
         before HYSPLIT runs (:meth:`arlmet.File.check`).
       - ``no more meteorology``: the particles reached a time that no met
         file holds. HYSPLIT says so in the ``WARNING`` file of the kept
         workdir.
       - A HYSPLIT message, such as ``start point not within (x,y,t) any
         data file``: the receptor is outside the files, in time or in
         space.
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
``keep_workdir: true`` under ``execution:`` to keep every run's workdir.

A run cut off partway leaves its log, which starts with a line saying when
and on which node it started. ``status()`` calls a simulation
``interrupted`` when its log exists and it has no particles and no failure
record.

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
   short of a source you expect may need a longer run. Particles also stop
   early when they all leave the met's domain or its crop, and the run is
   still complete. ``sim.particles["age"].abs().max() / 60`` is how many
   hours they went. If that is short of ``n_hours`` and the footprint
   needs more, widen ``subgrid_bounds``.

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
