Where To Run
============

The same project can run on your computer or on an HPC cluster. Only the
``execution`` section of ``config.yaml`` changes. Your receptors,
meteorology, footprints, and outputs stay the same.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Backend
     - Use it when
     - Setup
   * - :doc:`local`
     - You're working in a notebook or script, or have up to a few hundred
       simulations. This is the default.
     - Nothing
   * - :doc:`slurm`
     - You have thousands of simulations and access to an HPC cluster that
       uses Slurm.
     - An ``execution`` section with your account and partition

.. toctree::
   :maxdepth: 1
   :hidden:

   local
   slurm

Commands you'll use
-------------------

``stilt init <project>``
   Create a project folder with a starter ``config.yaml`` and
   ``receptors.csv``.

``stilt run <project>``
   Run every simulation that isn't finished yet, and wait until they are
   done, on your computer or as a Slurm job array. From Python, this is
   ``project.run()``.

``stilt submit <project>``
   Submit the unfinished simulations to Slurm as a job array, and return as
   soon as it is submitted. From Python, this is ``project.submit()``.

``stilt status <project>``
   Count finished and remaining simulations, and the failed ones by reason.

Options for ``stilt run``:

- ``--backend`` and ``--n-workers`` override ``config.yaml`` for this run.
- ``--no-skip`` reruns every simulation, finished or not.
- ``--compute-root DIR`` runs HYSPLIT in ``DIR``, a scratch folder that is
  emptied after each successful run.

When a simulation fails
-----------------------

A failed simulation doesn't stop the others. The worker records why it
failed, keeps HYSPLIT's working folder under ``scratch/`` in the output
directory, and goes on. The simulation stays unfinished, so the next
``stilt run`` tries it again. Fix the cause (often missing meteorology) and
run again. Finished simulations are skipped, and a simulation that succeeds
loses its failure record.

``stilt status`` counts the failures by reason:

.. code-block:: text

   Project: /path/to/my_project  total=1200  completed=1150  pending=50
   failed: MISSING_MET_FILES 42, MET_TRUNCATED 8  (sim.failure says why)

In Python, ``sim.failure`` says why one simulation failed, and
``sims.failures()`` lists every failed simulation in a selection:

.. code-block:: python

   >>> sim.failure
   {'step': 'particles', 'error': 'MeteorologyError', 'reason': 'MISSING_MET_FILES',
    'message': 'Insufficient number of meteorological files found. ...',
    'time': '2026-10-03T21:14:05+00:00', 'log': None, 'scratch': None}
   >>> project.simulations.failures()[["receptor", "variant", "reason"]]

``step`` is ``particles`` when HYSPLIT failed, which fails every variant that
shares those particles, and ``footprint`` when only that variant's footprint
did. ``log`` and ``scratch`` point at HYSPLIT's log and working folder in
the output directory, and ``sim.log`` reads the log. An unexpected error,
such as a bug, also has a ``traceback``.

A footprint can be empty because no particle reached the grid. That is not a
failure. PYSTILT records it with the reason, and the simulation counts as
finished (see :doc:`../outputs`).
