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
   * - :doc:`containers`
     - You run many containers at once, such as a Kubernetes Job.
     - A volume the containers share, and ``stilt run --task``

.. toctree::
   :maxdepth: 1
   :hidden:

   local
   slurm
   containers

Commands you'll use
-------------------

``stilt init <project>``
   Create a project folder with a starter ``config.yaml`` and
   ``receptors.csv``.

``stilt run <project>``
   Run every simulation that isn't finished yet, and wait until they are
   done, on your computer or as a Slurm job array. From Python, this is
   ``project.run()``, which returns the status table of the simulations it
   ran (``state`` is ``complete``, ``failed``, or ``pending``).

``stilt submit <project>``
   Submit the unfinished simulations to Slurm as a job array, and return as
   soon as it is submitted. From Python, this is ``project.submit()``.

``stilt status <project>``
   Count finished and remaining simulations, and the failed ones by reason.

``stilt output ls <project>``
   List the settings folders in the output directory, which variants use
   them, and how the unused ones differ.

Options for ``stilt run``:

- ``--backend`` and ``--n-workers`` override ``config.yaml`` for this run.
- ``--no-skip`` reruns every simulation, finished or not.
- ``--compute-root DIR`` runs HYSPLIT in ``DIR``, a scratch folder that is
  emptied after each successful run.
- ``--receptors FILE`` runs only the receptors listed in ``FILE``, one id
  per line.
- ``--task I/N`` runs task ``I`` of ``N`` here, as one task of a job array
  (:doc:`containers`).

``stilt run`` exits with 0 when every simulation it ran is complete, 1 when
some failed, and 2 when some did not finish because the run was stopped.

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
``status()`` lists every failed simulation in a selection:

.. code-block:: pycon

   >>> sim.failure
   {'step': 'particles', 'reason': 'MET_COVERAGE',
    'message': 'HYSPLIT: start point not within (x,y,t) any data file',
    'time': '2026-10-03T21:14:05+00:00'}
   >>> st = project.status()
   >>> st[st.state == "failed"][["receptor", "variant", "reason"]]

``step`` is ``particles`` when the transport model failed, which fails every variant that
shares those particles, and ``footprint`` when only that variant's footprint
did. ``reason`` is a short name for the cause, or the error's class when it
has none. For a HYSPLIT failure, ``message`` is the line of HYSPLIT's log
that says what went wrong. ``sim.log`` reads the whole log, and
``sim.kept_workdir`` is HYSPLIT's working folder, kept in the output
directory. An unexpected error, such as a bug, also has a ``traceback``.

A footprint can be empty because no particle reached the grid. That is not a
failure. PYSTILT records it with the reason, and the simulation counts as
finished (see :doc:`../outputs`).
