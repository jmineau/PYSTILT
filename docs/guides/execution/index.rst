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
   ran (``state`` is ``complete``, ``failed``, ``interrupted``, or
   ``pending``).

``stilt submit <project>``
   Submit the unfinished simulations to Slurm as a job array, and return as
   soon as it is submitted. From Python, this is ``project.submit()``.

``stilt status <project>``
   Count finished and remaining simulations, and the failed ones by reason.
   List the settings folders in the output directory, which variants use
   them, and how the unused ones differ. ``--json`` prints the same as
   JSON, for a program to read.

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
some failed, and 3 when some did not finish because the run was stopped. It
exits with 2 when its command line is wrong.

When a simulation fails
-----------------------

A failed simulation doesn't stop the others. The worker records why it
failed and goes on, and the next ``stilt run`` tries it again.
``stilt status`` counts the failures by reason, and :doc:`../checking`
says what each reason means and where to look.
