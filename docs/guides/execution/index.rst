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
   Run every simulation that isn't finished yet. On your computer it returns
   when they are all done. On Slurm it submits a job array and returns right
   away. Add ``--wait`` to wait for the job. ``model.run()`` does the same
   from Python.

``stilt status <project>``
   Count finished and remaining simulations.

Options for ``stilt run``:

- ``--backend`` and ``--n-workers`` override ``config.yaml`` for this run.
- ``--no-skip`` reruns every simulation, finished or not.
- ``--compute-root DIR`` runs HYSPLIT in ``DIR``, a scratch folder that is
  emptied after each successful run.

When a simulation fails
-----------------------

A failed simulation doesn't stop the others. The error goes to the end of
that simulation's log in the output directory, and its HYSPLIT working
folder is kept under ``scratch/`` there. The simulation stays unfinished, so the next
``stilt run`` tries it again. Fix the cause (often missing meteorology) and
run again. Finished simulations are skipped.

In Python, ``sim.outcome`` gives a short failure reason and ``sim.log`` the
full log.

A footprint can be empty because no particle reached the grid. That is not a
failure. PYSTILT records it with the reason, and the simulation counts as
finished (see :doc:`../outputs`).
