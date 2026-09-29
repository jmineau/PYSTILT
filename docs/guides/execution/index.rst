Where To Run
============

The same project can run on your computer, on an HPC cluster, or in the
cloud (experimental). Only the ``execution`` section of ``config.yaml``
changes. Your receptors, meteorology, footprints, and outputs stay the same.

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
   * - :doc:`kubernetes`
     - You're running in the cloud. Experimental.
     - A container image, a PostgreSQL database, and a cloud bucket

.. toctree::
   :maxdepth: 1
   :hidden:

   local
   slurm
   kubernetes

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
- ``--compute-root DIR`` runs HYSPLIT in ``DIR`` and copies the outputs into
  the project afterward.

Commands PYSTILT runs for you
-----------------------------

You won't usually type these. ``stilt run`` launches them on each Slurm task
or cloud worker, and they are listed here so you recognize them in job
scripts and logs.

``stilt push-worker``
   Runs every variant of each receptor in one list. Each Slurm array task
   runs one list.

``stilt pull-worker``
   Takes receptors one at a time from a shared PostgreSQL queue until the
   queue is empty. With ``--follow`` it keeps waiting for new work.

``stilt serve``
   The same as ``stilt pull-worker --follow``.

``stilt register``
   Saves the project's settings and receptors without running anything. If
   a queue is set up, it also adds the receptors to the queue.

``stilt rm --variant NAME``
   Deletes a variant's outputs so it runs again from scratch. Repeat
   ``--variant`` to delete several (:doc:`../configuration`).

When a simulation fails
-----------------------

A failed simulation doesn't stop the others. The error goes to the end of
that simulation's ``stilt.log``. The simulation stays unfinished, so the next
``stilt run`` tries it again. Fix the cause (often missing meteorology) and
run again. Finished simulations are skipped.

In Python, ``sim.outcome`` gives a short failure reason and ``sim.log`` the
full log.

A footprint can be empty because no particle reached the grid. That is not a
failure. PYSTILT writes a ``.empty`` file in its place, and the simulation
counts as finished (see :doc:`../outputs`).
