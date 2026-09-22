Where To Run
============

The same project can run on your computer, on an HPC cluster, or (experimentally)
in the cloud. Only the ``execution`` section of ``config.yaml`` changes; your
receptors, meteorology, footprints, and outputs stay the same.

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
     - You're running in the cloud. **Experimental.**
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
   Run every simulation that isn't finished yet. On your computer it waits
   until they are done; on Slurm it submits the jobs and returns (add
   ``--wait`` to wait). ``model.run()`` does the same from Python.

``stilt status <project>``
   Count finished and remaining simulations.

``stilt run`` also accepts ``--backend`` and ``--n-workers`` to override
``config.yaml`` for one run, and ``--no-skip`` to rerun everything.

Commands PYSTILT runs for you
-----------------------------

You won't usually type these; they are what ``stilt run`` launches on each
Slurm task or cloud worker. They're listed so you recognize them in job
scripts and logs.

``stilt push-worker``
   Runs one fixed list of simulations. Each Slurm array task runs one.

``stilt pull-worker`` / ``stilt serve``
   Take simulations one at a time from a shared queue (PostgreSQL) until it is
   empty, or, for ``serve``, indefinitely. Used by cloud workers.

``stilt register``
   Saves the project's settings and receptors without running anything.

When a simulation fails
-----------------------

A failed simulation doesn't stop the others. Its error is in the
simulation's ``stilt.log``, and it stays unfinished, so the next
``stilt run`` tries it again. Fix the cause (often missing meteorology) and
run again; finished simulations are skipped. In Python,
``sim.status`` gives a short failure reason and ``sim.log`` the full log.

A footprint that is empty because no particle reached the grid is not a
failure. It is recorded with a ``.empty`` file and counts as finished (see
:doc:`../outputs`).
