Tutorial: Scaling Up On A Slurm Cluster
=======================================

A year of hourly receptors at one site is almost 9,000 simulations. That's
too many for a laptop, but a job array on an HPC cluster handles it easily.
This tutorial moves a project to Slurm.

What you'll learn
-----------------

- how to set up a project on a cluster
- how to switch it from local to Slurm execution
- how to monitor it and recover from failures

Step 1: Create the project on the cluster
-----------------------------------------

Log in to the cluster and activate the environment where PYSTILT is
installed. Then create a project on a filesystem the compute nodes can see:

.. code-block:: bash

   stilt init /path/to/shared/slv_2023

Edit ``config.yaml`` to set the meteorology and footprint grid, as in
:doc:`../getting_started/quickstart`. Then fill in ``receptors.csv``. You can
also generate receptors in Python and pass them to :class:`stilt.Model`, as
in :doc:`wbb_stationary`.

Step 2: Add Slurm settings
--------------------------

Add an ``execution`` section to ``config.yaml``. Use your own account and
partition, and the commands you normally use to activate your environment:

.. code-block:: yaml

   execution:
     backend: slurm
     n_workers: 200            # 200 array tasks, about 45 receptors each
     account: my-account
     partition: my-partition
     time: "04:00:00"
     mem_per_cpu: 2G
     array_parallelism: 50     # at most 50 tasks running at once
     setup:
       - module load miniforge3
       - conda activate my-env

To choose ``time``, first time a few simulations in a small test project.
Multiply by the number of receptors per task and add a safety margin. If
tasks run out of time, nothing is lost. Step 5 shows how to resubmit.

Step 3: Submit
--------------

.. code-block:: bash

   stilt run /path/to/shared/slv_2023

This prints the Slurm job ID and returns. You can log out while the job
runs.

Step 4: Monitor
---------------

.. code-block:: bash

   squeue -u "$USER"                         # Slurm's view of the array
   stilt status /path/to/shared/slv_2023     # finished vs remaining simulations

Task output is in ``slurm/logs/<date_time>/`` inside the project, with one
folder per submission. If every task fails immediately, look there first.
The most common cause is a ``setup`` that doesn't activate the environment,
so the tasks can't find ``stilt``.

Step 5: Resubmit what's left
----------------------------

Some simulations may be unfinished when the array ends. Their tasks may have
run out of time or been preempted, or the meteorology they need may be
missing. Once the job has left the queue, run the same command again:

.. code-block:: bash

   stilt run /path/to/shared/slv_2023

Only the unfinished simulations are submitted. Repeat until
``stilt status`` shows none remaining. If a simulation keeps failing, read
its ``stilt.log``.

Next
----

- Every Slurm option: :doc:`../guides/execution/slurm`
- Analyze the results: :doc:`../guides/outputs`
