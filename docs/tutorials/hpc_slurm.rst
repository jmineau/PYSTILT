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

Log in to the cluster, activate the environment where PYSTILT is installed,
and create a project on a filesystem the compute nodes can see:

.. code-block:: bash

   stilt init /path/to/shared/slv_2023

Edit ``config.yaml`` (meteorology and footprint grid, as in
:doc:`../getting_started/quickstart`) and fill in ``receptors.csv``, or
generate receptors in Python and pass them to :class:`stilt.Model` as in
:doc:`wbb_stationary`.

Step 2: Add Slurm settings
--------------------------

Add an ``execution`` section to ``config.yaml``. Use your own account and
partition, and the commands you normally use to activate your environment:

.. code-block:: yaml

   execution:
     backend: slurm
     n_workers: 200            # 200 array tasks, about 45 simulations each
     account: my-account
     partition: my-partition
     time: "04:00:00"
     mem_per_cpu: 2G
     array_parallelism: 50     # at most 50 tasks running at once
     setup:
       - module load miniforge3
       - conda activate my-env

To choose ``time``: time a few simulations locally first (for example by
running a small test project), multiply by the number per task, and add a
safety margin. If tasks run out of time, nothing is lost; the next step
covers resubmitting.

Step 3: Submit
--------------

.. code-block:: bash

   stilt run /path/to/shared/slv_2023

This prints the Slurm job ID and returns. You can log out; the job runs on
its own.

Step 4: Monitor
---------------

.. code-block:: bash

   squeue -u "$USER"                         # Slurm's view of the array
   stilt status /path/to/shared/slv_2023     # finished vs remaining simulations

Task output is in ``slurm/logs/<date_time>/`` inside the project, one
directory per submission. If every task fails
immediately, check there first: the most common cause is ``setup`` not
activating the environment, so ``stilt`` can't be found.

Step 5: Resubmit what's left
----------------------------

When the array finishes, some simulations may not have: tasks that ran out
of time, were preempted, or hit missing meteorology. Run the same command
again:

.. code-block:: bash

   stilt run /path/to/shared/slv_2023

Only the unfinished simulations are submitted. Repeat until
``stilt status`` shows none remaining. For simulations that keep failing,
read their ``stilt.log``.

Next
----

- Every Slurm option: :doc:`../guides/execution/slurm`
- Analyze the results: :doc:`../guides/outputs`
