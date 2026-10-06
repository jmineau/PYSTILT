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
also generate receptors in Python, as in :doc:`wbb_stationary`, and add
them with ``stilt.Project("/path/to/shared/slv_2023").add_receptors(receptors)``.

Step 2: Add Slurm settings
--------------------------

Add an ``execution`` section to ``config.yaml``. Use your own account and
partition:

.. code-block:: yaml

   execution:
     backend: slurm
     n_workers: 200            # 200 array tasks, about 45 receptors each
     account: my-account
     partition: my-partition
     time: "04:00:00"
     mem: 2G
     array_parallelism: 50     # at most 50 tasks running at once

The tasks run with the Python you submit from, so activate the environment
PYSTILT is installed in before Step 3. Nothing has to be activated inside
the job.

To choose ``time``, first time a few simulations in a small test project.
Multiply by the number of receptors per task and add a safety margin. If a
task runs out of time or is preempted, nothing is lost: it goes back in the
queue and continues with the receptors it has not finished.

Step 3: Submit
--------------

.. code-block:: bash

   stilt submit /path/to/shared/slv_2023

This prints the Slurm job ID and returns. You can log out while the job
runs. ``stilt run`` submits the same job array and then waits for it to
finish, printing the status at the end.

Step 4: Monitor
---------------

.. code-block:: bash

   squeue -u "$USER"                         # Slurm's view of the array
   stilt status /path/to/shared/slv_2023     # finished vs remaining simulations

Task output is in ``_slurm/<date_time>_<id>/`` inside the project, with one
folder per submission. If every task fails immediately, look in the
``<task>.log`` files there first.

Step 5: Resubmit what's left
----------------------------

Some simulations may be unfinished when the array ends. A task may have run
out of time more than a few times, or the meteorology a receptor needs may
be missing. Once the job has left the queue, run the same command again:

.. code-block:: bash

   stilt submit /path/to/shared/slv_2023

Only the unfinished simulations are submitted. Repeat until
``stilt status`` shows none remaining. If a simulation keeps failing, read
its log (``sim.log`` in Python).

Next
----

- Every Slurm option: :doc:`../guides/execution/slurm`
- Analyze the results: :doc:`../guides/outputs`
