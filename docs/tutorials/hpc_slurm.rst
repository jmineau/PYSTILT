Tutorial: HPC / Slurm Execution
===============================

This tutorial shows the standard Slurm pathway for large receptor sets on a
shared filesystem.

What you'll learn
-----------------

- how to scaffold a project
- how to switch from local execution to ``backend: slurm``
- how ``stilt run`` maps onto chunk files and array tasks
- how to monitor and rerun a project safely

Scaffold a project
------------------

.. code-block:: bash

   stilt init /path/to/slv_project

Then edit ``config.yaml`` and populate ``receptors.csv``.

Add Slurm execution settings
----------------------------

.. code-block:: yaml

   execution:
     backend: slurm
     n_workers: 200
     partition: notchpeak
     account: my_account
     time: "00:20:00"
     mem_per_cpu: 2G
     cpus_per_task: 1
     array_parallelism: 50

Submit
------

.. code-block:: bash

   stilt run /path/to/slv_project

This normally returns as soon as ``sbatch`` accepts the array job.

``--wait`` is available:

.. code-block:: bash

   stilt run /path/to/slv_project --wait

but it is mainly a convenience for debugging or small demonstrations, not the
usual HPC pattern.

What happens under the hood
---------------------------

Each array task consumes one immutable chunk file through:

.. code-block:: bash

   stilt push-worker /path/to/slv_project --chunk /path/to/task.txt --cpus 1

That means:

- no worker-side queue polling is required for Slurm
- each array task has a fixed work assignment
- reruns are driven by output status and ``skip_existing``

Monitor and rerun
-----------------

.. code-block:: bash

   squeue -u "$USER"
   stilt status /path/to/slv_project

To resume after interruption, simply run the same command again. Completed work
is skipped by default.
