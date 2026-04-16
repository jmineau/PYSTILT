.. _tutorial_slurm:

Tutorial: HPC / Slurm Execution
================================

For large receptor sets you will usually want to drain the queue on a Slurm
cluster instead of running locally. This tutorial uses the University of
Utah's `CHPC <https://www.chpc.utah.edu/>`_ cluster as a reference, but the
same pattern applies on any Slurm system.

**What you'll learn:**

* how to scaffold a project directory and write a starting ``config.yaml``
* how to switch ``execution.backend`` from ``local`` to ``slurm``
* how ``stilt run`` maps to Slurm array workers
* how to monitor and resume runs

**Prerequisites:**

* PYSTILT installed in your environment
* a Slurm-managed cluster with shared filesystem access to the project
  directory and ARL meteorology files


Step 1: Create a Project
------------------------

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         import shutil
         import stilt

         PROJECT = "/path/to/slv_project"

         config = stilt.ModelConfig(
             n_hours=-24,
             numpar=100,
             mets={
                 "hrrr": stilt.MetConfig(
                     directory="/data/met/hrrr",
                     file_format="hrrr_%Y%m%d.arl",
                     file_tres="1h",
                 ),
             },
             footprints={
                 "slv": stilt.FootprintConfig(
                     grid=stilt.Grid(
                         xmin=-113.5,
                         xmax=-111.0,
                         ymin=40.0,
                         ymax=41.5,
                         xres=0.005,
                         yres=0.005,
                     ),
                 ),
             },
         )

         model = stilt.Model(project=PROJECT, config=config)
         model.config.to_yaml(model.directory / "config.yaml")
         shutil.copy("receptors.csv", model.directory / "receptors.csv")

   .. tab-item:: CLI

      The quickest way to scaffold the project is:

      .. code-block:: bash

         stilt init /path/to/slv_project

      Then edit ``config.yaml`` and add receptor rows to ``receptors.csv``.


Step 2: Configure Slurm Execution
---------------------------------

Open ``slv_project/config.yaml`` and add an ``execution`` block like this:

.. code-block:: yaml

   n_hours: -24
   numpar: 100

   mets:
     hrrr:
       directory: /data/met/hrrr
       file_format: hrrr_%Y%m%d.arl
       file_tres: 1h

   footprints:
     slv:
       grid:
         xmin: -113.5
         xmax: -111.0
         ymin: 40.0
         ymax: 41.5
         xres: 0.005
         yres: 0.005

   execution:
     backend: slurm
     n_workers: 200
     n_tasks: 200
     partition: notchpeak
     account: my_account
     time: "00:20:00"
     mem_per_cpu: 2G
     cpus_per_task: 1
     array_parallelism: 50

The important fields are:

* ``n_workers``: total worker capacity requested by ``stilt run``
* ``n_tasks``: maximum Slurm array size submitted for one run
* ``cpus_per_task``: number of simulations each worker claims at once
* ``array_parallelism``: optional cap on simultaneously active array tasks

All other keys in ``execution`` are passed through to ``sbatch`` with
underscores converted to dashes. For example:

* ``partition`` -> ``#SBATCH --partition=...``
* ``mem_per_cpu`` -> ``#SBATCH --mem-per-cpu=...``
* ``job_name`` -> ``#SBATCH --job-name=...``

If ``job_name`` is omitted, PYSTILT uses ``pystilt-{project}`` automatically,
where ``{project}`` is a slugified project name.

.. tip::

   STILT simulations are typically single-threaded and modest in memory.
   Start with ``cpus_per_task: 1`` and conservative array parallelism, then
   increase parallelism after you see how quickly the queue drains on your
   cluster.


Step 3: Submit the Run
----------------------

Submit from a login node or from your own lightweight controller job:

.. code-block:: bash

   stilt run /path/to/slv_project

By default this returns as soon as ``sbatch`` accepts the array job. To block
until the array is gone from the scheduler:

.. code-block:: bash

   stilt run /path/to/slv_project --wait

What happens under the hood
---------------------------

For the Slurm backend, ``stilt run`` does not execute the simulations inline.
Instead it writes a small submission script and calls ``sbatch``. Each Slurm
array task runs:

.. code-block:: bash

   stilt worker /path/to/slv_project --cpus <cpus_per_task>

Each worker task claims pending simulations from the repository, runs them,
marks terminal state back into the repository, and exits when the queue is
drained.


Monitoring Progress
-------------------

.. code-block:: bash

   squeue -u "$USER"
   stilt status /path/to/slv_project
   stilt claims /path/to/slv_project
   stilt attempts /path/to/slv_project

``stilt status`` reports queue state from the repository, which is often more
useful than raw Slurm counts once some work has already completed.


Resuming Interrupted Runs
-------------------------

PYSTILT records terminal state in the repository. If the scheduler preempts
the array or a node fails, you can rerun the same command:

.. code-block:: bash

   stilt run /path/to/slv_project

Completed simulations are skipped by default. To force a full rerun:

.. code-block:: bash

   stilt run /path/to/slv_project --no-skip


CHPC Notes
----------

On CHPC, start with conservative settings on shared partitions such as
``notchpeak`` or ``ember``. ``array_parallelism`` is a good first lever for
avoiding over-claiming shared resources while still keeping the STILT queue
moving.


.. seealso::

   :doc:`../user_guide/execution` for the backend matrix and configuration
   reference.
