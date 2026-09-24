On An HPC Cluster (Slurm)
=========================

For thousands of simulations, run them as a Slurm job array. PYSTILT splits
the unfinished simulations into groups, writes the job script, and submits
it. You don't write any Slurm scripts yourself.

Your project folder must be on a filesystem that the compute nodes can see
(a shared home, group, or scratch space).

Set it up
---------

Add an ``execution`` section to ``config.yaml``:

.. code-block:: yaml

   execution:
     backend: slurm
     n_workers: 200              # number of array tasks
     account: my-account
     partition: my-partition
     time: "02:00:00"            # time limit per array task
     mem: 4G
     setup:                      # shell commands run at the start of each task
       - module load miniforge3
       - conda activate my-env

``setup`` matters: each task runs the ``stilt`` command, so it must be able to
find the Python environment where PYSTILT is installed.

Then submit:

.. code-block:: bash

   stilt run ./my_project

``stilt run`` prints the Slurm job ID and returns once the job is submitted.
Add ``--wait`` to keep watching until it finishes.

Options
-------

``n_workers`` (required)
   How many array tasks to split the simulations into. With 10,000
   simulations and ``n_workers: 200``, each task runs 50 one after another.
   Set ``time`` long enough for one task's share.

``cpus_per_task``
   CPUs per array task (default 1). With more than one, each task runs that
   many simulations at the same time.

``array_parallelism``
   The most tasks allowed to run at once, to stay within your group's limits.
   ``array_parallelism: 50`` becomes ``--array=0-199%50``.

``setup``
   Shell commands to run at the start of each task, before PYSTILT: loading
   modules, activating an environment, setting environment variables.

Anything else
   Every other key is passed to ``sbatch`` as a flag, with underscores turned
   into dashes: ``mem_per_cpu: 2G`` becomes ``--mem-per-cpu=2G``, and
   ``qos: normal`` becomes ``--qos=normal``.

Watch progress and rerun
------------------------

.. code-block:: bash

   squeue -u "$USER"            # Slurm's view
   stilt status ./my_project    # finished vs remaining simulations

If tasks time out, are preempted, or fail, just run the same command again:

.. code-block:: bash

   stilt run ./my_project

Only unfinished simulations are submitted. Use ``--no-skip`` to force
everything to run again.

What PYSTILT writes
-------------------

Each submission adds files under one ``<date_time>`` key, so a later
submission never overwrites an earlier one's chunks or logs:

.. code-block:: text

   my_project/
     chunks/<date_time>/task_0.txt, task_1.txt, ...   # simulation IDs for each task
     slurm/submit_<date_time>.sh                      # the script given to sbatch
     slurm/logs/<date_time>/0.out, 0.err, ...         # output from each task

Each array task runs ``stilt push-worker`` on its ``task_N.txt`` list. If a
task fails, look in ``slurm/logs/<date_time>/`` for task-level problems (for example, the
environment not activating) and in each simulation's ``stilt.log`` for
HYSPLIT problems.

Limitations
-----------

The project must be a local or shared-filesystem folder. Projects stored in
cloud buckets (``s3://``, ``gs://``) can't use the Slurm backend.
