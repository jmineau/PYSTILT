On An HPC Cluster (Slurm)
=========================

For thousands of simulations, run them as a Slurm job array. PYSTILT splits
the unfinished receptors into lists, writes the job script, and submits it.
You don't write any Slurm scripts yourself.

Your project folder must be on a filesystem that the compute nodes can see,
such as a shared home, group, or scratch space.

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

Each task runs the ``stilt`` command, so ``setup`` must activate the Python
environment where PYSTILT is installed.

Then submit:

.. code-block:: bash

   stilt run ./my_project

``stilt run`` prints the Slurm job ID and returns once the job is submitted.
Add ``--wait`` to keep watching until it finishes.

Options
-------

``n_workers`` (required)
   How many array tasks to split the receptors into. With 10,000 receptors
   and ``n_workers: 200``, each task runs 50 receptors one after another.
   Each receptor runs once per variant, so set ``time`` long enough for one
   task's share. If fewer receptors are left than ``n_workers``, PYSTILT
   submits one task per receptor.

``cpus_per_task``
   CPUs per array task (default 1). With more than one, each task runs that
   many receptors at the same time.

``array_parallelism``
   The most tasks allowed to run at once, to stay within your group's limits.
   ``array_parallelism: 50`` becomes ``--array=0-199%50``.

``setup``
   Shell commands to run at the start of each task, before PYSTILT. Use it to
   load modules, activate an environment, or set environment variables.

Any other key
   Passed to ``sbatch`` as a flag, with underscores turned into dashes.
   ``mem_per_cpu: 2G`` becomes ``--mem-per-cpu=2G``, and ``qos: normal``
   becomes ``--qos=normal``. A key set to ``true``, such as
   ``exclusive: true``, becomes a bare flag (``--exclusive``). If you don't
   set ``job_name``, it is ``pystilt-`` followed by the project folder name.

Watch progress and rerun
------------------------

.. code-block:: bash

   squeue -u "$USER"            # Slurm's view
   stilt status ./my_project    # finished vs remaining simulations

If tasks time out, are preempted, or fail, run the same command again once
the job has left the queue:

.. code-block:: bash

   stilt run ./my_project

Only unfinished receptors are submitted. Don't resubmit while the first job
is still running. The receptors it hasn't finished yet would be submitted a
second time. Use ``--no-skip`` to force everything to run again.

What PYSTILT writes
-------------------

Each submission gets its own ``<date_time>`` stamp, so a later submission
never overwrites an earlier one's lists or logs:

.. code-block:: text

   my_project/
     chunks/<date_time>/task_0.txt, task_1.txt, ...   # receptor ids for each task
     slurm/submit_<date_time>.sh                      # the script given to sbatch
     slurm/logs/<date_time>/0.out, 0.err, ...         # output from each task

Each array task runs ``stilt push-worker`` on its ``task_N.txt`` list. With
``--wait``, PYSTILT deletes ``chunks/<date_time>/`` once the job leaves the
queue.

If a task fails, look in ``slurm/logs/<date_time>/`` for problems with the
task itself, such as the environment not activating. Look in each
simulation's ``stilt.log`` for HYSPLIT problems.

Limitations
-----------

The project directory and the output directory must be on a filesystem
every array task can reach, such as the cluster's shared storage.
