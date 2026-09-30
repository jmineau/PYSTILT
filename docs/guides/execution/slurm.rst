On An HPC Cluster (Slurm)
=========================

For thousands of simulations, run them as a Slurm job array. PYSTILT splits
the unfinished receptors among the array tasks and submits the job. You
don't write any Slurm scripts yourself.

Your project folder, its output directory, and the Python environment must
be on a filesystem the compute nodes can see, such as a shared home, group,
or scratch space.

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

Then submit from the environment PYSTILT is installed in:

.. code-block:: bash

   stilt run ./my_project

``stilt run`` prints the Slurm job ID and returns once the job is submitted.
Add ``--wait`` to keep watching until it finishes. Each task runs with the
same Python that submitted it, so there is nothing to activate inside the
job.

Options
-------

``n_workers``
   How many array tasks to split the receptors into. With 10,000 receptors
   and ``n_workers: 200``, each task runs 50 receptors one after another.
   Each receptor runs once per variant, so set ``time`` long enough for one
   task's share. If fewer receptors are left than ``n_workers``, PYSTILT
   submits one task per receptor.

``cpus``
   CPUs per array task (default 1). With more than one, each task runs that
   many receptors at the same time. ``cpus_per_task`` is accepted too.

``time``, ``mem``, ``partition``, ``account``, ``qos``
   The ``sbatch`` options of the same names.

``array_parallelism``
   The most tasks allowed to run at once, to stay within your group's
   limits. ``array_parallelism: 50`` becomes ``--array=0-199%50``. It is 256
   when unset.

``setup``
   Shell commands to run at the start of each task, before PYSTILT. Use it
   to load modules or set environment variables.

``slurm``
   Any other ``sbatch`` option, by name:

   .. code-block:: yaml

      execution:
        backend: slurm
        n_workers: 200
        slurm:
          exclude: node17,node42
          constraint: skl
          exclusive: true      # a bare flag, --exclusive

A setting PYSTILT does not know is an error, so a misspelled ``partition``
is caught before anything is submitted.

Preempted and timed-out tasks
-----------------------------

A task that is preempted, or reaches its time limit, is put back in the
queue and picks up where it stopped: receptors it already finished are
skipped. The cluster may hold a requeued task in the queue for some minutes
before it starts again. A task that keeps running out of time is given up
on after a few tries, so give ``time`` some room.

Watch progress and rerun
------------------------

.. code-block:: bash

   squeue -u "$USER"            # Slurm's view
   stilt status ./my_project    # finished vs remaining simulations

If receptors fail, fix the cause and run the same command again once the
job has left the queue:

.. code-block:: bash

   stilt run ./my_project

Only unfinished receptors are submitted. Don't resubmit while the first job
is still running. The receptors it hasn't finished yet would be submitted a
second time. Use ``--no-skip`` to force everything to run again.

From Python, ``model.run(wait=False)`` returns a handle. ``handle.wait()``
blocks until the job is done and raises if a task did not complete, and
``handle.jobs`` are the `submitit <https://github.com/facebookincubator/submitit>`_
jobs, one per task.

What PYSTILT writes
-------------------

Each submission gets its own folder, so a later one never overwrites an
earlier one's logs:

.. code-block:: text

   my_project/
     slurm/<date_time>_<id>/
       <job>_submission.sh          # the script given to sbatch
       <job>_<task>_0_log.out       # output from each task
       <job>_<task>_0_log.err       # progress lines and errors
       <job>_<task>_submitted.pkl   # the task's receptors, read by the task
       <job>_<task>_0_result.pkl    # what it returned

If a task fails, look in its ``_log.err`` for problems with the task
itself. Look in each simulation's log in the output directory for HYSPLIT
problems (see :doc:`index`). The folders are safe to delete once a job has
left the queue.
