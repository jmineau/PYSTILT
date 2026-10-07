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

   stilt submit ./my_project

``stilt submit`` prints the Slurm job ID and returns once the job is
submitted. ``stilt run`` submits it too, then waits until it finishes. Each
task runs with the
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
   many receptors at the same time. A local run uses ``cpus`` the same way.

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
queue and picks up where it stopped. Receptors it already finished are
skipped. Slurm warns each task two minutes before its time limit with the
signal ``USR1``, and stops a preempted task with ``SIGTERM``. Either way
the task stops its runs and requeues itself with ``scontrol requeue``. The
cluster may hold a requeued task in the queue for some minutes before it
starts again. A task requeues itself at most 10 times, so give ``time``
some room. ``scancel`` stops a task without requeuing it.

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

From Python, ``project.run()`` submits the job array, waits until every
task has ended (it asks ``sacct``), and returns the status table of the
simulations it submitted. ``project.submit()`` submits and returns the
Slurm job id at once:

.. code-block:: python

   job_id = project.submit()    # None when nothing needs to run

Follow it with ``squeue -j <job_id>`` and stop it with ``scancel <job_id>``.
``submit()`` raises a ``ValueError`` when the backend is ``local``.

What PYSTILT writes
-------------------

Each submission gets its own folder, so a later one never overwrites an
earlier one's logs:

.. code-block:: text

   my_project/
     _slurm/<date_time>_<id>/
       job.sh            # the script given to sbatch
       receptors.txt     # the receptors submitted, one per line
       execution.yaml    # the execution settings of this submission
       <task>.log        # each task's progress lines and errors

``job.sh`` is a plain ``sbatch`` script. Task ``i`` of ``N`` runs

.. code-block:: bash

   python -m stilt run my_project --receptors .../receptors.txt \
       --task $SLURM_ARRAY_TASK_ID/N --execution .../execution.yaml

with the Python that submitted it. To rerun one task by hand, on a compute
node, run that line with the task's number in place of
``$SLURM_ARRAY_TASK_ID``.

If a task fails, look in its ``<task>.log`` for problems with the task
itself. Its first line says when and on which node it started, and the
job and task ids; a requeued task adds another such line. Look in each
simulation's log in the output directory for transport model problems
(see :doc:`index`). The folders are safe to delete once a job has
left the queue.
