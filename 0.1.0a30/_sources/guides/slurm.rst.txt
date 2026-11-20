On An HPC Cluster (Slurm)
=========================

For thousands of simulations, run them as a Slurm job array. PYSTILT splits
the unfinished releases among the array tasks and submits the job. You
don't write any Slurm scripts yourself.

Your project folder, its output directory, and the Python environment must
be on a filesystem the compute nodes can see, such as a shared home, group,
or scratch space.

Set it up
---------

Add an ``execution`` section to ``config.yaml``. ``backend`` says where the
project runs, and every setting of that place goes in the block of its
name, here ``slurm``:

.. code-block:: yaml

   execution:
     backend: slurm
     slurm:
       n_workers: 200              # number of array tasks
       account: my-account
       partition: my-partition
       time: "02:00:00"            # time limit per array task
       mem: 4G

Only the block ``backend`` names is read. A config can also hold a
``local`` block (:doc:`running`), and ``stilt run --backend local`` then
runs the project on this machine with it.

Then check the project and submit it from the environment PYSTILT is
installed in. ``stilt check`` finds a met folder that is not where you said,
or met files a release needs that are not there, before the job array
starts and its simulations fail one by one (:doc:`checking`):

.. code-block:: bash

   stilt check ./my_project
   stilt submit ./my_project

``stilt submit`` prints the Slurm job ID and returns once the job is
submitted. ``stilt run`` submits it too, then waits until it finishes.
Each task runs with the same Python that submitted it, so there is nothing
to activate inside the job.

Options
-------

The keys of the ``slurm`` block:

``n_workers``
   How many array tasks to split the releases into. With 10,000 releases
   and ``n_workers: 200``, each task runs 50 releases one after another.
   Each release runs once per variant, so set ``time`` long enough for one
   task's share. If fewer releases are left than ``n_workers``, PYSTILT
   submits one task per release. We size the tasks to run an hour or two,
   because each task first loads Python and the project, and a shorter
   task spends more of its time on that.

``cpus``
   CPUs per array task (default 1). With more than one, each task runs that
   many releases at the same time. A local run has its own ``cpus``, in
   the ``local`` block.
   Each worker is one process that uses one CPU, so the job script sets
   ``OMP_NUM_THREADS=1`` for it. Without that, NumPy's BLAS starts a thread
   pool the size of the node in every worker (64 threads on a 104-core node)
   and the threads sit idle. A task with more CPUs than workers
   (``cpus_per_task: 8`` with ``cpus: 2``) gives each worker its
   share, 4 here. To choose another value, export ``OMP_NUM_THREADS`` in
   ``setup``.

``time``
   The time limit of each array task, written to ``--time`` as you give
   it. It is required: without it a task gets the partition's default
   limit, which can be days, and holds its node that long if it hangs. To
   choose it, time a few simulations in a small test project, multiply by
   the releases per task and the variants, and add a margin. A task that
   runs out of time loses nothing (below).

``mem``, ``partition``, ``account``, ``qos``
   The ``sbatch`` options of the same names.

``array_parallelism``
   The most tasks allowed to run at once, to stay within your group's
   limits. ``array_parallelism: 50`` becomes ``--array=0-199%50``. Unset,
   Slurm's own limits apply.

``setup``
   Shell commands to run at the start of each task, before PYSTILT. Use it
   to load modules or set environment variables. They run after the script
   exports ``OMP_NUM_THREADS`` (see ``cpus``), so they can change it.

Any other key
   Any other ``sbatch`` option, by its name:

   .. code-block:: yaml

      execution:
        backend: slurm
        slurm:
          n_workers: 200
          time: "02:00:00"
          exclude: node17,node42
          constraint: skl
          exclusive: true      # a bare flag, --exclusive

   ``clusters: kingspeak`` sends the job to another cluster of a site that
   has several. PYSTILT follows the job there. Give ``squeue`` and
   ``scancel`` the cluster too: ``squeue -M kingspeak -j <job_id>``.

``fetch``
   The Slurm resources of the job that gets the met files before the
   array, or ``false`` for no such job (below).

A key that reads as a misspelled setting is an error, so ``partion`` is
caught before anything is submitted and is not sent to ``sbatch`` as an
option. A Slurm setting written beside ``backend``, outside the ``slurm``
block, is an error that names the block.

.. _slurm-fetch:

Met files that download
-----------------------

When a met has ``download`` set, ``stilt submit`` first submits a job that
gets the met files, and the array waits for it. Otherwise every array task
would download the files its own releases need, and tasks whose releases
share a day would download the same files at the same time. The fetch job
is one task that runs ``stilt fetch`` over the releases the array runs, so
each file is downloaded once. ``stilt submit`` prints both job IDs:

.. code-block:: text

   Submitted fetch job: 7712; the job array waits for it to end. Log in /path/my_project/_slurm/20261115_103000_ab12cd/fetch.log
   Submitted job: 7713 (200 tasks); logs in /path/my_project/_slurm/20261115_103000_ab12cd
   Submitted job: 7713

The fetch job uses the array's ``partition``, ``account``, ``qos``, ``time``
and ``mem`` unless ``fetch`` gives its own. Of the other ``sbatch`` options
it takes only ``clusters``, since a data-transfer partition may not have the nodes or
features (``exclude``, ``constraint``) the array asks for. If your cluster has a data-transfer partition, we
recommend sending the fetch job there, because compute nodes often
download slowly. Give it a ``time`` long enough for every download.

.. code-block:: yaml

   execution:
     backend: slurm
     slurm:
       n_workers: 200
       account: my-account
       partition: my-partition
       time: "02:00:00"
       fetch:
         partition: my-dtn-partition
         account: my-dtn-account
         time: "12:00:00"

The array starts when the fetch job ends, however it ended
(``--dependency=afterany``). If the archive lacked some files, the runs
that need them fail with the met failure they would get anyway, and the
other runs go ahead. If the fetch job was stopped, ran out of time, or could
not reach the archive, it keeps the files it got and each task downloads
what is still missing. If ``sbatch`` refuses the array, PYSTILT cancels the
fetch job. The fetch job's ID is also in ``fetch.id`` in the submission
folder; to stop both jobs, ``scancel`` both.

A met without ``download`` gets no fetch job: its files are already in its
``directory``, and ``stilt check`` finds any that are missing. Neither does
a submission whose particles all exist, since making footprints needs no
met, unless it runs with ``--no-skip``, which runs the transport model
again: then the fetch job gets the met of every run. ``fetch: false`` turns the fetch job off, for example when something
else fills the met directory.

Preempted and timed-out tasks
-----------------------------

A task that is preempted, or reaches its time limit, is put back in the
queue and picks up where it stopped. Releases it already finished are
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

If releases fail, fix the cause and run the same command again once the
job has left the queue:

.. code-block:: bash

   stilt run ./my_project

Only unfinished releases are submitted. Repeat until ``stilt status``
shows none remaining. Don't resubmit while the first job is still running:
the releases it hasn't finished yet would be submitted a second time. If
every task fails at once, look in the tasks' logs (below) first. Use
``--no-skip`` to force everything to run again.

From Python, ``project.run()`` submits the job array, waits until every
task has ended (it asks ``sacct``), and returns the status table of the
simulations it submitted. When some look unfinished then, it lists the
output again a minute later, since a network filesystem can show the
submitting node new files late. ``project.submit()`` submits and returns the
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
       releases.txt     # the releases submitted, one per line
       releases.parquet # their rows of releases.csv, already checked
       execution.yaml    # the execution settings of this submission
       <task>.log        # each task's progress lines and errors
       fetch.sh          # the fetch job's script, when a met downloads
       fetch.log         # the fetch job's log: the files it got
       fetch.id          # the fetch job's Slurm ID

``job.sh`` is a plain ``sbatch`` script. Task ``i`` of ``N`` runs

.. code-block:: bash

   python -m stilt run my_project --releases .../releases.parquet \
       --task $SLURM_ARRAY_TASK_ID/N --execution .../execution.yaml \
       --output /path/to/output

with the Python that submitted it. A task reads its releases from
``releases.parquet``, so it does not read and check all of
``releases.csv`` again. To rerun one task by hand, run that line on a
compute node with the task's number in place of ``$SLURM_ARRAY_TASK_ID``.

If a task fails, look in its ``<task>.log`` for problems with the task
itself. Its first line says when and on which node it started, and the
job and task ids; a requeued task adds another such line. Look in each
simulation's log in the output directory for transport model problems
(:doc:`checking`). The folders are safe to delete once a job has left the
queue.
