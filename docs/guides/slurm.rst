Slurm
=====

The Slurm backend is the current HPC path for large receptor sets on shared
filesystems.

How it works
------------

``stilt run`` with ``backend: slurm`` does not execute simulations inline.
Instead it:

1. registers pending simulations
2. writes immutable chunk files under ``<output>/chunks/<batch_id>/``
3. renders a submission script under ``<project>/slurm/``
4. submits a Slurm array whose tasks run ``stilt push-worker``

Minimal config
--------------

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

Important knobs
---------------

``n_workers``
   Number of chunk shards to create, and therefore the maximum array-task
   count.

``cpus_per_task``
   Passed through both to Slurm and to ``stilt push-worker --cpus``.

``array_parallelism``
   Limits simultaneously active array tasks via the ``%N`` Slurm syntax.

Any additional keys in ``execution`` are forwarded to ``sbatch`` with
underscores converted to dashes.

Submitting from the CLI
-----------------------

.. code-block:: bash

   stilt run /path/to/project

This is normally fire-and-forget. The CLI prints the submitted job ID and
returns after ``sbatch`` accepts the array.

``--wait`` exists:

.. code-block:: bash

   stilt run /path/to/project --wait

but it is mainly for debugging, small demonstrations, or controller-style
workflows. It is not the common HPC usage pattern for production Slurm runs.

Monitoring and reruns
---------------------

Use both scheduler and durable-index views:

.. code-block:: bash

   squeue -u "$USER"
   stilt status /path/to/project

Rerunning the same command is safe. Completed simulations are skipped by
default. Use ``--no-skip`` only when you want to force a full rerun.

Current constraint
------------------

Slurm currently requires both the project root and the durable output root to
be local or shared filesystem paths. Cloud URIs are rejected for this backend.
