Slurm
=====

The ``slurm`` backend is the HPC path for large receptor sets on shared
filesystems.  It uses push dispatch: the coordinator writes immutable chunk
files and submits a Slurm array job whose tasks each call ``stilt
push-worker``.

How it works
------------

Running ``stilt run`` with ``backend: slurm``:

1. Registers pending simulations in the output index.
2. Writes immutable chunk files under ``<output>/chunks/<batch_id>/``.
3. Renders a submission script under ``<project>/slurm/``.
4. Submits a Slurm array job — one task per chunk — via ``sbatch``.

Workers run ``stilt push-worker`` independently; no inter-task communication
is required after submission.

Configuration
-------------

Minimal config:

.. code-block:: yaml

   execution:
     backend: slurm
     n_workers: 200
     partition: mypartition
     account: myaccount
     time: "00:20:00"

Full example with common knobs:

.. code-block:: yaml

   execution:
     backend: slurm
     n_workers: 200
     partition: mypartition
     account: myaccount
     time: "00:20:00"
     mem: 2G
     cpus-per-task: 2
     array_parallelism: 50

Key options
-----------

``n_workers``
   Number of chunk shards to create, and therefore the maximum array-task
   count.  Each worker processes its chunk sequentially; tune this alongside
   ``array_parallelism`` to control cluster load.

``cpus-per-task``
   Passed through both to Slurm (``#SBATCH --cpus-per-task``) and to
   ``stilt push-worker --cpus`` so that each task uses a matching local process
   pool for within-chunk parallelism.

``array_parallelism``
   Limits simultaneously active array tasks via the ``%N`` Slurm syntax
   (e.g. ``--array=0-199%50``).  Useful for staying within fair-share limits.

Any additional keys in the ``execution`` block are forwarded to ``sbatch`` as
``--key=value`` flags, with underscores converted to dashes.

Submitting from the CLI
-----------------------

Fire-and-forget (common for production runs):

.. code-block:: bash

   stilt run /path/to/project

The CLI prints the submitted job ID and returns after ``sbatch`` accepts the
array.

Block until the array finishes (useful for debugging or scripted workflows):

.. code-block:: bash

   stilt run /path/to/project --wait

Monitoring and reruns
---------------------

Use the Slurm scheduler and the output index together:

.. code-block:: bash

   squeue -u "$USER"
   stilt status /path/to/project

Rerunning the same ``stilt run`` is safe — completed simulations are skipped
by default.  Use ``--no-skip`` only when you want to force a full rerun.

Current constraint
------------------

The Slurm backend currently requires both the project root and the output
output root to be local or shared filesystem paths.  Cloud URIs
(``s3://``, ``gs://``, etc.) are not supported for this backend.
