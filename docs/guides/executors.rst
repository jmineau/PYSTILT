Execution Backends Overview
===========================

PYSTILT currently supports three execution backends through one shared durable
project model:

- ``local``
- ``slurm``
- ``kubernetes``

Execution model
---------------

``local`` and ``slurm`` use **push** dispatch
   The coordinator computes the list of pending simulation IDs and sends work to
   workers directly.

``kubernetes`` uses **pull** dispatch
   Workers claim pending simulations from a claim-capable index backend.

Local executor
--------------

The default executor is local:

.. code-block:: yaml

   execution:
     backend: local
     n_workers: 4

or from the CLI:

.. code-block:: bash

   stilt run ./project --backend local --n-workers 4

Behavior:

- ``n_workers: 1`` runs inline in the current process
- ``n_workers > 1`` uses a local process pool
- the CLI always blocks until local workers finish
- this is the right default for notebooks and workstation batch runs

Slurm executor
--------------

The Slurm backend uses push dispatch. It writes immutable chunk files and
submits an array job whose tasks call ``stilt push-worker``.

See :doc:`slurm` for a dedicated guide. In short:

.. code-block:: yaml

   execution:
      backend: slurm
     n_workers: 48
     cpus_per_task: 2
     array_parallelism: 12
     account: my-account
     partition: notchpeak
     time: "02:00:00"

Important constraints:

- Slurm currently requires both project and output roots to be local paths
- the coordinator writes submission scripts under ``<project>/slurm/``
- chunk files are written under ``<output>/chunks/<batch_id>/``
- ``stilt run`` usually returns immediately after ``sbatch``; ``--wait`` is
  available, but it is not the common HPC usage pattern

Kubernetes executor
-------------------

The Kubernetes executor uses pull dispatch. It creates a Kubernetes Job whose
pods call ``stilt pull-worker`` against a PostgreSQL-backed durable index.

See :doc:`kubernetes` for a dedicated guide. The short version is:

.. code-block:: yaml

   execution:
     backend: kubernetes
     image: ghcr.io/example/pystilt-worker:latest
     namespace: stilt
     n_workers: 8
     db_secret: pystilt-db

This mode assumes a shared-worker deployment model:

- a PostgreSQL-backed index
- ``PYSTILT_DB_URL`` available to workers
- durable outputs reachable from every pod

For Kubernetes, ``output_dir`` can be a cloud URI and ``compute_root`` should
usually be a pod-local scratch directory.

HPC and service commands
------------------------

The CLI surfaces the executor model as a few primitives:

``stilt run``
   Register pending work and launch workers using the configured executor.

``stilt register``
   Publish inputs and register simulations without launching workers.

``stilt push-worker``
   Consume one immutable chunk file. Mostly used by Slurm submissions.

``stilt pull-worker``
   Claim and execute pending simulations from the durable index.

``stilt serve``
   Keep polling indefinitely for new claimable work.

Choosing a backend
------------------

Use ``local`` when you want the simplest path and the data fit on one machine.

Use ``slurm`` when the project and durable outputs are on local or shared HPC
filesystems and array jobs are the natural scheduling unit.

Use ``kubernetes`` when worker pods need to scale independently against a
shared PostgreSQL-backed queue and durable storage.
