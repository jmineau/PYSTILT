Execution
=========

PYSTILT supports three execution backends. All three share the same output
project model — the same config, the same output layout, and the same CLI
commands.  The backend controls only *how work gets dispatched to workers*.

Backends
--------

.. toctree::
   :maxdepth: 1

   local
   slurm
   kubernetes

Dispatch models
---------------

**Push dispatch** (``local``, ``slurm``)
   The coordinator enumerates pending simulation IDs and sends work directly to
   workers — either inline in the current process or by writing chunk files for
   a Slurm array.

**Pull dispatch** (``kubernetes``)
   Workers independently claim pending simulations from a shared output index
   backend.  The coordinator registers work and returns; pods drain the queue
   autonomously.

Choosing a backend
------------------

``local``
   Default. Best for notebooks, workstation runs, and small receptor sets.
   Runs inline with ``n_workers: 1`` or uses a local process pool. No
   infrastructure required.

``slurm``
   Best for large receptor sets on HPC clusters with shared filesystems.
   Writes immutable chunk files and submits a Slurm array job whose tasks each
   call ``stilt push-worker``. Project and output roots must be local or
   shared-filesystem paths.

``kubernetes``
   For cloud-native or container-scale deployments backed by a PostgreSQL index
   and object-store outputs. Requires more infrastructure than the
   other two backends.

   .. note::
      The Kubernetes backend is not yet fully implemented. See
      :doc:`kubernetes` for the current status.

CLI primitives
--------------

These commands surface the executor model regardless of backend:

``stilt run``
   Register pending simulations and launch workers using the configured
   executor.  For ``local``, blocks until done.  For ``slurm``, submits the
   array and returns (fire-and-forget); use ``--wait`` to block.

``stilt register``
   Publish project inputs and register simulations without launching any
   workers.  Useful for separating the planning step from execution.

``stilt push-worker``
   Execute one immutable chunk of simulation IDs without queue polling or
   heartbeats.  Used by Slurm task array elements.

``stilt pull-worker``
   Claim and execute pending simulations from the output index.  Used by
   Kubernetes pods and long-lived local workers.

``stilt serve``
   Like ``pull-worker --follow``: keeps polling indefinitely for new claimable
   work.  Use for always-on queue consumers.

Simulation state and delivery guarantees
-----------------------------------------

These semantics apply across all backends.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Area
     - Current behavior
   * - Delivery guarantee
     - At-least-once processing. A simulation can be retried after interruption
       or failure.
   * - Trajectory status
     - ``pending → running → complete`` or ``failed``.
   * - Footprint status
     - ``complete``, ``complete-empty``, or ``failed`` per footprint name.
   * - Empty footprint
     - Treated as terminal success (``complete-empty``), not failure. No NetCDF
       file is written or expected for empty footprints.
   * - Reruns
     - ``skip_existing=True`` avoids rework for already complete outputs.
       ``skip_existing=False`` forces a full rerun regardless of prior state.
