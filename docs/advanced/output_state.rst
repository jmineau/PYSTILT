Output State And Shared Workers
===============================

PYSTILT keeps no database of what has run. What exists is decided by the
outputs themselves, so local processes, Slurm tasks, Kubernetes pods, and
notebook code all agree without coordination.

The project is the registry
---------------------------

A project is one root holding ``config.yaml`` and ``receptors.csv``. The
simulations it defines are those receptors crossed with the configured met
streams; ``model.simulations`` enumerates exactly that set. ``Model.register()``
writes both files into the project store (local directory or ``s3://`` /
``gs://`` URI) so any worker can rebuild the model from the root alone, and
merges new receptor batches into ``receptors.csv``.

Completion is by key
--------------------

Every output has one address: a store key such as
``simulations/by-id/<sim_id>/<sim_id>_traj.parquet``. A simulation is
*complete* when every output it must produce exists under its keys
(:meth:`stilt.Simulation.is_complete`): the trajectory, the error trajectory
when wind-error params are set, and one netCDF or ``.empty`` marker per
configured footprint. Nothing is listed; each check is one existence probe.

Compute versus store
--------------------

Workers run HYSPLIT under ``compute_root`` (a local scratch parent) and call
:meth:`stilt.Simulation.publish` to copy outputs into the store. For a local
project the default compute root *is* the store's ``simulations/by-id``
directory, so publishing is a no-op. For a cloud project it uploads.

The work queue
--------------

Claim-based workers (``stilt pull-worker``, ``stilt serve``, Kubernetes) need a
backend that can atomically lock one pending simulation at a time. That is a
lean PostgreSQL work queue (:class:`stilt.service.PostgresQueue`: enqueue →
claim ``FOR UPDATE SKIP LOCKED`` → done/failed), present only when
``PYSTILT_DB_URL`` is set. It tracks *work status* only; completion is still by
key. Local and Slurm workflows never touch it.

Runtime environment
-------------------

Runtime-only settings are separate from ``config.yaml``:

- ``PYSTILT_DB_URL`` — the work queue
- ``PYSTILT_CACHE_DIR`` — local cache for downloads from a remote store
- ``PYSTILT_COMPUTE_ROOT`` — worker scratch parent

Nothing here changes a simulation's result, only where and how it runs.
