.. _service:

Service Runtime
===============

PYSTILT now exposes a thin service-facing runtime surface in
``stilt.service``. This is not a second orchestration framework; it is a
named wrapper around the same repository-backed submission, worker, and status
behavior already used by :class:`~stilt.Model` and the CLI.

Use it when you want:

- an explicit queue/service API in Python,
- project-level and batch-level progress summaries,
- claim visibility for long-lived workers,
- a stable place to hang cloud/service docs without creating a new package.


Python service facade
---------------------

The most explicit way to use the service layer is to wrap a fully configured
project and then interact with the queue through :class:`stilt.Service`.

.. code-block:: python

   import stilt
   import pandas as pd

   receptor = stilt.Receptor(
       time=pd.Timestamp("2023-07-15 18:00", tz="UTC"),
       latitude=40.766,
       longitude=-111.848,
       altitude=10,
   )

   service = stilt.Service(
       project="./my_project",
       receptors=[receptor],
       config=stilt.ModelConfig(
           n_hours=-24,
           numpar=100,
           mets={
               "hrrr": stilt.MetConfig(
                   directory="/data/met/hrrr",
                   file_format="hrrr_%Y%m%d.arl",
                   file_tres="1h",
               )
           },
           footprints={
               "default": stilt.FootprintConfig(
                   grid=stilt.Grid(
                       xmin=-113.0,
                       xmax=-110.5,
                       ymin=40.0,
                       ymax=42.0,
                       xres=0.01,
                       yres=0.01,
                   )
               )
           },
       ),
   )

   # Register work without starting workers.
   sim_ids = service.submit(batch_id="daily_20260414")
   print(len(sim_ids))

   pending = service.status()
   print(pending.pending, pending.running, pending.completed)

   # Drain the queue once.
   service.drain(cpus=4)

   # Inspect queue and batch state after the worker finishes.
   status = service.status()
   print(status.pending, status.running, status.completed)

   batch = service.batch_status("daily_20260414")
   print(batch.batch_id, batch.completed, batch.total, batch.percent_complete)

   for batch in service.batches():
       print(batch.batch_id, batch.completed, batch.total)

   for claim in service.active_claims():
       print(claim.sim_id, claim.worker_id)

If the project already exists on disk with ``config.yaml`` and
``receptors.csv``, reopening it is just:

.. code-block:: python

   service = stilt.Service(project="./my_project")
   service.serve(cpus=4)


CLI visibility
--------------

The CLI now exposes the same service-oriented views:

.. code-block:: bash

   # Project + batch counts
   stilt status ./my_project

   # Active queue claims
   stilt claims ./my_project

   # Execution-attempt history
   stilt attempts ./my_project

   # Include expired claim rows if you are debugging lease recovery
   stilt claims ./my_project --include-expired


Cloud/service deployment shape
------------------------------

The intended first-party service story is:

1. use PostgreSQL-backed repository state,
2. store durable outputs in an object store or shared durable filesystem,
3. register work with ``stilt submit`` or :meth:`stilt.Service.submit`,
4. run long-lived workers with ``stilt serve`` or
   :meth:`stilt.Service.serve`,
5. scale workers from repository queue depth.

For Kubernetes, :class:`~stilt.executors.KubernetesExecutor` and
``stilt.service.kubernetes`` both build on the same queue semantics:

- worker pods claim simulations from the repository,
- workers publish durable artifacts,
- queue depth is derived from pending simulations rather than from a stale
  ``trajectory_status IS NULL`` heuristic.

See ``examples/cloud/`` in the repo for a minimal starting point, including
sample Kubernetes worker Deployment and KEDA ScaledObject manifests.

You can also build those manifests directly in Python via
``stilt.service.kubernetes``:

.. code-block:: python

   from stilt.service.kubernetes import (
       scaled_object_manifest,
       service_deployment_manifest,
   )

   deployment = service_deployment_manifest(
       "gs://my-bucket/my-project",
       image="ghcr.io/my-org/pystilt-worker:latest",
       namespace="stilt",
       cpus=1,
   )
   scaler = scaled_object_manifest("stilt-my-project", namespace="stilt")


What this module is not
-----------------------

``stilt.service`` is intentionally thin. It does not yet try to replace a
full staged orchestration layer like ``stiltctl``:

- no scene/domain ingestion,
- no deployment stack management,
- no application-specific Jacobian or inverse-model workflows.

Those remain higher-level concerns above the core runtime substrate.
