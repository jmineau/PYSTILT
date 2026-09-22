In The Cloud (Kubernetes)
=========================

.. warning::

   The Kubernetes backend is **experimental and not yet fully tested**. The
   pieces described here exist in the code but haven't been validated end to
   end. Use :doc:`local` or :doc:`slurm` for real work. Contributions are
   welcome.

This page is for people running PYSTILT on cloud infrastructure. Unlike the
local and Slurm backends, cloud workers don't get a fixed list of
simulations. Each one repeatedly takes the next unfinished simulation from a
shared queue in a PostgreSQL database until the queue is empty.

PYSTILT supports two Kubernetes-oriented patterns:

- **Batch Jobs** via :class:`stilt.execution.KubernetesExecutor` — creates a
  bounded Kubernetes Job to drain currently pending work.
- **Long-lived worker Deployments** via ``stilt serve`` and
  ``stilt.service.kubernetes`` manifest helpers — for always-on queue
  consumers that poll for new work indefinitely.

Both patterns depend on a shared PostgreSQL-backed work queue.

Run a batch of workers
----------------------

The executor path creates a Kubernetes Job whose pods run:

.. code-block:: text

   stilt pull-worker <project>

Minimal config:

.. code-block:: yaml

   execution:
     backend: kubernetes
     image: ghcr.io/example/pystilt-worker:latest
     namespace: stilt
     n_workers: 8
     db_secret: pystilt-db

Use this when you want a single bounded drain of currently pending
simulations: all pods start together, process until the queue is empty, and
exit.

Always-on workers
-----------------

For always-on execution, ``stilt.service.kubernetes`` exposes helper functions
for rendering:

- worker Job manifests
- follow-mode worker Deployments
- ``stilt serve`` Deployments
- KEDA ``ScaledObject`` manifests for autoscaling

This path is separate from ``KubernetesExecutor``: it targets deployments
where pods keep polling for new work as it arrives, rather than draining a
fixed batch.

What you need
-------------

Both patterns currently assume:

- ``PYSTILT_DB_URL`` is available to workers (typically via a Kubernetes
  Secret).
- The project is reachable from every pod.  In practice the project root
  is a cloud URI such as ``s3://`` or ``gs://``.
- Workers have a writable ``compute_root`` for staging meteorology and HYSPLIT
  files.  A pod-local path such as ``/tmp/pystilt`` works well here.
