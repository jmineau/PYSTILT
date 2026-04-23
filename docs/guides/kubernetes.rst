Kubernetes
==========

PYSTILT currently supports two Kubernetes-oriented patterns:

- **batch Jobs** through :class:`stilt.execution.KubernetesExecutor`
- **long-lived worker Deployments** through ``stilt serve`` and
  ``stilt.service.kubernetes`` helpers

Both depend on a shared PostgreSQL-backed durable index.

Batch-mode KubernetesExecutor
-----------------------------

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

This mode is best when you want one bounded batch drain of currently pending
work.

Service-style workers
---------------------

For always-on execution, the current package also exposes helper functions in
``stilt.service.kubernetes`` for rendering:

- worker Job manifests
- follow-mode worker Deployments
- ``stilt serve`` Deployments
- KEDA ``ScaledObject`` manifests

That path is conceptually separate from ``KubernetesExecutor``: it is for a
service-style deployment where pods keep polling for new work.

Required runtime pieces
-----------------------

Kubernetes-oriented deployments currently assume:

- ``PYSTILT_DB_URL`` is available to workers, usually from a Secret
- durable outputs are reachable from every pod
- workers have a writable ``compute_root`` for staging meteorology and HYSPLIT files

In practice, ``output_dir`` is often a cloud URI and ``compute_root`` is a pod
local path such as ``/tmp/pystilt``.

Alpha caveat
------------

The Kubernetes and service runtime is still one of the least-settled parts of
the package. The helper functions are the most trustworthy source of truth in
the current codebase; deployment conventions around them may still evolve.
