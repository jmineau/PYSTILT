Migrating From stiltctl
=======================

The clearest architectural change is that queue-backed execution is now part of
PYSTILT itself rather than a separate control-plane package.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - stiltctl concept
     - stiltctl pattern
     - PYSTILT equivalent
   * - Work submission
     - service-oriented submit API
     - ``Model.register_pending()`` or ``stilt register``
   * - Batch worker
     - queue worker job
     - ``stilt pull-worker`` or ``stilt push-worker``
   * - Long-lived worker
     - service deployment
     - ``stilt serve``
   * - Kubernetes manifests
     - Helm / KEDA / helper tooling
     - ``stilt.service.kubernetes`` helper functions
   * - Output registry
     - PostgreSQL queue tables
     - PostgreSQL-backed output simulation index via ``PYSTILT_DB_URL``

Why it matters
--------------

- one package now owns the science-facing model and the worker runtime
- local, HPC, and cloud execution all share one output project model
- fewer cross-package compatibility problems

What to re-check
----------------

- database connectivity and secrets
- whether your deployment is push-style or pull-style
- whether project, output, and compute roots are distinct in your environment
- any Kubernetes YAML that assumed older CLI flags or resource names

The service/runtime layer is still one of the least-settled parts of the alpha,
so treat deployment conventions as moving parts rather than frozen interfaces.
