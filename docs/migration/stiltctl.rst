Coming From stiltctl
====================

For stiltctl users running STILT on cloud infrastructure. In PYSTILT the
work queue and workers are part of the package itself, not a separate
service. That part of PYSTILT is still experimental
(:doc:`../guides/execution/kubernetes`).

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - stiltctl concept
     - stiltctl pattern
     - PYSTILT equivalent
   * - Work submission
     - service-oriented submit API
     - ``Model.register()`` or ``stilt register``
   * - Batch worker
     - queue worker job
     - ``stilt pull-worker`` or ``stilt push-worker``
   * - Long-lived worker
     - service deployment
     - ``stilt serve``
   * - Kubernetes manifests
     - Helm / KEDA / helper tooling
     - ``stilt.service.kubernetes`` helper functions
   * - Tracking what has run
     - PostgreSQL queue tables
     - the outputs themselves (:doc:`../advanced/output_state`), plus a PostgreSQL work queue via ``PYSTILT_DB_URL`` for cloud workers

Why it matters
--------------

- one package now owns the science-facing model and the worker runtime
- local, HPC, and cloud runs all use the same project folder layout
- fewer cross-package compatibility problems

What to re-check
----------------

- database connectivity and secrets
- whether the project root is a cloud URI and where workers get scratch (``compute_root``)
- any Kubernetes YAML that assumed older CLI flags or resource names

The service/runtime layer is still one of the least-settled parts of the alpha,
so treat deployment conventions as moving parts rather than frozen interfaces.
