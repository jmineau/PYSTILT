Coming From stiltctl
====================

This page is for stiltctl users who run STILT on cloud infrastructure. In
PYSTILT the work queue and the workers are part of the package. There is no
separate service to deploy. This part of PYSTILT is still experimental (see
:doc:`../guides/execution/kubernetes`).

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
     - ``stilt pull-worker`` drains the queue. ``stilt push-worker`` runs a
       fixed list of receptors, which is how the Slurm backend works.
   * - Long-lived worker
     - service deployment
     - ``stilt serve``
   * - Kubernetes manifests
     - Helm / KEDA / helper tooling
     - helper functions in ``stilt.service.kubernetes``
   * - Tracking what has run
     - PostgreSQL queue tables
     - the output files (:doc:`../advanced/output_state`). Cloud workers
       also use a PostgreSQL work queue, set with ``PYSTILT_DB_URL``, to
       share out the work.

Local, HPC, and cloud runs all use the same project folder layout and the
same package, so the model and the workers cannot drift out of step.

When you move a deployment over, check:

- the database connection and its secrets
- where the project and output directories are on the shared filesystem,
  and where workers get scratch space (``compute_root``)
- any Kubernetes YAML that uses old CLI flags or resource names

The worker and service code is one of the least settled parts of PYSTILT.
Expect deployment details to change between releases.
