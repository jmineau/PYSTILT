Coming From stiltctl
====================

This page is for stiltctl users who run STILT on cloud infrastructure.
PYSTILT does not run a service: there is no work queue, no long-lived
worker, and no Kubernetes deployment. It runs a project's unfinished
receptors on one machine or as a Slurm job array, and decides what is
finished from the output files.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - stiltctl concept
     - stiltctl pattern
     - PYSTILT equivalent
   * - Work submission
     - service-oriented submit API
     - ``stilt run``, ``project.run()``, or ``project.submit()`` (:doc:`../guides/running`)
   * - Batch worker
     - queue worker job
     - one task of a Slurm job array or a Kubernetes indexed Job,
       ``stilt run --task I/N`` (:doc:`../guides/containers`)
   * - Long-lived worker
     - service deployment
     - none. Run ``stilt run`` again when receptors are added; finished
       ones are skipped.
   * - Tracking what has run
     - PostgreSQL queue tables
     - the output files (:doc:`../advanced/output_state`)

An earlier PYSTILT had a PostgreSQL work queue, pull workers, and
Kubernetes manifests. They were removed. If you need a queue-backed
deployment, the unit of work to build it on is a command line:
``stilt run <project> --receptors ids.txt`` or ``--task I/N``, which any
worker with the project's filesystem can run. Its exit code says whether
every simulation finished (:doc:`../guides/containers`).
