In The Cloud (Kubernetes)
=========================

.. warning::

   The Kubernetes backend is experimental. The pieces described here exist
   in the code but haven't been tested end to end. Use :doc:`local` or
   :doc:`slurm` for real work. Contributions are welcome.

This page is for people running PYSTILT on cloud infrastructure. Cloud
workers don't get a fixed list of receptors the way local and Slurm workers
do. Each worker takes the next receptor from a shared queue in a PostgreSQL
database, runs all of its variants, and repeats until the queue is empty.

There are two ways to run workers:

- A batch Kubernetes Job, started by ``stilt run``. Its pods work through the
  receptors in the queue and exit.
- Long-running Deployments that keep waiting for new receptors. You build
  these from the manifest helpers in ``stilt.service.kubernetes`` and apply
  them yourself.

What you need
-------------

- A PostgreSQL database for the queue, with its URL in ``PYSTILT_DB_URL``.
  Set it on the machine where you run ``stilt run`` too, so the receptors
  get added to the queue.
- A Kubernetes Secret holding that URL under the key ``PYSTILT_DB_URL``.
  Workers read it from there.
- A project directory and output directory that every pod can reach, on a
  shared filesystem.
- A container image with PYSTILT installed.
- A writable folder in each pod for meteorology and HYSPLIT files. Pass one
  with ``stilt run --compute-root /tmp/pystilt`` (or set
  ``PYSTILT_COMPUTE_ROOT``). The same path is given to every pod.

Run a batch of workers
----------------------

.. code-block:: yaml

   execution:
     backend: kubernetes
     image: ghcr.io/example/pystilt-worker:latest
     namespace: stilt            # default: default
     n_workers: 8                # pods in the Job, all started together
     db_secret: pystilt-db       # the Secret with PYSTILT_DB_URL (default)

Any other key is copied into the pod spec as written, for example
``serviceAccountName`` or ``nodeSelector``.

``stilt run`` then adds the project's receptors to the queue and creates a
Job named ``stilt-<project name>``. Each pod runs:

.. code-block:: text

   stilt pull-worker <project>

The pods finish when the queue is empty. ``stilt run --wait`` waits for the
Job to finish.

Always-on workers
-----------------

For workers that keep running and pick up new receptors as you register
them, ``stilt.service.kubernetes`` has functions that return manifests as
Python dicts:

- ``worker_deployment_manifest``: a Deployment running
  ``stilt pull-worker --follow``
- ``service_deployment_manifest``: a Deployment running ``stilt serve``
- ``scaled_object_manifest``: a KEDA ``ScaledObject`` that scales a
  Deployment with the number of receptors waiting in the queue
- ``secret_manifest``: a Secret holding ``PYSTILT_DB_URL``
- ``worker_job_manifest``: the batch Job that ``stilt run`` creates

Write them to YAML and apply them with ``kubectl``. Add work with
``stilt register``.

Limitations
-----------

- The Job name comes from the project name. A finished Job has to be
  deleted (``kubectl delete job stilt-<project name>``) before ``stilt run``
  can start a new one for the same project.
- ``--no-skip`` has no effect. Pods always skip simulations that are already
  finished.
