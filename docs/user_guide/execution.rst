.. _execution:

Execution Backends
==================

PYSTILT selects execution backends through ``ModelConfig.execution`` or an
explicit executor passed to :meth:`~stilt.Model.run`.

All backends share the same queue behavior:

1. workers claim pending simulations from the repository,
2. each worker builds a per-simulation runtime payload,
3. trajectory and footprint outputs are persisted,
4. terminal state is written directly via repository ``mark_*`` methods.

This keeps one-off, batch, and streaming modes behaviorally aligned.


Backend matrix
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Backend key
     - Runtime executor
     - Typical use
   * - ``local``
     - :class:`~stilt.executors.LocalExecutor`
     - Single-process local runs for ``n_workers <= 1`` and multi-core local workstation or shared node runs for ``n_workers > 1``.
   * - ``slurm``
     - :class:`~stilt.executors.SlurmExecutor`
       - HPC array-task chunk execution via ``stilt push-worker``.
   * - ``kubernetes``
     - :class:`~stilt.executors.KubernetesExecutor`
     - Cloud-native worker jobs or long-lived deployments.


Local Execution
---------------

Default local inline execution:

.. code-block:: python

   config = stilt.ModelConfig(
       ...,
       execution={"backend": "local", "n_workers": 1},
   )

Local multi-core execution:

.. code-block:: python

   config = stilt.ModelConfig(
       ...,
       execution={
           "backend": "local",
           "n_workers": 8,
       },
   )

   model = stilt.Model(project="./my_project", receptors=receptors, config=config)
   model.run()

You can override backend choice at run time:

.. code-block:: bash

   stilt run ./my_project --backend local --n-workers 8


Slurm Execution
---------------

For large HPC workloads use the ``slurm`` backend.

.. code-block:: python

   config = stilt.ModelConfig(
       ...,
       execution={
           "backend": "slurm",
           "n_workers": 200,            # max workers requested by run()
           "n_tasks": 1000,             # safety cap for array task count
           "partition": "notchpeak",
           "account": "lin-group27",
           "time": "01:00:00",
           "mem_per_cpu": "4G",
           "cpus_per_task": 1,
           "array_parallelism": 100,    # optional array throttle (%100)
       },
   )

   model.run(wait=False)

Keys other than ``backend``, ``n_workers``, ``n_tasks``, ``cpus_per_task``,
and ``array_parallelism`` are passed through to ``sbatch`` with underscores
converted to dashes. If ``job_name`` is omitted, PYSTILT submits the array as
``pystilt-{project}``.

CLI examples:

.. code-block:: bash

   stilt run ./my_project                 # submit and return
   stilt run ./my_project --wait          # submit and block
   stilt push-worker ./my_project --chunk /path/to/task_0.txt --cpus 4


Kubernetes Execution
--------------------

Use ``kubernetes`` for cloud worker pods and shared PostgreSQL repository
queues.

.. code-block:: python

   config = stilt.ModelConfig(
       ...,
       execution={
           "backend": "kubernetes",
           "image": "ghcr.io/my-org/pystilt-worker:latest",
           "namespace": "stilt",
           "n_workers": 10,
           "autoscale": True,
           "db_secret": "pystilt-db",
       },
   )

   # batch mode (k8s Job)
   model.run()

   # streaming mode (k8s Deployment) via explicit executor call
   # handled by serve/worker follow mode inside Deployment pods

Kubernetes workers should be paired with cloud/object storage and PostgreSQL
repository state for multi-pod coordination.


Queue and payload behavior
--------------------------

Workers run per-simulation with a serialized run payload that includes receptor,
meteorology source, run parameters, selected footprint configs, artifact-store
context, and repository handle. This gives stable per-simulation execution
context and reduces config drift in long-lived queue workers.

For streaming deployments, keep worker images and project configuration in sync
between submissions to preserve reproducible behavior.

.. tip::

   Start with conservative Slurm settings and tune upward based on observed
   queue throughput and per-simulation runtime.
   Array parallelism caps help avoid over-claiming shared partitions.

.. seealso::

   :doc:`../tutorials/hpc_slurm` for a complete Slurm walkthrough and
   practical HPC examples.
