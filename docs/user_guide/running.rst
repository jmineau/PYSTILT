.. _running:

Running Simulations
===================

PYSTILT uses one execution model across local, HPC, and cloud environments:

1. simulations are registered in the repository,
2. workers claim pending simulations,
3. each worker computes trajectory and optional footprints,
4. workers write terminal state directly back to the repository.

This section covers Python and CLI workflows for one-off runs, batch queues,
and streaming workers.


Alpha guarantees
----------------

- Delivery semantics are at-least-once.
- A claimed simulation can be retried after interruption or worker loss.
- Footprint terminal states are explicit per footprint name:
  ``complete``, ``complete-empty``, or ``failed``.
- ``skip_existing=True`` avoids rerunning already complete outputs.


Python API
----------

Run everything in one call
^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~stilt.Model.run` registers all met x receptor combinations and starts
workers using the configured backend.

.. code-block:: python

   handle = model.run()                     # default skip_existing from config
   handle = model.run(skip_existing=False)  # force rerun
   handle = model.run(wait=False)           # submit and return handle

:meth:`~stilt.Model.run` returns a :class:`~stilt.executors.JobHandle`
that can be waited on later.

Queue-first workflow
^^^^^^^^^^^^^^^^^^^^

You can separate registration from execution:

.. code-block:: python

   sim_ids = model.submit(batch_id="daily_20260413")
   print(len(sim_ids))

Then run queue workers using CLI or your chosen deployment system.

If you want a named queue-facing surface in Python, wrap the same project in
the thin :class:`~stilt.Service` facade:

.. code-block:: python

   service = stilt.Service(model=model)
   service.submit(batch_id="daily_20260413")
   service.drain(cpus=4)
   print(service.status())

Status and filtering
^^^^^^^^^^^^^^^^^^^^

Use repository-backed and filesystem-backed views together:

.. code-block:: python

   total = model.repository.count()
   completed = len(model.repository.completed_trajectories())
   pending = len(model.repository.pending_trajectories())

   print(total, completed, pending)

   # Load only simulations that have a successful terminal footprint state
   # for the selected footprint name (complete or complete-empty).
   sims = model.get_simulations(footprint="default")
   print(len(sims))


Command-Line Interface
----------------------

.. code-block:: bash

  # Initialize a project (creates config.yaml and receptors.csv)
  stilt init /path/to/my_project

  # Register receptors as pending simulations
  stilt submit /path/to/my_project --batch-id campaign_apr13

  # Run trajectories and footprints with local workers (blocks)
  stilt run /path/to/my_project --backend local --n-workers 8

  # Run through Slurm backend from config, returning after submission
   stilt run /path/to/my_project

  # Optionally block until Slurm jobs finish
  stilt run /path/to/my_project --wait

  # Dedicated queue worker process
  stilt worker /path/to/my_project --cpus 4

  # Long-lived streaming worker
  stilt serve /path/to/my_project --cpus 4

  # Rebuild repository from on-disk outputs after interruption/manual edits
  stilt rebuild /path/to/my_project

  # Check simulation and batch progress counts
  stilt status /path/to/my_project
  stilt claims /path/to/my_project
  stilt attempts /path/to/my_project

All CLI commands support ``--help`` for full option documentation.


Three common workflows
----------------------

1. **One-off local or notebook run**

   Use :meth:`~stilt.Model.run` or ``stilt run`` when you want the current
   process to submit work and wait for completion.

2. **Queue-first batch run**

   Use :meth:`~stilt.Model.submit`, :meth:`~stilt.Service.submit`, or
   ``stilt submit`` to register simulations first, then drain them with
   ``stilt worker`` or :meth:`~stilt.Service.drain`.

3. **Long-lived service workers**

   Use ``stilt serve`` or :meth:`~stilt.Service.serve` when you want workers to
   keep polling for new submissions.


Simulation Status
-----------------

Each simulation is identified by a :class:`~stilt.SimID` with the format::

   <met>_<YYYYMMDDHHMM>_<location_id>

For example: ``hrrr_202307151800_40.766N111.848W10m``.

Possible :attr:`~stilt.Simulation.status` values:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Meaning
   * - ``None``
     - Simulation directory does not yet exist.
   * - ``"complete"``
     - Trajectory parquet file exists.
   * - ``"failed:<reason>"``
     - HYSPLIT ran but produced no valid output (see :func:`~stilt.errors.identify_failure_reason`).

For footprint state, the repository tracks per-footprint terminal outcomes:

- ``complete``: footprint file produced successfully.
- ``complete-empty``: successful run with no footprint artifact produced.
- ``failed``: footprint generation failed.


Handling Failures
-----------------

Failed simulations and worker interruptions are expected in large campaigns.
At-least-once semantics means retry safety matters.

Recommended pattern:

1. keep writes idempotent,
2. rerun with default ``skip_existing=True`` after interruptions,
3. use ``stilt rebuild`` if filesystem and repository state diverge.

Inspect failures:

.. code-block:: python

   import stilt

   for sim_id, sim in model.simulations.items():
       if sim.status == "failed":
           print(sim_id, sim.log)

The :func:`~stilt.errors.identify_failure_reason` function classifies
HYSPLIT log output into typed :class:`~stilt.errors.FailureReason` values:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - FailureReason
     - Meaning
   * - ``MISSING_MET_FILES``
     - Fewer than ``n_min`` ARL files were found for this receptor.
   * - ``VARYING_MET_INTERVAL``
     - The time interval of available ARL files is inconsistent.
   * - ``NO_TRAJECTORY_DATA``
     - HYSPLIT produced no trajectory output.
   * - ``FORTRAN_RUNTIME_ERROR``
     - HYSPLIT crashed with a Fortran runtime error.
   * - ``EMPTY_LOG``
     - HYSPLIT exited without writing a log.
   * - ``UNKNOWN``
     - Failure reason could not be determined from the log.
