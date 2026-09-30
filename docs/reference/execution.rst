Execution
=========

Most users only need ``model.run()`` or ``stilt run``
(:doc:`../guides/execution/index`). The functions below do the work
underneath.

- :func:`~stilt.execution.run_trajectories` runs HYSPLIT for one
  :class:`~stilt.Simulation` and writes its particles and log.
- :func:`~stilt.execution.write_footprint` makes the footprint from those
  particles and writes it.
- :func:`~stilt.execution.run_simulation` does both, skipping what exists.
- :func:`~stilt.execution.run_receptor` runs every variant of one receptor.
  Workers are always handed receptors.
- :func:`~stilt.execution.run_receptors` runs a list of receptors, in this
  process or in a process pool.
- :func:`~stilt.execution.pull_receptors` takes receptors from the
  PostgreSQL work queue until it is empty.

The executors start that work on this machine, on Slurm, or on Kubernetes.

Worker functions
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.run_trajectories
   stilt.execution.write_footprint
   stilt.execution.run_simulation
   stilt.execution.run_receptor
   stilt.execution.run_receptors
   stilt.execution.pull_receptors
   stilt.execution.SimulationResult
   stilt.execution.ReceptorResult

Executors
---------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.LocalExecutor
   stilt.execution.SlurmExecutor
   stilt.execution.KubernetesExecutor
