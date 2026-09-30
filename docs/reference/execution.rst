Execution
=========

Most users only need ``model.run()`` or ``stilt run``
(:doc:`../guides/execution/index`). The functions below do the work
underneath.

- :func:`~stilt.execution.run` is what ``model.run()`` calls. It saves the
  model's settings and receptors to the project
  (:func:`~stilt.execution.register`), finds the receptors with missing
  results, and starts workers for them.
- :func:`~stilt.execution.run_trajectories` runs HYSPLIT for one
  :class:`~stilt.Simulation` and writes its particles and log.
- :func:`~stilt.execution.write_footprint` makes the footprint from those
  particles and writes it.
- :func:`~stilt.execution.run_simulation` does both, skipping what exists.
- :func:`~stilt.execution.run_receptor` runs every variant of one receptor.
  Workers are always handed receptors.
- :func:`~stilt.execution.run_receptors` runs a list of receptors, in this
  process or in a process pool.

On Slurm the receptors are split into batches
(:class:`~stilt.execution.Batch`), one per task of a job array submitted
with `submitit <https://github.com/facebookincubator/submitit>`_. A local
run happens in this process.

Worker functions
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.run
   stilt.execution.register
   stilt.execution.resolve_compute_root
   stilt.execution.run_trajectories
   stilt.execution.write_footprint
   stilt.execution.run_simulation
   stilt.execution.run_receptor
   stilt.execution.run_receptors
   stilt.execution.SimulationResult
   stilt.execution.ReceptorResult

Batches and handles
-------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.Batch
   stilt.execution.LocalHandle
   stilt.execution.SlurmHandle
