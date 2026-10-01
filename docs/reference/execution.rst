Execution
=========

Most users only need ``project.run()``, ``project.submit()``, or
``stilt run`` (:doc:`../guides/execution/index`). The functions below do the
work underneath.

- :func:`~stilt.execution.run` is what ``project.run()`` calls. It finds the
  receptors with missing results, runs them, and waits until they finish.
- :func:`~stilt.execution.submit` is what ``project.submit()`` calls. It
  submits those receptors to Slurm as one job array and returns the
  submitit jobs at once.
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
   stilt.execution.submit
   stilt.execution.resolve_compute_root
   stilt.execution.run_trajectories
   stilt.execution.write_footprint
   stilt.execution.run_simulation
   stilt.execution.run_receptor
   stilt.execution.run_receptors
   stilt.execution.SimulationResult
   stilt.execution.ReceptorResult

Batches
-------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.Batch

Transport model
----------------

The worker runs a simulation through the model its settings name. HYSPLIT
is the one transport model.

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.transport.TransportModel
   stilt.transport.ModelRun
   stilt.transport.get_model
   stilt.transport.hysplit.HysplitModel
