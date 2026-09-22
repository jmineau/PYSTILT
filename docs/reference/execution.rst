Execution
=========

Most users only need ``model.run()`` or ``stilt run``
(:doc:`../guides/execution/index`). Underneath,
:func:`~stilt.execution.run_simulation` runs one
:class:`~stilt.Simulation` from start to finish and saves its outputs,
:func:`~stilt.execution.run_simulations` runs many, and the executors decide
where that happens: on this machine, on Slurm, or on Kubernetes.

Worker functions
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.run_simulation
   stilt.execution.run_simulations
   stilt.execution.pull_simulations
   stilt.execution.SimulationResult

Executors
---------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.LocalExecutor
   stilt.execution.SlurmExecutor
   stilt.execution.KubernetesExecutor
