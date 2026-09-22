Execution
=========

The execution layer is one worker function plus backends that launch it.
:func:`~stilt.execution.run_simulation` runs a
:class:`~stilt.Simulation` end to end and publishes its outputs;
:func:`~stilt.execution.run_simulations` runs many for a model, inline or in
one process pool; the executors decide where that happens.

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
