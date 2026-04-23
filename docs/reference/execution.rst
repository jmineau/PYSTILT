Execution
=========

The execution layer unifies local runs, Slurm arrays, and Kubernetes workers
behind one task model.

Executors
---------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.LocalExecutor
   stilt.execution.SlurmExecutor
   stilt.execution.KubernetesExecutor

Queue helpers
-------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.pull_simulations
   stilt.execution.push_simulations
   stilt.execution.SimulationTask
   stilt.execution.SimulationResult
