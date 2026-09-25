Execution
=========

Most users only need ``model.run()`` or ``stilt run``
(:doc:`../guides/execution/index`). Underneath,
:func:`~stilt.execution.run_simulation` runs one
:class:`~stilt.Simulation` from start to finish and saves its outputs,
:func:`~stilt.execution.run_receptor` runs every variant of one receptor,
:func:`~stilt.execution.run_receptors` runs many receptors, and the
executors decide where that happens: on this machine, on Slurm, or on
Kubernetes. The unit of work handed to workers is a receptor.

Worker functions
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

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
