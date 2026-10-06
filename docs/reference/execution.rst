Execution
=========

Most users only need ``project.run()``, ``project.submit()``, or
``stilt run`` (:doc:`../guides/execution/index`). The functions below do the
work underneath.

- :func:`~stilt.execution.run` is what ``project.run()`` calls. It finds the
  receptors with missing results, runs them, and waits until they finish.
  ``task=(i, n)`` runs share ``i`` of ``n`` here (:func:`~stilt.execution.task_share`),
  which is what ``stilt run --task i/n`` does.
- :func:`~stilt.execution.submit` is what ``project.submit()`` calls. It
  writes a job array script (:func:`~stilt.execution.job_script`) whose
  tasks each run ``stilt run --task``, submits it with ``sbatch``, and
  returns the job id at once.
- :func:`~stilt.execution.run_particles` runs HYSPLIT for one
  :class:`~stilt.Simulation` and writes its particles and log.
- :func:`~stilt.execution.make_footprint` makes the footprint from those
  particles and writes it.
- :func:`~stilt.execution.run_receptor` runs every variant of one receptor,
  skipping what exists. Variants that share particles run HYSPLIT once and
  make their footprints from the particles in memory. A failure is recorded
  with the simulation (:attr:`stilt.Simulation.failure`). Workers are
  always handed receptors.
- :func:`~stilt.execution.run_receptors` runs a list of receptors, in this
  process or in a process pool.

On Slurm each task of the job array runs one share of the receptors
(:func:`~stilt.execution.task_share`) with ``stilt run --task``. A local
run happens in this process.

Worker functions
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.run
   stilt.execution.submit
   stilt.execution.task_share
   stilt.execution.resolve_compute_root
   stilt.execution.run_particles
   stilt.execution.make_footprint
   stilt.execution.run_receptor
   stilt.execution.run_receptors

Slurm job arrays
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   stilt.execution.job_script

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
