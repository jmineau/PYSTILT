Local Executor
==============

The ``local`` backend is the default.  It runs simulations directly on the
current machine, either in a single process or across a local process pool.
This is the right choice for notebooks, workstation batch runs, and anything
that fits on one machine.

Configuration
-------------

.. code-block:: yaml

   execution:
     backend: local
     n_workers: 4   # omit or set to 1 for single-process

Or override at the CLI without touching config.yaml:

.. code-block:: bash

   stilt run ./project --backend local --n-workers 4

``n_workers: 1`` runs all simulations in one worker; ``n_workers > 1`` uses a
``multiprocessing`` pool.  The CLI always blocks until all local workers finish.

Python and notebook usage
-------------------------

Build a model in memory and call ``run()``:

.. code-block:: python

   import stilt

   model = stilt.Model(
       project="./case",
       receptors=[
           stilt.PointReceptor(
               time="2023-01-01 12:00:00",
               longitude=-111.85,
               latitude=40.77,
               altitude=5.0,
           )
       ],
       config=stilt.ModelConfig(
           mets={
               "hrrr": stilt.MetConfig(
                   directory="/data/hrrr",
                   file_format="%Y%m%d_%H",
                   file_tres="1h",
               )
           }
       ),
   )

   model.run()

``model.run()`` is equivalent to ``stilt run`` from the CLI — it persists the
inputs and dispatches every incomplete simulation through the configured backend.

Querying outputs
----------------

``model.simulations`` is every receptor crossed with every met stream.  Use it
to inspect or filter completed work without re-running:

.. code-block:: python

   ids = model.simulations.ids()
   incomplete = model.simulations.incomplete()

For cross-simulation access, use the model-level collections:

.. code-block:: python

   trajectory_paths = model.trajectories.paths()
   footprints = model.footprints["default"].load()

Register first, execute later
------------------------------

For more control, split registration from execution:

.. code-block:: python

   sim_ids = model.register()

At that point ``config.yaml`` and ``receptors.csv`` are in the project store
and any worker can rebuild the model from the root.  You can then:

- call ``model.run()`` immediately
- call ``run_simulations(model, sim_ids, n_cores=...)`` yourself
- drain a queue later with ``pull_simulations()``

This pattern is useful when generating receptors programmatically before
handing off to a long-running workflow.

When to use Python vs the CLI
------------------------------

Prefer the **Python API** when:

- iterating in a notebook
- generating receptors programmatically
- you want direct access to model collections and output objects after a run

Prefer the **CLI** when:

- the project already lives on disk with a config.yaml
- launching workers from batch scripts or container entrypoints
- you want a clean shell boundary around registration, status, and serving
