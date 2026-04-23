Inline And Notebook Usage
=========================

PYSTILT is designed so the same model object works in quick notebook
experiments and in larger executor-driven runs.

Core inline pattern
-------------------

For one-off analysis, build a model in memory and call ``run()``:

.. code-block:: python

   import stilt

   model = stilt.Model(
       project="./case",
       receptors=[
           stilt.Receptor(
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

Querying simulations
--------------------

``model.simulations`` is a lazy collection backed by the durable index:

.. code-block:: python

   ids = model.simulations.ids()
   subset = model.simulations.select(mets="hrrr")
   incomplete = model.simulations.incomplete()

Each selected item is a ``Simulation`` object with lazy output accessors.

Cross-simulation access
-----------------------

Use model-level collections when you want to work across many simulations:

.. code-block:: python

   trajectory_paths = model.trajectories.paths()
   footprints = model.footprints["default"].load()

This is usually cleaner than iterating through every simulation directory by
hand.

Register first, execute later
-----------------------------

For more control, especially in executor-driven workflows, split registration
from execution:

.. code-block:: python

   sim_ids = model.register_pending(scene_id="tower-20230715")

At that point the project inputs and simulation registry are durable, and you
can decide whether to:

- call ``model.run()``
- dispatch ``push_simulations()`` directly
- drain claims with ``pull_simulations()``

Choosing between Python and CLI
-------------------------------

Prefer inline Python when:

- you are iterating in a notebook
- receptors are generated programmatically
- you want direct access to model collections and output objects

Prefer the CLI when:

- the project already lives on disk
- you are launching workers from batch scripts or containers
- you want a stable shell boundary around registration, status, and serving
