Observations, Scenes, And Transforms
====================================

The ``stilt.observations`` package is a science-facing layer that sits above
the transport core. Its job is to normalize observation products into a small
set of stable concepts:

- ``Observation``
- ``Scene``
- ``Sensor``
- receptor builders
- weighting and chemistry transforms

What it is for
--------------

Use the observation layer when your inputs are measurements or retrievals, not
already-formed STILT receptors.

The current alpha package includes:

- ``PointSensor`` for in-situ or tower-style measurements
- ``ColumnSensor`` for vertical or slant-column workflows
- scene grouping helpers based on time gaps, swaths, or metadata
- declarative per-footprint transforms

What it is not for
------------------

PYSTILT intentionally keeps product-specific file readers outside the core
package. The intended boundary is:

1. your reader or normalization code creates ``Observation`` objects
2. sensors group them into scenes
3. sensors build receptors
4. the transport/runtime core executes those receptors

Scene-aware registration
------------------------

Scenes are an in-memory grouping. Build receptors from a scene and register
them with the model:

.. code-block:: python

   sensor = PointSensor(name="tower", supported_species=("co2",))
   observations = [
       sensor.make_observation(
           time="2023-01-01 12:00:00",
           latitude=40.77,
           longitude=-111.85,
           altitude=30.0,
           observation_id="tower-001",
       )
   ]

   [scene] = sensor.group_scenes(observations)
   receptors = [sensor.build_receptor(obs) for obs in scene.observations]
   sim_ids = model.register(receptors=receptors)

The returned ids identify the scene's simulations for later queries.

Weighting and chemistry
-----------------------

Averaging kernels, pressure weighting, and lifetime decay are *particle
transforms*, declared per footprint in ``config.yaml`` or passed to
:meth:`stilt.Simulation.generate_footprint`. An ``Observation`` can carry its
retrieval's own transforms in ``observation.transforms`` for the caller to
pass along. See :doc:`transforms` for the built-ins, the column-weighting
science, and how to write your own.
