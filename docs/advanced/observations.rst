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

Scenes are first-class enough to matter operationally. You can register a batch
of receptors with a scene ID:

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
   sim_ids = model.register_pending(receptors=receptors, scene_id=scene.id)

That scene ID then flows into grouped status queries.

Declarative transforms
----------------------

Each footprint can declare particle transforms in ``config.yaml``:

.. code-block:: yaml

   footprints:
     column:
       grid: slv
       transforms:
         - kind: vertical_operator
           mode: ak_pwf
           levels: [0.0, 1000.0, 2000.0]
           values: [0.2, 0.5, 0.3]
         - kind: first_order_lifetime
           lifetime_hours: 4.0

The built-in transform specs are intentionally small:

- ``vertical_operator``
- ``first_order_lifetime``

That keeps configuration readable while still covering useful column and simple
chemistry workflows.
