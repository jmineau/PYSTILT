Observation Layer
=================

PYSTILT includes a narrow observation-domain layer in ``stilt.observations``.
It sits above the transport/runtime core and gives science-facing workflows a
place for:

- normalized measurement records via :class:`stilt.observations.Observation`
- scene grouping via :class:`stilt.observations.Scene`
- geometry and operator metadata
- generic sensor families that normalize observations and build
  transport-facing :class:`stilt.Receptor` objects

This layer is intentionally smaller than full X-STILT parity. It does not yet
include product-specific adapters such as OCO or TROPOMI readers.


Core model
----------

The intended boundary is:

- :class:`stilt.observations.Observation`
  normalized measurement record
- :class:`stilt.observations.Scene`
  grouping of related observations
- :class:`stilt.observations.Sensor`
  sensor-family logic that stamps sensor metadata, validates species, and
  converts observations into receptors
- :class:`stilt.Receptor`
  transport-facing release geometry used by STILT/HYSPLIT

``Scene.batch_id`` exists as a convenience bridge for queue submission, but
``Scene`` is still just a science-layer grouping object.


Sensor-first normalization
--------------------------

For most user-facing workflows, sensors should construct normalized
observations rather than asking you to stamp ``sensor=`` and ``species=``
manually for every record.

When a sensor is configured with exactly one supported species,
``make_observation()`` will use it automatically. If a sensor family supports
multiple species, pass ``species=...`` explicitly.

.. code-block:: python

   from stilt.observations import PointSensor

   sensor = PointSensor(name="tower", supported_species=("co2",))
   observation = sensor.make_observation(
       time="2023-01-01 12:00:00",
       latitude=40.77,
       longitude=-111.85,
       altitude=30.0,
       observation_id="tower-001",
   )

   receptor = sensor.build_receptor(observation)

   assert receptor.kind == "point"
   assert receptor.altitude == 30.0

If you already have a separate normalization layer or product-specific parser,
constructing :class:`~stilt.observations.Observation` directly is still fine.
The sensor helpers are there to keep the common path less repetitive, not to
replace custom ingestion code.


Scene grouping and queue submission
-----------------------------------

The observation layer does not replace :class:`stilt.Model`. Instead, it feeds
normal PYSTILT transport workflows.

.. code-block:: python

   import stilt
   from stilt.observations import PointSensor

   sensor = PointSensor(name="tower", supported_species=("co2",))
   observations = [
       sensor.make_observation(
           time="2023-01-01 12:00:00",
           latitude=40.77,
           longitude=-111.85,
           altitude=30.0,
           observation_id="tower-001",
       ),
       sensor.make_observation(
           time="2023-01-01 12:05:00",
           latitude=40.78,
           longitude=-111.84,
           altitude=30.0,
           observation_id="tower-002",
       ),
   ]

   [scene] = sensor.group_scenes(observations)

   receptors = [sensor.build_receptor(obs) for obs in scene.observations]

   model = stilt.Model(project="./my_project")  # existing project config on disk
   sim_ids = model.submit(receptors=receptors, batch_id=scene.batch_id)

   completed, total = model.repository.batch_progress(scene.batch_id)
   print(scene.id, total, completed, len(sim_ids))

This is the intended near-term observation bridge:

1. normalize observations through a sensor or a custom parser
2. group them into scenes
3. build receptors through a generic sensor family
4. submit those receptors into the normal PYSTILT runtime


Generic observation filtering
-----------------------------

For simple non-spatial filtering, use
:func:`stilt.observations.filter_observations`.

.. code-block:: python

   from stilt.observations import filter_observations

   selected = filter_observations(
       observations,
       sensors="tower",
       species=("co2", "ch4"),
       time_range=("2023-01-01 00:00:00", "2023-01-02 00:00:00"),
       metadata={"orbit": "001"},
   )

This preserves input order and currently supports exact-match filtering on:

- sensor
- species
- observation id
- time range
- metadata fields
- quality fields
- an optional custom predicate


Column and slant workflows
--------------------------

Use :class:`stilt.observations.ColumnSensor` for vertical-column or slant-style
release geometries.

- ``mode="vertical"`` builds a standard column receptor
- ``mode="slant"`` builds a multipoint LOS receptor from
  :class:`stilt.observations.LineOfSight` plus
  :class:`stilt.observations.ViewingGeometry`

These are generic geometry families only. Product-specific retrieval adapters
remain deferred.


Weighting models
----------------

PYSTILT exposes a small generic weighting interface in
``stilt.observations``. This keeps weighting logic near the observation layer
without coupling it to any one sensor or chemistry workflow.

The current pieces are:

- :class:`stilt.observations.WeightingContext`
- :class:`stilt.observations.VerticalOperatorWeighting`
- :func:`stilt.observations.apply_weighting`
- :func:`stilt.observations.apply_vertical_operator` as the convenience wrapper

.. code-block:: python

   from stilt.observations import (
       VerticalOperator,
       VerticalOperatorWeighting,
       WeightingContext,
       apply_weighting,
   )

   operator = VerticalOperator(
       mode="ak_pwf",
       levels=[0.0, 1000.0],
       values=[0.2, 0.8],
   )
   context = WeightingContext(operator=operator, coordinate="xhgt")
   weighted_particles = apply_weighting(
       particles,
       VerticalOperatorWeighting(),
       context=context,
   )

This is the intended interface for later portable chemistry/lifetime work:

- today: vertical operator weighting
- later: chemistry-aware weighting models that still operate on the same
  particle-weighting contract


Chemistry models
----------------

PYSTILT also exposes a small chemistry/lifetime interface in
``stilt.observations``. This is not the X-STILT chemistry application stack;
it is just the reusable particle-level model underneath it.

The current pieces are:

- :class:`stilt.observations.ChemistryContext`
- :class:`stilt.observations.FirstOrderLifetimeChemistry`
- :func:`stilt.observations.apply_chemistry`

.. code-block:: python

   from stilt.observations import (
       ChemistryContext,
       FirstOrderLifetimeChemistry,
       apply_chemistry,
   )

   chemistry = FirstOrderLifetimeChemistry(lifetime_hours=4.0)
   context = ChemistryContext(species="no2", time_column="time", time_unit="min")
   chemically_weighted = apply_chemistry(
       particles,
       chemistry,
       context=context,
   )

The first concrete model is intentionally generic:

- first-order exponential lifetime decay
- driven only by particle transport age
- no inventory coupling
- no background logic
- no application-specific postprocessing


Uncertainty budgets
-------------------

PYSTILT also exposes a small observation-level uncertainty model in
``stilt.observations``. This is not X-STILT's calibration or transport-error
workflow; it is just a portable place to store component uncertainties without
forcing them into ad hoc metadata.

The current pieces are:

- :class:`stilt.observations.UncertaintyComponent`
- :class:`stilt.observations.UncertaintyBudget`
- optional ``Observation.uncertainty_budget``

.. code-block:: python

   from stilt.observations import Observation, UncertaintyBudget

   budget = UncertaintyBudget.from_mapping(
       {"measurement": 1.2, "transport": 2.8, "background": 0.7},
       units="ppm",
   )
   observation = Observation(
       sensor="oco2",
       species="xco2",
       time="2023-01-01 12:00:00",
       latitude=40.7,
       longitude=-111.9,
       uncertainty_budget=budget,
   )

   assert observation.uncertainty == budget.total

The first pass is intentionally small:

- named uncertainty components
- root-sum-square totals
- no radiosonde calibration
- no transport/background estimation workflow
- no inversion covariance assembly


Declarative footprint transforms
--------------------------------

For plain ``config.yaml`` workflows, PYSTILT now supports a small set of
predefined per-footprint transforms. These are applied after trajectories are
available and before the footprint is rasterized.

.. code-block:: yaml

   footprints:
     column:
       grid: slv
       transforms:
         - kind: vertical_operator
           mode: ak_pwf
           levels: [0.0, 1000.0, 2000.0]
           values: [0.2, 0.5, 0.3]
           coordinate: xhgt
         - kind: first_order_lifetime
           lifetime_hours: 4.0
           time_column: time
           time_unit: min

This is intentionally declarative and narrow:

- built-in transform kinds only
- validated in config loading
- applied per footprint product
- no arbitrary Python callables in YAML

Advanced Python workflows can still add runtime transforms directly when
calling :meth:`stilt.Simulation.generate_footprint`.


Current scope
-------------

``stilt.observations`` currently aims to provide:

- normalized observations
- geometry and operator metadata
- scene grouping
- observation-selection helpers
- observation-to-receptor transforms
- generic point and column sensor families

It intentionally does not yet provide:

- product-specific sensor/file adapters
- inventory coupling
- background estimation
- chemistry workflows beyond small reusable transforms
- inverse-model logic
