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
           values: [1.0, 0.9, 0.7]
         - kind: first_order_lifetime
           lifetime_hours: 4.0

The built-in transform specs are intentionally small:

- ``vertical_operator``
- ``first_order_lifetime``

That keeps configuration readable while still covering useful column and simple
chemistry workflows.

Vertical weighting for column receptors
---------------------------------------

A column instrument averages the mixing ratio over air mass, but HYSPLIT
releases column particles uniformly in *height*, so a plain particle mean
over-weights the thin upper layers.  The ``vertical_operator`` transform fixes
that with two ingredients:

``pwf`` — pressure weighting function
   Derived from the particles themselves, following X-STILT.  A hypsometric
   curve is fit to the particles' first-step heights and pressures and
   evaluated at each release height; each particle then carries the slab of air
   centred on it, so a particle near the ground counts for more than one aloft.
   Nothing needs to be supplied.  Pass ``surface_pressure`` (hPa) to reference
   the profile to the retrieval's surface pressure instead of the fitted value.
   Requires ``pres`` and ``zagl`` in ``varsiwant`` (both are defaults).

``ak`` — averaging kernel
   Supplied by the user as ``levels`` / ``values`` (normalized, dimensionless).
   ``levels`` are release heights AGL in metres by default; set
   ``coordinate: pres`` when the kernel is on pressure levels.  Fold
   instrument-specific factors (e.g. TCCON's wet-air scaling) into ``values``.

``ak_pwf`` combines both and is the usual choice for satellite and TCCON
columns.

Two things are worth knowing about the result:

- The weights sum to the fraction of the atmosphere's mass the column covers
  (about 0.3 for a 0-3 km column), not to one.  Air above the column top
  cannot be reached by surface fluxes within the back-trajectory, so the
  footprint is complete; the remaining fraction belongs to the prior profile
  if you are building a full column simulation.
- The weighted footprint's magnitude does not depend on ``numpar``.

The transformed particle table carries ``xpres`` (release pressure, hPa) and
``pwf`` columns so you can inspect the weighting directly:

.. code-block:: python

   weighted = apply_vertical_operator(particles, VerticalOperator(mode="pwf"))
   weighted.drop_duplicates("indx")[["xhgt", "xpres", "pwf"]]

A column whose bottom sits above the ground still measures the air beneath it,
so the lowest particle carries that whole sub-column.  Start the receptor at
or near the surface unless you intend that.
