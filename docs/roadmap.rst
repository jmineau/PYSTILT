Roadmap
=======

.. note::

   PYSTILT is alpha software (the ``0.1.0a`` series).  The public API may
   change while the package settles.  No backward compatibility guarantees
   before v1.0.

Current status
--------------

The **core transport** is stable and exercised by the test suite:

- HYSPLIT trajectory and footprint generation
- Numerical parity with `uataq/stilt <https://github.com/uataq/stilt>`_ (R-STILT) at ``rtol=1e-7`` per cell
- Local and SLURM execution paths
- Simulation registry (skip-existing, status tracking)
- Observation layer for science-facing workflows

The core runtime simplification (a lean manifest-plus-Postgres registry with
by-key completion) landed in June 2026.  Active development has moved to the
science boundary between footprints and inversion state grids (see
*Future plans* below); the execution and observation tracks are maintained
but not expanding.

Execution and orchestration (from stiltctl)
-------------------------------------------

`stiltctl <https://github.com/jmineau/air-tracker-stiltctl>`_ is a cloud-native
STILT orchestration system.  PYSTILT borrows its thin CLI → model → worker
call path: no separate service facade, no broad queue abstraction — just a
direct connection from the CLI to ``Model`` and from ``Model`` to the
execution backend.

.. list-table::
   :header-rows: 1
   :widths: 60 20

   * - Feature
     - Status
   * - Pull-mode queue workers (``stilt pull-worker``)
     - Implemented
   * - Long-lived streaming mode (``stilt serve``)
     - Implemented
   * - PostgreSQL-backed simulation registry for distributed coordination
     - Implemented
   * - Scene-based submission grouping (``stilt register --scene-id``)
     - Implemented
   * - Thin CLI → ``Model`` → worker call path
     - Implemented
   * - Kubernetes worker deployment
     - Partial
   * - Cloud object store outputs (GCS, S3)
     - In scope

Column and satellite science (from X-STILT)
--------------------------------------------

`X-STILT <https://github.com/uataq/X-STILT>`_ extends STILT for column and
slant-path satellite retrievals.  PYSTILT absorbs X-STILT's observation-layer
design and column-weighting concepts.  Full X-STILT feature parity is
**not** a goal.

.. list-table::
   :header-rows: 1
   :widths: 60 20

   * - Feature
     - Status
   * - ``stilt.observations`` layer (``Observation``, ``Scene``, ``PointSensor``, ``ColumnSensor``)
     - Implemented
   * - Column receptor support
     - Implemented
   * - Vertical operator particle transforms (averaging kernel / pressure weighting)
     - Implemented
   * - First-order lifetime decay transform
     - Implemented
   * - Declarative per-footprint transforms in config YAML
     - Implemented
   * - Slant-column receptor support
     - In scope (pending HYSPLIT vertical-coordinate validation)
   * - Additional transform types
     - In scope
   * - Specific sensor adapters (OCO-2/3, TROPOMI, TCCON)
     - Deferred
   * - Inventory coupling and background estimation
     - Deferred

Future plans
------------

In priority order:

- **Spatial geometries and footprint aggregation** (implemented): footprints
  are computed on a rectilinear raster and aggregated onto any state geometry
  (:class:`stilt.Grid`, :class:`stilt.Mesh` from shapefiles / H3 hexagons /
  point windows, :class:`stilt.Zones` super-cells) through cached sparse
  overlap weights, with ``Grid.from_geometry`` deriving the raster for a
  geometry, ``FootprintConfig.geometry`` naming it in YAML, and
  ``Trajectories.footprint`` regenerating footprints from stored particles.
  Still to come: a YAML form for ``Zones``.
- **Slant receptor geometry**: satellite-geometry receptors once
  HYSPLIT vertical-coordinate behavior is validated.
- **Observation-layer maturation**: specific sensor
  adapters, weighting pipelines, and chemistry hooks after the observation
  foundation proves stable.
