Roadmap
=======

.. note::

   PYSTILT is alpha software (v0.1.0a1).  The public API may change while the
   package settles.  No backward compatibility guarantees before v1.0.

Current status
--------------

The **core transport** is stable and exercised by the test suite:

- HYSPLIT trajectory and footprint generation
- Numerical parity with `uataq/stilt <https://github.com/uataq/stilt>`_ (R-STILT) at ``rtol=1e-7`` per cell
- Local and SLURM execution paths
- Simulation registry (skip-existing, status tracking)
- Observation layer for science-facing workflows

Active development is focused on finishing the core runtime simplification
(plan 036: simulation registry and bounded index) before expanding into new
science or packaging features.

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

The following work is planned but blocked on stabilizing the current runtime
surface first:

- **Spatial-target and footprint aggregation**: ergonomic
  projection/aggregation bridge between footprints and non-rectilinear flux grids.
- **Slant receptor geometry**: satellite-geometry receptors once
  HYSPLIT vertical-coordinate behavior is validated.
- **Observation-layer maturation**: specific sensor
  adapters, weighting pipelines, and chemistry hooks after the observation
  foundation proves stable.
