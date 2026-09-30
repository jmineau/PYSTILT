Roadmap
=======

.. note::

   PYSTILT is alpha software (the ``0.1.0a`` series). The public API may
   change between releases until v1.0.

Current status
--------------

These parts are stable and covered by the test suite:

- HYSPLIT trajectories and footprints
- Footprints that match `uataq/stilt <https://github.com/uataq/stilt>`_
  (STILT-R) at ``rtol=1e-7`` per cell
- Running locally and on Slurm
- Reruns that skip finished simulations, judged by which output files exist
- Variants, which run the same receptors under several named settings in one
  project. Transport-error ensembles and extra footprints are variants too.
- Helpers for column and satellite observations

Footprints can also be summed onto other geometries. A footprint is computed
on a rectangular grid and then summed onto a :class:`stilt.Grid`, a
:class:`stilt.Mesh` (from shapefiles, H3 hexagons, or windows around points),
or :class:`stilt.Zones` (groups of cells), using cached overlap weights.
``Grid.from_geometry`` picks a grid for a geometry, the ``geometry``
footprint setting names one in YAML, and ``Trajectories.footprint`` makes new
footprints from saved particles.

Most new work is on column and slant-column workflows for satellite and
ground-based instruments (see *Future plans* below).

Execution (call path from stiltctl)
-----------------------------------

`stiltctl <https://github.com/jmineau/air-tracker-stiltctl>`_ runs STILT on
cloud infrastructure. PYSTILT keeps its call path: the CLI calls ``Model``,
and ``Model`` hands receptors to workers. Its queue-backed and Kubernetes
execution was implemented and then removed. Batches of receptors now run on
this machine or as a Slurm job array submitted through
`submitit <https://github.com/facebookincubator/submitit>`_.

.. list-table::
   :header-rows: 1
   :widths: 60 20

   * - Feature
     - Status
   * - Thin CLI → ``Model`` → worker call path
     - Implemented
   * - Local runs, in one process or a process pool
     - Implemented
   * - Slurm job arrays, with preempted tasks resubmitted
     - Implemented
   * - Queue-backed workers (PostgreSQL), Kubernetes deployment
     - Removed
   * - Cloud object store outputs (GCS, S3)
     - Not planned

Column and satellite science (from X-STILT)
--------------------------------------------

`X-STILT <https://github.com/uataq/X-STILT>`_ extends STILT for column and
slant-path satellite retrievals. PYSTILT ports X-STILT's ideas for handling
observations and weighting columns. It does not try to match every X-STILT
feature.

.. list-table::
   :header-rows: 1
   :widths: 60 20

   * - Feature
     - Status
   * - ``stilt.observations`` helpers (overpass grouping, sounding selection, jitter, slant geometry)
     - Implemented
   * - Column receptor support
     - Implemented
   * - Averaging-kernel and pressure-weighting particle transforms
     - Implemented
   * - First-order lifetime decay transform
     - Implemented
   * - Declarative transforms in config YAML (default or per variant)
     - Implemented
   * - Slant-column receptor support
     - Implemented (see the *Slant Columns* guide)
   * - Slant altitudes from a retrieval's pressure levels (``pressure_altitudes``)
     - Implemented
   * - User-defined transforms via ``kind: my.module.Class``
     - Implemented
   * - Per-sounding averaging kernels in batch runs (``averaging_kernel`` with ``table:``)
     - Implemented
   * - Product readers (OCO-2/3, TROPOMI, TCCON)
     - Implemented (see *Reading Retrieval Products*); other instruments as one module each
   * - Transport error on the modelled enhancement (``transport_error``)
     - Implemented
   * - Modelled enhancement from a flux field (``Footprint.enhancement``)
     - Implemented
   * - Background from a mole-fraction field at the trajectory endpoints (``background``)
     - Implemented
   * - Satellite-derived plume background (forward trajectories)
     - Implemented (see the *Plume Background* guide)
   * - Forward runs (positive ``n_hours``) for plume and dispersion studies
     - Implemented (see the *Plume Background* guide)
   * - Emission-error propagation to the modelled enhancement
     - A recipe on ``Footprint.enhancement`` (see the *Transport Error* guide); the
       correlated case is fips's ``prior_obs_error``
   * - Inventory readers
     - Out of scope; a flux field is an xarray array

Future plans
------------

In priority order:

1. Readers for the instruments people bring, such as EM27/SUN and
   MethaneAIR/MethaneSAT. Which ones come first depends on who is using
   column receptors.
2. A YAML form for :class:`stilt.Zones`, so the ``geometry`` footprint
   setting can name groups of cells.
