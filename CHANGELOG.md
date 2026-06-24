# Changelog

All notable changes to PYSTILT are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a4] - 2026-06-24

### Added

- **``Trajectories.endpoints()``** returns one row per particle at the far end of
  its path (the largest ``|time|`` from release), with the absolute endpoint
  ``time``, ``lati``, ``long``, ``zagl``, ``endpoint_age_min``, and ``run_time``.
  This is where to sample a boundary/background field (e.g. CarbonTracker) for a
  backward run. Every particle contributes one endpoint, including those that left
  the domain early (an early exit is a real inflow point, not an incomplete run).

## [0.1.0a3] - 2026-06-17

### Added

- **``Grid.to_xarray()``** returns the footprint grid as a CF-style xarray
  ``Dataset`` of cell centers (1-D ``lon``/``lat`` or projected ``x``/``y``
  coordinates matching the native footprint grid, plus a ``crs`` grid-mapping
  variable). This is the interchange form of the grid: pass it straight to
  ``Footprint.aggregate`` as the target, or hand it to other tools via the shared
  xarray/CF grid convention. ``pyproj`` is required only for non-longlat grids.

### Fixed

- **``Footprint.aggregate`` now conservatively regrids onto the target grid
  instead of nearest-point sampling.** A footprint is an *extensive, per-cell*
  sensitivity (its units carry ``m²``), so coarsening it means **summing** native
  cells, not averaging. Each native cell is apportioned to the target cells it
  overlaps by area fraction (a sum-conserving conservative regrid): aligned
  coarsening reduces exactly to a block-sum, misaligned/finer targets split
  native cells by overlap, and mass outside the target grid is dropped (never
  folded into edge cells). The old ``sel(..., method="nearest")`` kept a single
  native pixel per cell, undercounting coarse-grid sensitivities by
  ``~(target_res/native_res)²`` — e.g. ~25-43× too small for 0.01° footprints on
  a 0.05° inversion grid, silently weakening STILT Jacobian rows.
  - The first argument is now ``target`` (was ``coords``) and accepts an **xarray
    grid** (``lon``/``lat`` or ``x``/``y`` coordinates; ``NaN`` cells in a 2-D
    ``DataArray`` are masked out) *or* a plain list of ``(x, y)`` cell centers.
    The bare-coords path still infers ``resolution`` from the coordinate spacing,
    so positional callers (e.g. the fips Jacobian builder) keep working with no
    change. The ``(x, y)``-indexed, one-column-per-time-bin return shape is
    unchanged.
- PYSTILT's ``MeteorologyError`` now uses HYSPLIT's exact "Insufficient number of
  meteorological files found" wording, so missing-met failures caught Python-side
  (before HYSPLIT runs) are classified as ``MISSING_MET_FILES`` like HYSPLIT's
  own. ``Simulation.status`` previously reported ``failed:UNKNOWN`` for this
  common case.
- Repaired two stale CI tests left over from the index dissolution:
  ``test_failure_missing_met`` now asserts failure through the log-derived
  ``Simulation.status`` (the by-key store has no "failed" state), and
  ``test_pull_simulations_requires_runtime_queue_backend`` matches the current
  Postgres-work-queue error message. Tests only.
- Documentation build: removed the dead ``stilt.index`` autosummary entries and
  stale SQLite / output-index prose left over from the index dissolution, so the
  Sphinx docs build cleanly again. Also made ``stilt.manifest`` pyright-clean
  (``pd.Index``-wrapped ``DataFrame(columns=...)``). No runtime change.

## [0.1.0a2] - 2026-06-09

### Changed

- **Completion is computed by key from the store, not from an index.** A
  simulation is complete iff every output it is configured to produce exists
  (`stilt.completion.is_complete`). When wind-error params are set, that set
  includes the error trajectory.
- **The registry is now `.stilt/manifest.parquet`** (`stilt.manifest`) — a small
  parquet of registration metadata (sim_id / met / receptor / scene /
  footprints), read and written through the `Store` so it works on local
  filesystems and cloud object stores alike. Registration metadata only;
  completion is never stored.
- **`model.index` → `model.queue`.** The Postgres backend is now a lean,
  status-only work queue (`stilt.service.PostgresQueue`: enqueue → claim
  [`FOR UPDATE SKIP LOCKED`] → done/failed), present only when `PYSTILT_DB_URL`
  is set. Local projects have no database — registry is the manifest, completion
  is by key.
- **Error-trajectory backfill runs only the error pass.** Enabling error params
  on a project that already has trajectories now runs the error trajectory
  alone — reusing the existing main when the config (ignoring error params)
  matches — instead of recomputing every main trajectory.

### Removed

- The `stilt.index` subpackage, `SimulationIndex` / `IndexCounts` /
  `OutputSummary`, the SQLite index backend, and the SQL
  completion-predicate / dialect machinery. The on-disk index database is gone.

### Fixed

- A simulation with a main trajectory and footprints but no error trajectory
  (when wind-error params are configured) was incorrectly treated as complete and
  skipped under `skip_existing`. It is now re-dispatched so the error trajectory
  is backfilled.

## [0.1.0a1] - 2026-05-12

First public alpha of PYSTILT — a typed Python implementation of the
[R-STILT](https://github.com/uataq/stilt) framework for Stochastic
Time-Inverted Lagrangian Transport modeling.

### Core transport

- `Receptor` — typed release point with support for single-point, multi-point,
  and vertical-column configurations; AGL and MSL altitude references
- `Simulation` — runs HYSPLIT, reads back trajectories, and computes footprints
  in a single object; lazy I/O with parquet trajectories and netCDF footprints
- `Trajectories` — particle-track container with plume-dilution weighting
  (`hnf_plume`) and self-describing Arrow/parquet serialization
- `Footprint` — rasterized surface-influence function with Gaussian smoothing,
  time integration, domain clipping, and named multi-footprint support
- `Model` — project-level coordinator; owns receptors, met sources, config,
  output index, and execution dispatch

### Configuration

- `ModelConfig` — flat YAML project config; all transport, met, footprint, and
  execution parameters in one file
- `STILTParams` / `TransportParams` / `ErrorParams` — typed parameter objects
  that map directly to HYSPLIT `SETUP.CFG` entries; config-time validation
  (e.g. `hnf_plume` cross-checks `varsiwant` columns at construction)
- `FootprintConfig` — grid, projection, smoothing, and time-integration spec
  with a content-addressed hash for output naming
- `MetConfig` — ARL met archive configuration with optional subgrid extraction

### HYSPLIT interface

- Bundled `hycs_std` binaries for Linux and macOS (Apple Silicon supported via
  Rosetta 2); custom binary path accepted via `exe_dir`
- `HYSPLITDriver` — writes `CONTROL` and `SETUP.CFG`, symlinks data files,
  streams stdout to a log, and parses `PARTICLE_STILT.DAT`
- Error-trajectory support via `WINDERR` / `ZIERR` / `ZICONTROL`
- Process-group cleanup with `SIGTERM` → `SIGKILL` escalation on timeout

### Execution

- Local parallel execution (joblib) for notebook and workstation use
- Slurm array-job executor for HPC clusters
- Pull-mode workers over a claim-capable PostgreSQL index (`stilt pull-worker`,
  `stilt serve`) for distributed streaming pipelines
- Push-mode chunk dispatch for immutable Slurm batch runs
- Kubernetes executor scaffolding (not yet functional)

### Output storage and indexing

- SQLite index (default, WAL mode, NFS-safe) and PostgreSQL index for
  distributed workers; upsert-idempotent registration
- `LocalStore` and `FsspecStore` for local and cloud (S3/GCS/ABS) output roots
- Content-addressed footprint naming; simulation status tracking with
  failure-reason extraction from HYSPLIT logs

### Observations (`stilt.observations`)

- `Observation` and `Scene` containers for normalized measurement data
- Sensor helpers, receptor builders, and line-of-sight geometry for slant-path
  columns (X-STILT style)
- Footprint-weighted concentration operators and uncertainty propagation
- First-order chemistry and vertical-operator particle transforms

### CLI

- `stilt run` / `stilt submit` — local and HPC dispatch
- `stilt pull-worker` / `stilt serve` — claim-based worker and REST service
- `stilt register` — register receptors against an existing project index
- `python -m stilt` alias

### Tests

- **R-STILT parity**: 20 fidelity scenarios in `tests/fixtures/r_stilt_reference.py`
  validate PYSTILT footprints against a pinned commit of
  [uataq/stilt](https://github.com/uataq/stilt) (`e2feb358`) at `rtol=1e-7`
  per cell; scenarios cover point/column/multipoint receptors, 6 h and 24 h
  backward runs, forward runs, HNF plume dilution on/off, smoothing factors
  0–2, longlat and UTM projected grids, AGL/MSL altitude references, winter
  and summer HRRR meteorology, hourly and integrated time bins, and error
  trajectories with `siguverr`
- **Synthetic footprint tests**: hand-crafted particle DataFrames in
  `tests/r_stilt/test_footprint_synth.py` isolate specific code paths
  (single-particle Gaussian, boundary fenceposts, dateline crossing, global
  grid, latitude bandwidth scaling)
- **Unit and integration tests** cover HYSPLIT driver, config validation,
  meteorology staging, output storage, SQLite and PostgreSQL index backends,
  execution dispatch, and the observation layer
