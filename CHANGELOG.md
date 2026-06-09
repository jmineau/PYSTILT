# Changelog

All notable changes to PYSTILT are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
