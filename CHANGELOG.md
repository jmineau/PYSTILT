# Changelog

All notable changes to PYSTILT are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Background from trajectory endpoints**
  (`stilt.observations.background`): sample a mole-fraction field (a
  global model or a curtain, as an xarray array; or values you sampled
  yourself, one per particle) at each particle's endpoint and average
  over the particles with the same transforms that weight the footprint,
  so the background and the enhancement add to the modelled mole
  fraction. `transport_error` takes the same field as `background=` so
  the wind errors' effect on the endpoints is part of the error. The new
  Background guide has the field layout and the choices made.
  `stilt.trajectory.endpoint_rows` and `stilt.flux.nearest_cell` are
  the shared pieces.

## [0.1.0a16] - 2026-09-22

### Added

- **Wind-error statistics** (`stilt.observations.variogram`,
  `fit_variogram`): derive `siguverr`, `tluverr`, `zcoruverr` and
  `horcoruverr` from analysis-minus-observation winds the way Lin and
  Gerbig (2005) did, by fitting exponential variograms of the error over
  height, time and distance. The new Wind Error Statistics guide has the
  recipe: the standard deviation and vertical scale from radiosonde
  profiles, the time and horizontal scales from hourly surface stations
  (sondes twelve hours apart cannot resolve a time scale of a few hours),
  with the Salt Lake Valley HRRR 2024 values. Sampling the meteorology at
  the observations is `arlmet.sample_points(..., earth_relative=True)`
  (arlmet 0.1.0a8+)

## [0.1.0a15] - 2026-09-22

### Changed

- **`transport_error` estimates Lin and Gerbig's (2005) variance difference
  directly** instead of X-STILT's regression-scaled version. Validation on a
  real HRRR column (`stilt/validation/transport_error/`) showed the
  regression turns sampling noise into a positive error: it fits only the
  levels whose variance happened to rise, so under no perturbation at all it
  reported a third of the enhancement in summer and three quarters in
  winter. The result now carries the signed `variance` (negative values are
  noise, aggregate them with a median), `noise` (the estimator's own
  standard deviation, from random halves of the unperturbed particles),
  `sd` (clipped square root), and `enhancement_perturbed`. `percentile` now
  defaults to `1.0` (every particle) and X-STILT's behaviour is available as
  `percentile=0.99, regression=True`. The Transport Error guide now explains
  that HYSPLIT decorrelates the wind error over distance travelled, so a
  `horcoruverr` of a few kilometres produces no detectable perturbation, and
  that the scales must come from an analysis-versus-radiosonde comparison.

## [0.1.0a14] - 2026-09-22

### Added

- **Transport error on the modelled enhancement**, X-STILT's method (Wu et
  al., 2018). `stilt.observations.transport_error(particles,
  error_particles, flux, transforms=, context=, levels=, length_scale=,
  percentile=)` takes a simulation's main and wind-perturbed particle
  tables and a flux field, computes each particle's enhancement, and turns
  the extra ensemble spread under the perturbation into a transport-error
  standard deviation: per release level, with the regression scaling of the
  variance difference, and combined over levels with an exponential
  vertical correlation (356 m default). The footprint's transforms are
  applied to both tables first so the error is weighted like the footprint.
  Returns a `TransportError` with `sd`, `enhancement`, and the per-level
  table. See the new *Transport Error* guide.
- `Footprint.enhancement(flux)`: the modelled enhancement per footprint time
  step for an `xarray` flux field on `lat`/`lon` (optionally `time`).
  `stilt.flux.sample_flux` and `stilt.flux.particle_enhancement` are the
  building blocks (nearest cell; outside the field counts as zero).
- `Simulation.transform_context(name, error=False)`: the context a
  simulation hands its transforms, for applying a footprint's transforms
  outside `generate_footprint`.
- **One averaging kernel per receptor inside batch runs.** `averaging_kernel`
  takes `table: kernels.parquet` (or `.csv`) instead of inline
  `levels`/`values`: a long table of `receptor, level, value` rows in the
  project, looked up by the receptor id when each footprint is generated.
  The path is relative to the project root and is resolved through the new
  `TransformContext.store`, so `stilt run`, Slurm arrays and Kubernetes
  workers apply each sounding's own kernel; a receptor with no row is an
  error. `stilt.transforms.averaging_kernel_table(receptors, levels, values)`
  builds the table from the registered receptors and the product's kernels
  (`levels` per receptor, or one shared grid). This replaces the post-run
  `generate_footprint(..., transforms=obs.transforms)` loop.
- **Slant Columns guide** (`docs/guides/slant_columns.rst`): building a
  slant receptor from viewing angles for a ground-based solar tracker
  (EM27/SUN) or an off-nadir satellite sounding, the altitude and azimuth
  conventions, and how to weight the result. `ViewingGeometry` and
  `build_slant_receptor` now document the azimuth convention (clockwise from
  north, bearing from the ground point toward the instrument or the sun),
  and a test pins it.

- `stilt.observations.slant_points(longitude, latitude, altitudes, zenith=,
  azimuth=, anchor=None)`: the slant line-of-sight geometry as a pure
  function, so a ground-based instrument can build a receptor from plain
  numbers with `Receptor.from_points` and no `Observation`.

### Changed

- **Documentation rewritten for new users.** Plain-language landing page and
  "What Is STILT?"; a quickstart that goes from one receptor to a footprint
  map; a glossary; user-guide pages organized by task; a STILT-R settings
  mapping table; and a Slurm guide covering the `setup:` option. Examples
  were checked against the current API.
- The R implementation is referred to as **STILT-R** throughout docs,
  docstrings, and comments.
- **Slant receptor pipeline simplified.** `build_slant_receptor(observation,
  altitudes)` takes the altitude samples directly; the path is anchored at
  `observation.altitude` (now required) in `observation.altitude_ref`, and
  the builder warns when that is AGL. `ViewingGeometry` is two required
  fields, `zenith_angle` and `azimuth_angle`, validated on construction;
  product-specific angles belong in `Observation.metadata`.
- Roadmap: slant-column receptor support is marked implemented. Release
  heights along a slant are recovered from the particles (0.1.0a12), and a
  close-spaced slant is exercised through HYSPLIT in the test suite.

### Changed

- **`stilt.observations` works on tables, not observation objects.** A
  product reader keeps its soundings as a DataFrame; `group_by_overpass(times)`
  returns overpass labels aligned to the input for `df.groupby`,
  `select_observations_spatial(longitudes, latitudes, ...)` returns the
  selected row indices, and `jitter_points(polygon, n)` returns points across
  a pixel outline. `slant_points` is unchanged. Product filtering is a pandas
  mask.
- `dump_transform` (and the `transforms` attribute stored in footprint
  netCDF files) omits fields that are `None`.

### Removed

- `Observation`, `Scene`, `HorizontalGeometry`, `ViewingGeometry`,
  `group_observations`, `filter_observations`, `jitter_observation`, and
  `build_slant_receptor`. The library read only the receptor inputs from an
  `Observation` (time, location, altitude, angles, pixel outline) and the
  kernel, which now lives in the kernel table; the rest was carried for the
  user, who already has it in their own table. Slant receptors are
  `Receptor.from_points(time, slant_points(...), altitude_ref="msl")`.
  Synthesising a pixel outline from a resolution and orientation is gone with
  `HorizontalGeometry`; pass the product's corner coordinates (or your own
  shapely polygon) to `jitter_points`.
- `TransformContext.observation`: never populated. `TransformContext.store`
  takes its place.
- `LineOfSight` and `Observation.line_of_sight`: the altitude samples are an
  argument to `build_slant_receptor`, and clipping is the caller's choice of
  samples (`surface_altitude=` / `model_top_altitude=` are gone).
- `ViewingGeometry.solar_zenith_angle`, `solar_azimuth_angle`,
  `relative_azimuth_angle`, `scan_angle`, `glint_angle`: never read.
- `build_point_receptor`, `build_column_receptor`,
  `build_multipoint_receptor`: they only forwarded observation fields to a
  receptor constructor; call `PointReceptor` / `ColumnReceptor` /
  `Receptor.from_points` directly.

## [0.1.0a13] - 2026-09-21

### Fixed

- **Grids lost their last row or column when the bounds were not exact in
  binary.** The cell count `floor((max - min) / res)` used a tolerance scaled
  to the cell count, but the rounding error in `max - min` scales with the
  bounds' magnitude, so `ymin=40.45, ymax=40.93, yres=0.01` gave 47 rows
  instead of 48, and a one-cell extent like `40.0-40.01` raised. About 40% of
  0.01° extents starting on the 0.01° grid were affected; integer-degree
  bounds were always exact. The tolerance now scales with the bounds, so
  `Grid.axes` and footprint rasters have the intended cells and agree with
  STILT-R's `seq()` count.

## [0.1.0a12] - 2026-09-21

### Added

- **User-defined particle transforms from `config.yaml`.** A transform `kind`
  containing a dot is an import path (`kind: mypkg.transforms.MyWeighting`);
  the class is imported and built from the remaining keys, so custom
  weightings reach Slurm and Kubernetes workers, which rebuild the model from
  config alone. A stored footprint whose transform cannot be imported still
  loads (the entry becomes an `UnresolvedTransform`); a project config naming
  one fails validation. See the new *Particle Transforms* guide.
- `exe_dir` setting (`STILTParams` / `config.yaml`) to run a custom `hycs_std`
  build instead of the bundled binary. It reaches local, Slurm and queue
  workers, is recorded with the trajectory parameters, and only `hycs_std` is
  linked from the directory. A reused simulation directory is relinked when
  the build changes.

### Changed

- **The storage, registry, and execution layers were collapsed onto
  `Project` and `Simulation`.** A project is one root — a local
  directory or an `s3://`/`gs://` URI — and a store key is the only address an
  output has. The simulations a model defines are its receptors crossed with
  its met streams; whether each is complete is read from the outputs by key
  through `Simulation.is_complete()`, which is now the single definition of
  completion (it was written seven times). Removed: `stilt.completion`,
  `stilt.manifest` (the `.stilt/manifest.parquet` registry), `stilt.queries`,
  the `stilt.storage` package (now flat `stilt.store` and `stilt.project`),
  `stilt.execution.{tasks,execute,phases,entrypoints}` (now
  `stilt.execution.worker`), `Model.layout` / `Model.storage` /
  `Model.manifest`, the separate `output_dir` root everywhere, scene grouping
  (`scene_id`, `scene_counts()`, `--scene-id`, `--by-scene`), the no-op
  `stilt rebuild` / `Model.run(rebuild=)`, and the unread flat
  `simulations/particles` / `simulations/footprints` symlink views.
- `Model.register_pending()` is `Model.register()`. Registering an explicit
  receptor batch now merges it into the project's `receptors.csv` instead of
  overwriting it, which fixes a bug where a second batch made the first
  batch's simulations unreachable through `model.simulations`.
- `Model.project` is a `Project`; `Model.simulation(sim_id)` builds a handle.
  `Simulation.resolve_output` is `Simulation.resolve`; `Simulation.files` and
  `storage_key` are gone in favour of `key()`, `has_trajectory`,
  `has_footprint()`, `expected_outputs()`, `is_complete()`, `publish()`.
  Constructing a `Simulation` no longer creates its directory.
- `SimulationResult` is `(sim_id, status, error)`; the ten other fields were
  never read. `execute_task`/`execute_batch`/`push_simulations` are
  `run_simulation`/`run_simulations`. The local executor runs
  `run_simulations` on a background thread (one process pool, not two) so the
  CLI can print progress. `Executor.start()` lost `output_dir`.
- `PostgresQueue.register()` takes sim ids; the queue no longer stores a
  receptor copy. `StatusCounts` moved to `stilt.model` with
  `total`/`completed`/`pending` only.
- `Model.status()` and `model.simulations.incomplete()` now count against the
  same set (receptors × mets); previously one used the manifest and the other
  did not.
- **Particle transforms are one class each.** `stilt.transforms` now holds
  three pydantic transforms whose fields are their YAML keys and whose
  `apply(particles, context)` does the work: `AveragingKernel`
  (`kind: averaging_kernel`), `PressureWeighting` (`kind: pressure_weighting`)
  and `FirstOrderLifetime` (`kind: first_order_lifetime`). The old
  `vertical_operator` kind with its `mode` switch is gone: `mode: ak_pwf` is
  now the first two listed in order, `mode: pwf` is the second alone, and
  `mode: none` / `uniform` is an empty list. Transforms apply once, in list
  order, to the unweighted particles and return a new frame; the
  `foot_before_weight` / `foot_before_chemistry` restore guard is gone
  (`ak_weight`, `xpres` and `pwf` diagnostic columns remain). Removed the
  spec / adapter / model / context layers that sat between YAML and the
  arithmetic: `stilt.config.transforms`, `stilt.observations.{apply,
  weighting, chemistry, operators}`, `VerticalOperator`,
  `apply_vertical_operator`, `*TransformSpec`, `ParticleTransformContext`,
  `build_particle_transforms`, `apply_particle_transforms`, and the unused
  `WeightingModel` / `ChemistryModel` protocols. The science functions
  (`particle_pwf`, `ak_weights`, `release_coordinate`) are public in
  `stilt.transforms`. `Observation.operator` is `Observation.transforms`.
- `Simulation.generate_footprint(transform_context=)` is `context=`
  (a `TransformContext`).
- **The observation layer is the X-STILT port and nothing else.** Removed the
  `Sensor` / `BaseSensor` / `PointSensor` / `ColumnSensor` facade,
  `UncertaintyBudget` / `UncertaintyComponent` and
  `Observation.uncertainty_budget`, and the `make_scene` /
  `group_scenes_by_{key,swath,metadata,time_gap}` helpers. `Scene` stays as a
  small frozen dataclass (`id`, time-ordered `observations`, `metadata`,
  `time`, `time_range`) with one method, `receptors(build)`, which maps any
  observation-to-receptor callable over its members; two groupers remain,
  `group_by_overpass(max_gap="30min")` (X-STILT's overpass finder) and
  `group_observations(key=...)`. A new instrument is a reader that yields
  `Observation`s plus, when needed, your own builder and transform; the
  observations guide has the worked example.
- **Config fields are plain pydantic `Field`s.** `cfg_field` and its
  `visibility` / `target` / `namelist` metadata are gone, along with
  `iter_documented_config_fields`, `CONFIG_DOC_MODELS`,
  `build_setup_entries` and `build_control_entries`. What HYSPLIT reads from
  `SETUP.CFG` is now `STILTParams.setup_entries()`: every `TransportParams`
  field except `STILTParams.CONTROL_FIELDS` and `ZICONTROL_FIELDS`, plus
  `numpar` and `varsiwant`.
- `RuntimeSettings` is one `pydantic-settings` class read from `PYSTILT_*`
  (`db_url`, `cache_dir`, `compute_root`); `RuntimeSettings.from_env()`,
  `resolve_runtime_settings()`, and the never-used `max_rows` /
  `PYSTILT_MAX_ROWS` are removed.
- `TrajectoryError` and `FootprintError` were never raised; the HYSPLIT
  errors now subclass `SimulationError` directly.

### Fixed

- **Multipoint and slant receptors assigned release heights (`xhgt`) to the
  wrong particles when release points were close together.** HYSPLIT's first
  output row is one timestep after release, by which time bulk advection has
  moved particles 200-600 m, further than the 50-300 m spacing of a typical
  slant column; matching particles to release points by horizontal position
  put `xhgt` off by 190-700 m RMS, so averaging kernels were applied to the
  wrong particles. `xhgt` is now recovered from release-time (`t=0`) rows when
  the HYSPLIT build writes them (exact), else by release height when the
  altitudes are distinct (~20 m), else by horizontal position with a warning
  when points are closer than 1 km. Point and column receptors were never
  affected.
- Removed the fallback that split particles as `numpar // n_locations` per
  release point. HYSPLIT rounds the per-location count up and truncates the
  last location, so that split was wrong.
- **A receptor CSV could silently split into two receptors when its rows
  straddled a pandas chunk boundary.** `read_receptors` left `r_idx` to type
  inference; pandas types a large file one chunk at a time, so in a file
  mixing numeric and string ids a receptor whose rows straddled a chunk
  boundary came back part int, part str, and grouping split it into two
  receptors with half the points each, with no warning beyond a
  `DtypeWarning`. Found in the SLV TRAX project: a 38-point multipoint
  receptor sitting across the 2**18-row boundary of a file mixing integer
  crossing ids and string dwell ids ran as two 19-point receptors (7,979
  loaded instead of 7,978). `r_idx` is only a grouping key, so it is now
  always read as text.

## [0.1.0a11] - 2026-09-19

### Fixed

- `VerticalOperator` now rejects an unknown `mode` at construction. Because it
  is a plain dataclass its `Literal` annotation is not enforced at runtime, so
  a mode retired in 0.1.0a10 (`integration`, `tccon`) fell straight through
  `apply_vertical_operator` leaving `foot` unweighted while still adding
  `foot_before_weight` — a silent no-op that looked like weighting had been
  applied. Retired modes now raise and name their replacement. The declarative
  `transforms` path was never affected; pydantic validated it.

## [0.1.0a10] - 2026-09-19

### Changed

- **Pressure weighting is now derived from the particles** (X-STILT's
  approach). `VerticalOperator` modes `pwf` and `ak_pwf` fit a hypsometric
  curve to the particles' first-step heights and pressures and weight each
  particle by the pressure gap it represents, so the user no longer supplies a
  PWF profile and the result is independent of `numpar` and profile
  resolution. `levels` / `values` now hold only the averaging kernel. New
  optional `surface_pressure` (hPa) pins the column bottom; `pressure_levels`
  is removed. The `integration` and `tccon` modes are removed (`pwf` and
  `ak_pwf` with instrument factors folded into `values`). Transformed particle
  tables gain `xpres` and `pwf` columns.

### Added

- Integration tests for pressure weighting against real HYSPLIT trajectories
  (`tests/test_pwf_integration.py`), covering hypsometric fit quality through
  a winter inversion, weight physicality, `numpar` independence, and the
  particle scatter that makes the fit necessary.

## [0.1.0a9] - 2026-09-17

### Added

- **Spatial geometries** for footprint aggregation. Footprints
  are still computed on a rectilinear raster; moving one onto another
  geometry is a cached sparse overlap-weight matrix (`stilt.geometry.
  overlap_weights`), so Jacobian building is one matmul per footprint.
  - `stilt.Mesh`: arbitrary polygon cells with ids and a CRS. Constructors:
    `from_file` / `from_geodataframe` (shapefiles, GeoPackage; geopandas),
    `from_h3(resolution, bounds)` (hexagons; optional `h3`),
    `from_windows(coords, size, ids=)` (point sources), `from_grid`.
  - `stilt.Zones`: labels merging the cells of a `Grid` or `Mesh`
    into super-cells.
  - `Grid` gained `axes`, `cells`, `index`, `is_longlat`, and
    `Grid.from_geometry(geometry, cells_per_target=4)` /
    `from_geometries`, which derive the native raster that resolves a
    geometry (envelope snapped to whole cells, resolution rounded down to
    one significant figure).
  - `Footprint.aggregate` accepts `Grid`, `Mesh`, and `Zones`
    (`stilt.SpatialTarget` is the union with xarray grids and coordinate
    lists), reprojects geometries in another CRS onto the native raster,
    and warns when the smallest target cell spans fewer than two native
    cells.
- `Trajectories.footprint(config)` regenerates a footprint from stored
  particles on a new grid, for target geometries finer than any stored
  raster.
- `FootprintConfig.geometry`: a declarative geometry spec (`kind: file`,
  `h3`, or `windows`) naming the state geometry a footprint serves. When
  `grid` is omitted it is derived with `Grid.from_geometry` (tunable via
  `cells_per_target`); `geometry.build()` returns the `Mesh`. The built
  geometry's content hash is recorded as `geometry_hash`, written to the
  footprint netCDF alongside the spec, and `Footprint.aggregate` warns when
  the mesh it is given has a different hash (the geometry source changed
  after the footprint was computed). Grid-only footprints are unaffected.
- `pystilt[geometry]` extra (geopandas, h3, pyproj), included in `complete`.
- Polygon overlap weights use `exactextract` automatically when it is
  installed (not a dependency; ~100x faster on large rasters, identical
  fractions), else shapely. `overlap_weights(..., backend=...)` forces one.

## [0.1.0a8] - 2026-09-15

### Fixed

- `apply_vertical_operator(coordinate="pres")` weighted each trajectory row
  by that row's pressure, so a particle's weight drifted along its path.
  Every coordinate is now taken from the particle's release row (nearest the
  receptor time) and applied to the whole trajectory.

### Changed

- Renamed `stilt.selection` to `stilt.queries` and
  `stilt.observations.receptors` to `stilt.observations.builders`. Public
  re-exports from `stilt` and `stilt.observations` are unchanged.
- `RuntimeSettings` and `stilt.service` docstrings now state that they are
  deployment wiring only (queue URL, compute root, cache, Kubernetes
  manifests), not science configuration or a general service API.

### Documentation

- Roadmap refreshed: the runtime simplification is done; the
  footprint-to-state-geometry bridge is the active track.

## [0.1.0a7] - 2026-09-15

### Fixed

- `MultiPointReceptor` rejects points that share a horizontal location.
  HYSPLIT chains same-lat/lon starting locations into one vertical line
  source and only releases between the last two heights, so stacking
  several altitudes at one location silently dropped all but the top
  segment. Use `ColumnReceptor` or one `PointReceptor` per height instead.
- `apply_vertical_operator` PWF modes (`pwf`, `ak_pwf`, `integration`,
  `tccon`) now treat `values` as layer weights: each particle takes its
  nearest level's value, shared among the particles at that level
  (`value × n_particles / n_at_level`), instead of linearly interpolating and
  multiplying by `n_particles` outright. Weighted footprints built from a
  coarse retrieval profile no longer scale with `numpar`; per-particle
  profiles (the X-STILT convention) are unchanged.

## [0.1.0a6] - 2026-08-28

### Added

- `CITATION.cff` and `.zenodo.json` citation metadata so GitHub releases are
  automatically archived (and DOI-minted) on Zenodo

## [0.1.0a5] - 2026-08-28

### Fixed

- `read_receptors` raises `ValueError` when rows sharing an `r_idx` have
  different release times, instead of silently using the first row's time
- `Model.register_pending()` no longer rewrites the project's `receptors.csv`
  when registering receptors loaded from it

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
[STILT-R](https://github.com/uataq/stilt) framework for Stochastic
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

- **STILT-R parity**: 20 fidelity scenarios in `tests/fixtures/r_stilt_reference.py`
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
