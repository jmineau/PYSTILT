# Changelog

All notable changes to PYSTILT are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- A config must declare its variants (breaking). Without a `variants:`
  section the config no longer runs one variant per met; it fails with a
  message that says what to write (`variants: {hrrr: {}}`). Files PYSTILT
  writes always had the section; a hand-written `config.yaml` without it
  needs two lines (#150).
- An unknown key in `config.yaml`, at the top or in a variant, is an error
  that names the nearest setting and what it belongs to: `'smooth_factr'
  is not a setting. Did you mean 'smooth_factor', a footprint setting?`
  It used to reach HYSPLIT's config and be reported as one of its fields
  (#150).
- `numpar`, `hnf_plume`, and `veght` are on the base `TransportConfig`,
  with `n_hours` and `seed`, so a variant that runs another transport
  model inherits them. No hash changes (#150).
- `stilt.variants` is gone (breaking): `Variant` is in `stilt.config`,
  `stilt.variants.resolve(config)` is `config.resolve(directory)`, and
  `ProjectConfig.variant(name)` and `Declared` are private. A relative
  geometry file in a footprint's settings starts from the project
  directory, where it started from the working directory; the recorded
  path and the hashes are unchanged (#150).

- `project.simulations` is a plain pandas DataFrame, and the verbs are on
  `Project` (breaking): `project.status(sel)`, `incomplete(sel)`,
  `particles(sel)` (was `load_particles`), `footprints(sel)` (was
  `load_footprints`), and `jacobian(sel, target, time_bins)`. A selection
  is any table with `receptor` and `variant` columns (pandas, polars,
  pyarrow) or a boolean mask over `project.simulations`; leave it out for
  every simulation. A pandas selection keeps its own columns in
  `status()`. `Simulations`, its `.frame`, and its iteration are gone;
  look a row up with `project.simulation(receptor, variant)` (#150).
- `SimID` is gone (breaking). `sim.id` is the tuple `(receptor, variant)`,
  `str(sim)` is `receptor/variant`, and `project.footprints()` is keyed by
  the tuple. A run's working directory is
  `<compute root>/<receptor>/<variant>` as before (#150).
- `project.jacobian` reads and sums footprints in `workers` threads, the
  number of CPUs by default, where it used `execution.cpus`; `batch` sets
  how many receptors are read together (64) (#150).

- `project.run()` and `stilt.execution.run` return the status table of
  the simulations they ran, the rows of `project.simulations.status()`
  (breaking). `SimulationResult` is gone: a worker records each failure in
  the output directory, and the table reads them. `run_receptor` returns
  one line per failure for the progress log, `run_receptors` and
  `Batch` return nothing, and a stopped receptor raises
  `KeyboardInterrupt` (#150).

- `Output` has no folder classes (breaking). `stilt.output.Particles` and
  `Footprints` are gone, with `Output.particles()`, `footprints()`,
  `find_particles()`, `find_footprints()`, `particle_sets()`, and
  `footprint_sets()`. Each method takes the kind of result
  (`"particles"` or `"footprints"`) and the variant:
  `output.path(kind, variant, receptor_id)`, `present(kind, variant,
  receptor_ids)`, `complete(variant, receptor_ids)`, `table(kind,
  variant, receptor_ids)`, `folder(kind, variant)`, `folders(kind)`,
  `failure`, `failures`, `record_failure`, `clear_failure`, and the
  writers `write_particles`, `write_footprint`, `write_empty_footprint`,
  `write_log`, and `keep_workdir`. The output directory is
  `output.directory`, where it was `output.path` (#150).
- One rule says a simulation is complete, `stilt.output.completed`:
  `Output.complete`, `Simulation.is_complete()`, and `status()` apply it,
  so the test that held two copies together is gone (#150).

- Docs: the quickstart and the README make one footprint with
  `run_trajectories` and `calc_footprint` before they make a project
  (#150).

- `sim.generate_footprint` is `sim.calc_footprint`, `stilt.calc_footprint`
  with the simulation's particles, receptor, and settings filled in
  (breaking). It takes the settings to change as keywords
  (`sim.calc_footprint(smooth_factor=2.0)`, `grid=`, `time_integrate=`,
  `transforms=`), each replacing the variant's own; it no longer takes a
  `FootprintConfig`, and `transforms` replaces the variant's transforms
  rather than adding to them (#150).

- `stilt.__all__` lists what a user calls (breaking): the project and its
  configs, the receptors, `Simulation`, `Grid`, `Bounds`, `Mesh`, `Zones`,
  `run_trajectories`, `calc_footprint`, `read_particles`,
  `read_footprint`, `averaging_kernel_table`, and `StiltError`. `Output`,
  `Simulations`, `Variant`, `SimID`, `Met`, `Geometry`,
  `particles_metadata`, and `write_particles` are imported from their
  modules (`stilt.output.Output`, `stilt.meteorology.Met`, ...) (#150).

- `stilt.footprint.calculate` is `stilt.calc_footprint`, STILT-R's name,
  and takes its settings as keywords (breaking):
  `calc_footprint(particles, receptor, grid, smooth_factor=1.0,
  time_integrate=False, transforms=())`. It no longer takes a
  `FootprintConfig` (#150).

- `Simulations.jacobian` reads and sums footprints in batches of 64
  receptors, `execution.cpus` batches at a time, and stacks the rows, so
  its memory no longer grows with the selection. On 2,000 footprints of
  about 350,000 cells each it peaked at 105 GB and took 188 s; it now
  takes 4 GB and 112 s on one thread, 32 s on eight, with the same matrix.
  The time bin is found once per receptor-hour, and the sum onto the
  target is one sparse product per batch (#148).

- Multipoint receptor ids are computed for a whole table at once, and
  the ids are byte for byte the same (checked on every receptor of two
  large projects). Reading a `receptors.csv` of 63,000 multipoint
  receptors (2.6 million rows) takes 9 s, where it took 12.5 s (#148).

- `Simulations.status()` opens no result file (breaking). It took about
  25 minutes on a project of 64,000 footprints, reading every footprint's
  metadata for its `empty` column and building a simulation per row; it
  now takes seconds (#141). Its columns are `particles`, `footprint`, and
  `state` (`complete`, `failed`, or `pending`), plus `step`, `reason`, and
  `message` for failed simulations. The `empty` and `complete` columns are
  gone: an empty footprint is complete, `sim.empty_reason` says why it is
  empty, and a Jacobian lists the empty ones. `failures()` is gone too;
  use `st[st.state == "failed"]`.

- A failure record is one flat file per failed result, in the logs folder
  of the folder that failed: `logs/settings=<particles key>/` for HYSPLIT,
  `logs/settings=<footprint key>/` for a footprint (breaking). A footprint
  failure is found by its folder's settings hash, so a project that names
  the variant differently finds it. A record holds `step`, `reason`,
  `message`, and `time`, and a `traceback` for an unexpected error.
  `reason` is always set (the cause, or the error's class), so `error` is
  gone; `log` and `scratch` are gone too, since `sim.log` and the new
  `sim.scratch_path` give them. For a HYSPLIT failure the message is the
  line of its log that matched. An expected failure logs one line instead
  of a traceback. `Particles.write_failure` and `failed` are replaced by
  `record_failure`, `clear_failure`, and `failures` on both folder kinds
  (#148).

- Docs: a receptor's and a `Mesh`'s fields are described once, in their
  field descriptions, and the reference pages list them from there.
  `Project.run` and `Project.submit` point to `stilt.execution.run` and
  `submit` for their parameters. The `PointReceptor` example builds a
  receptor with keyword arguments, the only way it can be built (#134).
- `Project.init(path, starter=True)` writes the commented starter
  `config.yaml` that `stilt init` writes (`stilt.config.STARTER_CONFIG`),
  and `stilt init` calls it. `stilt run` resolves the local scratch
  directory once, so the banner shows the directory the run uses (#134).
- `stilt.transport.TransportConfig` is a pydantic base class, where it was
  a protocol (#134). It holds `n_hours`, `seed`, `UNRECORDED`, and default
  `settings()` and `realizations(n)`; `HysplitConfig` subclasses it, and a
  second model's config inherits what it does not change. A seeded run's
  `SETUP.CFG` now lists `SEED` before the other entries; the values are
  unchanged.
- `HYSPLITDriver` is gone (breaking). `write_inputs(workdir, receptor,
  config, met_files)` writes the files `hycs_std` reads, and
  `read_particle_dat(path, columns)` reads a `PARTICLE_STILT.DAT` as a
  particle table; both are in `stilt.transport.hysplit`, with
  `finish_particles`. `HysplitModel.run` holds the working directory and
  runs `hycs_std` between them (#134).
- `kmsl` is no longer a setting (breaking). HYSPLIT's `KMSL` is written
  from each receptor's `altitude_ref`, which the setting could only
  contradict. A `config.yaml` that sets it now fails to load; stored
  settings that record `kmsl: null` still read (#134).
- The HYSPLIT driver decides which input file each setting goes to; the
  config's fields no longer carry routing tags, and `fields_in` is gone.
  The `WINDERR` and `ZIERR` lines follow the explicit
  `WIND_ERROR_SETTINGS` and `ZI_ERROR_SETTINGS`, not the order fields are
  declared in. The files HYSPLIT reads are unchanged (#134).
- `aggregate` and `jacobian` sum footprint cells through one function, so
  binning hours and applying the overlap weights is written once.
  `Footprints.jacobian` is gone (breaking): a selection's `jacobian` reads
  the folder's table and calls `stilt.footprint.jacobian` itself, which
  takes any table of footprint cells (#134).
- Transforms take the receptor and the project directory (breaking). A
  transform's method is `apply(particles, receptor=None, directory=None)`;
  `TransformContext`, the `ParticleTransform` protocol, and
  `Project.transform_context` are gone. `calculate`,
  `Simulation.generate_footprint`, `background` and `transport_error` take
  `receptor=` (where they do not already) and `directory=` in place of
  `context=` (#134).
- Receptors are checked once (#134). A receptor built from the rows of a
  checked table (`receptors.csv`, `receptors_from_frame`) no longer runs
  the same checks again. `receptors_from_rows(rows)` builds every receptor
  of a `receptor_rows` table and replaces `receptor_from_rows` (breaking).
  The check that a multipoint receptor's points have distinct locations
  runs on arrays. On a file of 2.5 million multipoint rows (82,000
  receptors), checking the table went from 21 to 13 seconds and reading
  every receptor from 35 to 16 seconds.
- **One failure record per simulation** (breaking). When a run fails, the
  worker writes `<receptor id>.failure.yaml` beside the receptor's log:
  the step that failed, the exception, a short `reason` (such as
  `MET_COVERAGE` or `TIMEOUT`), the message, and the HYSPLIT log and
  scratch folder kept in the output directory. A success removes it.
  `sim.failure` reads it, `sims.failures()` lists the failed simulations,
  `status()` has a `reason` column, and `stilt status` counts failures by
  reason. The worker no longer appends a `PYSTILT ERROR` block to the
  HYSPLIT log, and error messages no longer name the scratch folder that
  is removed after the run (#134).
- **Variants that share particles run as one group** (#134). HYSPLIT runs
  at most once per group, and every footprint of the group is made from
  the particles in memory, where each variant read them back from Parquet.
  `run_receptor` does this directly; `run_simulation` is gone, and
  `SimulationResult` no longer has `ran_hysplit` or `phase`.
- `HYSPLITTimeoutError`, `HYSPLITFailureError`, `NoParticleOutputError` and
  `EmptyParticleOutputError` are `SimulationError` with a `reason`
  (`TIMEOUT`, the `FailureReason` from the log, or `NO_PARTICLE_DATA`)
  (breaking). `MeteorologyError` keeps its class, with reason
  `MISSING_MET_FILES` (#134).
- The core no longer imports HYSPLIT's package at all; the one exception to
  that import contract, `Simulation.outcome`, is gone (#134).

- An `Output` reads each settings folder's `_settings.yaml` once. A
  lookup that misses (a variant that has not run yet) lists the tree again
  but reads only folders it has not seen, where it used to read and hash
  every folder again on each miss. A footprint folder finds its particles
  folder in the same cache (#134).
- `Particles` and `Footprints` share one base for what every settings
  folder does (`file`, `has`, `receptors`, `table`, `path`, equality);
  each keeps only what differs. `Footprints.hash` is an attribute set when
  the folder is read, as `Particles.hash` already was (#134).

- **Each part owns its config** (breaking). `FootprintConfig` and the
  geometry specs are in `stilt.footprint.config`, `MetConfig` in
  `stilt.meteorology`, and
  `ExecutionConfig` in `stilt.execution.config`; the `stilt.config` package
  is one module holding `ProjectConfig`. Import them from `stilt`
  (`stilt.Grid`, `stilt.FootprintConfig`, `stilt.MetConfig`, and now
  `stilt.ExecutionConfig`) or from their new modules; `stilt.config` no
  longer re-exports them. `Bounds` and `Grid` stay in `stilt.spatial`, the
  raster and CRS layer (#134).
- **A variant's transport settings are checked when it resolves**
  (breaking). `VariantConfig` and `config.variant_configs` are gone.
  `ProjectConfig` still checks at load what needs no transport model
  (variant names, mets, `model`, `realizations`, `from:`, grid merging,
  footprint settings without a grid) and offers each declared variant
  merged with the defaults as `config.variant(name)`. `project.variants`
  (`stilt.variants.resolve`) validates each variant's transport settings
  once, with its model's config class, and expands `realizations`, so a
  misspelled HYSPLIT setting inside a variant is reported there.
  `Project.init` resolves before it writes `config.yaml`, so it still
  refuses a bad config (#134).
- `ProjectConfig` is no longer a `FootprintConfig`. The top-level footprint
  settings are `config.footprint`, as the model's are `config.transport`;
  `config.grid` is now the raw value from the file, as `config.numpar`
  already was (#134).

- `HYSPLITDriver` needs the `directory` to run in. Without one it made a
  temporary directory that nothing removed (#134).
- `MetConfig.subgrid_buffer` must be 0 or more; a negative buffer is a
  validation error when the config loads (#134).
- A footprint's `.stilt` accessor reads the receptor and settings
  attributes once per array, so a transform that cannot be imported warns
  once rather than on every access (#134).
- A footprint folder whose `_settings.yaml` names a transform this machine
  cannot import still lists and reads, with a warning, as footprint files
  already did (#134).
- `import stilt` no longer imports HYSPLIT's driver; `stilt.transport.get_model`
  loads it when a run needs it (#134).
- `stilt.observations.selection.haversine_km` is the great-circle distance
  the variogram code uses, public in its module (#134).

- **Particle files record their run's settings** (breaking). A particle
  file's metadata holds the same settings as its folder's `_settings.yaml`
  (`stilt:settings`): the transport model's settings, the met's, the model
  build, and the realization number, in place of the model's whole config
  (`stilt:params`). `particles_metadata(path)` returns
  `(receptor, settings, met_files)`;
  `stilt.identity.transport_from_settings(settings)` rebuilds the config.
  `write_particles` takes the settings, and `Particles.write` no longer
  takes a config. `particles_metadata` raises on a particle file written
  before this change, which records no settings; `read_particles` still
  reads it. #132 describes the one-time rewrite of an existing output
  directory.
- **A transport model owns its config, and a model is a variant axis**
  (breaking). `stilt.config.TransportParams` is
  `stilt.transport.hysplit.HysplitConfig`. `ProjectConfig` gains `model`
  (`hysplit` unless set); the model's parameters stay flat, top-level keys,
  checked by its config class, and are in `config.transport` (so
  `config.numpar` is `config.transport.numpar`). A variant may name another
  `model`; it then gives that model's parameters itself and inherits only
  the met and the footprint settings, so models compare in one project.
  `project.simulations` has a `model` column. Existing `config.yaml` files
  read as before, and the hashes of existing folders are unchanged.
- The near-field plume correction (`hnf_plume`) is applied by the HYSPLIT
  model, whose config holds it (`stilt.transport.hysplit.model.finish_particles`).
  `stilt.particles.prepare(raw, receptor)` no longer takes the config.
  Particle files record their model (`stilt:model`); files without it are
  read as HYSPLIT's.
- `stilt.transport.get_model` reads the `stilt.transport.MODELS` table, where
  another model's port adds its entry.
- **Variants resolve in one place, and settings have one home** (breaking).
  `project.variants` holds `stilt.Variant` objects, which carry a variant's
  configs, its met, the transport model build (`stilt.transport.ModelInfo`),
  and two hashes: `particles_hash` and `footprint_hash`, computed once.
  `stilt.variants.resolve(config)` replaces `ProjectConfig.resolve_variants()`.
  `stilt.config.VariantConfig` is now one declared variant merged with the
  defaults, before the geometry and the model build are read
  (`config.variant_configs`). The settings records and their hashes are in
  `stilt.identity`. The hashes of existing folders are unchanged.
- `TransportSettings`, `MetSettings`, `FootprintConfig.resolve()`, and the
  geometry specs' `build()` are removed. `Mesh.from_spec(spec)` reads a
  geometry spec. `FootprintConfig.geometry_hash` is gone; the hash is on the
  `Variant` and in each footprint's settings (`foot.stilt.geometry_hash`).
- `MetConfig.directory` is optional on the class, and a project requires it
  for every met when its config loads, so a met read back from a folder's
  settings validates. A project also checks `subgrid_dir` for cropped local
  mets, which `MetConfig` alone no longer does.
- `Output.particles(variant)`, `Output.footprints(variant)`,
  `Output.find_particles(variant)`, and `Output.find_footprints(variant)`
  take a `Variant`. `Footprints.hash_for` is `stilt.identity.footprint_hash`.
- `stilt.config` and `stilt.receptors` import nothing above them, checked by
  an import-linter contract.
- `stilt.receptors` is a package: `models` (the receptor types and their
  ids), `table` (the receptor table and CSV files), and `validation` (the
  checks a receptor must pass, now written once and shared by the models
  and the table). Everything `stilt.receptors` exported is still exported
  from it.
- `Mesh`, `Zones`, `Geometry`, `overlap_weights`, and `check_resolution` are
  in `stilt.footprint` (module `stilt.footprint.targets`), since only
  footprints are summed onto them. `stilt.Mesh`, `stilt.Zones`, and
  `stilt.Geometry` are unchanged; `from stilt.spatial import Mesh` becomes
  `from stilt.footprint import Mesh`. `stilt.spatial` keeps `Bounds`,
  `Grid`, and the CRS helpers, the values configuration is made of.
- `stilt.footprint` is a package: `gridding` (`calculate`), `aggregation`
  (`aggregate`, `jacobian`), `io` (the array and footprint files), and
  `accessor`. Everything `stilt.footprint` exported is still exported
  from it; only private helpers moved.
- **`stilt.flux` is `stilt.sampling`** (breaking for module imports). It
  samples any gridded field at points, a surface flux or a mole fraction.
  `sample_flux(flux, x, y, times)` is `sample_field(flux, x, y,
  times=times, fill_value=0.0)`. `fill_value` is the value for points
  outside the field and for missing cells, `NaN` by default.
  `particle_enhancement(particles, flux)` is
  `particles.stilt.enhancement(flux)`, the per-particle twin of
  `foot.stilt.enhancement(flux)`.
- `VerticalReference` is `stilt.receptors.VerticalReference` (it was
  `stilt.config.VerticalReference`), so `stilt.receptors` no longer
  imports `stilt.config`. `stilt.config.kmsl_from_vertical_reference` is
  removed; the HYSPLIT driver sets `KMSL` from the receptor itself.
- **`Mesh.to_grid` replaces `Grid.from_geometry`** (breaking).
  `Grid.from_geometry(mesh, ...)` becomes `mesh.to_grid(...)`, with the same
  arguments. `Zones.to_grid` hands off to its base, and returns a grid base
  as it is, since a footprint on that grid overlaps every zone exactly. A
  `Grid` no longer knows about meshes and zones. `Grid.min_cell_width`,
  `Zones.bounds`, and `Zones.min_cell_width` are removed; only the grid
  derivation read them.
- `stilt.observations.winds` is `stilt.observations.variograms`. The public
  names (`variogram`, `fit_variogram`, `VariogramFit`) are unchanged.
- The documentation site shows the latest release rather than `main`.
- `foot.stilt.plot.facet()` is xarray's faceted plot, in the footprint's
  own coordinates (`x`/`y` for a projected grid). Its keyword arguments go
  to `DataArray.plot`.
- `stilt.flux.horizontal_dims` is `stilt.spatial.horizontal_dims`, and
  `Grid.dims` gives a footprint's dimension names on that grid.
- `stilt.SpatialTarget` (it was another name for `stilt.Geometry`),
  `Grid.from_geometries`, and `Grid.resolution`.
- `stilt.output.Jacobian` is `stilt.footprint.Jacobian`. A Jacobian built
  for a footprint folder now also warns when the target mesh is not the one
  the footprint grid was chosen for, as `foot.stilt.aggregate` does.
- **`Grid.projection` is `Grid.crs`**, matching `Mesh.crs` (breaking for
  code that reads `grid.projection`). `config.yaml`, stored settings, and
  footprint files that say `projection` still load; PYSTILT writes `crs`.
  `Grid.from_geometry(projection=)` is `Grid.from_geometry(crs=)`. Existing
  footprint folders are still found.
- **`stilt.spatial`** (breaking for module imports). `Grid`, `Bounds`,
  `Mesh`, `Zones`, the CRS helpers, and the overlap weights are in one
  module, `stilt.spatial`, replacing `stilt.geometry` and
  `stilt.config.spatial`. `stilt.Grid`, `stilt.Mesh` and the other
  package-root names are unchanged; `from stilt.geometry import Mesh`
  becomes `from stilt.spatial import Mesh`. The geometry specs moved into
  `stilt.config.footprint` and are still exported from `stilt.config`.
  `is_longlat_crs` is `stilt.spatial.is_longlat`, the one test for
  longitude/latitude, so a grid with `projection: EPSG:4326` is now
  treated as longitude/latitude, as a mesh already was.
- **One `TransportParams` class** (breaking). `STILTParams` and the three
  classes it combined (`ModelParams`, the old `TransportParams`, and
  `ErrorParams`) are one class, `stilt.config.TransportParams`, with the
  same fields under the same names, so `config.yaml` and the run hashes do
  not change. Each field records the HYSPLIT file it goes to (or that
  PYSTILT uses it), and `stilt.config.params.fields_in(file)` lists them.
  The file builders moved to `stilt.transport.hysplit.driver` as functions:
  `params.setup_entries()` is `setup_entries(params)`, and so are
  `setup_seed`, `ziscale_factors`, `zicontroltf`, `winderr`, `zierr`, and
  `winderrtf`.
- **`stilt run` always waits, and `stilt submit` returns** (breaking for
  scripts). They now match `project.run()` and `project.submit()`.
  `stilt run --backend slurm` submits the job array and waits for it; the
  new `stilt submit` submits it and returns. `--wait` is gone.
- **`timeout` and `keep_scratch` move under `execution:`, and `rm_dat` is
  removed** (breaking). They change no result and say how runs are carried
  out. A successful run's working directory is removed anyway, so `rm_dat`
  had an effect only with `keep_scratch`, which now keeps the directory as
  HYSPLIT left it. Move the two keys in `config.yaml`:

  ```yaml
  execution:
    timeout: 900
    keep_scratch: true
  ```

  A transport model's `run()` takes `timeout=`.
- **One name for each part of a simulation** (breaking). `sim.params`,
  `sim.footprint_config`, and `sim.receptor_id` are removed. Use
  `sim.variant.transport`, `sim.variant.footprint`, and `sim.receptor.id`.
- **A selection of simulations loads its own results**
  ([#107](https://github.com/jmineau/PYSTILT/issues/107); breaking).
  `project.simulations` is a `Simulations`: the table plus the project it
  came from. Select it as before, then ask the selection:

  ```python
  july = sims[(sims.variant == "hrrr") & sims.time.between(a, b)]
  july.status()
  july.load_footprints()      # {simulation id: DataArray}
  july.load_particles()       # one table, with receptor and variant columns
  for sim in july: ...
  ```

  `project.status()`, `incomplete()`, `load_particles()`,
  `load_footprints()`, and `jacobian()` are gone; call them on
  `project.simulations` or a selection. `jacobian(target, time_bins)`
  takes a selection of one variant. A selection understands columns and
  row masks; for any other pandas operation use `sims.frame`, and
  `stilt.Simulations(project, frame)` makes a selection from a table again.
  `Particles.table()` reads many receptors' particles at once, like
  `Footprints.table()`.
- **Particles and footprints are plain data**
  ([#107](https://github.com/jmineau/PYSTILT/issues/107); breaking).
  `sim.particles` is a pandas DataFrame and `sim.footprint` an xarray
  DataArray, so pandas and xarray work on them directly with no `.data`
  step. PYSTILT's own methods are under `.stilt`:
  `foot.stilt.aggregate(...)`, `foot.stilt.enhancement(flux)`,
  `foot.stilt.receptor`, `foot.stilt.plot.map()`,
  `particles.stilt.endpoints()`, `particles.stilt.plot.map()`. The
  `Trajectories` and `Footprint` classes are gone. A footprint keeps its
  receptor id as the coordinate `foot.receptor`, so
  `xr.concat(feet, dim="receptor")` stacks footprints labelled by receptor.

  | Before | After |
  |---|---|
  | `traj.data`, `foot.data` | `sim.particles`, `sim.footprint` themselves |
  | `Footprint.calculate(...)`, `traj.footprint(config)` | `stilt.footprint.calculate(particles, receptor, config)`, or `sim.generate_footprint(config)` |
  | `Trajectories.from_particles(...)` | `stilt.particles.prepare(raw, receptor, params)` |
  | `Trajectories.from_parquet(path)`, `traj.to_parquet(path)` | `stilt.read_particles(path)`, `stilt.particles_metadata(path)`, `stilt.write_particles(...)` |
  | `Footprint.from_netcdf(path)`, `foot.to_netcdf(path)` | `stilt.read_footprint(path)`, `foot.stilt.to_netcdf(path)` |
  | `foot.integrate_over_time()` | `foot.sum("time")` |
  | `foot.time_range` | `foot.indexes["time"]` |
  | `traj.met_files` | `sim.met_files` |
  | `traj.endpoints()`, with `time` as a timestamp, `endpoint_age_min` and `run_time` | `particles.stilt.endpoints()`: each particle's last row, every column kept (`time` stays minutes since release, `datetime` is the timestamp) |
  | `stilt.trajectory.endpoint_rows(p)` | `p.stilt.endpoints()` |
  | `show_traj=`, `traj_cmap=`, ... in `sim.plot.map()` | `show_particles=`, `particles_cmap=`, ... |

  Footprint files in the output directory now record their own settings,
  grid included, so `stilt.read_footprint(path)` opens one without its
  folder. It reads NetCDF files too.
- **Exceptions share one base class, `stilt.StiltError`, and live in
  `stilt.exceptions`** ([#80](https://github.com/jmineau/PYSTILT/issues/80);
  breaking). `stilt.errors` is renamed `stilt.exceptions`, with no alias.
  Each exception still subclasses its builtin, so `except RuntimeError`
  and `except FileNotFoundError` keep working. `EmptyFootprintError` is
  renamed `EmptyFootprint` and is no longer a `RuntimeError`, since an
  empty footprint is a result rather than a failure. A missing HYSPLIT
  executable raises the new `HYSPLITNotFoundError` (a `FileNotFoundError`)
  in all three places it is checked; a platform with no bundled build used
  to raise `RuntimeError`. `FailureReason` and `identify_failure_reason`
  move to `stilt.transport.hysplit`, beside the driver whose log they read.
- **"Particles" is the one name for a simulation's particle table**
  ([#107](https://github.com/jmineau/PYSTILT/issues/107); breaking). The
  folder of particle files under one set of settings is now `Particles`
  (was `Run`), so "run" only means running something. The renames are:

  | Before | After |
  |---|---|
  | `sim.trajectories`, `sim.has_trajectory`, `sim.trajectories_path` | `sim.particles`, `sim.has_particles`, `sim.particles_path` |
  | `project.load_trajectories()` | `project.load_particles()` |
  | the `trajectory` column of `project.status()` | `particles` |
  | `stilt.trajectory` | `stilt.particles` |
  | `stilt.execution.run_trajectories` | `stilt.execution.run_particles` |
  | `Output.run()`, `Output.find_run()`, `Output.runs()` | `Output.particles()`, `Output.find_particles()`, `Output.particle_sets()` |
  | `Run.particles_path()`, `has_particles()`, `read_particles()`, `write_particles()` | `Particles.file()`, `has()`, `read()`, `write()` |
  | `Footprints.footprint_path()`, `Footprints.run`, `Footprints.run_key` | `Footprints.file()`, `Footprints.particles`, `Footprints.particles_key` |
  | `EmptyTrajectoryError`, `FailureReason.NO_TRAJECTORY_DATA` | `EmptyParticleOutputError`, `FailureReason.NO_PARTICLE_DATA` |

  A simulation is one receptor, so it no longer hands out the folders that
  hold other receptors' results (`sim.run`, `sim.footprints`). Its own
  file paths are still public.
- **Receptor ids name everything that makes a receptor distinct**
  ([#105](https://github.com/jmineau/PYSTILT/issues/105); breaking for
  column and MSL receptors). A column id gives its bottom and top
  (`202101150600_-112_40.5_X0-3000`), and heights above mean sea level end
  the id with `msl` (`_100msl`, `_X0-3000msl`, `multi_<hash>msl`). Before,
  two columns at one place and time shared an id, and so did an AGL and an
  MSL point at one height. Two projects sharing an output directory could
  then use each other's particles without a warning. Ids of AGL points and
  AGL multipoint receptors do not change. Each particle file now has a
  `receptor` column, so a scan of the `particles/` tree with pyarrow,
  DuckDB, polars, or R can tell receptors apart. Reading one file drops the
  column.
- **Results live in an output directory, not inside the project**
  ([#74](https://github.com/jmineau/PYSTILT/issues/74), design in
  [#67](https://github.com/jmineau/PYSTILT/issues/67); breaking). A
  project directory holds `config.yaml` and `receptors.csv`; its results go
  to the directory `output:` names (`./output` by default), which several
  projects can share. The directory has one tree per kind of result
  (`particles/`, `footprints/`, `logs/`), a `settings=<variant>-<hash>`
  folder per set of settings, and `date=YYYY-MM-DD` folders below, so each
  tree reads as one dataset. Particles are one Parquet file per receptor;
  footprints are stored sparse in float32, with an empty footprint as a file
  with no rows and its reason. Results of earlier versions under
  `simulations/by-id` are not read; there is no migration in the alpha.
- **A run is identified by its settings, not its variant name.** Two
  variants with the same transport settings share one HYSPLIT run per
  receptor and differ only in the footprint made from it, so `from:` is no
  longer needed and is rejected with advice. Editing a setting no longer
  raises `ConfigChangedError`: the next `stilt run` writes into a new
  folder beside the old one, and `stilt status` lists folders the config no
  longer uses. PYSTILT never deletes them.
- **HYSPLIT runs on scratch.** The working directory (`compute_root`,
  `PYSTILT_COMPUTE_ROOT`, or `$TMPDIR/pystilt/<project>`) is removed after
  a successful run and copied under the output directory's `scratch/`
  after a failed one; `keep_scratch: true` keeps every run's.
- Reading `sim.trajectories` or `sim.footprint` before the result is
  written raises `FileNotFoundError` instead of returning `None`, so the
  same object reads the result once the run lands. `None` now means only
  that the variant makes no footprint or that the footprint is empty. Check
  `sim.has_trajectory` / `sim.has_footprint` first while a run may still be
  going.
- **Receptors as a table.** `stilt.receptors.receptors_to_frame` returns
  receptors with a row per release point, and
  `stilt.receptors.receptors_from_frame` builds them from one.
- Every particle and footprint file records the hash of the settings it was
  made with and the PYSTILT version that wrote it (`stilt:hash` and
  `stilt:pystilt` in the Parquet metadata), so a file copied out of the
  output directory still says where it came from.
- `project.jacobian(target, time_bins, variant=...)` sums a variant's
  footprints onto a target in one pass, as a sparse matrix with
  labelled rows and columns (`stilt.output.Jacobian`). The time bins must
  be closed on the left, since a footprint time is the start of its hour;
  other bins raise a `ValueError`.
- **`VariantConfig` is composed, not flattened.** A resolved variant holds
  `transport` (a `TransportSettings`, whose hash names the run) and
  `footprint` (a `FootprintConfig`, or `None`) instead of sixty flat
  fields; `stilt_params()` and the `footprint` property on
  `FootprintConfig` are gone. The flat surface of `config.yaml` is
  unchanged.
- **`Simulation` is a value: receptor, variant, and output directory.** It
  reads results and reports completion, and runs nothing. HYSPLIT runs and
  footprint writing live in `stilt.execution` (`run_trajectories`,
  `write_footprint`, `run_simulation`); `Simulation.generate_footprint`
  calculates without writing.
- **Loading a config no longer reads its `geometry`**
  ([#64](https://github.com/jmineau/PYSTILT/issues/64)). The mesh is built
  when the variants are resolved (`ProjectConfig.resolve_variants`), once per
  geometry however many variants inherit it, instead of in validation on
  every load. A worker can load a config whose geometry file it cannot
  read. `FootprintConfig(geometry=...)` no longer fills in `grid` and
  `geometry_hash` itself; call `FootprintConfig.resolve()`, which
  `Footprint.calculate` also does. Geometry specs keep their built mesh as
  `spec.mesh`. The derived grid is no longer written to `config.yaml` by
  `to_yaml`. A variant can no longer change part of a grid derived from the
  default geometry (`grid: {xres: 0.1}`); set `cells_per_target` or give a
  full grid. Footprint folder hashes are unchanged.
- `TransformContext.store` is now `TransformContext.directory`, the project
  directory that relative file names in transform settings are taken from.
- `Simulation.trajectories_path`, `footprint_path`, and `log_path` point
  into the output directory and are `None` before the run's folder exists.
  `generate_footprint` with other settings writes to its own folder and no
  longer replaces `sim.footprint`
  ([#65](https://github.com/jmineau/PYSTILT/issues/65)).
- **One wheel per platform**
  ([#61](https://github.com/jmineau/PYSTILT/issues/61)). PYSTILT used to
  publish one `py3-none-any` wheel that held both HYSPLIT builds, so pip
  installed it anywhere. It now publishes a Linux x86-64 wheel
  (`manylinux_2_17_x86_64`) and an Intel macOS wheel (`macosx_11_0_x86_64`),
  each with only its own `hycs_std`. On any other platform, Apple Silicon
  with an arm64 Python included, pip installs the source archive, which has
  no HYSPLIT binary. PYSTILT imports, and a run raises the "no bundled
  HYSPLIT binary" error until `exe_dir` names your own build. Maintainers
  build all three with `just dist`.
- **Met naming** (breaking). In a `mets` entry, `source:` is now
  `download:` and `backend:` is now `download_from:`, so the name says what
  it does and `backend` means only the execution backend. `MetStream` is now
  `Met`, and `MetConfig.source_kwargs` is `download_options`. A named entry
  under `mets` is called a *met* throughout the docs.
- **Requires arlmet 0.1.0b1**, which renamed its download sources to
  archives. The `cloud` extra installs `arlmet[archives]` in place of `s3fs`,
  and `fsspec` is no longer a core dependency. PYSTILT 0.1.0a22 does not work
  with arlmet 0.1.0b1: pin `arlmet<0.1.0b1` if you stay on it.
- **Status and run planning list folders instead of checking files**
  ([#60](https://github.com/jmineau/PYSTILT/issues/60)). `stilt status`,
  `project.incomplete()`, and the planning step of `stilt run`
  read which receptors are done from a listing of the date folders the
  selection falls in, rather than checking two files per simulation.
  `Run.receptors()` and `Footprints.receptors()` take `among=` to check
  only some receptors. `stilt status` no longer
  opens every footprint file. `status()` still reports `empty`, which needs
  the file.
- **HYSPLIT sits behind a transport model interface**
  ([#87](https://github.com/jmineau/PYSTILT/issues/87), decision 8 of
  [#67](https://github.com/jmineau/PYSTILT/issues/67)). The worker runs a
  simulation through `stilt.transport.get_model(name)`, where the name is
  the one the run's settings record. `stilt.transport.hysplit.HysplitModel`
  is the one transport model. Each model is a subpackage of
  `stilt.transport`, so `stilt.hysplit` is now `stilt.transport.hysplit`.
  HYSPLIT now reads the meteorology files where they are (the cropped
  copies when a met is cropped) instead of through links made in each run's
  working directory, so a kept `scratch/` folder no longer has a `met/`
  folder. `Met.stage_files_for_simulation` is replaced by
  `Met.files`.
- **Slurm runs go through submitit; the queue and Kubernetes are removed**
  ([#87](https://github.com/jmineau/PYSTILT/issues/87), decision 7 of
  [#67](https://github.com/jmineau/PYSTILT/issues/67); breaking). A Slurm
  run is one job array of batches of receptors, submitted with
  [submitit](https://github.com/facebookincubator/submitit) (a new
  dependency). Each task runs with the Python that submitted it, so `setup:`
  no longer has to activate an environment, and a task that is preempted or
  runs out of time is submitted again and skips what it finished. Logs and
  submission files are in `slurm/<date_time>_<id>/` in the project;
  `chunks/` is gone. `project.run()` waits for the job and
  `project.submit()` returns the submitit jobs at once.
- **`execution:` settings are checked**
  ([#63](https://github.com/jmineau/PYSTILT/issues/63); breaking). The
  section is now `backend`, `n_workers`, `cpus`, `time`, `mem`, `partition`,
  `account`, `qos`, `array_parallelism`, `setup`, and `slurm`. A setting it
  does not have is an error instead of being ignored or passed to `sbatch`.
  Move any other `sbatch` option, such as `exclude` or `requeue`, under
  `slurm:`, and rename `cpus_per_task` to `cpus`. `n_workers` is the number
  of Slurm array tasks, and `cpus` the receptors each task runs at once. A
  local run is one task, so its processes are now set by `cpus` (and
  `stilt run --cpus`), not `n_workers`.
- **`Project` replaces `Model`; a project's receptors and simulations are
  tables** ([#99](https://github.com/jmineau/PYSTILT/issues/99), step 5 of
  [#67](https://github.com/jmineau/PYSTILT/issues/67); breaking).
  `stilt.Project.init(path, config=..., receptors=..., **settings)` makes a
  project: it writes `config.yaml` once and refuses a directory that has
  one. `stilt.Project(path)` opens it and only reads. To change a setting,
  edit `config.yaml`; PYSTILT never rewrites it. `project.add_receptors()`
  appends to `receptors.csv` and replaces `register()`.
  `project.receptors` and `project.simulations` are DataFrames, one row per
  receptor and one per receptor and variant, selected with pandas
  (`sims[sims.variant == "hrrr"]`, `sims[sims.site == "WBB"]`).
  `project.status(sims)`, `project.incomplete(sims)`,
  `project.load_trajectories(sims)`, and `project.load_footprints(sims)`
  take such a selection; `project.receptor(id)` and
  `project.simulation(id, variant)` return one object. Receptors are built
  only when asked for: `receptors.csv` is read and checked as a table
  (`stilt.receptors.receptor_rows`), so opening a project of 117 000 point
  receptors takes 1.3 s instead of 6.6 s, and one of 2.5 million rows of
  multipoint receptors 18 s instead of 53 s. Ids are unchanged. `project.run()`
  blocks on either backend and returns the receptor results;
  `project.submit()` sends the work to Slurm and returns the submitit jobs.
  `ModelConfig` is now `ProjectConfig`. `Model` (its module, `stilt.model`,
  now holds the transport model interface), `stilt.collections` (`SimulationCollection`, `ReceptorCollection`,
  `OutputCollection`), `stilt.execution.register`, and the run handles
  (`JobHandle`, `LocalHandle`, `SlurmHandle`) are removed. Pass the scratch
  directory to the run (`project.run(compute_root=...)`,
  `stilt run --compute-root`, or `PYSTILT_COMPUTE_ROOT`);
  `stilt.execution.resolve_compute_root` says where that is.
- **Receptors are frozen pydantic models**
  ([#86](https://github.com/jmineau/PYSTILT/issues/86),
  [#66](https://github.com/jmineau/PYSTILT/issues/66); breaking).
  `PointReceptor`, `ColumnReceptor`, and `MultiPointReceptor` take their
  arguments by name only (`PointReceptor(time=..., longitude=...,
  latitude=..., altitude=...)`); `Receptor.from_points` still takes a list
  of points. They can no longer be changed in place: use
  `receptor.model_copy(update={...})`, and pass labels as `attrs=` instead
  of assigning `receptor.attrs`. An unknown argument is a `ValidationError`.
  A receptor is no longer iterable and has no `len()`: `receptor.coords()`
  lists its `(lat, lon, alt)` points. `receptor.id` and `receptor.location_id`
  are plain strings; `ReceptorID` and `LocationID` are removed
  (`stilt.receptors.parse_receptor_id` splits an id into its time and
  location). `to_dict()` names the type under `kind` (`"point"`); dicts
  stored by earlier versions still load. `MultiPointReceptor.longitudes`,
  `latitudes`, and `altitudes` are tuples.
- **The id of a multipoint receptor covers its heights to 0.01 m**
  ([#50](https://github.com/jmineau/PYSTILT/issues/50)). Heights used to be
  cut to whole metres in the id, so two receptors that differed by less
  than a metre shared one and the second was dropped without a warning.
  Ids of receptors with whole-metre heights are unchanged. Two different
  receptors that still share an id are now refused when receptors are read
  or added.

### Removed

- `Simulation.outcome` and `stilt.transport.hysplit.identify_failure_reason`,
  which guessed why a run failed from phrases in a log shared by every
  variant on the same particles. `sim.failure` replaces them (#134).

- `stilt.particles.prepare`. It added a `datetime` column that nothing on
  the way to the particle file or the footprint read; `read_particles`
  adds `datetime` when it reads (#134).
- `BuiltinTransform` and `KERNEL_TABLE_COLUMNS` from
  `stilt.transforms.__all__` (#134).

- `stilt.observations.slant`. `slant_points`, `pressure_altitudes`, and
  `jitter_points` are in `stilt.observations.placement`; import them from
  `stilt.observations` as before.
- The single-value `temperature` of `pressure_altitudes`. Pass one
  temperature per pressure level, or none for the standard atmosphere.
- `stilt.observations.particle_background`, `endpoint_weights`, and
  `fill_missing`. `background(...).per_particle` is the field at each
  endpoint, and `background(...).weights` the weights.
- `stilt.project.project_slug`, now a private helper of the runner, which
  names Slurm jobs with it. `run_receptors` requires `compute_root`, as
  `resolve_compute_root` returns it.
- `stilt.execution.ReceptorResult`. `project.run()`, `stilt.execution.run`,
  and `run_receptors` return a flat list of `SimulationResult`, receptor by
  receptor; `run_receptor` returns that receptor's list.
- `stilt.execution.write_footprint` is `make_footprint`, and takes no
  `config=` or `transforms=`: it makes the variant's own footprint.
  `sim.generate_footprint` makes footprints with other settings.
  `stilt.footprint.write_footprint` writes a footprint file.
- `stilt.particles.prepare` no longer adds the release height `xhgt`. The
  HYSPLIT model adds it before returning its particles
  (`stilt.transport.hysplit.release.add_release_heights`), since it depends
  on how HYSPLIT orders particles. Call that function first when preparing
  particles read straight from HYSPLIT's output.
- `stilt.config.hysplit_version` is `stilt.transport.hysplit.model.hysplit_version`.
  `TransportSettings.build` asks the transport model for its version.

- The PostgreSQL work queue (`stilt.service`, `PYSTILT_DB_URL`,
  `stilt register`, `stilt pull-worker`, `stilt serve`), the Kubernetes
  backend and manifests, `stilt push-worker` and chunk files, and the
  `Executor` classes (`LocalExecutor`, `SlurmExecutor`, `KubernetesExecutor`,
  `get_executor`). The `cloud` extra no longer installs `gcsfs`, `psycopg`,
  or `kubernetes`. Closes the scope of
  [#4](https://github.com/jmineau/PYSTILT/issues/4),
  [#6](https://github.com/jmineau/PYSTILT/issues/6),
  [#7](https://github.com/jmineau/PYSTILT/issues/7),
  [#55](https://github.com/jmineau/PYSTILT/issues/55), and
  [#56](https://github.com/jmineau/PYSTILT/issues/56).
- `simulations/variants.yaml`, `ConfigChangedError`, `Model.check_config`,
  `Model.remove`, `Model.orphans`, `stilt rm`, the `.empty` marker,
  `Simulation.publish`, `Simulation.parent`, `stilt.store` and object-store
  project roots (`s3://`, `gs://`), `RuntimeSettings.cache_dir`, and
  `VariantConfig.derived_from` / `record`.
- `ConfigValidationError`, which nothing raised any more
  ([#80](https://github.com/jmineau/PYSTILT/issues/80)).
- `stilt.execution.sigterm_as_interrupt`, now private. The worker uses it to
  clean up when Slurm stops a task.
- `validate_vertical_reference`, `Met.files`, `ErrorParams.error_enabled`
  (use `winderrtf > 0`), `ProjectConfig.footprint` (use a variant's
  `footprint`), `VariantConfig.realization` (use
  `variant.transport.realization`), and `STILTParams.realization_seed`.
- `read_footprint(config=)`. A footprint file records its own settings,
  and the output folder reads it like any other file. Footprint files
  written before files recorded their settings need them added first.
- `Particles.find_footprints(config)`. `Output.find_footprints` now takes a
  variant: `output.find_footprints(variant)`.
- `Simulation.is_backward` (`sim.time_range` gives the period either way)
  and `Particles.footprint_sets()` (filter `Output.footprint_sets()` by
  `particles_key`).
- Reading receptors stored with their kind under `"type"`, as particle
  and footprint files from the old `simulations/by-id/` layout do.
  `Receptor.from_dict` needs the `kind` key.
- `RuntimeSettings` and the `pydantic-settings` dependency. The runner reads
  `PYSTILT_COMPUTE_ROOT` itself. An empty `PYSTILT_COMPUTE_ROOT` now means
  unset rather than the current directory.

### Fixed

- Relative paths depended on the working directory: a relative met
  `directory` or `subgrid_dir` was found from wherever Python started, and
  a relative averaging-kernel `table` too unless `directory=` was passed.
  They now start from the project directory, and met directories expand
  `$VARIABLES` like the other paths (a behaviour change for a relative met
  directory that worked only from the right working directory).
  `Simulation` holds its project directory, so `sim.generate_footprint()`
  and `make_footprint` find a kernel table without a `directory=`
  argument, which they no longer take. An absolute `output` is resolved,
  so one directory reached through a link is one `Output` (#148).
- `project.run(execution=...)` and `project.submit(execution=...)` passed
  only `cpus` to the workers, which read `timeout` and `keep_scratch` back
  from `config.yaml`, so an override of either was ignored. `Batch` and
  `run_receptors` now take the `ExecutionConfig` in place of `cpus` and
  `n_cores` (breaking for code that calls them directly) (#148).
- `foot.stilt.enhancement` took the grid axes by position, so a footprint
  with its dimensions in another order gave a wrong enhancement (zero in
  a test), and a footprint summed over time raised a `KeyError`. It now
  works by dimension name, gives the total for a time-summed footprint,
  and raises when the flux varies in time and the footprint does not.
  `foot.stilt.aggregate` returned zeros for a footprint without a `time`
  dimension; it now raises (#148).
- `Simulation.outcome` reported `failed:UNKNOWN` for a variant whose
  footprint was never made when another variant on the same particles had
  run. Its replacement, `sim.failure`, reads the variant's own record
  (#134).

- Downloaded meteorology now includes the next file when the receptor is in
  the last hour of a file, as local meteorology and STILT-R do. HYSPLIT
  interpolates the release time between two hours, so a backward run at,
  for example, 17:30 with 6-hour HRRR files needs the 18:00 file (#134).
- A run that fails before writing anything, such as on missing
  meteorology, no longer leaves an empty copy of its directory under the
  output directory's `scratch/` (#134).

- A grid derived for zones over a projected grid covered a box near the
  equator, because the base grid's longitude/latitude bounds were read as
  projected coordinates. Deriving one for a `Grid` raised `AttributeError`.
- `foot.stilt.aggregate` and `jacobian` warned that zones made of the
  footprint grid's own cells were under-resolved, although the overlap is
  exact.
- When HYSPLIT fails for a receptor, the other variants that share its
  transport settings fail with the same error instead of running HYSPLIT
  again. A receptor with N such variants no longer runs, or times out,
  N times.
- A footprint with a single hourly layer that is not the receptor's own
  hour, such as a backward run whose particles all leave the grid within
  the first hour, is stamped at that hour, as in STILT-R. It was stamped at
  the receptor time.
- A calculated footprint's coordinates are rounded as a stored one's are
  when read back, so the two line up (`sim.generate_footprint() -
  sim.footprint` no longer misaligns some cells).
- A run that ends without particles reports `failed:NO_PARTICLE_DATA` in
  `sim.outcome` and `stilt status`. PYSTILT's own errors for it were not
  recognised and came out as `failed:UNKNOWN`.
- `foot.stilt.plot.map()` and `facet()` work on a footprint on a projected
  grid (`x` and `y`). They read `lon` and `lat` and raised.
- A `config.yaml` loads on a machine where its `exe_dir` is not reachable.
  Loading checks the variants without building them; the HYSPLIT build is
  needed only once the variants are resolved.
- `ziscale: [[0.8, 0.9]]` (STILT-R's nested form) and `ziscale: [0.8, 0.9]`
  are now the same run. They hashed differently before. Existing output is
  still found.
- Changing a met's `download_from` or `n_min` no longer makes every run
  look new. Neither changes the particles, so neither is part of a run's
  hash. Existing output is still found.

### Added

- `stilt.run_trajectories(receptor, met, **params)` runs the transport
  model for one receptor without a project and returns its particles, as
  STILT-R's `calc_trajectory` does. With `stilt.calc_footprint`, a
  footprint takes two calls (#150).

- `Simulation.settings`: the run and footprint settings a simulation's
  results are made with, as the output folders record them (#134).

- `Receptor.to_json()` and `Receptor.from_json()`, the form result files
  record a receptor in (#134).

- **`data_dir`**: a folder of HYSPLIT data tables (`ASCDATA.CFG`,
  `LANDUSE.ASC`, `ROUGLEN.ASC`, `TERRAIN.ASC`) to use in place of the
  bundled ones, as `exe_dir` does for `hycs_std`. A table the folder does
  not hold comes from the bundled set. Each table that differs from the
  bundled one is recorded by its SHA-256 in the run's settings
  (`ModelInfo.data_files`) and is part of its hash; runs on the bundled
  tables keep their hashes. Transport models report these with
  `data_files(params)`.

- **`TransportSettings`, what identifies a run**
  ([#74](https://github.com/jmineau/PYSTILT/issues/74)). The transport
  fields that change a run's particles, the settings of its meteorology
  (`MetSettings`, which `MetConfig` now builds on), and the transport
  model and version that produced them (`ModelInfo`). Its hash names the run's folder
  in the output directory, and a stored `settings.yaml` loads back through
  it, so a field added later with a default still matches. A custom
  `exe_dir` needs a `version` file beside `hycs_std`. Nothing uses this yet
  beyond `stilt.output`.

### Removed

- **Python 3.10 is no longer supported** (breaking). PYSTILT now requires
  Python 3.11 or newer; 3.10 reaches end of life in October 2026. The
  `typing-extensions` dependency is gone, and CI tests 3.11 through 3.14
  ([#62](https://github.com/jmineau/PYSTILT/issues/62)).

### Fixed

- **A run on a damaged met file fails instead of counting as done**
  ([#98](https://github.com/jmineau/PYSTILT/issues/98)). When a met file
  holds only one time period, HYSPLIT warns and stops the particles where
  the met runs out. The run used to write the short trajectories and its
  footprint, which then missed every hour after that point. Now, when the
  warning is in the log and no particle reaches the end of the run, the
  run fails with the reason `MET_TRUNCATED` and is run again next time.
  The warning alone is not a failure, since the damaged file may cover
  hours the particles never reach.
- **`stilt init` writes a receptors file that loads**
  ([#49](https://github.com/jmineau/PYSTILT/issues/49)). The starter
  `receptors.csv` held a `# Example: ...` line, which the reader parsed as
  a receptor, so the first `stilt run` after adding a row failed. The file
  now holds only the header, and `stilt init` prints the example instead.
- **Machines without a bundled HYSPLIT build get a clear error**
  ([#61](https://github.com/jmineau/PYSTILT/issues/61)). The bundled
  binary was chosen by operating system only, so an aarch64 Linux machine
  was handed the x86-64 build and failed with an operating-system error on
  the first run. The architecture is now checked too, and the error says to
  set `exe_dir` to your own `hycs_std` build (it used to say `PATH`, which
  PYSTILT never reads). Apple Silicon Macs keep using the macOS build
  through Rosetta.
- **HYSPLIT settings take the types HYSPLIT reads**
  ([#57](https://github.com/jmineau/PYSTILT/issues/57)). `rhb`, `rht`, and
  `tout` are whole numbers, so a value such as `rhb: 80.5` is rejected here
  instead of stopping HYSPLIT with a namelist error. `delt`, `dxf`, `dyf`,
  `hscale`, `p10f`, `qcycle`, `splitf`, `vscale`, `vscales`, `vscaleu`,
  `wbbh`, `wbwf`, and `wbwr` now accept fractions such as `delt: 0.5`. The
  default `SETUP.CFG` is unchanged.
- **Cropped meteorology is cached per crop and written safely**
  ([#53](https://github.com/jmineau/PYSTILT/issues/53)). Crops of your own
  met files were found by file name only, so changing `subgrid_bounds`,
  `subgrid_buffer`, or `subgrid_levels` silently reused the old crop. Each
  crop now goes in its own folder inside `subgrid_dir` (`Met.crop_dir`),
  named by a hash of the crop box and levels. A crop is written to a
  temporary name and renamed into place, so parallel workers never read a
  half-written file, and its file handle is now closed. Breaking:
  `subgrid_dir` is required when cropping your own files (it used to
  default to a folder inside the met archive), crops saved by earlier
  versions are not found (move them into `crop_dir` to keep them).
  `subgrid_levels` now applies to downloaded files too (it was silently
  ignored), which needs arlmet 0.1.0a9.
- **Two workers can start the same run at once**
  ([#90](https://github.com/jmineau/PYSTILT/issues/90)). Workers that created
  a run folder at the same moment shared one temporary file name for its
  settings, so one of them failed its first receptor with a
  `FileNotFoundError` on `_settings.tmp`. Every writer now uses its own
  temporary name, for settings, particles, footprints, and the
  `to_parquet` / `to_netcdf` exports.
- **Appending an MSL receptor keeps its altitude reference**
  ([#51](https://github.com/jmineau/PYSTILT/issues/51)). Adding a receptor
  with heights above sea level to a `receptors.csv` that had a plain
  `altitude` column and no `altitude_ref` column wrote it without its
  reference, so it read back as above ground. The column is now added, with
  `agl` on the existing rows.
- `model.plot.availability(ax=...)` formats the dates on the figure of the
  axes you pass. It used to format whichever figure was current
  ([#58](https://github.com/jmineau/PYSTILT/issues/58)).
- **`Footprint.aggregate` rejects time bins not closed on the left**
  ([#58](https://github.com/jmineau/PYSTILT/issues/58)). Every bin was
  summed as if closed on the left, whatever its `closed` said. A bin from
  `pd.interval_range`, which is closed on the right by default, took the
  hour at its start and left out the hour at its end. Such bins now raise
  a `ValueError`. Build them with `closed="left"`, as the docs now do.

## [0.1.0a22] - 2026-09-29

This release simplifies the code before a larger redesign
([#48](https://github.com/jmineau/PYSTILT/issues/48)). Most entries remove
a second way of doing something. Items marked breaking change a public
name or signature.

### Fixed

- **Unknown keys in a met entry are an error**
  ([#52](https://github.com/jmineau/PYSTILT/issues/52)). A misspelled key
  such as `subgrid_enabel` was accepted and ignored. A met without a
  `source` now rejects any unknown key. With a `source`, the extra keys
  must be options that arlmet source takes, such as `domain` for `nams`.
  A `config.yaml` with such a typo no longer loads until it is fixed.
- **Every footprint applies and records the same transforms**
  ([#54](https://github.com/jmineau/PYSTILT/issues/54)). `Footprint.calculate`
  and `Trajectories.footprint` now apply `config.transforms` before
  gridding. Before, only `Simulation.generate_footprint` applied them, yet
  every written footprint listed them. `generate_footprint` makes its
  footprint through `Trajectories.footprint` and records its extra
  `transforms` along with the variant's. Both take a `context` for the
  transforms.
- **Ctrl-C and SIGTERM stop a local run cleanly.** `LocalExecutor` ran
  receptors on a background thread that `Model.run` then waited on.
  Signals reach only the main thread, so an interrupt ended the process
  without the worker's `interrupted` handling. `LocalExecutor.start` now
  runs the receptors itself, so `model.run(wait=False)` no longer returns
  early for a local run.
- **A relative directory such as `runs/a` is made absolute**
  ([#58](https://github.com/jmineau/PYSTILT/issues/58)). Only a bare name
  was, so a worker started from another directory looked in the wrong
  place.
- **`sample_flux` wraps longitudes** to the flux field's convention, so a
  flux on a 0 to 360 grid is no longer read as zero for negative
  longitudes.

### Changed

- **One footprint settings class.** Breaking. `FootprintParams` is merged
  into `FootprintConfig`, whose `grid` is now optional. `ModelConfig` and
  `VariantConfig` inherit it. `Footprint` raises `ValueError` for settings
  without a grid. `FootprintConfig.replace(...)` is gone; use
  `config.model_copy(update={...})`.
- **A transform named by import path must be an importable pydantic
  model.** Breaking. `load_transform` raises `ImportError` for a class it
  cannot import and `TypeError` for one that is not a pydantic model, so a
  config naming either fails to load. `UnresolvedTransform` is removed.
  `Footprint.from_netcdf` still reads a footprint that recorded such a
  transform: it warns and keeps that entry of `config.transforms` as its
  settings mapping.
- **`MetStream` takes its `MetConfig`.** Breaking. `MetStream(name, config)`
  replaces the thirteen keyword arguments and `MetStream.from_config`, and
  the settings are read from `stream.config`. `MetID` is removed; met names
  are plain strings, checked by `ModelConfig`.
- **`ConfigValidationError` is a `ValueError`.** Breaking. It and
  `ConfigChangedError` no longer subclass `SimulationError`, which marks a
  failed run.
- **`HYSPLITFailureError` names the log.** Breaking. It takes the HYSPLIT
  log path in place of an optional `sim_id`, and its message says which
  failure was found and where the log is.
- **The GGG readers need the species.** Breaking. `read_ggg_oof` defaulted
  to `xch4` and `read_ggg_netcdf` and `read_tccon` to `xco2`. Now each call
  names it, so none silently reads another gas. `read_tccon` is an alias of
  `read_ggg_netcdf`.
- **`Model(receptors="file.csv")` reads the file at once** and
  `Model.register` writes the receptors out, with extra columns kept as
  `attrs`. It no longer copies the file byte for byte.
- **`FirstOrderLifetime` reads particle `time` in minutes.** Breaking. Its
  `time_column` and `time_unit` settings and `HOURS_PER` are removed.
- **`pyproj` is a required dependency**, as it already was through arlmet.
  The `projection` extra is removed.
- **`sample_field` and `vertical_dim` live in `stilt.flux`**, next to
  `sample_flux`, which is `sample_field` with missing values as zero.
- **`stilt run`** builds its executor once and waits only for a submitted
  Slurm or Kubernetes job. Its banner reads "Execution mode: local, one line
  per receptor as it finishes" for a local run.

### Removed

Code with no callers, or a second way to do the same thing. Breaking where
public.

- xarray-grid and `(x, y)`-list targets of `Footprint.aggregate`, and its
  `resolution` argument. Pass a `stilt.Grid`, `stilt.Mesh`, or
  `stilt.Zones`; anything else raises `TypeError`. To keep only some cells
  of a grid, select rows of the result or use `Zones`.
- The `regression` option of `observations.transport_error`, which
  reproduced X-STILT's upward-biased fit through the positive levels.
- The `backend` argument of `overlap_weights`. exactextract is still used
  when it is installed.
- `OutputCollection.missing()`. Use `simulations.incomplete()` or
  `simulations.status()` to see what has not run.
- `ReceptorCollection.source_path` and the `source` argument of
  `Project.add_receptors`.
- The `n_workers` argument of `Executor.start`. Each executor takes
  `n_workers` when it is built. `LocalHandle.done`.
- `LocalStore.path`, which returned the same path as `local_path`. `Store`
  is no longer `runtime_checkable`.
- `VariantConfig.differences()`, `MetConfig.differences()`, and
  `MetConfig.record()`. `Model.check_config` compares the record itself.
- `FootprintParams.FIELDS`, `PostgresQueue.db_url`, the
  `SIMULATION_LOG_FILENAME` and `SIMULATION_MET_DIRNAME` constants of
  `stilt.project`, and `stilt.__author__` and `stilt.__email__`.

## [0.1.0a21] - 2026-09-29

### Fixed

- **Pressure weighting of multipoint and slant receptors**
  ([#46](https://github.com/jmineau/PYSTILT/issues/46)). All particles
  released from one point share a release height, so the slab between
  neighbouring release pressures had zero width and only one particle per
  point carried any weight. Each distinct release height now gets its slab
  and the particles released there split it evenly, so the weighted
  footprint no longer depends on how many particles each point released.
- **Pressure weighting of MSL receptors**
  ([#47](https://github.com/jmineau/PYSTILT/issues/47)). The hypsometric
  fit was made against height above ground and evaluated at release
  heights above sea level, so every release pressure of a receptor with
  `altitude_ref="msl"` came out too low by the terrain height.
  `PressureWeighting` now reads `altitude_ref` from the transform context
  and, for MSL receptors, fits against `zagl + zsfc` and closes the bottom
  slab at the terrain under the lowest point. Such receptors need `zsfc`
  in `varsiwant`, which the slant column guide already asks for;
  `particle_pwf` takes an `altitude_ref` argument.

## [0.1.0a20] - 2026-09-29

### Added

- **Reproducible runs and error realizations**
  ([#28](https://github.com/jmineau/PYSTILT/issues/28)). `seed` now does what
  it says: with `krand: 2` two runs with the same seed are bit-identical and
  different seeds give different turbulence and wind-error draws. HYSPLIT's
  generator re-initializes only from a negative namelist value and collapses
  every value `>= -1` onto one stream, so PYSTILT writes `SEED = -(|seed| + 1)`
  (`STILTParams.setup_seed`). Under `krand: 2` with a seed, error
  realization `k` runs with `seed + k` (`STILTParams.realization_seed`;
  realization 0 shares the main run's seed, as STILT-R's error run does),
  so several realizations no longer require `krand: 4`: the seeded route
  is reproducible, the clock-seeded one is not. The fidelity fixture writes
  the same mapped value on the STILT-R side.
- **Mixed-layer height section** in the Transport Error guide: `ziscale` as
  a shared, all-particle bracket on the mixed layer, one variant per factor,
  and why a column and a surface receptor respond differently. The X-STILT
  migration table now maps `get.zierr` to `ziscale`.
- **Radiosondes from IGRA2** in the Wind Error Statistics guide: a snippet
  that reads NOAA's Integrated Global Radiosonde Archive through siphon into
  the table the recipe starts from, so the recipe works for any sonde
  station. It replaces X-STILT's `grab.raob`. No new API or dependency.

### Changed

- **Empty footprints are recorded once**
  ([#59](https://github.com/jmineau/PYSTILT/issues/59)). Breaking. When no
  particle reaches the grid, the simulation writes only the
  `<rid>_foot.empty` marker, which now holds the reason. The zero-valued
  NetCDF that was written next to it is gone, and with it the `is_empty` and
  `empty_reason` attributes and the `Footprint.is_empty` property.
  `Footprint.calculate` raises `EmptyFootprintError` instead of returning
  zeros, `Simulation.generate_footprint` returns `None` and writes the
  marker, `sim.footprint` is `None`, and `sim.empty_reason` reads the
  marker. `status()` gained an `empty` column. Existing projects: delete the
  zero-valued `.nc` files that sit next to `.empty` markers.
- **Documentation rewritten in a plainer voice**
  ([#45](https://github.com/jmineau/PYSTILT/issues/45)). The user guide,
  getting-started pages, tutorials, migration pages, README, and public
  docstrings and config field descriptions were rewritten to read like the
  STILT-R docs, and checked against the code as they were rewritten. Many
  examples that no longer ran are fixed (both tutorials, the README
  quickstart, the slant-column guide). AGENTS.md has a new "Voice" section
  describing the style. The STILT-R migration page moved from
  `migration/r_stilt` to `migration/stilt_r`.
- **A project is receptors × variants**
  ([#37](https://github.com/jmineau/PYSTILT/issues/37),
  [#32](https://github.com/jmineau/PYSTILT/issues/32),
  [#33](https://github.com/jmineau/PYSTILT/issues/33)). Breaking. The
  top-level settings in `config.yaml` are defaults, and a new `variants`
  section names sets of overrides (`hrrr-zi08: {ziscale: 0.8}`); with no
  `variants` there is one per met, named after it. A simulation is one
  receptor under one variant, one HYSPLIT call, with id
  `<receptor_id>/<variant>` and outputs in
  `simulations/by-id/<receptor_id>/<variant>/`. What changes:
  - The wind-error run is a variant of its own (the error fields set on it)
    instead of a second run attached to every simulation.
    `error_realizations` becomes `realizations: N` on any variant, which
    runs it as `<name>-0 .. <name>-(N-1)` with `seed + k`. Changing the
    number of realizations no longer reruns anything else (#32), and every
    realization has its own trajectory and, if wanted, footprint (#33).
  - Footprint settings (`grid`, `smooth_factor`, `time_integrate`,
    `transforms`, `geometry`) are flat defaults like the transport ones, and
    each simulation has at most one footprint, `<receptor_id>_foot.nc`. The
    named `footprints` dict, the `grids` dict, `FootprintConfig.error` and
    `foot_names` are gone. `grid: null` means trajectory only. A variant with
    `from: <other>` makes another footprint from that variant's particles
    without running HYSPLIT.
  - `Model.register()` refuses to change the settings of a variant that is
    already in the project's `config.yaml` (`ConfigChangedError`), so outputs
    always match the config they are filed under. `register(allow_changes=True)`
    and `stilt register --force` override.
  - Collections: `model.simulations` is a selection over receptors ×
    variants with `sel(receptor=, variant=, time=, location=, where=)`,
    `incomplete()` and `status()` (a DataFrame); `.trajectories` and
    `.footprint` give one output over the selection. `TrajectoryCollection`,
    `FootprintCollection`, `model.footprints[name]` and the `ids` / `select` /
    `missing` / `paths` methods are removed.
  - `SimID` is a `(receptor, variant)` pair. `Simulation.footprint` and
    `footprint_path` take no name; `get_footprint`, `error_trajectories`,
    `all_error_trajectories` and `Trajectories.is_error` are removed (a
    trajectory's stored params say whether it was perturbed).
    `TransformContext` carries `variant` instead of `footprint_name` and
    `is_error`.
  - Workers are handed receptors: `run_receptor` / `run_receptors` /
    `pull_receptors` replace `run_simulations` / `pull_simulations`, and the
    queue and Slurm chunk files hold receptor ids. The HYSPLIT driver makes
    exactly one call per `execute()`. A failed realization now fails its
    simulation instead of being dropped into the log.
  - A simulation whose trajectory went missing is rerun even when its
    footprint exists; before, a missing error realization could never be
    backfilled once every footprint existed.
- **The settings a variant ran with are recorded apart from `config.yaml`**
  ([#38](https://github.com/jmineau/PYSTILT/issues/38)). `register()` writes
  the fully resolved settings of every variant to `simulations/variants.yaml`
  and compares against that record, so a setting changed in `config.yaml`
  with an editor is refused just like one changed in Python (before, the
  command-line path compared the file with itself and accepted anything).
  `config.yaml` and `receptors.csv` are now the user's files: a `config.yaml`
  loaded from the project is never rewritten, one given to `Model` in Python
  is written without defaults (it used to become a dump of every HYSPLIT
  setting), and receptors added later are appended to `receptors.csv` in its
  own columns, keeping `r_idx` and any extra column. `allow_changes` and
  `stilt register --force` are gone; the way to rerun a variant under the
  same name is `Model.remove(name)` / `stilt rm --variant NAME`, which
  deletes its outputs (and those of variants derived from it) and its record
  entry. `Simulation.delete()` does the same for one simulation.
  `stilt status` lists variants in the record that `config.yaml` no longer
  declares. `skip_existing` is no longer a config setting (it stays on
  `Model.run()`, `stilt run --no-skip`, and `stilt push-worker --no-skip`).
  A variant's `grid` override now updates the default grid field by field.
  Met settings that change results (`subgrid_*`, `n_min`, `file_tres`) are
  part of the record; `directory` is not.
- **Names and surface cleanup after variants**
  ([#40](https://github.com/jmineau/PYSTILT/issues/40)).
  - A variant that declares `realizations` is always a numbered group, even
    at `1` (`hrrr-err-0`), so raising the count later only adds simulations
    instead of renaming `hrrr-err` to `hrrr-err-0`.
  - `stilt init` writes a `variants:` section with one empty entry, since
    declaring any variant replaces the automatic one-per-met list.
  - `OutputCollection.paths()` and `load()` return dictionaries keyed by
    simulation id instead of bare lists.
  - One `status`: `Model.status()` returns the per-simulation table
    (`StatusCounts` is gone), `stilt status` prints totals and per-variant
    counts, and `Simulation.status` is now `Simulation.outcome`.
  - `sel()` raises `KeyError` for a receptor id or variant name the project
    does not have; the time, location, and predicate filters may still
    select nothing.
  - The worker's `complete-empty` status is gone: an empty footprint is
    `complete` with its `.empty` marker.
  - The Postgres queue column is `receptor_id` (it has held receptor ids
    since the variants change).
  - Removed: `Model.params`, `Model.trajectories` and `Model.footprint`
    (use `model.simulations.trajectories` / `.footprint`), `stilt run
    --config PATH`, receptors given as bare tuples (pass `Receptor`
    objects or a CSV path), and the unused `VARIANT_KEYS`.
- **`Simulation` takes its variant**
  ([#41](https://github.com/jmineau/PYSTILT/issues/41)).
  `Simulation(receptor, config, met=..., parent=..., directory=..., store=...)`
  takes the resolved `VariantConfig` instead of `meteorology`, `params`,
  `footprint` and `variant` separately; `sim.config` is the variant,
  `sim.params` and `sim.footprint_config` are read from it, and the
  `exe_dir` argument is gone (`STILTParams.exe_dir` already covers it).
  `expected_outputs()`, `has_output()` and `missing_outputs()` are replaced
  by `is_derived` and the `makes_footprint` property;
  `generate_footprint()` takes a `FootprintConfig`, not keyword arguments.
- **`stilt run` no longer polls the project for progress.** It used to walk
  every simulation's outputs every five seconds, which on a large project is
  minutes per poll over a network filesystem. The worker now logs one line
  per finished receptor, Slurm progress is the scheduler's, and the status
  summary is printed once at the end. Iterating a `SimulationCollection`
  is linear now (it was quadratic in the number of simulations). Removed:
  `Model.name` (use `model.project.name`), `uri_join`, and
  `Project.__fspath__`.
- **Cloud projects read their inputs and cache their outputs by key.**
  `FsspecStore.local_path` downloads a key into the cache directory under its
  own path instead of through fsspec's `simplecache`, and writing or deleting
  a key drops its cached copy; `config.yaml` and `receptors.csv` are read
  straight from the store. Before, a rewritten `receptors.csv` could be
  served stale from the cache for the life of the directory.
  Projects made with earlier versions need their outputs moved into the new
  layout (`by-id/<met>_<receptor>/` to `by-id/<receptor>/<met>/`, one
  footprint per simulation, the error parquet to a `<met>-err` variant); the
  maintainer can provide a conversion script on request.
- **`zicontroltf` is derived from `ziscale`** and is no longer a setting.
  The mixed-layer height is scaled whenever `ziscale` is not 1.0, the way
  `winderrtf` follows the wind-error parameters. A saved `config.yaml`
  that still has a `zicontroltf` line no longer loads: delete the line,
  and set `ziscale: 1.0` if it holds `0` (STILT-R's unset value and
  PYSTILT's old default). A `ziscale` of 0 is rejected.
- `krand` is checked against HYSPLIT's documented modes (0-4, 10-13). HYSPLIT
  does not validate it, and any other value silently makes every turbulence
  draw the same constant (or hangs). `seed` requires `krand: 2`: the bundled
  `hycs_std` discards the seed under `krand: 4` and 10-13 and uses it only
  for the initial turbulent velocity under `krand: 1`.
- **Receptor labels.** Columns of `receptors.csv` that PYSTILT does not use
  are kept on each receptor as `Receptor.attrs` and written back by
  `receptors_to_csv` / appended rows, so `sel(where=lambda r:
  r.attrs["scene"] == ...)` gathers a scene or a site without a second
  table.
- `config.yaml` written from a Python-built model always has a `variants`
  section (one empty entry per met when none was declared), each met holds
  only the fields that were given, and `mets` / `variants` come first.
  `register()` warns when the record holds variants `config.yaml` no longer
  declares. A variant named after a met stream uses it without `met:`.
  `stilt rm --variant` can be repeated.

### Fixed

- **Trajectories written by earlier versions did not load**
  ([#44](https://github.com/jmineau/PYSTILT/issues/44)). The params stored
  in a trajectory's Parquet metadata were validated strictly, so every file
  written while `zicontroltf` was still a setting failed with
  `extra_forbidden`. Stored params are a record of the run: settings this
  version no longer has are now skipped on read (the file keeps them).
- **A variant that raised `numpar` kept the default's `maxpar`.** `maxpar`
  was filled in from `numpar` when the defaults were built, and variants
  inherited the filled value, so `hrrr-np3k: {numpar: 3000}` over a default
  of 1000 wrote `MAXPAR = 1000` and HYSPLIT capped the run there. `maxpar`
  now stays unset unless given, and `SETUP.CFG` gets each variant's own
  `numpar`. The record stores the value HYSPLIT received, so a variant run
  under the old cap is reported as changed.
- **A variant that set `geometry` was rastered on the inherited grid**
  ([#42](https://github.com/jmineau/PYSTILT/issues/42)). Overrides merged
  onto the resolved defaults, which already carried `grid` and
  `geometry_hash`, so the geometry was never built. A variant's own
  `geometry` now drops both and derives its raster and hash.
- **`stilt rm` failed on an orphaned `from:` variant**
  ([#43](https://github.com/jmineau/PYSTILT/issues/43)) whose parent was no
  longer declared in `config.yaml`. `Model.remove()` now resolves every
  handle, parents and mets included, from the record.
- **A rerun of a missing trajectory kept the old footprint**
  ([#39](https://github.com/jmineau/PYSTILT/issues/39)). When HYSPLIT runs
  for a simulation, its footprint is now remade from the new particles even
  under `skip_existing`, and so are the footprints of variants derived from
  it with `from:`. Before, a lost or deleted trajectory came back next to a
  footprint made from different particles.
- **`ziscale` runs over 150 hours overflowed HYSPLIT's ZICONTROL array**
  ([#36](https://github.com/jmineau/PYSTILT/issues/36)). HYSPLIT holds at
  most 150 hourly factors and reads more without a bounds check, and a
  scalar `ziscale` was repeated for every hour of the run. A config that
  would write more than 150 factors is now rejected; a list of up to 150
  covers the first hours of a longer run.

## [0.1.0a19] - 2026-09-24

### Changed

- Removed `ModelConfig.basic` (it only forwarded to the constructor with
  defaults the fields already declare) and `Project.simulation_dir` (a
  simulation's directory comes from the model's compute root, which can sit
  outside the project on a Slurm worker). Neither had a caller.
- The GGG reader samples under `tests/data/products` are synthetic files on
  the GGG2020 `.oof` and `*.private.nc` layouts rather than slices of an
  instrument team's retrievals. They keep the formats' quirks, including the
  `.oof` header's variable count exceeding the number of columns written.

### Fixed

- **Backward runs staged a met file they did not need**
  ([#29](https://github.com/jmineau/PYSTILT/issues/29)). The next file
  after the release is needed only when the release falls in the last hour of
  its file, where HYSPLIT interpolates against the next file's first hour.
  STILT-R's `find_met_files` checks for that; PYSTILT added the next file for
  any release off a file boundary. With 6-hourly HRRR, a 19:06 release pulled
  in the next day's 00z file, and a defect in that file failed the run.
  Selection now matches STILT-R. Hourly met is unaffected.
- **The archive met search matched backup copies**
  ([#30](https://github.com/jmineau/PYSTILT/issues/30)). The prefix glob also
  found files like `20200107_18-23_hrrr~20260403182134~` next to the real
  file, and a corrupt backup failed the run. Names ending in `~` are now
  skipped.
- **Slurm array logs were overwritten by the next submission**
  ([#31](https://github.com/jmineau/PYSTILT/issues/31)). Logs were named by
  task id alone in `slurm/logs/`, so every array against a project reused the
  same files. Each submission now writes to `slurm/logs/<date_time>/`, the
  same key as its `chunks/<date_time>/` and `submit_<date_time>.sh`.

## [0.1.0a18] - 2026-09-23

### Added

- **Emission-error recipe** in the Transport Error guide: the enhancement's
  uncertainty from a per-cell flux sigma is the same footprint product,
  bracketed by the fully correlated sum (X-STILT's `cal.emiss.err`) and the
  independent root-sum-square; the correlated case is fips's new
  `prior_obs_error`. No new API.
- **Several error realizations per simulation** (`error_realizations: N`
  in `config.yaml`): the perturbed transport runs N times, each a different
  perturbation draw, stored as `{sim}_error.parquet`,
  `{sim}_error_1.parquet`, …; the main run is shared. Requires `krand: 4`
  (the default) — HYSPLIT randomizes the wind-error draw only in that mode
  and the bundled `hycs_std` ignores the namelist `seed` for it, so any
  other mode is rejected at config time rather than silently repeating one
  draw. Completion requires every realization and `skip_existing` reruns
  only the missing ones, so a preempted job resumes. `transport_error` takes
  the list and averages each level's perturbed mean and variance before
  differencing. Its `noise` scales as `sqrt((1 + 1/N) / 2)`: the unperturbed
  side is shared, so the gain is bounded by `sqrt(2)`. New:
  `Simulation.error_trajectory(k)`, `Simulation.all_error_trajectories`,
  `Simulation.error_realizations`, `Simulation.missing_error_realizations`,
  `TransportError.realizations`.
- **GGG readers for EM27/SUN** (`stilt.observations.readers.ggg`):
  `read_ggg_oof` reads a GGG2020 `.oof` (one instrument-day, how EGI delivers
  EM27/SUN retrievals) into the readers' sounding table; it carries no
  kernel or prior, so those columns are left out rather than faked.
  `read_ggg_netcdf` reads the run's `*.private.nc` — kernels are stored as
  a table against slant xgas and are interpolated per spectrum the way
  GGG's public writer does, priors are shared through `prior_index` — as
  well as the public files, so `read_tccon` is now that function under the
  network's name. Priors come back in the species' units (the public
  `prior_ch4` is ppb, `xch4` ppm). Checked against Salt Lake City EM27 days
  and EGI's example private files; the slant guide's EM27 recipe now reads
  real files.

### Fixed

- **Near-field plume dilution in forward runs**: `hnf_plume` accumulated the
  plume's vertical spread from the far end of the track back to the release
  point when `n_hours` was positive, so forward footprints carried the wrong
  dilution depth. The cumulative sum now walks each particle in order of
  elapsed time since release, which is the same order as before for backward
  runs — their footprints are unchanged. The plume-background guide no longer
  tells forward users to disable the correction.
- **`Simulation.time_range`** spanned `n_hours + 1` hours for a forward run;
  it now spans `n_hours` in both directions.

### Changed

- `Simulation.error_trajectories_path` is a method taking the realization
  index (`()` is the unsuffixed first realization); `HYSPLITResult.error_particles`
  is a dict keyed by realization, empty when no error run happened.
- `stilt init` writes `file_tres: 6h` in the starter config, matching every
  example in the docs (six-hour HRRR blocks); it said `1h`, so a new user
  following the quickstart hit a met-file mismatch on the first run.
- Forward runs (positive `n_hours`) are listed as a supported mode in the
  README and roadmap; they were only mentioned inside the plume-background
  guide.
- The `seed` field's description says what was measured: with the bundled
  `hycs_std` and `krand=2` it changes nothing, and `krand=4` randomizes the
  seed itself, so it is not a route to reproducible or distinct draws.

## [0.1.0a17] - 2026-09-22

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
- **Slant altitudes from a retrieval's pressure levels**
  (`stilt.observations.pressure_altitudes`): convert a sounding's pressure
  levels (OCO-2 `pressure_levels`, TROPOMI's surface pressure and interval)
  to MSL altitudes for `slant_points`, anchored at the sounding's surface
  pressure and altitude, with the standard-atmosphere lapse rate, an
  isothermal scale height, or the retrieval's temperature profile. Levels
  below the surface or above `top=` are dropped and the result starts at
  the surface. The slant columns guide has the recipe.
- **Product readers** (`stilt.observations.readers`, one module per
  instrument): `read_tropomi_ch4` (operational S5P L2 CH4 orbits and the
  TROPOMI+GOSAT blended files), `read_oco2` (OCO-2/3 Lite XCO2) and
  `read_tccon` (GGG2020 public site files) return a table of soundings
  with shared columns: location, time, surface altitude and pressure,
  value and uncertainty, quality, viewing angles, and the averaging
  kernel, pressure levels and prior as arrays from the surface up. The
  TROPOMI and TCCON readers are tested on slices of real files kept under
  `tests/data/products`; the OCO-2 reader on the documented Lite layout.
  New guide *Reading Retrieval Products* lists the columns and how to add
  an instrument.
- **Plume background from forward runs**
  (`stilt.observations.plume_polygon`, `plume_background`): outline the
  plume a city puts over a satellite swath from the positions of
  forward-run particles at overpass time (a kernel density cut at a
  fraction of its maximum, the largest piece kept) and take the
  background as the median of the good soundings beside the plume, per
  side or pooled, with the spread and retrieval error in quadrature.
  Forward runs are `n_hours > 0`; the new Plume Background guide has the
  release recipe (a jittered box over the city every half hour before the
  overpass) and the choices made. Set `hnf_plume: false` for forward runs
  for now: its cumulative sum walks the particle rows in backward order.

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
