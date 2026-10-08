# Changelog

All notable changes to PYSTILT are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Migration

This release redesigns how a project is laid out, run, and read. A
0.1.0a22 project needs its `config.yaml` updated (the keys below), and its
results in `simulations/by-id/` are not read: run it again, or read them
with the old release. The *Projects and variants* guide describes the new
layout; `docs/migration/` covers moving from STILT-R, X-STILT, and stiltctl.

### Changed

- `Project` replaces `Model` (breaking). `stilt.Project.init(path,
  receptors, settings)` writes `config.yaml` and `receptors.csv` once;
  `stilt.Project(path)` opens a project. `ModelConfig` is `ProjectConfig`.
  `project.simulations` is a pandas DataFrame; select rows as in pandas and
  pass them to `project.status(sel)`, `particles(sel)` (`columns=` reads
  fewer), `footprints(sel)`, or `jacobian(sel, target, time_bins)`. One
  simulation is
  `project.simulation(receptor_id, variant)` (was `model.simulations[...]`).
  The collection classes, `SimID`, `ReceptorID`, and `LocationID` are gone.
- Results go to an output directory (breaking). `output:` in
  `config.yaml` names it (`./output` unless set; several projects can share
  one, and it can be an `s3://` or other fsspec URL). It holds
  `particles/`, `footprints/`, and `logs/`, each with a folder per set of
  settings (`settings=<variant>-<hash>/date=.../<receptor id>.parquet`)
  named by a hash of what changes the result, so a changed setting makes a
  new folder and never overwrites. `variants.yaml`, `ConfigChangedError`,
  `stilt rm`, and the `.empty` marker are gone. `stilt status` opens no
  result file: seconds where it took 25 minutes on 64,000 footprints.
- Results are plain data (breaking). `sim.particles` is a DataFrame
  (was `sim.trajectories`, a `Trajectories`) and `sim.footprint` a
  DataArray (was a `Footprint`), with PYSTILT's methods in the `.stilt`
  accessor (`foot.stilt.plot.map()`, `.enhancement(flux)`,
  `.aggregate(...)`); `foot.sum("time")` replaces `integrate_over_time()`.
  The particle columns `indx`, `time`, `long`, `lati`, and `xhgt` are
  `particle`, `age`, `lon`, `lat`, and `release_height`; `varsiwant` keeps
  HYSPLIT's codes. `age` is still minutes since release, negative on a
  backward run, and the UTC time of each row, built when the table is
  read, is `time` (was `datetime`). An averaging kernel's `coordinate` is `release_height`
  unless set, so a footprint made with one gets a new settings folder.
- A config declares its variants, and is checked when it loads
  (breaking). Without `variants:` a config no longer runs one variant per
  met. An unknown key is an error naming the nearest setting. `from:` is
  rejected: variants with the same transport settings share particles on
  their own. `realizations: N` makes one variant whose simulations have a
  `realization` column, where it made N numbered variants.
- Run settings are flat keys of the transport model (breaking).
  HYSPLIT's settings are `stilt.transport.hysplit.HysplitConfig`, written as
  top-level keys as before; `STILTParams` and `RuntimeSettings` are gone.
  `kmsl` comes from each receptor's `altitude_ref`. `timeout` moves under
  `execution:`, beside the new `keep_workdir`; `rm_dat` is gone.
- Meteorology settings (breaking). In a `mets` entry, `source:` is
  `download:` and `backend:` is `download_from:`. `n_min` and
  `subgrid_buffer` are gone: every file a run needs must be there, and the
  crop is `subgrid_bounds` exactly. The met config is HYSPLIT's,
  `stilt.transport.hysplit.MetConfig`. `~` and `$VARIABLES` expand in a
  met's directories, and a relative one starts from the project.
- A run records its met as the weather product and the crop. The
  product is the source id in the ARL file headers (`HRRR`, `NAM`, ...),
  read from the first file in the met's `directory`, or the archive's for
  `download`. Moving or renaming the met files keeps every result, and two
  archives named alike no longer share results. The met's `directory` must
  hold its files on the machine that opens the project.
- A run cut short by its meteorology fails (breaking; #98, #169, #189).
  A missing met file fails the simulation before HYSPLIT runs, naming the
  hours with no file. A met file cut short fails it when HYSPLIT says the
  meteorology ran out (`no more meteorology` in its `WARNING` file). Both
  are `MET_COVERAGE`. Before, such a run wrote a partial footprint and
  counted as complete. A run whose particles all leave the met's domain,
  or its crop, before `n_hours` is complete.
- Failures are recorded (breaking). A failed simulation does not stop
  the others: `<receptor id>.failure.yaml` under `logs/` says why
  (`sim.failure`), and each simulation is `complete`, `failed`,
  `interrupted`, or `pending`; `stilt status` counts each, per variant.
  Exceptions share one base, `stilt.StiltError`, in `stilt.exceptions`
  (was `stilt.errors`).
- Running (breaking). `stilt run` (and `project.run()`) runs and
  waits, locally or as a Slurm job array; `stilt submit` (and
  `project.submit()`) submits the array and returns. `--wait/--no-wait` is
  gone. A submission is a folder `_slurm/<stamp>/` with its `job.sh` and
  `receptors.parquet`, the checked rows its tasks read in place of
  `receptors.csv`; `execution.time` is required, other `sbatch` options go
  under `execution.slurm`, and a task at its time limit or preempted
  requeues itself. A job sent to another cluster (`clusters:` under
  `execution.slurm`) is followed there. `stilt run` exits 0 (complete), 1
  (some failed), or 3 (some interrupted); every command exits 2 on a wrong
  command line. A failed run's workdir is kept under `scratch/` in the
  output directory.
- Receptors are frozen pydantic models, and an id names everything that
  makes a receptor distinct (a multipoint's heights to 0.01 m), so some ids
  differ from 0.1.0a22's (breaking).
- Footprint and geometry names (breaking). `Footprint.calculate` is
  `stilt.calc_footprint`, STILT-R's name. `Grid.projection` is `Grid.crs`.
  `Grid.from_geometry(mesh, ...)` is `mesh.to_grid(...)`. `stilt.flux` is
  `stilt.sampling`.
- `sim.background(field)` and `sim.transport_error(error, flux)` belong to
  the simulation, and a transform's method is `apply(particles,
  receptor=None, directory=None)` (`TransformContext` is gone) (breaking).
- Python 3.11 or newer is required, and the `cloud` extra is `download`
  (breaking). One wheel per platform, each with only its own HYSPLIT
  build; a machine without one gets a clear error (#61).
- The version comes from git tags (setuptools-scm). Between releases,
  `stilt.__version__` is a dev version such as `0.1.0a23.dev5+g1a2b3c4`
  instead of the last release's number.

### Added

- The API reference gives every attribute and property its own page, with the
  class's tables linking to them, and a subclass links to the members it
  inherits instead of repeating them. Fields a class docstring already
  describes no longer show as empty rows.
- `stilt.run_trajectories(receptor, met, params)` and
  `stilt.calc_footprint(particles, receptor, grid)`: one receptor without a
  project, as STILT-R's `calc_trajectory` and `calc_footprint`.
- A transport model interface (`stilt.transport`). `model:` names a
  built-in model or the import path of a model class in another package, a
  variant can name another model, and a model may run many receptors per
  call (`batched = True`), as an emulator would.
- `data_dir`: a folder of HYSPLIT data tables to use in place of the
  bundled ones. Its files are part of a run's settings.
- `stilt run --task I/N` runs one share of the receptors in this process,
  for a scheduler other than Slurm. `stilt.execution.pending` lists the
  receptors a run would run, and `wait(job_id)` waits for a Slurm job.
- `stilt status --json`; `--output` on `run`, `submit`, and `status`
  (`Project(path, output=...)`); `stilt submit --receptors FILE`.
  `stilt status` lists the settings folders and the variants using each
  (`project.folders()`).
- `project.jacobian(sel, target, time_bins)` sums footprints onto a grid,
  mesh, or zones as a sparse matrix, its columns named levels from the
  target (`time, lon, lat` or `time, cell`), in bounded memory. Its
  `workers` threads default to the CPUs the process may use (in a Slurm
  job, the job's), at most 8, as `open_footprints` does.
- `project.footprints(sel)` and `stilt.footprint.open_footprints(paths)`:
  many footprints as one dataset. Result files read alone:
  `stilt.read_particles`, `stilt.read_footprint`, and
  `stilt.particles.particles_metadata` (receptor, settings, met files).
- `stilt.observations.receptors_from_soundings` and `project.add_table`
  for satellite and column receptors, and a tutorial, *A Satellite
  Column, Start To Finish*.
- An empty footprint (no particle reached the grid) is a file marked
  `stilt:empty`; it is complete, and `project.footprints()` lists it apart.
- The user guide is reorganized, one idea per page. New: *Checking a run*
  (every failure reason and what to do), *Projects and variants*,
  *Summing footprints* (areas and the Jacobian), *Your own HYSPLIT build*,
  *Trajectory-style analyses* (residence time, PSCF, CWT), ERA5 in the
  meteorology guide, *Coming from Jena STILT*, the *Output layout*
  reference for readers without PYSTILT, querying the output with DuckDB,
  and a table of every `config.yaml` key.
- The documentation has a version dropdown. The site opens at the latest
  release, `dev/` follows `main`, and each release keeps its own pages.

### Removed

- The PostgreSQL work queue and the Kubernetes backend (`stilt register`,
  `pull-worker`, `push-worker`, `serve`, `PYSTILT_DB_URL`); run `stilt run
  --task I/N` from another scheduler. `MetStream`, `Model.check_config`,
  `Model.remove`, and `Model.orphans`.

### Fixed

- A run reads up to 128 met files. `CONTROL` lists them as one grid in
  time; listed as one grid per file, as STILT-R does, NOAA's HYSPLIT
  builds read at most 12 and the bundled one 99. The particles are the
  same.
- A run whose met files are damaged, or do not hold every hour it needs,
  fails before HYSPLIT starts, as `MET_COVERAGE`, naming the file or the
  hours (#189). A file can keep its name and still be cut after a whole
  hour, hold an hour written partway, or have records lost to null bytes.
  HYSPLIT then stopped the particles early, and when some had left the
  met's domain first, the run counted as complete. Each file is checked
  once in a process with arlmet's `File.check()`, so arlmet 0.1.0b3 is
  needed.
- The near-field correction (`hnf_plume`) reads a `veght` above 1 as
  meters above ground, as HYSPLIT does (#167). It multiplied it by the
  mixed-layer height, as STILT-R does. Nothing changes for a `veght` of 1
  or less.
- `stilt init` writes a receptors file that loads (#49). HYSPLIT settings
  take the types HYSPLIT reads (`rhb`, `rht`, ...) (#57).
- Crops of your own met files are cached per crop box and written
  atomically, so two crops never mix and a stopped crop leaves nothing
  behind (#53). Two workers can start the same run at once (#90).
- Appending an MSL receptor keeps its altitude reference (#51), and
  aggregating a footprint rejects time bins not closed on the left (#58).
- Downloaded met includes the next file when the receptor is in the last
  hour of a file, as local met and STILT-R do.
- `ziscale: [[0.8, 0.9]]` (STILT-R's nested form) and `[0.8, 0.9]` are one
  run. A footprint with one hourly layer that is not the receptor's hour
  is placed at its hour.
- A grid derived for zones on a projected grid covered a box near the
  equator. Footprint maps work on a projected grid, and
  `plot.availability(ax=...)` formats the figure of the axes you pass.
- A `config.yaml` loads where its `exe_dir` is not reachable.

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
