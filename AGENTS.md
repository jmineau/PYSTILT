# AGENTS.md

Orientation for contributors and coding agents working on PYSTILT. It covers
the architecture, the invariants that must not break, and the dev workflow.
User-facing documentation lives in `docs/` and at
<https://jmineau.github.io/PYSTILT>; contribution mechanics and recipes for
adding config fields, execution backends, and particle transforms are in
[CONTRIBUTING.md](CONTRIBUTING.md).

> **Keep this file current.** If you change the module layout, the build/test
> commands, or learn a new invariant or gotcha, update the matching section in
> the same change. A section that no longer matches the code is worse than no
> section: fix it or delete it.
>
> Personal or machine-specific notes (local paths, cluster setup) belong in an
> untracked file, not here. Anything matching `*.local.md` is gitignored for
> this (e.g. `CLAUDE.local.md`); add your tool's local files to `.gitignore` if
> they are not covered. Some agents stop reading `AGENTS.md` once a local
> instruction file exists, so import or reference it from yours.

## Overview

PYSTILT is a Python implementation of **STILT** (Stochastic Time-Inverted
Lagrangian Transport). It drives **HYSPLIT** for backward (or forward) particle
trajectories and computes gridded footprints for receptors. It is alpha
software (`0.1.0a*`): **there are no backward-compatibility guarantees before
v1.0**, so prefer the clean design over a compatibility shim.

### Naming

| Thing | Name |
|---|---|
| PyPI distribution | `pystilt` |
| Import name | `stilt` (`import stilt`, never `import pystilt`) |
| Particle columns | `particle`, `lon`, `lat` for HYSPLIT's and STILT-R's `indx`, `long`, `lati` (mapped in `read_particle_dat`); the rest keep STILT-R's names (`time`, `zagl`, `foot`, `mlht`, ...) |
| Source directory | `src/stilt/` |
| CLI entry point | `stilt` (Typer; see `[project.scripts]`) |
| Config | What the user writes: always a class (`ProjectConfig`, `MetConfig`, `FootprintConfig`, `ExecutionConfig`, a model's `HysplitConfig`), each next to the code that uses it |
| Settings | What a result was made with, as recorded: `_settings.yaml`, the `settings=` folders, a settings hash. Never a class |
| Parameters | Plain English for a config's fields; names no class or module |

Use `pystilt` only for installation, the repository, and documentation URLs;
tracebacks say `stilt`, pip and uv say `pystilt`. The R implementation
([uataq/stilt](https://github.com/uataq/stilt)) is **STILT-R**, never
"R-STILT", in docs, docstrings, comments, changelog entries, and commit
messages. Existing identifiers such as the `r_stilt` test fixtures and
`STILT_R_DIR` keep their names.

### Relationship to sister projects

- **[STILT-R](https://github.com/uataq/stilt)**: PYSTILT inherits its transport
  science. Footprints match STILT-R at `rtol=1e-7` per cell against a pinned
  upstream commit, checked by the `fidelity` test suite. The parity policy is
  in [docs/development.rst](docs/development.rst); read it before touching
  trajectory or footprint math.
- **[stiltctl](https://github.com/jmineau/air-tracker-stiltctl)**: source of the
  thin CLI → `Project` → worker call path. Its queue-backed and Kubernetes
  execution was implemented here and then removed (#67, #87).
- **[X-STILT](https://github.com/uataq/X-STILT)**: source of the observation
  layer and column science. PYSTILT ports the concepts, not the scripts, and
  does not aim for X-STILT feature parity.

## Architecture

**There is no index, manifest, or registry.** A project is a directory of
inputs (`stilt.project.Project`: `config.yaml` and `receptors.csv`) and an
output directory (`stilt.output.Output`) that `config.yaml` names and that
several projects can share. The simulations a project defines are
**receptors × variants**: `receptors.csv` crossed with the named variants in
`config.yaml`, which must declare at least one. Each variant resolves
into a `Variant` (`stilt.config`) with two records of settings
(`stilt.identity`): its run settings, whose hash identifies a *run* and
names its particles folder, and optional footprint settings. Variants with
equal run settings share one run per receptor and differ only in the
footprint made from it. Whether
a simulation is complete is decided **by the files in the output
directory**, by `stilt.output.completed`: the particles exist, and the
footprint when the variant has a grid. `Output.complete`,
`Simulation.is_complete`, and `status()` all apply it. `config.yaml` and `receptors.csv` are the user's
inputs: `Project.init` (or `stilt init`) writes `config.yaml` once, PYSTILT
never rewrites it, and `Project.add_receptors` only appends to
`receptors.csv`. Changing a setting never overwrites a result: it hashes to
a new folder.

**`Project` reads; the workers write results.** `Project(path)` opens a
project directory: its config, its receptors, the simulations they define,
and a view of their results. Its only writes are to its inputs:
`add_receptors` appends to `receptors.csv`, and `add_table` to
`tables/<name>.parquet`. It knows no scheduler or compute root.
`stilt.execution.run` and `submit` (which `Project.run()` and
`Project.submit()` call) find the receptors with missing results, resolve
the compute root, and start the workers, which are the only code that
writes results.

**Tables are plain DataFrames; the verbs are on `Project`.**
`project.simulations` is a pandas DataFrame, one row per simulation.
Select it as in pandas (`sims[sims.variant == "hrrr"]`) and pass the
selection to the project: `project.status(sel)`, `incomplete(sel)`,
`particles(sel)` (one long table), `footprints(sel)` (one lazy dataset
stacked on the hour, `open_footprints`), `jacobian(sel, ...)`.
A selection is any table with `receptor` and `variant` columns (pandas,
polars, pyarrow) or a boolean mask over `project.simulations`; the
methods read only those two columns from it. Do not wrap the table in a
class again: that is how the old collection classes grew. `Output` has no
folder classes: its methods take
the kind of result (`"particles"` or `"footprints"`), the variant, and
receptor ids (`path`, `present`, `complete`, `table`, the failure
records, and the writers). `project.receptors` stays a plain
DataFrame. `project.receptor(id)` and `project.simulation(id, variant)`
return the values; `sim.particles` and `sim.footprint` are data.

`stilt.__all__` (plus the `__all__` of each subpackage) is the public surface;
everything else is internal and can change.

### Module layout

```
src/stilt/
  cli.py             Typer CLI; a thin adapter over Project and execution
  project.py         Project: the project directory, its receptors and
                     simulations (DataFrames), run/submit, and a
                     selection's status, particles, footprints, Jacobian
  output.py          Output: the output directory, a folder per kind and
                     settings hash; finding a variant's folder, which
                     receptors have results (present, complete), reading
                     many files at once (table), failure records, writing
  simulation.py      Simulation: a frozen value (receptor, variant, output)
                     that knows where its results are and whether they exist
  config.py          ProjectConfig: reads config.yaml, splits the flat keys into
                     the model's config and the footprint's, rejects unknown
                     keys naming the nearest setting, checks each declared
                     variant; resolve() makes each a Variant (its configs,
                     the model build, its two hashes), reading geometries
                     and the model version once each
  identity.py        settings records and their hashes: what a run and a footprint
                     were made with, written to and read back from _settings.yaml
  receptors/         receptors and receptors.csv
    models.py        receptor types (frozen pydantic models: point, column,
                     multipoint), their times, and their ids
    table.py         the receptor table behind the CSV reader, writer, and
                     appender: `receptor_rows` checks it once, and
                     `receptors_from_rows` builds receptors from checked rows
                     without checking them again
    validation.py    the checks a receptor must pass, written once for many
                     points and shared by the models and the table
  particles/         the particle table (a DataFrame); __init__.py re-exports only
    table.py         its columns, particle files (read, write, metadata),
                     release heights, and the near-field plume correction
    accessor.py      the `.stilt` pandas accessor (endpoints, enhancement
                     from a flux field)
    background.py    the background at a receptor (`sim.background`)
    transport_error.py  the transport error of the enhancement
                     (`sim.transport_error`)
  footprint/         the footprint (a DataArray)
    config.py        FootprintConfig and the geometry specs
    gridding.py      `calc_footprint`, as STILT-R's (fidelity-guarded)
    aggregation.py   summing footprints onto other geometries: `aggregate`,
                     `jacobian`, their time binning and target weights
    io.py            the footprint array and its attributes; footprint files
                     (`read_footprint`, `write_footprint`) and CF-1.8 NetCDF;
                     `open_footprints`, many as one dataset on the hour
    targets.py       the geometries footprints are summed onto (Mesh, Zones),
                     the overlap weights, and `to_grid`
    accessor.py      the `.stilt` xarray accessor (enhancement from a flux
                     field, aggregation)
  sampling.py        sampling a gridded field (a flux, a mole fraction) at points
  spatial.py         rasters and CRS, no shapely: Bounds, Grid (with its cell
                     and CF helpers), horizontal_dims, is_longlat, same_crs,
                     haversine_km
  meteorology.py     MetConfig, run_window, and Met: ARL file discovery
                     (files_for a window), download, and
                     cropping (via arlmet); the wind-error statistics
                     (`variogram`, `fit_variogram`)
  transforms/        pre-footprint particle transforms, one module each
                     (averaging_kernel, pressure_weighting, lifetime); the
                     loader, YAML I/O, and apply_transforms in __init__.py
  exceptions.py      every exception class, all under StiltError
  visualization.py   matplotlib helpers (optional dependency)

  execution/         ExecutionConfig (config.py), the runner (plans what is missing,
                     runs it here or writes `_slurm/<stamp>/` and submits a
                     Slurm job array script whose tasks run `stilt run --task`)
                     and the worker (runs the transport model in a workdir
                     and writes results for one or many simulations)
  observations/      the X-STILT port, before or after the transport run:
                     product readers and SOUNDING_SCHEMA, overpass grouping
                     and sounding selection, slant geometry and
                     receptors_from_soundings, modelled_column, plume
                     backgrounds. Arrays in, plain values out; there is no
                     observation object.
  transport/         the TransportModel and TransportConfig protocols, ModelRun
                     (particles, log, met_files), ModelInfo, get_model with
                     its MODELS table of built-in names (another model is named
                     by import path), and run_model / run_trajectories, which
                     run a model and apply the core steps (__init__.py); one
                     subpackage per transport model, which owns its config
    hysplit/         HYSPLIT, the one model today: HysplitConfig (config.py, its
                     parameters), HysplitModel (finds its met files from the
                     MetConfig and window), the driver (driver.py:
                     write_inputs, which knows which file each setting goes
                     to, and read_particle_dat), failure reasons read from
                     its log (failures.py), and the bundled binaries (bin/)
                     and data tables (data/)

tests/               pytest; markers `integration`, `fidelity`, and `r_only`. Folders
                     follow src/stilt (execution/, footprint/, observations/,
                     particles/, transport/hysplit/); integration/ holds the
                     end-to-end runs, r_stilt/ the STILT-R comparisons, and
                     fixtures/ the factories and helpers tests share
docs/                Sphinx (pydata-sphinx-theme)
```

### Two ways work starts: do not conflate them

1. **A run** (`Project.run()`, `Project.submit()`, or `stilt run`):
   `stilt.execution.run` finds the receptors with missing results and
   either runs them in this process (`backend: local`) or submits them as
   one Slurm job array (`backend: slurm`): `submit` writes
   `_slurm/<stamp>/{receptors.txt,execution.yaml,job.sh}` and calls
   `sbatch`, each task runs `stilt run --receptors receptors.txt --task
   $SLURM_ARRAY_TASK_ID/N --execution execution.yaml`, and `run` waits by
   polling `sacct`. `submit` returns the job id at once. A task that stops
   with work left requeues itself (`scontrol requeue`) when it got SIGUSR1,
   which the script asks for two minutes before the time limit
   (`--signal=B:USR1@120`), or was preempted (`PreemptTime` set; this
   cluster preempts with SIGTERM, 30 s grace, and CANCEL, and Slurm checks
   `--signal` times only about once a minute, so USR1 may not come). A
   `scancel` only stops it. The unit of work is a receptor: `run_receptor` runs
   the transport model once per distinct transport hash, then writes the
   footprint of every variant that shares those particles. A batched model
   (`batched = True`, `run_many`) is the exception: `run_receptors` first
   runs each of its variant groups once for all the receptors that need
   particles, then the per-receptor loop does the rest. HYSPLIT is not
   batched. `run(task=(i, n))` (`stilt
   run --task i/n`) runs one share here whatever the backend; the share is
   taken from all the receptors before the complete ones are dropped
   (`task_share`), so tasks that start at different times never overlap.
   `stilt run` exits 0 (all complete), 1 (some failed), or 3 (some
   interrupted); every command exits 2 on a wrong command line, as Click
   does for an unknown option.
2. **Observation-driven**: a reader yields a DataFrame of soundings;
   `stilt.observations` helpers thin and group it; each row becomes a
   `Receptor` (`receptors_from_soundings`, which also gives the kernel
   table, kept with `project.add_table`); then `add_receptors` and run as
   usual. This layer sits *above* the transport core. Keep observation logic
   out of `project.py`.

The import contracts in `pyproject.toml` (run by `lint-imports` in CI)
hold this shape. One `layers` contract lists every top-level module from
the CLI down to `_paths` and `exceptions`; a module imports
only the layers below it, and modules that import each other share a
layer. It is exhaustive, so a new top-level module fails until it is
placed. Two short contracts hold what a layer cannot: only
`stilt.transport` (its `get_model`) imports HYSPLIT's package, and the
project config reads no results.

Slurm tasks rebuild the model from the project in another process on
another node, so anything a worker needs must be in the project or the
output directory, never only in memory.

### Configuration

- `ProjectConfig` is the root: `model` (the transport model, `hysplit`
  unless set), that model's parameters and the footprint (`FootprintConfig`)
  fields as flat defaults, `mets`, and `variants` (overrides of the
  defaults). The model's parameters are checked by its own config class
  (`stilt.transport.hysplit.HysplitConfig`; `config.transport`), reached
  through `stilt.transport.get_model`. A model is a variant axis, like a
  met: a variant that names another `model` gives that model's own
  parameters itself and inherits the met, the footprint fields, and the
  parameters PYSTILT's own code reads (the base `TransportConfig`:
  `n_hours`, `seed`, `hnf_plume`, `veght`). A parameter goes on the base
  only when the core reads it; `numpar` and every `SETUP.CFG` setting are
  HYSPLIT's. The top-level footprint fields
  are `config.footprint`, as the model's are `config.transport`. When it
  loads, `ProjectConfig` checks each declared variant against the defaults
  (names, met, model, `realizations`, grid merging), and rejects an unknown
  key with the nearest known one (`difflib`), reading no other file.
  `ProjectConfig.resolve(directory)` (behind `project.variants`) turns
  each into `Variant`s: it validates the variant's transport settings with
  its model's config class (once), keeps `realizations: N` as one
  ensemble variant (realization `k` runs with `seed + k`), reads each geometry once (a relative
  file from the project directory) to derive the grid, and asks the
  transport model its version once per build. `Project.init` resolves
  before it writes `config.yaml`. Stored `_settings.yaml` records keep
  ignoring keys this version does not know. Variants whose run settings
  match share the particles; `from:` is rejected. `grid: null` means
  particles only, and footprint settings without a grid are an error. There is no
  named-footprints dict.
- **Each part owns its config.** `FootprintConfig` is in
  `stilt.footprint.config`, `MetConfig` in `stilt.meteorology`, `ExecutionConfig` in `stilt.execution.config`, a
  model's config in its package. `Bounds` and `Grid` are in `stilt.spatial`,
  the raster and CRS layer that needs no shapely, since more than footprints
  use rasters (a flux put on the footprint grid). `stilt.config` composes them, so nothing
  below the project's layer imports `stilt.config` (the layers contract).
  Config classes do no I/O when they validate, so a `config.yaml` loads
  offline; reading a geometry or asking a model its build happens in
  `ProjectConfig.resolve`, on request.
- A config's fields that change no result are listed in its `UNRECORDED`
  class variable (`exe_dir` and `data_dir`; a met's directories,
  `download_from`, and `n_min`), and left out of the settings records. A
  project requires each met's `directory`, so a met config read back from a
  record, without one, still validates.
- Every field is a plain pydantic `Field(default, description=...)` and the
  public config stays flat (`ProjectConfig(numpar=..., seed=...)`). CONTRIBUTING
  explains how a field is routed to `SETUP.CFG`, `CONTROL`, `WINDERR`, or
  `ZIERR`.
- The scratch directory comes from the `PYSTILT_COMPUTE_ROOT` environment
  variable, which `resolve_compute_root` in the runner reads; `Project`
  does not.
- `ExecutionConfig` (`execution:` in `config.yaml`) says where receptors run
  and with what Slurm resources. It forbids unknown keys; other `sbatch`
  options go under its `slurm:` mapping.
- Particle transforms are declared as a default or per variant in YAML
  (`transforms: [{kind: ...}]`); `kind` may also be the import path of a user
  class.

### Project layout on disk

A project is a local directory; results go to the output directory its
`config.yaml` names (`./output` by default, relative to the project). Every
folder below a kind is hive-style, so each tree reads as one dataset:

```
<project>/
  config.yaml                 ProjectConfig (written once by Project.init or stilt init; never rewritten)
  receptors.csv               receptor list; add_receptors() appends new receptors
  tables/<name>.parquet       other inputs, such as kernels; add_table() appends
  _slurm/<stamp>/             one folder per Slurm submission: job.sh,
                              receptors.txt, execution.yaml, <task>.log

<output>/
  particles/settings=<variant>-<hash>/_settings.yaml
  particles/settings=<variant>-<hash>/date=YYYY-MM-DD/<receptor_id>.parquet
  footprints/settings=<variant>-<hash>/_settings.yaml      names the particles folder
  footprints/settings=<variant>-<hash>/date=YYYY-MM-DD/<receptor_id>.parquet
  logs/settings=<variant>-<hash>/date=YYYY-MM-DD/<receptor_id>.log
  scratch/settings=<variant>-<hash>/date=YYYY-MM-DD/<receptor_id>/   failed runs' working dirs
  particles/settings=<ensemble>-<hash>/realization=k/date=YYYY-MM-DD/...  an ensemble's realization k
```

A particles folder's hash is `Variant.particles_hash`; a footprint
folder's is `Variant.footprint_hash`, over the run settings' hash and the
footprint settings together. "Particles" is the one word for the particle table in code
(`sim.particles`, `project.particles()`, `has_particles`); "run" is only the verb, and
"trajectory" means one particle's path. Lookup
reads the stored `_settings.yaml` back through the current config classes
(`stilt.identity`) and re-hashes, so a field added later with a default
still matches. `compute_root`
is scratch: HYSPLIT runs there and the directory is discarded after success.
Footprints are sparse tables (`hour, y, x, foot`, float32); an empty
footprint is a file with no rows and the reason in its metadata, and counts
as complete.

## Invariants

- **STILT-R numerical parity** at `rtol=1e-7` per footprint cell. Run the
  `fidelity` suite before merging any change to trajectory or footprint math.
  NetCDF output is CF-1.8 and deliberately not byte-compatible with STILT-R.
- **Completion is by file.** A simulation is complete iff its files exist
  in the output directory. `stilt.output.completed` is the rule, written
  once; `Output.complete` applies it to a listing of the date folders, and
  `Simulation.is_complete` and `status()` call it. `has_particles` and
  `has_footprint` look for the same files one at a time. Never add a
  different rule for whether a result exists, a completion registry, or a
  manifest.
- **Bulk methods work from listings.** A method of `Project` that takes a
  selection works from folder listings and receptor ids, and never builds
  a `Simulation` per row: on a large project one receptor costs milliseconds, and
  `status()` once took 25 minutes on 64k footprints (#141). It opens a file
  per row only when the file's contents are the answer (`footprints`,
  `jacobian`), and builds receptors together (`Project._receptors`) when it
  needs them.
- **One code path for a disk and an object store, and no store class**
  (#135). `Output.directory` is a `Path` on a filesystem and a universal
  path (`UPath`) for a URL (`stilt._paths.location`). Code that touches
  the output checks `isinstance(path, Path)` for the local fast path
  (`os.scandir`, temporary file and rename, pyarrow on the path) and
  otherwise uses the store's fsspec filesystem (`path.fs`), putting a file
  directly. Parquet is read through `stilt._paths.readable`. A local
  output must stay exactly as fast; time a listing of a large folder when
  changing it.
- **Identity is content.** A results folder is its settings hash; a changed setting is
  a new folder, never an overwrite, and PYSTILT never deletes a results
  folder. (A kept workdir under `scratch/` is replaced by the next one
  kept for the same simulation.)
- **State lives in the project directory and the output directory.**
  Anything kept in a process-local variable is lost to a Slurm task, which
  opens the project again from its directory.
- **The CLI stays thin.** `cli.py` adapts arguments to `Project` and
  execution calls; orchestration logic does not belong there.
- **Meteorology I/O goes through [arlmet](https://github.com/jmineau/arl-met).**
  Do not reimplement ARL reading here.
- **Bundled HYSPLIT is package data.** Reach the binaries and tables through
  `importlib.resources` via the existing helpers, never a repo path. They ship
  through `[tool.setuptools.package-data]`; moving them means updating that.
  Each wheel is tagged for one platform and carries only that platform's
  `hycs_std` (`setup.py`); the sdist carries none (`MANIFEST.in`). A new
  platform build needs a case in `bundled_build` in `setup.py`, the driver's
  `_bundled_exe_dir`, and the `just dist` / `just check-dist` recipes.
- **Pydantic for all configuration**, with a description on every field.

## Development

### Commands

Driven by [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/):

| Command | What it does |
|---|---|
| `just install` | `uv sync --group dev` |
| `just test` | `uv run pytest -v` (unit tests only) |
| `just quality-check` | ruff, pyright, and the import contracts (`lint-imports`), then the tests |
| `just ruff` | `ruff check --fix` and `ruff format` on `src/stilt` |
| `just build-docs` | clean Sphinx HTML build into `docs/_build` |
| `just dist` | the sdist and one wheel per bundled HYSPLIT build, into `dist/` |
| `just check-dist` | check each wheel's tag and that it holds only its own `hycs_std` |
| `just pre-commit` | all pre-commit hooks on all files |
| `just clean` | remove build artifacts, caches, coverage, docs build |

CI (`.github/workflows/`): `tests.yml`, `quality.yml`, `docs.yml`, and
`publish.yml` for releases. `docs.yml` builds the docs on every pull request
but deploys the site only from a `vX.Y.Z` tag, so the site matches the
latest release; a docs change on `main` goes live with the next release.

### Tests

Plain `pytest` runs the unit tests and skips the rest. Three markers,
disjoint (a test has at most one); CI runs `-m "integration or fidelity"`
and then `-m r_only`:

- `-m integration`: PYSTILT end to end, with real met files and HYSPLIT
  (`tests/integration/`). Slow.
  The `met_dir` fixture needs `STILT_TEST_MET_DIR` pointing at a directory
  of HRRR ARL files, or `STILT_TEST_FETCH_MET=1` to download the seven 6 h
  blocks it needs into the ignored `tests/met_cache/`. Once downloaded, set
  `STILT_TEST_MET_DIR=tests/met_cache`; without either variable every
  integration test is skipped.
- `-m r_only`: the STILT-R comparisons that need R but no met files or
  HYSPLIT (the synthetic footprint tests). Fast; needs `STILT_R_DIR` and
  `Rscript`.
- `-m fidelity`: PYSTILT against STILT-R, both running HYSPLIT on the
  test met files. Slow; needs the met files as `integration` does,
  `STILT_R_DIR` pointing at a STILT-R checkout, and `Rscript` on `PATH`.

A local `.env` is loaded by `pytest-dotenv`, which is the place for
`STILT_R_DIR` and similar settings.

Test folders follow the package: tests for `stilt.execution` go in
`tests/execution/`, for HYSPLIT in `tests/transport/hysplit/`, and so on.
Modules without a subpackage stay at the top of `tests/`. `tests/observations/`
is self-contained: its tests use only its own `conftest.py` and `data/`, never
the shared fixtures in `tests/conftest.py`, so the observation layer can leave
the repository with its tests. Keep it that way.

Committed test data (`tests/observations/data/`) stays small and synthetic. Do
not commit real instrument retrievals or met files: they are large and often
not ours to redistribute. Synthetic samples should keep the real format's
quirks.

### Code conventions

- Python 3.11+ (`ruff target-version = "py311"`). Use the standard
  library for what 3.11 added (`typing.Self`, `enum.StrEnum`, `tomllib`,
  `datetime.UTC`) rather than `typing_extensions` or backports.
- Ruff rules `E, F, UP, B, SIM, I, D213`; line length is left to the formatter.
- **NumPy-style docstrings** (Sphinx napoleon is configured for NumPy only),
  with the summary on the line after the opening quotes (`D213`).
- Pyright in `basic` mode against the project `.venv`. `py.typed` ships.
  CI type-checks under Python 3.11, the oldest supported version. To
  reproduce it, note that `pyproject.toml` pins pyright to `.venv`, so
  `--pythonpath` does not switch environments; build a 3.11 environment
  elsewhere (`UV_PROJECT_ENVIRONMENT=<dir>/.venv uv sync --python 3.11
  --group dev`) and run `pyright --venvpath <dir> src/stilt`.
  Fix types at the source rather than reaching for `typing.cast`.
- Keep the `from __future__ import annotations` headers.
- Exception classes live in `stilt/exceptions.py`. Each subclasses
  `StiltError` and the builtin that describes it; a failed run is a
  `SimulationError`. Plain input checks raise builtins such as `ValueError`.
- Runtime dependencies live in `[project]`; optional extras are `geometry`,
  `visualization`, `download`, `sparse`, and `complete`. The `dev` dependency group pulls
  in `pystilt[complete]` plus the test, lint, type, and docs tooling.

### Documentation

Sphinx sources in `docs/`: `getting_started/` (install, quickstart,
concepts), `guides/` (task-oriented how-tos), `tutorials/` (end-to-end
worked examples), `advanced/` (design and internals), and `reference/` (API
pages built from docstrings). A user-facing change needs a guide or reference
update, and `just build-docs` should build without new warnings.

#### Voice

Docs, docstrings, config field descriptions, and CLI help all follow this
voice. Most readers are scientists who want to run the model and trust the
result; write for them. Good examples are the
[STILT-R docs](https://uataq.github.io/stilt/) and the xarray, pandas, and
MetPy user guides.

- **Start with what it is for.** Open a page or docstring with what the
  thing does, in plain words. Then show an example. Internals and edge cases
  go last, or in `docs/advanced/`.
- **Keep sentences short.** One idea per sentence. If a sentence needs a
  colon, a semicolon, and a parenthesis, split it.
- **Use the reader's words.** Receptor, particles, footprint, meteorology.
  Internal names (store key, publish, resolve) belong only on pages about
  internals.
- **Show it.** A short code block, with its output when that helps, is
  clearer than a paragraph describing it.
- **Say it plainly.** No metaphors ("a dial worth turning"), no selling
  ("powerful", "seamless"), no filler ("note that", "it's worth noting").

Use these words in user-facing text, and keep the internal ones on pages
about internals:

| Say | Not |
|---|---|
| settings folder | key, hash, identity |
| transport model (in full) | model, when the weather model or a modelled value is near |
| particles (the table); run (the verb); trajectory (one particle's path) | run as a noun for the result |
| workdir (one simulation's folder); compute root (where workdirs are made) | scratch, except the output's `scratch/` folder |
| log, met files | provenance |
| receptor id; simulation (a receptor under a variant) | simulation ID as a folder |
| the settings of a variant | resolve, record, `model_copy`, frame, handle |

Some habits make text read as machine-written. Avoid them:

- Colon reveals: "PYSTILT does one thing: it follows the air." Write
  "PYSTILT follows the air."
- "X, not Y" contrasts when nobody suggested Y.
- A bold sentence at the start of a paragraph that states its point.
- Dashes (`—` or `--`) joining clauses. Use a period or parentheses.
- Lists of three out of habit, and closing sentences that restate the
  paragraph.
- Asides about what the code does "deliberately" or used to do.
- Validation statistics in a user guide. Give the reader the setting to use
  and cite the study.
- Labels invented on one line and referred to later ("the first case").

Docstrings follow the NumPy style. The summary line says what the object is
or what the function returns. Parameter descriptions give the meaning and
units, not the type again. Check every example and claim against the code
when you write it.

### Commits, changelog, releases

- Commit messages follow Conventional Commits (`fix(slurm): ...`,
  `docs: ...`).
- User-visible changes go under `## [Unreleased]` in `CHANGELOG.md`, which
  follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
- The version is a static string in `pyproject.toml`; pushing a `vX.Y.Z` tag
  publishes to PyPI via `publish.yml`. **Do not bump the version, cut a
  release, or push a tag unless the maintainer asks.**

### Issues, pull requests, and roadmap

Open work is tracked in
[GitHub issues](https://github.com/jmineau/PYSTILT/issues), not in this file.
Feature status lives in one place, the roadmap tables in
[docs/roadmap.rst](docs/roadmap.rst); when a feature lands, update its row.
The README links to it and keeps no table of its own.

- Code that looks over-engineered goes on the running list in
  [#48](https://github.com/jmineau/PYSTILT/issues/48) (label `simplify`),
  not into an unrelated change.
- **Do not act on GitHub for the user unless asked.** No new issues, pull
  requests, comments, or review replies on their behalf. Summarize findings
  for the user and let them post in their own words.
- Before working on an issue, read it (`gh issue view <n>`) for the current
  scope and discussion.
- A pull request fixes one thing, links the issue it resolves
  (`Fixes #N`), and has a short description of what changed and why.

## Judgment calls

- Keep changes small and focused. Don't bundle unrelated cleanups into a fix.
- Prefer code that is easy to read and debug over clever or abstract code;
  most users are scientists who will read the source when a result looks
  wrong.
- Every behavior change gets a test; every bug fix gets a test that fails
  without it.
- Prefer readability over micro-optimizations unless a profile shows the
  cost. Real runs are dominated by HYSPLIT and I/O.

## Gotchas and science notes

- **Results are plain data.** `sim.particles` is a pandas DataFrame and
  `sim.footprint` an xarray DataArray; PYSTILT's methods on them live in
  `.stilt` accessors. A footprint carries its receptor id as a scalar
  coordinate (kept through arithmetic) and the full receptor and settings
  as attributes (`stilt_receptor`, `stilt_footprint`), which the accessor
  reads. Do not add wrapper classes back. Every result file records what
  it needs to be read alone (`read_particles`, `read_footprint`).
- **`project.simulations` is built once and copied.** Each access returns
  a copy, so a user's new column stays theirs. `add_receptors` drops the
  cache; a `config.yaml` edited by hand needs a new `Project(path)`.
- **Empty footprints are successes, and not footprints.** When no particle
  reaches the grid, `calc_footprint` raises `EmptyFootprint` and
  the worker's `make_footprint` writes a footprint file with no rows,
  marked `stilt:empty` in its metadata. `sim.is_complete` and
  `sim.has_footprint` are true, `sim.footprint` is `None`, and
  `project.footprints()` gives it no row and lists it in
  `attrs["empty"]`. Never synthesize a zero-valued footprint for it: a zero
  enhancement would flow into a comparison or an inversion unnoticed.
- **A failure record is a note, not a result.** When a step fails the
  worker writes `<receptor id>.failure.yaml` in the logs of the folder
  whose result failed (`logs/settings=<key>/date=.../`: the particles'
  folder for the group, the footprint's for one variant), and removes it
  when that result is written. Completion never reads it: a simulation is
  done by its result files alone. `sim.failure`, the `state`, `step`,
  `reason`, and `message` columns of `status()`, and `stilt status` read
  it, only for simulations missing a result.
- **A result that is not written yet raises.** `sim.particles` and
  `sim.footprint` are `cached_property` on a frozen value; a missing file
  raises `FileNotFoundError` (never cached, so the next read tries again)
  rather than caching a `None` that would hide the result when it lands. `None`
  is only for final states (no grid, empty footprint). Do not write to a
  simulation's `__dict__` by hand; use `has_particles` / `has_footprint` to
  test presence.
- **Realizations are an axis, not names.** `realizations: N` makes one
  variant whose simulations are `(receptor, variant, k)`, `k` in `0..N-1`,
  the `realization` column of `project.simulations`. They share one
  settings folder, a `realization=k/` partition each; the run record says
  `ensemble: true` with the base seed, and realization `k` runs with
  `seed + k` (`Variant.transport_for(k)`). An ensemble of one is still an
  ensemble, so raising N only adds partitions. A single run records
  `realization: null`, as before, so its hash did not change.
- **HYSPLIT line-source chaining.** In `emspnt.f`, consecutive CONTROL
  starting locations at the same lat/lon become one vertical line source and
  only the last pair is released. That is how `ColumnReceptor` works (two
  lines: bottom and top), and why a `MultiPointReceptor` may not repeat a
  horizontal location (the constructor raises). The bundled build releases
  column particles bottom-to-top in `indx` order (the table's `particle`), which
  `add_release_heights` (`particles/table.py`) falls back to for `xhgt` when a
  model writes no `t = 0` release row.
- **Pressure weighting is derived from the particles.**
  `PressureWeighting` fits `ln p = b + a·z` to the particles' first-step
  `(zagl, pres)` and gives each distinct release height the pressure slab
  centred on it (`particle_pwf`), split evenly among the particles released
  there (a multipoint receptor releases many per point). The fit is made in
  the receptor's own datum: for `altitude_ref="msl"` it uses `zagl + zsfc`
  (so `zsfc` must be in `varsiwant`) and the ground closing the bottom slab
  is the terrain under the lowest point. `PressureWeighting.apply` reads
  `altitude_ref` from the receptor it is given; with none it assumes AGL. `AveragingKernel` holds only the kernel.
  `calc_footprint` divides by the particle count, so weights are scaled by
  `N`. Weights sum to the column's mass fraction (< 1) by design; the rest of
  the atmosphere is above the column top. This deliberately differs from
  X-STILT's "layer below each particle" convention, which gives a
  surface-released particle zero weight.
- **The hypsometric fit is load-bearing.** HYSPLIT's first output step is
  already one timestep of turbulence past release: on a real 1000-particle
  column, 394 of 999 adjacent particles are non-monotone in pressure versus
  release height. Raw first-step pressures give neighbouring particles weights
  spanning 145×; the fit gives 1.49×, the true hydrostatic ratio across 3 km.
  Never "simplify" `particle_pwf` to use `pres` directly
  (`tests/integration/test_pressure_weighting.py` guards this).
- **Exact release positions would not help PWF.** PARTICLE.DAT starts at
  `t = -DELT`, never `t = 0`, but the scatter is not only transport:
  `emspnt.f` places each particle uniformly at random inside its own
  `1/numpar` slab. The air a particle represents is its slab, known
  analytically from `numpar` and the column, which is what `xhgt` is. Particle
  data is needed only for the two-parameter `p(z)` fit.
- **Release heights come from the `t = 0` row.** The particle table's
  contract asks a model to write each particle's release at `time = 0`. From
  it `add_release_heights` (`particles/table.py`, a core step the worker applies
  to any model's particles, as is `correct_near_field`) finds which point of
  a multipoint receptor a particle left from, or which slab of a column;
  `xhgt` is that point's altitude or that slab's centre, never the random
  height inside the slab.
- **Multipoint and slant `xhgt` recovery** (`_multipoint_release_heights` in
  `particles/table.py`) prefers, in order: `t = 0` rows if present (exact), a match
  on height when release altitudes are distinct (about 20 m apart), then
  horizontal position with a warning under 1 km. The bundled HYSPLIT v5.1.0
  writes no `t = 0` row; until a published build does, `exe_dir`
  can point at a patched `hycs_std`, and nothing needs undoing when one lands.
  The existing multipoint tests space points about 17 km apart, the one regime
  where horizontal matching works; they do not cover close-spaced slants
  (`tests/transport/hysplit/test_release_assignment.py` does).
- **Transport-error settings** behave in ways that are easy to misread; see
  `docs/guides/transport_error.rst` before changing or validating them.
- **The HYSPLIT seed is remapped.** `SETUP.CFG` gets `SEED = -(|seed| + 1)`
  (`setup_seed` in the HYSPLIT driver), not the user's value: HYSPLIT sets its generator
  state to `-1 + SEED`, and under `krand=2` the generator (`ran1`)
  re-initializes only from a negative value and collapses every state `>= -1`
  onto one stream, so a positive `SEED` is inert. `krand=4` discards the seed
  (clock draw with ~5000 distinct values), and `krand=1` uses it only for the
  initial turbulent velocity, so `seed` requires `krand=2`. `krand` is
  restricted to HYSPLIT's documented modes because any other value silently
  degenerates the turbulence draws. Realization `k` of a variant runs with
  `seed + k`; realization 0, and a single wind-error variant, share the
  default seed, as STILT-R's error run does, which is what the `winderr`
  fidelity scenario (an `hrrr-err` variant) relies on. The R
  fidelity fixture applies the same seed mapping.
- HYSPLIT binaries in `src/stilt/transport/hysplit/bin/` are Linux x86-64.
  Real runs are heavy; on a shared HPC system run them through the Slurm
  backend or an allocation, not on a login node.
