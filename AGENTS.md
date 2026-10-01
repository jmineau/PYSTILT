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
| Source directory | `src/stilt/` |
| CLI entry point | `stilt` (Typer; see `[project.scripts]`) |

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
`config.yaml` (one per met when none are declared). A variant resolves into
transport settings (`TransportSettings`, whose hash identifies a *run*) and
optional footprint settings. Variants with equal transport settings share
one run per receptor and differ only in the footprint made from it. Whether
a simulation is complete is decided **by the files in the output
directory**, by `Simulation.is_complete()`: that method is the single
definition of "done". `config.yaml` and `receptors.csv` are the user's
inputs: `Project.init` (or `stilt init`) writes `config.yaml` once, PYSTILT
never rewrites it, and `Project.add_receptors` only appends to
`receptors.csv`. Changing a setting never overwrites a result: it hashes to
a new folder.

**`Project` reads; the workers write results.** `Project(path)` opens a
project directory: its config, its receptors, the simulations they define,
and a view of their results. Its only write is `add_receptors`, which
appends to `receptors.csv`. It knows no scheduler or scratch directory.
`stilt.execution.run` and `submit` (which `Project.run()` and
`Project.submit()` call) find the receptors with missing results, resolve
the compute root, and start the workers, which are the only code that
writes results.

**Plurals are DataFrames; singular things are objects.**
`project.receptors` and `project.simulations` are DataFrames selected with
pandas, and `status`, `incomplete`, `load_particles`, and
`load_footprints` take such a selection. `project.receptor(id)` and
`project.simulation(id, variant)` return the objects. Do not add a custom
collection class.

`stilt.__all__` (plus the `__all__` of each subpackage) is the public surface;
everything else is internal and can change.

### Module layout

```
src/stilt/
  cli.py             Typer CLI; a thin adapter over Project and execution
  project.py         Project: the project directory, its receptors and
                     simulations as DataFrames, status, loading, run/submit
  output.py          Output: the output directory. Particles and Footprints are
                     its folders, one per settings hash; sparse footprint
                     files; Jacobian assembly
  simulation.py      Simulation, SimID: a frozen value (receptor, variant, output)
                     that knows where its results are and whether they exist
  receptors.py       receptor types (frozen pydantic models: point, column,
                     multipoint), their ids, and the receptor table behind
                     the CSV reader, writer, and appender
  particles.py       the particle table (a DataFrame): prepare, read, and write
                     particle files; the `.stilt` pandas accessor
  footprint.py       the footprint (a DataArray): `calculate`, `read_footprint`,
                     CF-1.8 NetCDF, and the `.stilt` xarray accessor
                     (enhancement from a flux field, aggregation)
  flux.py            sampling a flux field at points or along particles
  geometry.py        aggregation targets (meshes, zones) and overlap weights
  meteorology.py     Met: ARL file discovery, download, and cropping (via arlmet)
  transforms.py      pre-footprint particle transforms (averaging kernel,
                     pressure weighting, lifetime decay) and their YAML I/O
  exceptions.py      every exception class, all under StiltError
  visualization.py   matplotlib helpers (optional dependency)

  config/            pydantic configuration: ProjectConfig and its parts
  execution/         the runner (saves a model's inputs, plans what is missing,
                     runs it here or submits batches to Slurm through submitit)
                     and the worker (runs HYSPLIT on scratch and writes results
                     for one or many simulations)
  observations/      the X-STILT port, all before or after the transport run:
                     product readers, overpass grouping and sounding
                     selection, slant geometry, transport error, wind-error
                     statistics, backgrounds, plume backgrounds. Arrays in,
                     plain values out; there is no observation object.
  transport/         the TransportModel protocol and get_model (__init__.py),
                     one subpackage per transport model
    hysplit/         HYSPLIT, the one model today: HysplitModel, the driver
                     (CONTROL / SETUP.CFG writers), failure reasons read from
                     its log (failures.py), and the bundled binaries (bin/)
                     and data tables (data/)

tests/               pytest; markers `integration` and `fidelity`
docs/                Sphinx (pydata-sphinx-theme)
```

### Two ways work starts: do not conflate them

1. **A run** (`Project.run()`, `Project.submit()`, or `stilt run`):
   `stilt.execution.run` finds the receptors with missing results and
   either runs them in this process (`backend: local`) or submits them as
   one Slurm job array through submitit (`backend: slurm`), one `Batch` of
   receptors per task, and waits. `submit` returns the jobs at once. The unit of work is a receptor: `run_receptor` runs
   HYSPLIT once per distinct transport hash, then writes the footprint of
   every variant that shares those particles.
2. **Observation-driven**: a reader yields a DataFrame of soundings;
   `stilt.observations` helpers thin and group it; each row becomes a
   `Receptor`; `averaging_kernel_table` writes the kernels into the project;
   then `add_receptors` and run as usual. This layer sits *above* the
   transport core. Keep observation logic out of `project.py`. An import-linter
   contract in `pyproject.toml` (run by `lint-imports` in CI) fails when a
   core module imports `stilt.observations`; a new top-level module goes on
   that contract's list.

Slurm tasks rebuild the model from the project in another process on
another node, so anything a worker needs must be in the project or the
output directory, never only in memory.

### Configuration

- `ProjectConfig` is the root: flat transport (`STILTParams`) and footprint
  (`FootprintConfig`) defaults, `mets`, and `variants` (overrides of the
  defaults). `ProjectConfig.resolve_variants()` turns them into one
  `VariantConfig` per simulation name, expanding `realizations: N` into
  `<name>-0..N-1` with `seed + k`. A `VariantConfig` is composed:
  `transport: TransportSettings` (hashed, names the particles folder) and
  `footprint: FootprintConfig | None`. Variants whose transport settings
  match share the particles; `from:` is rejected. `grid: null` means
  particles only, and footprint settings without a grid are an error. There is no
  named-footprints dict.
- Every field is a plain pydantic `Field(default, description=...)` and the
  public config stays flat (`ProjectConfig(numpar=..., seed=...)`). CONTRIBUTING
  explains how a field is routed to `SETUP.CFG`, `CONTROL`, `WINDERR`, or
  `ZIERR`.
- `RuntimeSettings` is one `pydantic-settings` class reading `PYSTILT_*`
  environment variables (`compute_root`). The runner and the workers read
  it; `Project` does not.
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
  slurm/<stamp>/              one folder per Slurm submission: script, task logs,
                              and submitit's pickles

<output>/
  particles/settings=<variant>-<hash>/_settings.yaml
  particles/settings=<variant>-<hash>/date=YYYY-MM-DD/<receptor_id>.parquet
  footprints/settings=<variant>-<hash>/_settings.yaml      names the particles folder
  footprints/settings=<variant>-<hash>/date=YYYY-MM-DD/<receptor_id>.parquet
  logs/settings=<variant>-<hash>/date=YYYY-MM-DD/<receptor_id>.log
  scratch/settings=<variant>-<hash>/date=YYYY-MM-DD/<receptor_id>/   failed runs' working dirs
```

A particles folder's hash is `TransportSettings.hash`; a footprint
folder's is the hash of the transport settings and the footprint settings
together. "Particles" is the one word for the particle table in code
(`Particles`, `sim.particles`, `has_particles`); "run" is only the verb, and
"trajectory" means one particle's path. Lookup
re-validates the stored `_settings.yaml` through the current models and
re-hashes, so a field added later with a default still matches. `compute_root`
is scratch: HYSPLIT runs there and the directory is discarded after success.
Footprints are sparse tables (`hour, y, x, foot`, float32); an empty
footprint is a file with no rows and the reason in its metadata, and counts
as complete.

## Invariants

- **STILT-R numerical parity** at `rtol=1e-7` per footprint cell. Run the
  `fidelity` suite before merging any change to trajectory or footprint math.
  NetCDF output is CF-1.8 and deliberately not byte-compatible with STILT-R.
- **Completion is by file.** A simulation is complete iff its files exist
  in the output directory. `Simulation.is_complete()` says so for one
  simulation, and `Project._present()` reads the same rule for
  many from a listing of the date folders the selection falls in (a test
  holds the two together). Never add a second
  "does this output exist" check, a completion registry, or a manifest; call
  the `Simulation` method.
- **Identity is content.** A results folder is its settings hash; a changed setting is
  a new folder, never an overwrite, and PYSTILT never deletes a folder.
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
`publish.yml` for releases.

### Tests

Plain `pytest` runs the unit tests. Two opt-in markers:

- `-m integration`: end-to-end runs with real met files and HYSPLIT. Slow.
  The `met_dir` fixture needs `STILT_TEST_MET_DIR` pointing at a directory
  of HRRR ARL files, or `STILT_TEST_FETCH_MET=1` to download the seven 6 h
  blocks it needs into the ignored `tests/met_cache/`. Once downloaded, set
  `STILT_TEST_MET_DIR=tests/met_cache`; without either variable every
  integration test is skipped.
- `-m fidelity`: live comparison against STILT-R. Slow; needs `STILT_R_DIR`
  pointing at a STILT-R checkout and `Rscript` on `PATH`.

A local `.env` is loaded by `pytest-dotenv`, which is the place for
`STILT_R_DIR` and similar settings.

Committed test data (`tests/data/`) stays small and synthetic. Do not commit
real instrument retrievals or met files: they are large and often not ours to
redistribute. Synthetic samples should keep the real format's quirks.

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
  `visualization`, `cloud`, and `complete`. The `dev` dependency group pulls
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
Feature status lives in the roadmap tables in [README.md](README.md) and
[docs/roadmap.rst](docs/roadmap.rst); when a feature lands, update both.

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
- **`project.simulations` is a cached DataFrame.** `add_receptors` drops the
  cache; a `config.yaml` edited by hand needs a new `Project(path)`.
- **Empty footprints are successes, and not footprints.** When no particle
  reaches the grid, `footprint.calculate` raises `EmptyFootprint` and
  the worker's `write_footprint` writes a footprint file with no rows and
  the reason in its metadata. `sim.is_complete()` is true, `sim.footprint` is
  `None`, `sim.empty_reason` says why, and `load_footprints()` leaves the
  simulation out. Never synthesize a zero-valued footprint for it: a zero
  enhancement would flow into a comparison or an inversion unnoticed.
- **A result that is not written yet raises.** `sim.particles` and
  `sim.footprint` are `cached_property` on a frozen value; a missing file
  raises `FileNotFoundError` (never cached, so the next read tries again)
  rather than caching a `None` that would hide the result when it lands. `None`
  is only for final states (no grid, empty footprint). Do not write to a
  simulation's `__dict__` by hand; use `has_particles` / `has_footprint` to
  test presence.
- **Declaring `realizations` makes a numbered group, even at 1.** `hrrr-err`
  with `realizations: 1` is `hrrr-err-0`, so raising the count later only
  adds simulations. Realization 0 is never aliased to the unsuffixed name.
- **Declared variants replace the per-met defaults.** With a `variants`
  section only its entries run; the starter config writes `hrrr: {}` so the
  unchanged run stays visible.
- **HYSPLIT line-source chaining.** In `emspnt.f`, consecutive CONTROL
  starting locations at the same lat/lon become one vertical line source and
  only the last pair is released. That is how `ColumnReceptor` works (two
  lines: bottom and top), and why a `MultiPointReceptor` may not repeat a
  horizontal location (the constructor raises). The bundled build releases
  column particles bottom-to-top in `indx` order, which
  `particles.prepare` relies on for `xhgt`.
- **Pressure weighting is derived from the particles.**
  `PressureWeighting` fits `ln p = b + a·z` to the particles' first-step
  `(zagl, pres)` and gives each distinct release height the pressure slab
  centred on it (`particle_pwf`), split evenly among the particles released
  there (a multipoint receptor releases many per point). The fit is made in
  the receptor's own datum: for `altitude_ref="msl"` it uses `zagl + zsfc`
  (so `zsfc` must be in `varsiwant`) and the ground closing the bottom slab
  is the terrain under the lowest point. `PressureWeighting.apply` reads
  `altitude_ref` from the `TransformContext`; with no context it assumes
  AGL. `AveragingKernel` holds only the kernel.
  `footprint.calculate` divides by the particle count, so weights are scaled by
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
  (`tests/test_pwf_integration.py` guards this).
- **Exact release positions would not help PWF.** PARTICLE.DAT starts at
  `t = -DELT`, never `t = 0`, but the scatter is not only transport:
  `emspnt.f` places each particle uniformly at random inside its own
  `1/numpar` slab. The air a particle represents is its slab, known
  analytically from `numpar` and the column, which is what `xhgt` is. Particle
  data is needed only for the two-parameter `p(z)` fit.
- **Multipoint and slant `xhgt` recovery** (`_multipoint_release_heights` in
  `particles.py`) prefers, in order: `t = 0` rows if present (exact), a match
  on height when release altitudes are distinct (about 20 m apart), then
  horizontal position with a warning under 1 km. The bundled HYSPLIT v5.1.0
  writes no `t = 0` row; until a published build does, `STILTParams.exe_dir`
  can point at a patched `hycs_std`, and nothing needs undoing when one lands.
  The existing multipoint tests space points about 17 km apart, the one regime
  where horizontal matching works; they do not cover close-spaced slants
  (`tests/test_hysplit_release_assignment.py` does).
- **Transport-error settings** behave in ways that are easy to misread; see
  `docs/guides/transport_error.rst` before changing or validating them.
- **The HYSPLIT seed is remapped.** `SETUP.CFG` gets `SEED = -(|seed| + 1)`
  (`STILTParams.setup_seed`), not the user's value: HYSPLIT sets its generator
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
