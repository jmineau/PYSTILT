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
  execution patterns: queue-backed workers over a Postgres work queue and a
  thin CLI → `Model` → worker call path.
- **[X-STILT](https://github.com/uataq/X-STILT)**: source of the observation
  layer and column science. PYSTILT ports the concepts, not the scripts, and
  does not aim for X-STILT feature parity.

## Architecture

**There is no index, manifest, or registry.** A project is one root
(`stilt.project.Project`) over a store (`stilt.store`). The simulations it
defines are `receptors.csv × config.mets`. Whether a simulation is complete is
decided **by key**, by `Simulation.is_complete()`: that method is the single
definition of "done". The optional Postgres work queue in `stilt.service`
tracks work status only.

`stilt.__all__` (plus the `__all__` of each subpackage) is the public surface;
everything else is internal and can change.

### Module layout

```
src/stilt/
  cli.py             Typer CLI; a thin adapter over Model / service / execution
  model.py           Model, the top-level orchestrator
  project.py         Project: one root (local dir or URI), its store and key
                     layout; loads/saves config.yaml and receptors.csv
  store.py           Store protocol and the local / fsspec implementations
  simulation.py      Simulation, SimID: per-receptor outputs, their store keys,
                     completion, and publishing from the compute root
  receptors.py       receptor types (point, multipoint, column) and IDs
  trajectory.py      Trajectories: particle output container + Parquet I/O
  footprint.py       Footprint: gridded CF-1.8 NetCDF output, enhancement from
                     a flux field, aggregation onto other spatial targets
  flux.py            sampling a flux field at points or along particles
  geometry.py        aggregation targets (meshes, zones) and overlap weights
  meteorology.py     MetStream: ARL file discovery and staging (via arlmet)
  transforms.py      pre-footprint particle transforms (averaging kernel,
                     pressure weighting, lifetime decay) and their YAML I/O
  collections.py     the query surface over receptors × mets and their outputs
  errors.py          failure reasons and structured error types
  visualization.py   matplotlib helpers (optional dependency)

  config/            pydantic configuration: ModelConfig and its parts
  execution/         the worker (run one or many simulations, or pull from the
                     queue) and backends/ (local, slurm, kubernetes)
  observations/      the X-STILT port, all before or after the transport run:
                     product readers, overpass grouping and sounding
                     selection, slant geometry, transport error, wind-error
                     statistics, backgrounds, plume backgrounds. Arrays in,
                     plain values out; there is no observation object.
  service/           optional Postgres work queue and Kubernetes manifests
  hysplit/           HYSPLIT driver (CONTROL / SETUP.CFG writers) plus the
                     bundled binaries (bin/) and data tables (data/)

tests/               pytest; markers `integration` and `fidelity`
docs/                Sphinx (pydata-sphinx-theme)
```

### Three execution paths: do not conflate them

1. **One-off** (notebook or script): `Model.run()` or `stilt run`. Blocks.
2. **Queue / service** (batch): `Model.register()` or `stilt register`
   enqueues; `stilt pull-worker` drains the queue, `stilt serve` runs
   long-lived. Requires `PYSTILT_DB_URL` pointing at PostgreSQL. The queue
   (`model.queue`) records status; completion is still by key. The Slurm
   backend instead pushes fixed chunks of simulation IDs to
   `stilt push-worker`, with no queue.
3. **Observation-driven**: a reader yields a DataFrame of soundings;
   `stilt.observations` helpers thin and group it; each row becomes a
   `Receptor`; `averaging_kernel_table` writes the kernels into the project;
   then register as usual. This layer sits *above* the transport core. Keep
   observation logic out of `model.py`.

Before changing behavior, ask which paths it touches. The worker paths
serialize state across processes, so optimizations that only work in-process
tend to break them.

### Configuration

- `ModelConfig` is the root. It composes `MetConfig`, a dict of named
  `FootprintConfig`s, and `Grid` / `Bounds`.
- Every field is a plain pydantic `Field(default, description=...)` and the
  public config stays flat (`ModelConfig(numpar=..., seed=...)`). CONTRIBUTING
  explains how a field is routed to `SETUP.CFG`, `CONTROL`, `WINDERR`, or
  `ZIERR`.
- `RuntimeSettings` is one `pydantic-settings` class reading `PYSTILT_*`
  environment variables (`db_url`, `cache_dir`, `compute_root`);
  `Model(runtime=...)` overrides it.
- Particle transforms are declared per footprint in YAML
  (`transforms: [{kind: ...}]`); `kind` may also be the import path of a user
  class.

### Project layout on disk

A project root is a local directory or an `s3://` / `gs://` URI. Every output
is a store key relative to it (see `project.py`):

```
<project>/
  config.yaml                 ModelConfig (user-authored)
  receptors.csv               receptor list; register() merges new batches
  simulations/
    by-id/<sim_id>/
      stilt.log               HYSPLIT log
      met/                    staged meteorology (compute-local only)
      *_traj.parquet          trajectories (+ *_error.parquet)
      *_<name>_foot.nc        footprints, or *_foot.empty markers
  chunks/, slurm/             Slurm push-dispatch artefacts (local projects)
```

`compute_root` is the only other location: workers run HYSPLIT there and
`Simulation.publish()` copies outputs into the store. For a local project the
default compute root *is* `simulations/by-id`, so publishing is a no-op. An
empty footprint writes a `.empty` marker, which counts as complete.

## Invariants

- **STILT-R numerical parity** at `rtol=1e-7` per footprint cell. Run the
  `fidelity` suite before merging any change to trajectory or footprint math.
  NetCDF output is CF-1.8 and deliberately not byte-compatible with STILT-R.
- **Completion is by key; the queue is status-only.** A simulation is complete
  iff its expected outputs exist in the store. Never add a second "does this
  output exist" check, a completion registry, or a manifest; call the
  `Simulation` method.
- **State lives in the project store** (config.yaml, receptors.csv, outputs)
  and, on the queue path, in Postgres. Anything kept in a process-local
  variable silently diverges between `run`, `pull-worker`, and `serve`.
- **The CLI stays thin.** `cli.py` adapts arguments to `Model` / service /
  execution calls; orchestration logic does not belong there.
- **Meteorology I/O goes through [arlmet](https://github.com/jmineau/arl-met).**
  Do not reimplement ARL reading here.
- **Bundled HYSPLIT is package data.** Reach the binaries and tables through
  `importlib.resources` via the existing helpers, never a repo path. They ship
  through `[tool.setuptools.package-data]`; moving them means updating that.
- **Pydantic for all configuration**, with a description on every field.

## Development

### Commands

Driven by [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/):

| Command | What it does |
|---|---|
| `just install` | `uv sync --group dev` |
| `just test` | `uv run pytest -v` (unit tests only) |
| `just quality-check` | ruff + pyright on `src/stilt`, then the tests |
| `just ruff` | `ruff check --fix` and `ruff format` on `src/stilt` |
| `just build-docs` | clean Sphinx HTML build into `docs/_build` |
| `just pre-commit` | all pre-commit hooks on all files |
| `just clean` | remove build artifacts, caches, coverage, docs build |

CI (`.github/workflows/`): `tests.yml`, `quality.yml`, `docs.yml`, and
`publish.yml` for releases.

### Tests

Plain `pytest` runs the unit tests. Two opt-in markers:

- `-m integration`: end-to-end runs with real met files and HYSPLIT. Slow.
  Met files are downloaded on demand into the ignored `tests/met_cache/`.
- `-m fidelity`: live comparison against STILT-R. Slow; needs `STILT_R_DIR`
  pointing at a STILT-R checkout and `Rscript` on `PATH`.

A local `.env` is loaded by `pytest-dotenv`, which is the place for
`STILT_R_DIR` and similar settings.

Committed test data (`tests/data/`) stays small and synthetic. Do not commit
real instrument retrievals or met files: they are large and often not ours to
redistribute. Synthetic samples should keep the real format's quirks.

### Code conventions

- Python 3.10+ (`ruff target-version = "py310"`).
- Ruff rules `E, F, UP, B, SIM, I, D213`; line length is left to the formatter.
- **NumPy-style docstrings** (Sphinx napoleon is configured for NumPy only),
  with the summary on the line after the opening quotes (`D213`).
- Pyright in `basic` mode against the project `.venv`. `py.typed` ships.
  Fix types at the source rather than reaching for `typing.cast`.
- Keep the `from __future__ import annotations` headers.
- Runtime dependencies live in `[project]`; optional extras are `projection`,
  `visualization`, `cloud`, and `complete`. The `dev` dependency group pulls
  in `pystilt[complete]` plus the test, lint, type, and docs tooling.

### Documentation

Sphinx sources in `docs/`: `getting_started/` (install, quickstart,
concepts), `guides/` (task-oriented how-tos), `tutorials/` (end-to-end
worked examples), `advanced/` (design and internals), and `reference/` (API
pages built from docstrings). A user-facing change needs a guide or reference
update, and `just build-docs` should build without new warnings.

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

- **`Model.simulations` is a lazy, mapping-like `SimulationCollection`**, not
  a list.
- **Empty footprints are successes.** `model.footprints[name].load(...)`
  treats `.empty` simulations as complete with no file; code that iterates
  results must accept a missing payload.
- **HYSPLIT line-source chaining.** In `emspnt.f`, consecutive CONTROL
  starting locations at the same lat/lon become one vertical line source and
  only the last pair is released. That is how `ColumnReceptor` works (two
  lines: bottom and top), and why a `MultiPointReceptor` may not repeat a
  horizontal location (the constructor raises). The bundled build releases
  column particles bottom-to-top in `indx` order, which
  `Trajectories.from_particles` relies on for `xhgt`.
- **Pressure weighting is derived from the particles.**
  `PressureWeighting` fits `ln p = b + a·z` to the particles' first-step
  `(zagl, pres)` and gives each particle the pressure slab centred on its
  release height (`particle_pwf`). `AveragingKernel` holds only the kernel.
  `Footprint.calculate` divides by the particle count, so weights are scaled by
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
  `trajectory.py`) prefers, in order: `t = 0` rows if present (exact), a match
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
  degenerates the turbulence draws. Error realization `k` runs with
  `seed + k`; realization 0 shares the main seed, as STILT-R's error run
  does, which is what the `winderr` fidelity scenario relies on. The R
  fidelity fixture applies the same seed mapping.
- HYSPLIT binaries in `src/stilt/hysplit/bin/` are Linux x86-64. Real runs are
  heavy; on a shared HPC system run them through the Slurm backend or an
  allocation, not on a login node.
