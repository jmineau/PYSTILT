# AGENTS.md

Orientation for contributors and coding agents working on PYSTILT: the
architecture, the invariants that must not break, and the development
workflow. User docs: `docs/` and <https://jmineau.github.io/PYSTILT>. Longer
references: [CONTRIBUTING.md](CONTRIBUTING.md) (setup, checks, recipes,
building, releasing), [docs/advanced/architecture.rst](docs/advanced/architecture.rst)
(module map, runs, configuration internals, implementation notes), and
[docs/development.rst](docs/development.rst) (STILT-R parity, writing the docs).

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

- **Names.** Say `pystilt` (the PyPI name) only for installing, the
  repository, and docs URLs; the import name and the CLI are `stilt`. The R
  implementation ([uataq/stilt](https://github.com/uataq/stilt)) is
  **STILT-R**, never "R-STILT"; identifiers such as the `r_stilt` fixtures and
  `STILT_R_DIR` keep their names.
- **Particle columns** get plain names for HYSPLIT's and STILT-R's `indx`,
  `time`, `long`, `lati`, `xhgt` (`docs/reference/particles.rst`).
- **Config** is what the user writes, always a class next to the code that
  uses it (`ProjectConfig`, `FootprintConfig`, `HysplitConfig`, ...).
  **Settings** are what a result was made with, as recorded; never a class.
- **Sister projects.** STILT-R gives the transport science;
  [stiltctl](https://github.com/jmineau/air-tracker-stiltctl) the thin CLI →
  `Project` → worker path; [X-STILT](https://github.com/uataq/X-STILT) the
  observation layer and column science (concepts ported, not feature parity).

## Architecture

**There is no index, manifest, or registry.** A project is a directory of
inputs (`Project`: `config.yaml`, `receptors.csv`) and an output directory
(`Output`) that `config.yaml` names and several projects can share. Its
simulations are **receptors × variants** (at least one variant). Each `Variant` (`stilt.config`) has
run settings, whose hash names its particles folder, and optional footprint
settings (`stilt.identity`); variants with equal run settings share one run
per receptor, and a changed setting hashes to a new folder
(`docs/reference/layout.rst`).

**`Project` reads; the workers write results.** `Project` writes only its
inputs (`config.yaml` once, at `Project.init`; `add_receptors` appends to
`receptors.csv`; `add_table` writes `tables/<name>.parquet`) and knows no
scheduler or compute root; `stilt.execution.run` and `submit` start the
workers. **Tables are plain DataFrames; the verbs are on `Project`**: a
selection of `project.simulations` (any table with `receptor` and `variant`
columns, or a mask) goes to `status`, `incomplete`, `particles`, `footprints`,
or `jacobian`. Do not wrap the table in a class again. `Output` has no folder
classes. `stilt.__all__` (and each subpackage's) is the public surface.

```
src/stilt/
  cli.py          Typer CLI, a thin adapter over Project and execution
  project.py      Project: inputs, the simulations table, run/submit, bulk reads
  check.py        check_project: what would stop a project's runs, from its inputs
  migration.py    read_stilt_r: STILT-R and X-STILT by-id output, read with rdata
  output.py       Output: the output directory; `completed`, the completion rule
  simulation.py   Simulation: a frozen (receptor, variant, output) value
  config.py       ProjectConfig and Variant;  identity.py  settings records, hashes
  receptors/      receptor models, receptors.csv, shared validation
  particles/      the particle table and files, release heights, near-field correction
  footprint/      FootprintConfig, calc_footprint (gridding.py), aggregation, io, targets
  transforms/     pre-footprint particle transforms, one module each
  transport/      TransportModel/TransportConfig protocols, MODELS, get_model, run_model
    hysplit/      config, met, driver, CONTROL and SETUP.CFG writers, failures, bin/, data/
  execution/      ExecutionConfig, runner (plan, local run, Slurm submit), worker
  observations/   the X-STILT port: readers, sounding selection, slant receptors, columns
  meteorology.py, sampling.py, spatial.py, visualization.py, exceptions.py, _paths.py
```

**A run** (`stilt run`, `Project.run()`, `submit()`) happens in this process
or as a Slurm job array whose tasks each run `stilt run --task i/N`. The unit
of work is a receptor: the worker runs the model once per run-settings hash,
then makes every footprint that shares the particles (`run_receptor`); a batched model
(`batched = True`) first runs a variant group's receptors in one call.
**Observation-driven** work sits above the core: `stilt.observations` turns
soundings into receptors, then a run goes as usual. Keep it out of `project.py`.

The import contracts in `pyproject.toml` (`just imports`) hold this shape: an
exhaustive `layers` contract (a new top-level module fails until placed), only
`stilt.transport` imports HYSPLIT's package, and the project config reads no
results and runs nothing. Slurm tasks rebuild the project in another process,
so anything a worker needs must be in the project or the output directory.

`ProjectConfig` holds the transport model (`hysplit` unless set), its
parameters and the footprint fields as flat defaults, `mets`, and `variants`.
Each part owns its config class; the base `TransportConfig` gets a parameter
only when the core reads it. Config classes do no I/O when they validate;
`ProjectConfig.resolve` reads geometries and asks the model its version.

## Invariants

- **STILT-R parity** at `rtol=1e-7` per footprint cell against a pinned commit
  (`docs/development.rst`). Run the `fidelity` suite before merging a change to
  trajectory or footprint math. NetCDF output is CF-1.8, deliberately not
  byte-compatible with STILT-R.
- **Completion is by file.** A simulation is complete iff its files exist in
  the output directory; `stilt.output.completed` is the one rule. Never add
  another, a completion registry, or a manifest.
- **Bulk methods work from listings** and receptor ids, never a `Simulation`
  per row; they open a file per row only when its contents are the answer
  (`footprints`, `jacobian`).
- **One code path for a disk and an object store, no store class.**
  `Output.directory` is a `Path` (local fast path) or a `UPath` (`path.fs`);
  see `stilt._paths`. A local output must stay as fast: time a large listing.
- **Identity is content.** PYSTILT never deletes or overwrites a results folder.
- **The CLI stays thin.** Orchestration does not belong in `cli.py`.
- **Meteorology I/O goes through [arlmet](https://github.com/jmineau/arl-met).**
- **Bundled HYSPLIT is package data** (Linux and macOS x86-64), reached via
  `importlib.resources`, never a repo path. Each wheel carries only its
  platform's `hycs_std`, the sdist none. A new platform needs a case in
  `bundled_build` (`setup.py`), `_bundled_exe_dir`, and `just dist`/`check-dist`.
- **Pydantic for all configuration**: every field a plain
  `Field(default, description=...)`, and the public config stays flat.

## Development

Driven by [`just`](https://github.com/casey/just) and
[`uv`](https://docs.astral.sh/uv/); `just` alone lists every recipe. CI
(`.github/workflows/`) runs the same recipes. The tooling comes from
[jmineau/python-template](https://github.com/jmineau/python-template)
(`.copier-answers.yml`; `copier update` pulls in its changes).

| Command | What it does |
|---|---|
| `just sync` | `uv sync`: the package and the dev tools into `.venv` |
| `just test` | the unit tests, in parallel (extra args go to pytest; `-n 0` for serial) |
| `just lint` / `just format` | ruff check and format check / fix and format |
| `just type-check` | pyrefly, against Python 3.11 (set in `pyproject.toml`) |
| `just imports` / `just docstr` | the import contracts / docstring coverage of the public API (95%) |
| `just quality-check` | lint, type check, import contracts, docstrings, unit tests |
| `just pre-commit` | every pre-commit hook on every file |
| `just build-docs` | a clean Sphinx build that runs every example; warnings are errors |
| `just dist` | the sdist and one wheel per bundled HYSPLIT build, then `just check-dist` |

### Tests

`just test` runs every test without one of three disjoint markers. Warnings
are errors (a test that expects one uses `pytest.warns`). CI also runs
`-m "integration or fidelity"` and `-m r_only`:

- `integration`: end to end with real met files and HYSPLIT. Needs
  `STILT_TEST_MET_DIR` (HRRR ARL files) or `STILT_TEST_FETCH_MET=1` (downloads
  seven 6 h blocks into the ignored `tests/met_cache/`); otherwise they skip.
- `fidelity`: PYSTILT against STILT-R, both running HYSPLIT. Needs the met,
  `STILT_R_DIR` (a STILT-R checkout built as `tests.yml` builds it), `Rscript`.
- `r_only`: STILT-R comparisons with no met or HYSPLIT (`STILT_R_DIR`, `Rscript`).

pytest loads a local `.env` for such variables. Test folders follow the
package. `tests/observations/` is self-contained (its own `conftest.py` and
`data/`, never `tests/conftest.py`'s fixtures). Committed test data stays
small: slices of public granules (made by
`tests/observations/data/products/make_samples.py`) or synthetic files with a
format's quirks; never met files or data that is not ours to redistribute.

### Code conventions

- Keep changes small; don't bundle unrelated cleanups into a fix. Every
  behavior change gets a test, every bug fix a test that fails without it.
- Prefer code that is easy to read and debug (users are scientists who read
  the source when a result looks wrong) over micro-optimizations, unless a
  profile shows the cost; HYSPLIT and I/O dominate real runs.
- Python 3.11+ (use `typing.Self`, `enum.StrEnum`, `datetime.UTC`, not
  backports). Ruff rules are in `pyproject.toml`; **NumPy-style docstrings**
  with the summary on the line after the opening quotes (`D213`). Keep a
  module's `from __future__ import annotations` header where it has one.
- pyrefly must pass on `src/`. Fix types at the source, not with
  `typing.cast`; for an error in a library's stubs (often pandas-stubs), put
  `# pyrefly: ignore[<code>]` on the line above, after a comment saying why.
- Exceptions live in `stilt/exceptions.py`, each subclassing `StiltError` and
  the builtin that describes it; a failed run is a `SimulationError`. Plain
  input checks raise builtins such as `ValueError`.
- Extras: `geometry`, `visualization`, `download`, `sparse`, `stilt-r`, and
  `complete`, which the `dev` group installs.

### Documentation

A user-facing change updates its guide or reference in `docs/` in the same
change, and `just build-docs` must pass. Examples run when the docs build, so
they use synthetic data, a small repository file, a public download, or the
test met cache, and seed their random numbers (`docs/development.rst`). Give
every public property and attribute a docstring, with no colon in the middle
of its first line (napoleon reads `text: more` as `type: description`).

**Voice** (docs, docstrings, field descriptions, CLI help): write for
scientists who want to run the model and trust the result. Open with what it
does, then an example. One idea per sentence; no metaphors, selling, or
filler; American spelling. The full guide, with the habits that read as
machine-written, is "Writing documentation" in `docs/development.rst`. Words:

| Say (user-facing) | Not |
|---|---|
| settings folder | key, hash, identity |
| transport model (in full) | model, when the weather model or a modeled value is near |
| particles (the table); run (the verb); trajectory (one particle's path) | run as a noun for the result |
| workdir (one simulation's folder); compute root (where workdirs are made) | scratch, except the output's `scratch/` folder |
| log, met files | provenance |
| receptor id; simulation (a receptor under a variant) | simulation ID as a folder |
| the settings of a variant | resolve, record, `model_copy`, frame, handle |

### Commits, releases, issues

- Conventional Commits (`fix(slurm): ...`, `docs: ...`); user-visible changes
  go under `## [Unreleased]` in `CHANGELOG.md`. The version comes from git tags
  (setuptools-scm). Releases are cut by the maintainer (CONTRIBUTING.md).
- Open work is in [GitHub issues](https://github.com/jmineau/PYSTILT/issues);
  read one before working on it. Feature status lives in `docs/roadmap.rst`
  (update its row when a feature lands). Over-engineered code goes on the
  issue labeled `simplify`. A pull request fixes one thing and links its issue.

## Gotchas

The reasons are in `docs/advanced/architecture.rst` ("Implementation notes").

- **Results are plain data**: `sim.particles` a DataFrame, `sim.footprint` a
  DataArray, with `.stilt` accessors; no wrapper classes. A missing result
  raises `FileNotFoundError` (not cached); `None` is only for final states, so
  test with `has_particles` / `has_footprint`.
- **`project.simulations` is built once and copied** on each access; a
  hand-edited `config.yaml` needs a new `Project`.
- **Empty footprints are successes.** `calc_footprint` raises `EmptyFootprint`
  and the worker writes a no-row file marked `stilt:empty`: complete, with
  `sim.footprint` `None` and a listing in `project.footprints().attrs["empty"]`.
  Never synthesize a zero footprint for it.
- **A failure record is a note** (`<receptor id>.failure.yaml` in the logs),
  removed when the result is written and never read by completion.
- **Met that ran out is a failure; a domain exit is not.** HYSPLIT's warnings
  are unreliable, so `Met.files_for` and `Met.check` stop a run on missing or
  damaged met before it starts (`FAILURE_PHRASES` maps the log afterwards).
  `stilt check` runs the file search for every receptor early; it opens no met
  file past one header, so a damaged file still fails at run time.
- **Realizations are an axis**: `realizations: N` is one variant with a
  `realization` column and `realization=k/` partitions, run with `seed + k`.
  HYSPLIT gets `SEED = -(|seed| + 1)` (`setup_seed`, mirrored by the R
  fidelity fixture), and `seed` needs `krand: 2`.
- **HYSPLIT chains same-location starts into a line source**: that is how
  `ColumnReceptor` works, and why a `MultiPointReceptor` may not repeat a
  horizontal location.
- **HYSPLIT stops releasing at `MAXPAR`** and fills a multipoint receptor's
  locations in order. An unset `maxpar` is chosen per receptor
  (`HysplitConfig.maxpar_for`), so it is recorded unset; `check_release` fails
  a multipoint run whose `WARNING` says `exceeding puff limit` (a one-location
  run writes it too, harmlessly).
- **Pressure weighting fits `ln p = b + a·z`** (`particle_pwf`); never use raw
  `pres` (`tests/integration/test_pressure_weighting.py` guards it).
- **Release heights come from the `age = 0` row** (`add_release_heights`), with
  a fallback for models that write none, as the bundled HYSPLIT does.
- **Transport-error settings** are easy to misread: read
  `docs/guides/transport_error.rst` before changing or validating them.
- **Never name a workdir from the simulation alone**: each run gets a new path
  in the compute root (`_new_workdir`), so projects can share one. Real runs
  are heavy: on a shared HPC system use Slurm or an allocation.
