# AGENTS.md

Orientation for contributors and coding agents working on PYSTILT: the
architecture, the invariants that must not break, and the development
workflow. User docs: `docs/` and <https://jmineau.github.io/PYSTILT>. Longer
references: [CONTRIBUTING.md](CONTRIBUTING.md) (setup, checks, recipes,
building, releasing), [docs/development/architecture.rst](docs/development/architecture.rst)
(module map, runs, finished work, configuration internals, implementation notes), and
[docs/development/stilt_r_parity.rst](docs/development/stilt_r_parity.rst) (STILT-R parity).

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
Lagrangian Transport). It computes gridded footprints for releases from
backward (or forward) particle trajectories, which
[runtraj](https://github.com/jmineau/runtraj) makes by driving **HYSPLIT**.
It is alpha software (`0.1.0a*`): **there are no backward-compatibility
guarantees before v1.0**, so prefer the clean design over a compatibility shim.

- **Two packages.** runtraj makes the particle runs: projects of releases
  and variants, settings folders, completion by file, `run` and `submit`,
  realizations, the met catalog, the transport models. PYSTILT is what STILT
  adds to the particles: footprints, the near-field correction, transforms,
  transport error, background, observations, the STILT-R reader. runtraj
  never imports `stilt`; its `AGENTS.md` has its architecture and gotchas.
  A change that crosses the line is two pull requests, runtraj's first.
- **Names.** Say `pystilt` (the PyPI name) only for installing, the
  repository, and docs URLs; the import name and the CLI are `stilt`. The R
  implementation ([uataq/stilt](https://github.com/uataq/stilt)) is
  **STILT-R**, never "R-STILT"; identifiers such as the `r_stilt` fixtures and
  `STILT_R_DIR` keep their names.
- **Release** is what STILT-R calls a receptor: `releases.csv`, the `release`
  column, `--releases`, `group` for `r_idx`, and the classes `PointRelease`,
  `ColumnRelease`, `MultiPointRelease` (runtraj's). Docs write "release (receptor)".
- **Config** is what the user writes, always a class next to the code that
  uses it (`FootprintConfig` here; `ProjectConfig`, `HysplitConfig` in
  runtraj). **Settings** are what a result was made with, as recorded; never a class.
- **Sister projects.** STILT-R gives the transport science;
  [X-STILT](https://github.com/uataq/X-STILT) the observation layer and
  column science (concepts ported, not feature parity).

## Architecture

**The footprint is a stage of a runtraj project.** A project is a directory
of inputs (`config.yaml`, `releases.csv`) and an output directory; its
simulations are releases × variants, and runtraj's one worker runs the
transport model, then each stage of the variant. `config.yaml` names the
footprint stage by import path (`class: stilt.footprint.FootprintStage`,
`stilt.footprint.FOOTPRINT_STAGE`), in the `footprint` block under `stages`,
so runtraj imports no stage and PYSTILT has no worker of its own. The stage's
settings (`FootprintConfig`) are hashed with the particles' hash, so a changed
footprint setting makes a new `footprint/` folder beside the old one
(`docs/reference/layout.rst`).

**`stilt.Project` and `stilt.Simulation` only add reads** to runtraj's
classes (`project.py`, `simulation.py`): `footprints`, `jacobian`,
`transport_error` on a selection of `project.simulations`; the footprint,
`calc_footprint`, the background, the transport error and the plot on a
simulation. They override only `simulation` (returns `stilt.Simulation`) and
`init`, which writes STILT-R's HYSPLIT values (`STILT_R_HYSPLIT`), where
runtraj's writes the settings as given (`tests/test_stilt_project.py`).
Workers and stages get runtraj's classes; nothing on disk names the
subclasses, so `runtraj.Project(path)` opens the same project. Results are
plain data: do not wrap a table or an array in a class.

`stilt.__all__` (and each subpackage's) is the public surface. It re-exports
runtraj's names a STILT user types (`ProjectConfig`, the release classes,
`Grid`, `run_trajectories`, `read_particles`, `RuntrajError`); the rest of
runtraj is imported from `runtraj` (`runtraj.models.hysplit.HysplitConfig`).

```
src/stilt/
  cli.py          the stilt command: runtraj's commands, and stilt init
  project.py      stilt.Project: footprints, jacobian, transport_error; init with STILT-R's values
  simulation.py   stilt.Simulation: footprint, calc_footprint, background, transport_error, plot
  particles.py    the particles.stilt accessor: runtraj's methods, and enhancement
  visualization.py  footprint and simulation plots
  footprint/      FootprintConfig, FootprintStage, calc_footprint (gridding.py), foot (add_foot,
                  the near-field correction), aggregation, io, targets
  transforms/     pre-footprint particle transforms, one module each
  migration.py    read_stilt_r, import_stilt_r: STILT-R and X-STILT by-id output, read with rdata
  stilt_r.py      StiltRModel, the stilt-r model of imported runs (runs nothing), named by import path
  observations/   the X-STILT port (readers, sounding selection, slant releases, columns);
                  releases along a moving platform's track (tracks.py)
  background.py, transport_error.py, sampling.py, exceptions.py (EmptyFootprint)
```

**Observation-driven** work sits above the footprint: `stilt.observations`
turns soundings into releases, then a run goes as usual. Keep it out of
`project.py`.

The import contract in `pyproject.toml` (`just imports`) is an exhaustive
`layers` contract: a new top-level module fails until placed.

**runtraj is taken from git at a commit** (`[tool.uv.sources]`) until its
first release; then PYSTILT requires a range of versions up to runtraj's next
minor version. STILT-R parity runs here, on every pull request and on each
change of the runtraj pin.

## Invariants

- **STILT-R parity** at `rtol=1e-7` per footprint cell against a pinned commit
  (`docs/development/stilt_r_parity.rst`). Run the `fidelity` suite before merging a change to
  footprint math, and after moving the runtraj pin. NetCDF output is CF-1.8, deliberately not
  byte-compatible with STILT-R.
- **Completion is by file, and the rule is runtraj's** (`runtraj.output.completed`):
  the model's output file and the file of every stage the variant uses. The
  footprint stage always writes a file. Never add a registry or a manifest.
- **Bulk methods work from listings** and release ids, never a `Simulation`
  per row; they open a file per row only when its contents are the answer
  (`footprints`, `jacobian`).
- **Identity is content.** PYSTILT never deletes or overwrites a results folder.
- **The CLI stays thin.** `stilt` is runtraj's commands and `stilt init`.
- **PYSTILT ships no HYSPLIT file**; runtraj finds the user's `hycs_std`.
- **Pydantic for all configuration**: every field a plain
  `Field(default, description=...)`, and the settings inside a block stay flat.

## Development

Driven by [`just`](https://github.com/casey/just) and
[`uv`](https://docs.astral.sh/uv/); `just` alone lists every recipe. CI
(`.github/workflows/`) runs the same recipes. The tooling comes from
[jmineau/python-template](https://github.com/jmineau/python-template)
(`.copier-answers.yml`; `copier update` pulls in its changes).

| Command | What it does |
|---|---|
| `just sync` | `uv sync`: the package and the dev tools into `.venv` |
| `just test` | the unit tests and the `>>>` examples under `src/`, in parallel (extra args go to pytest; `-n 0` for serial); `just sync-floors` first runs them with every dependency at its minimum |
| `just lint` / `just format` | ruff check and format check / fix and format |
| `just type-check` | pyrefly, against Python 3.11 (set in `pyproject.toml`) |
| `just imports` / `just docstr` | the import contract / docstring coverage of the public API (95%) |
| `just quality-check` | lint, type check, import contract, docstrings, unit tests |
| `just pre-commit` | every pre-commit hook on every file |
| `just build-docs` | a clean Sphinx build that runs every example; warnings are errors |
| `just dist` | the sdist and the wheel, checked with twine |

### Tests

`just test` runs every test without one of three disjoint markers. Warnings
are errors (a test that expects one uses `pytest.warns`). CI runs `-m integration`
in `tests.yml`, `-m fidelity` and `-m r_only` in `stilt-r.yml` (the paths at its top):

- `integration`: end to end with real met files and HYSPLIT. Needs a `hycs_std` on
  `PATH` (unit tests get a stand-in) and `STILT_TEST_MET_DIR` (HRRR ARL files) or
  `STILT_TEST_FETCH_MET=1` (downloads seven 6 h blocks, 23.9 GB in all, and keeps
  211 MB in the ignored `tests/met_cache/`); otherwise they skip.
- `fidelity`: PYSTILT against STILT-R, both running HYSPLIT. Needs the met,
  `STILT_R_DIR` (a STILT-R checkout built as `stilt-r.yml` builds it), `Rscript`.
- `r_only`: STILT-R comparisons with no met or HYSPLIT (`STILT_R_DIR`, `Rscript`).

The tests here are of STILT's side, and of a project that makes footprints;
a test that needs nothing of STILT's side belongs in runtraj.

Docstring examples (`>>>` lines under `src/`) run as tests too
(`--doctest-modules`), so an example that stops matching the code fails
`just test`. One that needs a HYSPLIT run, met files or the network gets
`# doctest: +SKIP` on its `>>>` line.

pytest loads a local `.env` for such variables. Test folders follow the
package. `tests/observations/` is self-contained (its own `conftest.py` and
`data/`, never `tests/conftest.py`'s fixtures). Committed test data stays
small: slices of public products (TROPOMI, TCCON, OCO-2/3) keep the granule's
own name and are cut by `tests/observations/data/products/make_samples.py`.
Retrievals that are not public (GGG, EM27) stay synthetic files with the real
format's quirks. Never commit met files.

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
- An exception a run raises on purpose is a `runtraj.RuntrajError`
  (`runtraj.exceptions`); PYSTILT's own is `EmptyFootprint`, in
  `stilt/exceptions.py`. Plain input checks raise builtins such as `ValueError`.
- Extras: `geometry`, `visualization`, `download`, `sparse`, `stilt-r`, and
  `complete`, which the `dev` group installs.

### Documentation

A user-facing change updates the docs in the same change: the guide when
what the user does changes, the reference (docstring Notes) otherwise. A bug
fix's edge case goes to the reference. `just build-docs` must pass. Examples run when the docs build, so
they use synthetic data, a small repository file, a public download, or the
test met cache, and seed their random numbers; an `.. ipython::` block has no multi-line statement as its 10th or 100th (`CONTRIBUTING.md`). Give
every public property and attribute a docstring, with no colon in the middle
of its first line (napoleon reads `text: more` as `type: description`).

**Voice** (docs, docstrings, field descriptions, CLI help): write for
scientists who want to run the model and trust the result. Open with the
reader's goal, recommend a default and say why ("you" for the reader, "we"
for our advice), and show real output. One point per paragraph; no
metaphors, selling, or filler; American spelling. The full guide, with the habits that read as
machine-written, is "Writing documentation" in `CONTRIBUTING.md`. Words:

| Say (user-facing) | Not |
|---|---|
| settings folder | key, hash, identity |
| transport model (in full) | model, when the weather model or a modeled value is near |
| particles (the table); run (the verb); trajectory (one particle's path) | run as a noun for the result |
| workdir (one simulation's folder); compute root (where workdirs are made) | scratch, except the output's `scratch/` folder |
| log, met files | provenance |
| release id; simulation (a release under a variant) | simulation ID as a folder |
| the settings of a variant | resolve, record, `model_copy`, frame, handle |

### Commits, releases, issues

- Conventional Commits (`fix(slurm): ...`, `docs: ...`). A pull request leaves
  `CHANGELOG.md` alone: at release, `just changelog` drafts the entries from the
  commit messages (`cliff.toml`) and the maintainer edits them in, so write each
  message as the line a user would read. The version comes from git tags
  (setuptools-scm). Releases are cut by the maintainer (CONTRIBUTING.md).
- `uv.lock` is updated by the Upgrade lock workflow (`lock-upgrade.yml`, monthly,
  opens the `chore/lock-upgrade` pull request), not by Dependabot; `STILT_R_SHA` by
  `stilt-r-upgrade.yml` (monthly, opens `chore/stilt-r-upgrade`).
- Open work is in [GitHub issues](https://github.com/jmineau/PYSTILT/issues);
  read one before working on it. Feature status lives in `docs/development/roadmap.rst`
  (update its row when a feature lands). Over-engineered code goes on the
  issue labeled `simplify`. A pull request fixes one thing and links its issue.

## Gotchas

The reasons are in `docs/development/architecture.rst` ("Implementation notes");
runtraj's `AGENTS.md` has the ones about runs, met files, HYSPLIT and the worker pool.

- **Results are plain data**: `sim.particles` a DataFrame, `sim.footprint` a
  DataArray, with `.stilt` accessors; no wrapper classes. A missing result
  raises `FileNotFoundError` (not cached); `None` is only for final states, so
  test with `has_particles` / `has_footprint`.
- **Empty footprints are successes.** `calc_footprint` raises `EmptyFootprint`
  and the footprint stage writes a no-row file marked `stilt:empty`: complete, with
  `sim.footprint` `None` and a listing in `project.footprints().attrs["empty"]`.
  Never synthesize a zero footprint for it.
- **A particle file records what runtraj wrote** (`runtraj:release`,
  `runtraj:hash`, `runtraj:version`); a footprint file adds PYSTILT's own
  (`stilt:footprint`, `stilt:hours`, `stilt:empty`, `stilt:pystilt`).
- **A stage's `config_class` reads back what its `settings` wrote**: a stored
  folder is found that way (`runtraj.stages.check_stage` in the stage's tests).
- **Two depths**: HYSPLIT's `foot` uses `veght`; the footprint stage computes
  its own (`footprint_depth`, `pbl_height`) when it needs one and stores it
  nowhere. On HYSPLIT's `foot`, `resolve` takes `veght` as `footprint_depth`.
- **Pressure weighting fits `ln p = b + a·z`** (`particle_pwf`); never use raw
  `pres` (`tests/integration/test_pressure_weighting.py` guards it).
- **Release heights come from the `age = 0` row** (runtraj's
  `add_release_heights`), with a fallback for models that write none, as
  HYSPLIT 5.1.0 does.
- **Transport-error settings** are easy to misread: read
  `docs/guides/transport_error.rst` before changing or validating them.
