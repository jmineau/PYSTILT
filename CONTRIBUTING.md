# Contributing to PYSTILT

Bug reports, documentation fixes, and code are all welcome. This page covers
how to set up a checkout, the checks a change has to pass, how to write the
docs, and how to add the most common kinds of extension.
[AGENTS.md](AGENTS.md) describes the architecture and the rules the code
relies on.

## Getting started

1. Fork the repository on GitHub and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PYSTILT.git
   cd PYSTILT
   ```
2. Install the package and the development tools into `.venv` with uv:
   ```bash
   uv sync
   ```
3. Install the pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```
4. For the tests that run HYSPLIT (`integration`, `fidelity`) and the docs
   build, put a HYSPLIT folder's `exec/` on `PATH`: STILT-R's
   `bin/linux_x64/hycs_std` in `exec/` and its `exe/` tables in `bdyfiles/`,
   at the commit `.github/workflows/stilt-r.yml` pins, as CI does
   (`.github/scripts/install_hysplit.sh`). Without one those tests are
   skipped. The unit tests need none: they put a stand-in HYSPLIT first on
   `PATH`.

## Making a change

1. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your change. New behavior needs a test, and a bug fix needs a test
   that fails without the fix. User-facing changes need a docs update.
3. Run the checks:
   ```bash
   just quality-check   # ruff, pyrefly, the import contracts, docstrings, and the unit tests (with the docstring examples)
   just pre-commit      # all pre-commit hooks
   just sync-floors     # optional, then `just test`: every dependency at its minimum, as CI's "minimum dependencies" job does
   ```
4. Commit with a [Conventional Commits](https://www.conventionalcommits.org/)
   message (`fix(slurm): ...`, `docs: ...`), push to your fork, and open a
   pull request. Leave [CHANGELOG.md](CHANGELOG.md) alone: the release notes
   are drafted from these messages when the maintainer releases.

## Writing documentation

The sources are in `docs/`: `getting_started/` (install, quickstart,
concepts), `guides/` (task-oriented how-tos, one task per page, with
`guides/migration/` for moving from STILT-R, X-STILT, stiltctl, and Jena
STILT), `tutorials/` (end-to-end worked examples), `reference/` (API pages
built from docstrings, and the table of every `config.yaml` key, made by the
`config-model` directive in `docs/_ext/`), and `development/` (architecture,
STILT-R parity, roadmap). A user-facing change updates the docs: the guide
when what the user does changes, the reference otherwise. `just build-docs`
must pass, and warnings are errors. The Documentation workflow builds the
docs on every pull request and publishes `main` as `dev/` and each release
under its version.

### Voice

Docs, docstrings, config field descriptions, and CLI help are written for
scientists who want to run the model and trust the result. Our models are the
[STILT-R docs](https://uataq.github.io/stilt/), the xarray and MetPy user
guides, and Herbie's tutorials.

- **Start from the reader's goal.** Open a page or section with what the
  reader wants to do and why it matters to their result. Then show it.
- **Recommend.** When there is a choice, say which one we would use and why
  ("Start with HRRR, because its 3 km grid resolves the valleys around a
  city"). "We" is the developers, "you" is the reader.
- **Give the reason in the same sentence.** Join cause and effect with
  "because" or "so". Keep one point per paragraph, and let a sentence be as
  long as its reason needs.
- **Show real output, and say what to notice in it.** Prefer an `ipython`
  block to a `code-block`, and a figure to a paragraph that describes one.
- **A guide holds what the reader acts on.** Exact rules, failure behavior,
  file-name rules and internals go in the docstring (Notes) and the
  reference. Measured accuracy goes on a Development page, with its test or
  study. A bug fix changes a guide only when it changes what the user does.
- **Say it once.** A caveat lives on one page, and the others link to it.
- **Use the reader's words.** Release, particles, footprint, meteorology.
  Internal names (store key, publish, resolve) belong only on pages about
  internals. Describe a config's fields (its parameters) in plain English,
  naming no class or module. `AGENTS.md` has the table of words to use.
- **Say it plainly.** No metaphors ("a dial worth turning"), no selling
  ("powerful", "seamless"), no filler ("note that", "it's worth noting").
  Use American spelling (meters, modeled, center).

Each kind of page has its own job:

- **Getting Started** tells one story, from a measurement to a footprint
  map, and says what to notice in each output.
- **Tutorials** answer a science question end to end. They run when the docs
  build and end in a figure.
- **User Guide** pages each cover one task, in the order a user meets them.
  A page opens with why the reader would do it, recommends a default, shows
  real output, and ends by linking to checking for when it goes wrong.
- **API Reference** pages are complete and exact, so the edge cases live
  there.
- **Development** pages hold the architecture, STILT-R parity, the roadmap,
  and measured accuracy.

Some habits make text read as machine-written. Avoid them:

- Colon reveals: "PYSTILT does one thing: it follows the air." Write
  "PYSTILT follows the air."
- "X, not Y" contrasts when nobody suggested Y.
- A bold sentence at the start of a paragraph that states its point.
- A page that begins "This page covers", and closing sentences that restate
  the paragraph.
- A run of sentences that each say what something does *not* do.
- Lists of three out of habit.
- Asides about what the code does "deliberately" or used to do.
- Labels invented on one line and referred to later ("the first case").

Before you commit a page, read it top to bottom as someone who has never run
STILT. If a paragraph is there because of a bug fix, move it to the
reference.

Docstrings follow the NumPy style. The summary line says what the object is
or what the function returns. Parameter descriptions give the meaning and
units, not the type again. Check every example and claim against the code
when you write it.

### API pages

Each class has a page with tables of its attributes and methods, and each
member a page of its own; a subclass lists what it defines and links to what
it inherits. `docs/_templates/autosummary/class.rst` is PYSTILT's own (it
renders config models through `config-model`). `docs/_ext/api_pages.py`,
from python-template, decides which members get a row (not pydantic's, so no
list of them is kept) and what is inherited.

A property or attribute's page is where its See Also and Examples go, as in
pandas. Give each public one a docstring that says what it is, and add See
Also or Examples where they help; one with no docstring shows an empty row.
Napoleon reads a first line of the form `text: more text` of a property, an
attribute, or module data as `type: description`, so keep a colon out of
the middle of that line.

Module data such as `PARTICLE_SCHEMA` is listed under the package that
exports it (`runtraj.particles`). autodoc looks for its `#:` comment in that
module, where there is none, so `reexported_data_comment` in
`docs/conf.py` takes the comment from the submodule that defines it.
Without the hook, a page shows the docstring of the object's type
("Built-in immutable sequence.").

### Examples in the docs

Examples run when the docs build, so they show real output, and the build
fails when one breaks.

- **A figure in a docstring:** put a `.. plot::` directive in the Examples
  section. Its code runs (doctest `>>>` lines work), and the figure appears
  under it.

  ```rst
  Examples
  --------
  .. plot::

     >>> import matplotlib.pyplot as plt
     >>> _ = plt.plot([0, 1, 2], [0, 1, 4])
  ```

- **Code in an .rst page:** an `.. ipython:: python` block runs and shows
  each line with its output, as in an IPython session. Changing a
  `.. code-block:: python` to `.. ipython:: python` is enough to make it
  run. For a figure, put `@savefig name.png` on the line above the plotting
  call.

  ```rst
  .. ipython:: python

     import matplotlib.pyplot as plt
     import stilt

     stilt.__version__

     @savefig squares.png width=5in
     plt.plot([0, 1, 2], [0, 1, 4]);
  ```

  A block that raises or warns fails the build; `:okexcept:` or
  `:okwarning:` under the directive allows one that is meant to. All .rst
  pages share one session, so each page imports and defines what it uses.
  Split a long block so that no multi-line statement is its 10th (or 100th)
  statement: the directive runs only the first line of that one.
- **Examples in docstrings are tests too:** every `>>>` line under `src/` runs
  with `just test` (`--doctest-modules`), so an example that stops matching
  the code fails. One that needs a HYSPLIT run, met files or the network gets
  `# doctest: +SKIP` on its `>>>` line.
- **When they run:** `just build-docs` runs everything, as CI does.
  `just docs-serve` reruns a page's blocks whenever that page is rebuilt.
- **Notebooks, and code cells in Markdown pages,** are off in this project.
  `copier update --data docs_notebooks=true` turns them on; it adds myst-nb,
  and Jupyter with it, to the dev tools.
- **Data:** the docs build on GitHub Actions, so an example uses synthetic
  data, a small file in the repository, or a public download, never a
  machine-specific path. Seed random numbers so the figures don't change from
  build to build.
- **The Getting Started page runs a real simulation** on the met files the
  tests cache (`tests/met_cache`, 14 and 15 January 2021 around Salt Lake),
  in a temporary folder. `just build-docs` needs them: set
  `STILT_TEST_MET_DIR` to that folder, or fetch them once with

  ```bash
  STILT_TEST_FETCH_MET=1 uv run pytest \
    tests/integration/test_pressure_weighting.py::test_hypsometric_fit_holds_on_a_real_winter_profile \
    --setup-only -q
  ```

  The fetch is large. NOAA's HRRR files cover the whole CONUS grid and are
  3.42 GB each; each is downloaded whole and then cropped to the 130 by 129
  cells around Salt Lake, which keeps 30.2 MB. The seven blocks the tests need
  therefore download 23.9 GB, all seven at once into `tests/met_cache`, so that
  folder's disk needs about 24 GB free while it runs. After the crop it holds
  211 MB. The time depends on your connection (at 10 MB/s it is about 40
  minutes). Fetch once, and keep the folder: a file already in it is not
  downloaded again.

  A release on the page must stay inside that time window. `docs.yml`
  restores the same cache the tests use.
- **The guides that open a project start from a sample project**
  (`docs/_ext/docs_examples.py`): four releases an hour apart on 15 January
  2021, run once per docs build (about 40 s) and copied into a temporary
  folder for each page. A page starts it from a hidden block
  (`docs_examples.temp_workdir(sample=True)`) and goes back with
  `os.chdir(docs_cwd)` at its end, so the page's files land in the
  temporary folder.
- **Optional dependencies:** the `dev` group installs
  `pystilt[complete]`, so the plotting extra (`visualization`) is there in
  CI. If an example needs another extra, add `pystilt[extra]` the same way.

## Adding configuration fields

`config.yaml` is blocks: `mets`, `models`, `stages`, and `variants` each map a
name to a block, and `ProjectConfig(...)` and `Project.init(...)` take the same
blocks. Inside a block the settings are flat: fields such as `seed`, `numpar`,
and `ziscale` are keys of the `hysplit` block under `models`, and `grid` and
`smooth_factor` of the footprint's under `stages`. Don't add nested parameter
objects inside a block.

A footprint setting is a field of `FootprintConfig`
(`src/stilt/footprint/config.py`): a plain pydantic
`Field(default, description=...)`. The transport model's settings
(`HysplitConfig`), the met's and the execution's are
[runtraj](https://github.com/jmineau/runtraj)'s, and a new one is a change
there.

## The particle runs are runtraj's

Transport models, the stage protocol, the output directory with its
completion rule, and how work is run (the local pool, Slurm) are
[runtraj](https://github.com/jmineau/runtraj)'s. Its `CONTRIBUTING.md` has
"Adding a transport model", "Adding a stage", "Output directory and
completion" and "Changing how work is run". The footprint is a stage
(`stilt.footprint.FootprintStage`), and `stilt.Project` and
`stilt.Simulation` subclass runtraj's with the reads of what it makes.

A change that needs both packages is two pull requests: runtraj's first,
then one here that moves the commit `pyproject.toml` takes runtraj from
(`[tool.uv.sources]`), updates `uv.lock` (`just lock`), and runs the STILT-R
parity suite.

## Adding particle transforms

Particle transforms run on the particles before the footprint is made. A
transform is any object with `apply(particles, release=None,
directory=None)`: it takes a particle `DataFrame`, the release, and the
project directory, and returns a new `DataFrame`, leaving the input
unchanged.

A built-in transform is one new file in `src/stilt/transforms/`, such as
`lifetime.py`: one pydantic class whose fields are the YAML keys, whose
`kind` is a `Literal` that names it, and whose `apply()` does the work.
Import it in `transforms/__init__.py` and add it to the `BuiltinTransform`
union, and it is ready to use. A transform changes the footprint through
`foot` and may add columns; the built-ins only scale `foot`, which
`sim.background` and `sim.transport_error` rely on (see the transforms
guide). Users can also point `kind` at their own class by import path
(`kind: my.module.Class`), so only transforms that many users need belong in
PYSTILT.

Test:

- parsing the config and writing it back to YAML
- the numbers the transform produces on a small particle table
- a footprint made with the transform configured

## Building distributions

PYSTILT is pure Python and includes no HYSPLIT, so one wheel and the source
archive serve every platform. Build and check them with:

```bash
just dist         # the sdist and the wheel, in dist/
```

## Releasing

The version comes from git tags, through setuptools-scm, so there is no
version string to bump.

1. Write the changelog: `just changelog` drafts the entries from the commit
   messages since the last release. Edit them into `CHANGELOG.md` under
   `## [Unreleased]`, then rename that heading to `## [X.Y.Z] - YYYY-MM-DD` and
   start a new empty `## [Unreleased]` above it. Commit (`chore(release): X.Y.Z`)
   and push to `main`.
2. Run `just release X.Y.Z`. It checks that the tree is clean, that `main` is in
   sync with GitHub, and that the version is newer than every existing tag, then
   pushes the tag `vX.Y.Z`.
3. The Publish workflow builds the tag with `just dist`, uploads it to PyPI and
   creates a GitHub Release from the CHANGELOG section. Zenodo archives the
   release and mints a DOI. The Documentation workflow publishes the docs as
   `X.Y.Z/` in the version dropdown.

Pre-releases (`0.1.0a23`, `0.1.0rc1`) work the same way. Until the first final
release, the newest pre-release is also the docs' default (`stable/`). Once a
final release is out, the pre-releases before it leave the dropdown (see
`DOCS_PRERELEASES` in `.github/workflows/docs.yml` to change that).

## Dependency updates

Dependabot opens one pull request a month for each of GitHub Actions and
pre-commit hooks. Merge it when CI passes.

`uv.lock` (the dev tools and every dependency) is updated by the Upgrade lock
workflow alone, not by Dependabot, which would leave a package at its old
version when the newest release needs a newer Python than the oldest one this
project supports. On the 1st of each month the workflow runs `uv lock --upgrade`
(`uv.lock` then holds the newest release for each Python; it never raises the
minimum versions in `pyproject.toml`) and, when the lock changes, opens the pull
request `chore/lock-upgrade` (or updates it). GitHub does not start workflows on
a pull request that its own token opened, so the workflow starts Tests, Code
Quality and Documentation on the branch itself; their results show on the pull
request. Merge it when they pass. You can also run it by hand: Actions > Upgrade
lock > Run workflow.

## Template

The tooling (CI workflows, pre-commit, justfile, packaging configuration) comes
from [jmineau/python-template](https://github.com/jmineau/python-template).
`.copier-answers.yml` records the template version; `copier update` pulls in
later template changes. Improvements that would help every package are best
made in the template.

## Pull requests

- Fix or add one thing per pull request, and link the issue it closes
  (`Fixes #N`).
- Leave `CHANGELOG.md` alone; the maintainer writes it at release, from your commit messages.
- Make sure the tests pass and coverage does not drop.

## Reporting bugs

Please include:

- your operating system and Python version
- steps to reproduce the problem
- what you expected and what happened
- any error messages or logs

## Feature requests and questions

Check the [existing issues](https://github.com/jmineau/PYSTILT/issues) first.
If nothing matches, open a new one describing the feature and why you need
it, or use the "question" label for questions.

## Code of conduct

Be respectful and constructive.

## License

Contributions are released under the project's MIT License.
