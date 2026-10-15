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

## Making a change

1. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your change. New behavior needs a test, and a bug fix needs a test
   that fails without the fix. User-facing changes need a docs update.
3. Run the checks:
   ```bash
   just quality-check   # ruff, pyrefly, the import contracts, docstrings, and the unit tests
   just pre-commit      # all pre-commit hooks
   ```
4. Commit with a [Conventional Commits](https://www.conventionalcommits.org/)
   message (`fix(slurm): ...`, `docs: ...`), push to your fork, and open a
   pull request.

## Writing documentation

The sources are in `docs/`: `getting_started/` (install, quickstart,
concepts), `guides/` (task-oriented how-tos, one idea per page, with
`guides/migration/` for moving from STILT-R, X-STILT, stiltctl, and Jena
STILT), `tutorials/` (end-to-end worked examples), `reference/` (API pages
built from docstrings, and the table of every `config.yaml` key, made by the
`config-model` directive in `docs/_ext/`), and `development/` (architecture,
STILT-R parity, roadmap). A user-facing change needs a guide or reference
update, and `just build-docs` must pass: warnings are errors. The
Documentation workflow builds the docs on every pull request and publishes
`main` as `dev/` and each release under its version.

### Voice

Docs, docstrings, config field descriptions, and CLI help all follow this
voice. Most readers are scientists who want to run the model and trust the
result; write for them. Good examples are the
[STILT-R docs](https://uataq.github.io/stilt/) and the xarray, pandas, and
MetPy user guides.

- **Start with what it is for.** Open a page or docstring with what the thing
  does, in plain words. Then show an example. Internals and edge cases go
  last, or in `docs/development/architecture.rst`.
- **Keep sentences short.** One idea per sentence. If a sentence needs a
  colon, a semicolon, and a parenthesis, split it.
- **Use the reader's words.** Receptor, particles, footprint, meteorology.
  Internal names (store key, publish, resolve) belong only on pages about
  internals. Describe a config's fields (its parameters) in plain English,
  naming no class or module. `AGENTS.md` has the table of words to use.
- **Show it.** A short code block, with its output when that helps, is
  clearer than a paragraph describing it.
- **Say it plainly.** No metaphors ("a dial worth turning"), no selling
  ("powerful", "seamless"), no filler ("note that", "it's worth noting").
  Use American spelling (meters, modeled, center).

Some habits make text read as machine-written. Avoid them:

- Colon reveals: "PYSTILT does one thing: it follows the air." Write
  "PYSTILT follows the air."
- "X, not Y" contrasts when nobody suggested Y.
- A bold sentence at the start of a paragraph that states its point.
- Dashes joining clauses. Use a period or parentheses.
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
exports it (`stilt.particles`). autodoc looks for its `#:` comment in that
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
    tests/transport/hysplit/test_release_assignment.py::test_hysplit_multipoint_release_points_follow_control_order \
    --setup-only -q
  ```

  A receptor on the page must stay inside that time window. `docs.yml`
  restores the same cache the tests use.
- **The guides that open a project start from a sample project**
  (`docs/_ext/docs_examples.py`): four receptors an hour apart on 15 January
  2021, run once per docs build (about 40 s) and copied into a temporary
  folder for each page. A page starts it from a hidden block
  (`docs_examples.temp_workdir(sample=True)`) and goes back with
  `os.chdir(docs_cwd)` at its end, so the page's files land in the
  temporary folder.
- **Optional dependencies:** the `dev` group installs
  `pystilt[complete]`, so the plotting extra (`visualization`) is there in
  CI. If an example needs another extra, add `pystilt[extra]` the same way.

## Adding configuration fields

The public config is flat. Fields such as `seed`, `numpar`, and `ziscale` are
passed straight to `ProjectConfig(...)` and `Project.init(...)`. Don't add nested
parameter objects to the public API.

Each HYSPLIT setting is a field of `HysplitConfig`
(`src/stilt/transport/hysplit/config.py`): a plain pydantic
`Field(default, description=...)`. Which input file it goes to is the
HYSPLIT driver's business (`src/stilt/transport/hysplit/driver.py`). A
setting is a `SETUP.CFG` entry unless the driver lists it in
`NOT_IN_SETUP`: the `CONTROL` settings (which `ControlFile` writes),
`ziscale` (`ZICONTROL`), the wind- and mixed-layer-error groups (`WINDERR`
and `ZIERR`, in the order of `WIND_ERROR_SETTINGS` and
`ZI_ERROR_SETTINGS`), and the settings PYSTILT uses itself. So a new
`SETUP.CFG` entry needs only the field, and its HYSPLIT type in the table in
`tests/test_config.py`. Any other new setting also needs a line in the
driver; a test fails if it would land in `SETUP.CFG` unknown to HYSPLIT.

## Adding a transport model

A transport model owns its config: a subclass of
`stilt.transport.TransportConfig` with the model's own parameters. The base
gives the parameters PYSTILT's own code reads whatever the model
(`n_hours`, `seed`, `hnf_plume`, `veght`); a parameter goes there only
when the core reads it, so another model is never handed a setting it
would ignore. It also gives an `UNRECORDED` set of the fields that change no
particle (empty unless the subclass names some), `settings()` (what a run
records), and `realizations(n)` (realization `k` with `seed + k`); override
those two only where the model differs. The model itself names that class as
`config_class`, names its met config as `met_config_class`, and gives
`version`, `data_files`, and
`run(receptor, config, met, window, workdir=None, timeout=None)`
(`stilt.transport.TransportModel`). The met config is a pydantic model that
checks one entry under `mets:`; its `settings()` returns what a run records
of the met (which weather it is, and anything else that changes the
particles, never where its files are kept). HYSPLIT's is
`stilt.transport.hysplit.MetConfig`, whose settings are the source id in
the ARL headers and the crop. A model that reads no files can take any
keys, with no directory. `met` is that config, its `directory` and
`subgrid_dir` absolute when it has them, and `window` the `(start, end)`
the run covers; the model finds its own met from them. `run` returns a `ModelRun`: the particle table (the
columns of `stilt.particles.PARTICLE_SCHEMA`, `foot`, and what the
transforms read, with a release row at `age = 0` when the model can write
one; see `docs/reference/particles.rst`), its log as text, and the met
files it read. A failed run raises `SimulationError` with its log. PYSTILT
adds the release heights and the near-field correction itself
(`stilt.transport.run_model`). A model that writes no files sets
`needs_workdir = False` and is handed `workdir=None`. A model that runs
many receptors in one call, such as an emulator on a GPU, sets
`batched = True` and gives `run_many(receptors, config, met, windows,
workdir=None, timeout=None)`, returning one table with a `receptor` column
(`stilt.transport.BatchedTransportModel`): the worker then hands it every
receptor of a variant that needs particles at once, and a receptor it
returns no rows for fails alone. A model in a package of its own needs no
registration: `config.yaml` names it by import path
(`model: mypkg.models.MyModel`), as a transform's `kind:` is, and every
process that opens the project imports it from there. The run settings
record the `model:` string as written, so the same class named by two
paths (`mypkg.models.MyModel` and a re-export, `mypkg.MyModel`) gets two
settings folders. Pick one path and keep it. A model built into
PYSTILT goes in `stilt.transport.MODELS` under a short name instead. Keep
its package out of the core: the import contracts in `pyproject.toml` say
how.

## Output directory and completion

A project is a local directory of inputs (`stilt.project.Project`):
`config.yaml` and `receptors.csv`. Results go to an output directory
(`stilt.output.Output`) that `config.yaml` names and that several projects
can share. Each results folder is named by the hash of its settings, so a
changed setting writes a new folder and never overwrites a result.

`Output` knows where each result's file is (`path`), which receptors have
one (`present`, from a listing of their date folders), and the one
definition of a finished simulation (`complete`, which applies
`stilt.output.completed`). `Simulation.is_complete()` and `status()` use
it. Don't add another "does this output exist" check, a registry, or a
manifest. Call these instead.

## Changing how work is run

`stilt.execution.run` (`execution/runner.py`) finds the receptors with
missing results. A local run calls `run_receptors` in this process. A Slurm
run goes through `submit`, which writes `receptors.txt`,
`receptors.parquet` (the submitted receptors' checked rows, which tasks
read in place of `receptors.csv`), `execution.yaml`, and `job.sh`
(`job_script`) into `_slurm/<stamp>/` and calls `sbatch`.

The unit of work is a command line: `stilt run <project> --receptors FILE
--task I/N`. Any scheduler that can start it with an index can run a
project; the task requeues itself on `SIGUSR1` inside a Slurm job. Another
scheduler means another script writer, not another worker.

New `execution:` settings go on `ExecutionConfig` (`execution/config.py`)
with a description. Settings that only `sbatch` understands do not need a
field: users put them under `slurm:`.

## Adding particle transforms

Particle transforms run on the particles before the footprint is made. A
transform is any object with `apply(particles, receptor=None,
directory=None)`: it takes a particle `DataFrame`, the receptor, and the
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

PYSTILT ships a compiled HYSPLIT, so each wheel is built for one platform and
holds only that platform's `hycs_std`. The source archive holds none. Build
and check all of them with:

```bash
just dist         # sdist plus one wheel per bundled HYSPLIT build, in dist/
```

`just dist` ends with `just check-dist`, which checks each wheel's platform
tag and binary, and that the sdist has none. Don't publish from
`python -m build` or a bare `uv build`: they make the wheel from the source
archive, so it has no binary.

## Releasing

The version comes from git tags, through setuptools-scm, so there is no
version string to bump.

1. Run `just changelog` to draft the entries from your commit messages, edit
   them into `CHANGELOG.md` under `## [Unreleased]`, then rename that heading
   to `## [X.Y.Z] - YYYY-MM-DD` and start a new empty `## [Unreleased]` above
   it. Commit (`chore(release): X.Y.Z`) and push to `main`.
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

Dependabot opens one pull request a month per kind of pin: GitHub Actions,
pre-commit hooks, and `uv.lock` (the dev tools; it never raises the minimum
versions in `pyproject.toml`). Merge it when CI passes.

## Template

The tooling (CI workflows, pre-commit, justfile, packaging configuration) comes
from [jmineau/python-template](https://github.com/jmineau/python-template).
`.copier-answers.yml` records the template version; `copier update` pulls in
later template changes. Improvements that would help every package are best
made in the template.

## Pull requests

- Fix or add one thing per pull request, and link the issue it closes
  (`Fixes #N`).
- Add user-visible changes to `CHANGELOG.md` under `## [Unreleased]`.
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
