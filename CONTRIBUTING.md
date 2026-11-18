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
- **Use the reader's words.** Receptor, particles, footprint, meteorology.
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
    tests/transport/hysplit/test_release_assignment.py::test_hysplit_multipoint_release_points_follow_control_order \
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

`config.yaml` is blocks: `mets`, `models`, `stages`, and `variants` each map a
name to a block, and `ProjectConfig(...)` and `Project.init(...)` take the same
blocks. Inside a block the settings are flat: fields such as `seed`, `numpar`,
and `ziscale` are keys of the `hysplit` block under `models`, and `grid` and
`smooth_factor` of the footprint's under `stages`. Don't add nested parameter
objects inside a block.

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
(`n_hours`, `seed`, `footprint_depth`, `pbl_height`, `pbl_height_min`);
a parameter goes there only when the core reads it, so another model is
never handed a setting it would ignore. It also gives an `UNRECORDED` set
of the fields that change no particle (empty unless the subclass names
some), a `RECORDED_WHEN_SET` set of the fields a run records only when they
are not at their defaults (so adding them moved no stored run to a new
settings folder), `settings()` (what a run
records), and `realizations(n)` (realization `k` with `seed + k`); override
those two only where the model differs. The model itself names that class as
`config_class`, names its met config as `met_config_class`, and gives
`sha256` (the checksum of the executable it runs, `None` for none: part of
the run's settings), `version` (what the build reports, recorded beside
them), `data_files`, and
`run(receptor, config, met, window, workdir=None, timeout=None)`
(`stilt.transport.TransportModel`). `sha256` raises `FileNotFoundError` when
the executable cannot be found; the project still opens, and its status
compares stored folders without the build. The met config is a pydantic model that
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
adds the release heights itself (`stilt.transport.run_model`), and the
footprint applies the near-field correction (`stilt.calc_footprint`). A model that writes no files sets
`needs_workdir = False` and is handed `workdir=None`. A model that runs
many receptors in one call, such as an emulator on a GPU, sets
`batched = True` and gives `run_many(receptors, config, met, windows,
workdir=None, timeout=None)`, returning one table with a `receptor` column
(`stilt.transport.BatchedTransportModel`): the worker then hands it every
receptor of a variant that needs particles at once, and a receptor it
returns no rows for fails alone. A model in a package of its own needs no
registration: its block under `models:` in `config.yaml` names it by import
path (`class: mypkg.models.MyModel`), as a transform's `kind:` is, and every
process that opens the project imports it from there. The run settings
record the `class:` string as written, so the same class named by two
paths (`mypkg.models.MyModel` and a re-export, `mypkg.MyModel`) gets two
settings folders. Pick one path and keep it. A model built into
PYSTILT goes in `stilt.transport.MODELS` under a short name instead. Keep
its package out of the core: the import contracts in `pyproject.toml` say
how.

## Adding a stage

A stage is what is made from each simulation's results after the transport
model runs: one file per receptor, from the particles, from the files of
other stages, or both. The footprint is one
(`stilt.footprint.FootprintStage`). Write a stage when you want a product
made for every receptor of a project, kept in the output directory beside
the particles, and counted by `stilt status`. For a calculation you run now
and then, read the particles (`project.particles`) instead.

A stage is a class with the members of `stilt.stages.Stage`; it subclasses
nothing and holds no state. This one writes each receptor's mean particle
height:

```python
from pydantic import BaseModel


class MeanHeightConfig(BaseModel):
    below: float = 5000.0  # meters above ground


class MeanHeight:
    name = "mean_height"  # its tree in the output, and its status() column
    config_class = MeanHeightConfig
    suffix = ".txt"  # the extension of its files

    def resolve(self, config, model, model_settings, directory):
        return config  # read files, and refuse a model that does not fit, here

    def settings(self, config):
        return config.model_dump(mode="json")  # what names its settings folder

    def check(self, config, receptors, directory):
        return []  # (check, receptor id or None, message), for `stilt check`

    def run(self, config, sim, path):
        zagl = sim.particles["zagl"]
        path.write_text(str(zagl[zagl < config.below].mean()))
```

```yaml
stages:
  mean_height:
    class: mypkg.stages.MeanHeight
    below: 3000
```

- **`name`** is the name of the stage's block under `stages:`, its tree in
  the output directory
  (`mean_height/settings=<variant>-<hash>/date=<day>/<receptor><suffix>`),
  its column in `project.status()`, and the `stage` of a failure record: one
  word for the stage everywhere, so a block under another name is refused.
  Lowercase letters, digits and underscores.
- **`suffix`** is the extension of the stage's files, with its dot.
- **`config_class`** is a pydantic model that checks the block's settings
  when `config.yaml` loads, without reading a file. A variant changes a
  setting under its own `stages:` key, and the variant's keys replace the
  block's; give the class an `override(block, own)` classmethod to merge
  them another way, as `FootprintConfig` changes a grid field by field.
- **`resolve`** runs once for each variant when the project reads its
  variants. Read the files the settings name there, and raise `ValueError`
  for a transport model whose particles the stage cannot use (`model` is
  its name and `model_settings` what a run of it records). Return the
  config the runs use.
- **`settings`** returns the record that names the stage's settings folder,
  as plain data. A changed value is a new folder, so leave out what changes
  no result, such as a path. `config_class` must read the record back to a
  config with the same record, because a stored folder is found that way.
  `stilt.stages.check_stage(stage, config)` checks it: call it in your
  tests, once for each kind of setting.
- **`check`** returns what would stop the stage for the project's
  receptors, for `stilt check`. Run nothing in it.
- **`run`** makes one simulation's file and writes it at `path`. It gets the
  simulation: `sim.particles`, `sim.receptor`, `sim.variant`, and
  `sim.directory`, where a relative file name in the settings starts. Do
  not change `sim.particles` in place, since variants share it. Write one
  file whatever there is to say: a simulation is complete when the model's
  output file exists, and so does the file of every stage its variant uses,
  so a stage that writes nothing leaves its simulations unfinished. Write a
  file that says "nothing here", as an empty footprint does. An error
  fails the simulation at this stage, and the worker records it. On a
  filesystem `path` is a hidden temporary name ending in `suffix`, moved
  into place when `run` returns; the stage's earlier file is removed before
  it runs, and so is what a `run` that raises has written. So no file made
  from other inputs, and no half-written one, counts as complete. A Parquet
  file may carry `stilt.identity.file_stamp(...)` in its metadata, the
  folder's hash and the PYSTILT version, as particle and footprint files do.
- **`reads`** (optional) names the stages whose files this one reads, such
  as `reads = ("footprint",)`. The stage runs after them and again whenever
  one was made again, and its settings folder is named from theirs too.
  Find a file with `sim.path("footprint")` and read it with the reader of
  the package that wrote it (`stilt.read_footprint`). A variant that sets a
  stage to null while another of its stages reads it is refused.

The stage's package must be installed wherever the project is opened, the
login node that reports status included: a settings folder is named from
the stage's settings, defaults included, and only its config class knows
them. A project whose config names a class that is not installed stops
with a message naming the package.

## Output directory and completion

A project is a local directory of inputs (`stilt.project.Project`):
`config.yaml` and `receptors.csv`. Results go to an output directory
(`stilt.output.Output`) that `config.yaml` names and that several projects
can share. Each results folder is named by the hash of its settings, so a
changed setting writes a new folder and never overwrites a result.

`Output` knows where each result's file is (`path`), which receptors have
one (`present`, from a listing of their date folders), and the one
definition of a finished simulation (`complete`, which applies
`stilt.output.completed`): the model's output file exists, and so does the
file of every stage the variant uses. `Simulation.is_complete` and
`status()` use it. Every method takes the kind of result, `"particles"` or
a stage's name, so a stage needs no method of its own; it writes its file
at the path `make_path` gives. Don't add another "does this output exist"
check, a registry, or a manifest. Call these instead.

## Changing how work is run

`stilt.execution.run` (`execution/runner.py`) finds the receptors with
missing results. A local run calls `run_receptors` in this process. A Slurm
run goes through `submit`, which writes `receptors.txt`,
`receptors.parquet` (the submitted receptors' checked rows, which tasks
read in place of `receptors.csv`), `execution.yaml`, and `job.sh`
(`job_script`) into `_slurm/<stamp>/` and calls `sbatch`. When a met
downloads, `fetch.sh` (`fetch_script`) is submitted first and the array
waits for it.

The unit of work is a command line: `stilt run <project> --receptors FILE
--task I/N`. Any scheduler that can start it with an index can run a
project; the task requeues itself on `SIGUSR1` inside a Slurm job. Another
scheduler means another script writer, not another worker.

A new `execution:` setting goes on its backend's block (`LocalConfig` or
`SlurmConfig` in `execution/config.py`) with a description; only a setting
that applies wherever a run happens goes on `ExecutionConfig` itself. A new
backend gets a block of its own, named as `backend:` names it. Settings that
only `sbatch` understands do not need a field: users write them in the
`slurm:` block by `sbatch`'s own names.

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
