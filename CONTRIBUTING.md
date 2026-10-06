# Contributing to PYSTILT

Bug reports, documentation fixes, and code are all welcome. This page covers
how to set up a checkout, the checks a change has to pass, and how to add the
most common kinds of extension. [AGENTS.md](AGENTS.md) describes the
architecture and the rules the code relies on.

## Getting started

1. Fork the repository on GitHub and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PYSTILT.git
   cd PYSTILT
   ```
2. Install the development dependencies with uv:
   ```bash
   uv sync --group dev
   ```
3. Install the pre-commit hooks:
   ```bash
   pre-commit install
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
   just quality-check   # ruff, pyright, the import contracts, and the unit tests
   just pre-commit      # all pre-commit hooks
   ```
4. Commit with a [Conventional Commits](https://www.conventionalcommits.org/)
   message (`fix(slurm): ...`, `docs: ...`), push to your fork, and open a
   pull request.

## Writing documentation

Docs and docstrings follow the voice described under "Voice" in
[AGENTS.md](AGENTS.md). Build the docs with
`just build-docs` and check that your change adds no new warnings.

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
gives the parameters every model shares (`n_hours`, `numpar`, `seed`,
`hnf_plume`, `veght`), an `UNRECORDED` set of the fields that change no
particle (empty unless the subclass names some), `settings()` (what a run
records), and `realizations(n)` (realization `k` with `seed + k`); override
those two only where the model differs. The model itself names that class as
`config_class` and gives `version`, `data_files`, and
`run(receptor, config, met, window, workdir=None, timeout=None)`
(`stilt.transport.TransportModel`). `met` is a `MetConfig` with absolute
directories and `window` the `(start, end)` the run covers; the model finds
its own met from them. `run` returns a `ModelRun`: the particle table (the
columns of `stilt.particles.PARTICLE_SCHEMA`, `foot`, and what the
transforms read, with a release row at `time = 0` when the model can write
one; see `docs/reference/particles.rst`), its log as text, and the met
files it read. A failed run raises `SimulationError` with its log. PYSTILT
adds the release heights and the near-field correction itself
(`stilt.transport.run_model`). Add it to `stilt.transport.MODELS` under the name
`model:` takes in `config.yaml`. Keep its package out of the core: the
import contracts in `pyproject.toml` say how.

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
run goes through `submit`, which writes `receptors.txt`, `execution.yaml`,
and `job.sh` (`job_script`) into `slurm/<stamp>/` and calls `sbatch`.

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

A built-in transform is one pydantic class in `transforms.py`. Its fields
are the YAML keys, `kind` is a `Literal` that names it, and `apply()` does
the work. Add the class to the `BuiltinTransform` union and it is ready to
use. Users can also point `kind` at their own class by import path
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
just check-dist   # each wheel's platform tag and binary, and no binary in the sdist
```

Don't publish from `python -m build` or a bare `uv build`: they make the wheel
from the source archive, so it has no binary.

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
