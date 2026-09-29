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
   just quality-check   # ruff, pyright, and the unit tests
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
passed straight to `ModelConfig(...)` and `Model(...)`. Don't add nested
parameter objects to the public API.

Each config field is a plain pydantic `Field(default, description=...)`.
`STILTParams.setup_entries()` writes every `TransportParams` field to
HYSPLIT's `SETUP.CFG`, except the fields listed in `STILTParams.CONTROL_FIELDS`
(written to `CONTROL`) and `ZICONTROL_FIELDS` (written to `ZICONTROL`).
`ErrorParams` fields go to `WINDERR` and `ZIERR`. If HYSPLIT reads your new
field from a file other than `SETUP.CFG`, add it to the matching set and to
the routing test in `tests/test_config.py`.

## Project store and completion

A project is one root directory or URI (`stilt.project.Project`) with a
`Store` (`stilt.store`: `LocalStore` or `FsspecStore`) that reads and writes
its files by key. `Simulation` owns the file names, the keys, and the single
definition of a finished simulation, `Simulation.is_complete()`. Don't add
another "does this output exist" check. Call that method instead. A new store
backend implements the six methods of the `Store` protocol.

## Adding execution backends

Execution backends implement the `Executor` and `JobHandle` protocols in
`src/stilt/execution/backends/protocol.py`. Each executor has a `dispatch`
mode:

- A `push` executor is given the list of pending receptors. It runs them
  through `stilt.execution.run_receptors`, directly or with the
  `stilt push-worker` command, and `Simulation.publish()` copies the outputs
  into the project.
- A `pull` executor starts workers that claim receptors from the Postgres
  queue. A worker keeps its claim until it records a result or releases the
  claim.

Register the backend in `resolve_backend` in `execution/backends/factory.py`,
export it from `execution/__init__.py`, and handle SIGTERM with
`sigterm_as_interrupt` the way the local, Slurm, and Kubernetes backends do.

`start()` should return a handle quickly. `wait()` should raise when the
backend reports a failure. A job that is no longer queued has not
necessarily succeeded. Add tests for a failed submission, failed jobs,
interruption or preemption, and calling `wait()` more than once.

Scheduler-backed executors should put a timeout on every subprocess call,
write their task and chunk files under a predictable directory, and delete
those files once the job is known to be finished.

## Adding particle transforms

Particle transforms run on the particles before the footprint is made. They
implement the `ParticleTransform` protocol in `src/stilt/transforms.py`: a
transform takes a particle `DataFrame` and a `TransformContext` and returns a
new `DataFrame`, leaving the input unchanged.

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
