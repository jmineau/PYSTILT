# Contributing to PYSTILT

Thank you for considering contributing to PYSTILT! We welcome contributions from the community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PYSTILT.git
   cd PYSTILT
   ```
3. Install development dependencies with uv:
   ```bash
   uv sync --group dev
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure they follow our coding standards:
   - Code is formatted with ruff
   - All tests pass
   - New features include tests
   - Documentation is updated if needed

3. Run quality checks:
   ```bash
   just quality-check
   ```

4. Run test suite:
   ```bash
   just test
   ```

5. Run pre-commit checks:
   ```bash
   just pre-commit
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request on GitHub

## Adding configuration fields

PYSTILT keeps the public config flat for alpha users: fields such as `seed`,
`numpar`, and `ziscale` should remain directly constructible through
`ModelConfig(...)` and `Model(...)`. Do not introduce nested user-facing
parameter objects unless the public API is deliberately redesigned.

Config fields are plain pydantic `Field(default, description=...)`. Every
`TransportParams` field is written to HYSPLIT's `SETUP.CFG` by
`STILTParams.setup_entries()` unless it is listed in
`STILTParams.CONTROL_FIELDS` (read from `CONTROL`) or `ZICONTROL_FIELDS`;
`ErrorParams` fields go to `WINDERR` / `ZIERR`. When you add a field that
HYSPLIT reads from somewhere other than `SETUP.CFG`, add it to the matching
set and to the routing test in `tests/test_config.py`.

## Project store and completion

A project is one root (`stilt.project.Project`) over a `Store`
(`stilt.store`: `LocalStore`, `FsspecStore`). Every output is addressed by a
store key; `Simulation` owns the filenames, keys, presence checks, and the one
definition of completion (`Simulation.is_complete()`). Do not add a second
"does this output exist" check anywhere else — call the `Simulation` method.
Adding a store backend means implementing the five-method `Store` protocol.

## Adding execution backends

Execution backends implement the `Executor` and `JobHandle` protocols in
`src/stilt/execution/backends/protocol.py`. The coordinator relies on the
executor's `dispatch` mode:
- `push` executors receive an explicit list of pending simulation IDs and run
  them through `stilt.execution.run_simulations` (directly or via the
  `stilt push-worker` CLI); outputs are published by `Simulation.publish()`
- `pull` executors launch workers that claim from the Postgres queue and should
  preserve claim transactions until a simulation result is recorded or released

Register a new backend in `execution/backends/factory.py::resolve_backend`,
re-export it from `execution/__init__.py`, and handle SIGTERM with
`sigterm_as_interrupt` as the local, Slurm, and Kubernetes backends do.

Backend `start()` methods should return quickly with a handle. `wait()` should
raise on backend-level failure states rather than treating “not queued anymore”
as success. Add tests for submission failure, terminal failure states,
interruption/preemption, and repeated `wait()` calls.

For scheduler-backed executors, avoid unbounded subprocess calls, write
temporary task/chunk files under a predictable directory, and clean them up when
the backend can prove the launched job is finished.

## Adding particle transforms

Pre-footprint particle transforms implement the `ParticleTransform` protocol in
`src/stilt/transforms.py`. A transform receives a particle `DataFrame` and a
`TransformContext`, then returns a new `DataFrame` without mutating the
caller's data.

A built-in transform is one pydantic class in `transforms.py`: its fields are
the YAML keys, `kind` is a `Literal` discriminator, and `apply()` does the
work. Add it to the `BuiltinTransform` union; nothing else needs wiring. Users
can reference their own class by import path (`kind: my.module.Class`), so
only generally useful transforms belong in PYSTILT.

Add tests for:
- config parsing and YAML round trip
- the transform's numerical behavior on a small particle table
- interaction with `Footprint.calculate()` when the transform is configured on a footprint

## Pull Request Guidelines

- Keep pull requests focused on a single feature or bugfix
- Write clear, descriptive commit messages
- Update the changelog if applicable
- Ensure all tests pass
- Maintain or improve test coverage
- Update documentation as needed

## Reporting Bugs

When reporting bugs, please include:
- Your operating system and Python version
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any error messages or logs

## Feature Requests

We welcome feature requests! Please:
- Check if the feature has already been requested
- Provide a clear description of the feature
- Explain why it would be useful
- Consider submitting a pull request to implement it

## Questions?

If you have questions, please:
- Check existing issues and discussions
- Open a new issue with the "question" label
- Reach out to the maintainers

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
