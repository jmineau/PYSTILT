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

Use `cfg_field()` when a field needs PYSTILT metadata such as `visibility`,
`target`, or a HYSPLIT `namelist` alias. Prefer class `DEFAULT_TARGET` values
for whole parameter groups, and override `target` only when a field routes
somewhere different, such as `control`, `zicontrol`, `winderr`, or `zierr`.

When adding a routed field, update the config routing tests so contributors can
see which generated HYSPLIT input file the field affects.

## Adding index backends

PYSTILT index backends implement the `SimulationIndex` protocol in
`src/stilt/index/protocol.py`. The SQLite and PostgreSQL implementations share
most CRUD behavior through `_SqlIndex` in `src/stilt/index/base.py`; a new SQL
backend should reuse that base unless it has a materially different transaction
model.

For a SQL backend, provide:
- a connection factory returning context-managed connections with dict-style row access
- placeholder, boolean, timestamp, and JSON-cast dialect constants
- `_execute_match_ids()` for efficient `sim_id` filtering in that dialect
- `_table_columns()` for schema validation
- tests for registration, counts, summaries, reset/rebuild behavior, and pruning

Completion semantics are shared through `build_index_predicates()` in
`src/stilt/index/sql.py`. Do not copy/paste the completed, pending, or
prune-eligible predicates into backend modules; pass only the dialect-specific
JSON traversal fragments.

## Adding execution backends

Execution backends implement the `Executor` and `JobHandle` protocols in
`src/stilt/execution/backends/protocol.py`. The coordinator relies on the
executor's `dispatch` mode:
- `push` executors receive an explicit list of pending simulation IDs and should
  publish output files before the coordinator rebuilds local state
- `pull` executors claim work from the shared index and should preserve claim
  transactions until a simulation result is recorded or released

Backend `start()` methods should return quickly with a handle. `wait()` should
raise on backend-level failure states rather than treating “not queued anymore”
as success. Add tests for submission failure, terminal failure states,
interruption/preemption, and repeated `wait()` calls.

For scheduler-backed executors, avoid unbounded subprocess calls, write
temporary task/chunk files under a predictable directory, and clean them up when
the backend can prove the launched job is finished.

## Adding particle transforms

Pre-footprint particle transforms implement the `ParticleTransform` protocol in
`src/stilt/transforms.py`. A transform receives a particle `DataFrame` and an
optional `ParticleTransformContext`, then returns a transformed `DataFrame`
without mutating caller-owned data unexpectedly.

Declarative transforms should have a config spec in `src/stilt/config` and a
builder branch in `build_particle_transform()`. Runtime-only transforms can be
passed as objects that implement `apply(...)`, but public/documented transforms
should prefer config specs so YAML round trips remain reproducible.

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
