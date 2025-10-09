# Contributing to PYSTILT

Thank you for considering contributing to PYSTILT! We welcome contributions from the community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PYSTILT.git
   cd PYSTILT
   ```
3. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
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

3. Run the test suite:
   ```bash
   pytest
   ```

4. Run pre-commit checks:
   ```bash
   pre-commit run --all-files
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request on GitHub

## Code Style

This project uses:
- **Ruff** for code formatting and linting (replaces Black, isort, and flake8)
- **Pyright** for static type checking

Pre-commit hooks will automatically check and format your code before each commit.

## Testing

- Write tests for all new features and bug fixes
- Place tests in the `tests/` directory
- Use pytest for testing
- Aim for high test coverage

Run tests with:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src/stilt --cov-report=html
```

Check code with ruff:
```bash
ruff check src/stilt
ruff format src/stilt
```

Check type coverage:
```bash
pyright src/stilt
```

Check docstring coverage:
```bash
docstr-coverage src/stilt --skip-magic --skip-init
```

## Documentation

- Update documentation for any changed functionality
- Add docstrings to all public functions, classes, and modules
- Use the NumPy docstring style
- Build docs locally to verify changes:
  ```bash
  cd docs
  make html
  ```

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
