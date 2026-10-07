# PYSTILT development tasks. CI runs these same recipes.

set positional-arguments

# Unit tests only: the slow suites need met files, HYSPLIT or STILT-R (AGENTS.md, Tests).
unit := "not integration and not fidelity and not r_only"

# Show available recipes
list:
    @just --list

# Install the project and dev tools into .venv
sync:
    uv sync

# Update uv.lock after changing dependencies
lock:
    uv lock

# Lint and check formatting (no changes)
lint:
    uv run ruff check
    uv run ruff format --check

# Fix lint and format the code
format:
    uv run ruff check --fix
    uv run ruff format

# Type check with pyrefly
type-check:
    uv run pyrefly check

# Check the import contracts in pyproject.toml (the package's layers)
imports:
    uv run lint-imports

# Require docstrings on the public API
docstr:
    uv run docstr-coverage src/stilt --skip-magic --skip-init --fail-under 95

# Run the unit tests in parallel (up to 8 workers; `-n 0` for serial)
test *args:
    uv run pytest -n auto --maxprocesses=8 -m "{{ unit }}" "$@"

# Run the unit tests with coverage (coverage.xml, junit.xml for Codecov)
cov *args:
    uv run pytest -n auto --maxprocesses=8 -m "{{ unit }}" --cov --cov-report=term --cov-report=xml --junitxml=junit.xml -o junit_family=legacy "$@"

# Build the HTML docs, failing on warnings
build-docs:
    rm -rf docs/_build docs/reference/_api
    MPLCONFIGDIR="${TMPDIR:-/tmp}/pystilt-mplconfig" uv run sphinx-build -M html docs docs/_build -W --keep-going

# Serve the docs at http://127.0.0.1:PORT, rebuilding on every save (Ctrl-C stops)
docs-serve port="8000":
    uv run sphinx-autobuild docs docs/_build/html --port "$1" --watch src --re-ignore 'reference/_api/'

# Everything the Code Quality workflow checks, plus the unit tests
quality-check: lint type-check imports docstr test

# Run every pre-commit hook on every file
pre-commit:
    uv run pre-commit run --all-files

# Wheel platform tag for each bundled HYSPLIT build. linux_x64/hycs_std is
# statically linked (no glibc symbol versions) and needs Linux 3.2 or newer,
# which every manylinux2014 (glibc 2.17) system has. macos_x64/hycs_std
# declares a minimum of macOS 10.16, which is macOS 11 (LC_BUILD_VERSION).
linux_wheel := "manylinux_2_17_x86_64"
macos_wheel := "macosx_11_0_x86_64"

# Build the sdist and one wheel per bundled HYSPLIT build into dist/, then check them
dist: && check-dist
    rm -rf build dist
    uv build --sdist
    uv build --wheel -C--build-option=--plat-name={{ linux_wheel }}
    uv build --wheel -C--build-option=--plat-name={{ macos_wheel }}
    uv run twine check --strict dist/*

# Check dist/: each wheel holds only its own hycs_std, the sdist none
check-dist:
    #!/usr/bin/env python3
    import tarfile
    import zipfile
    from pathlib import Path

    expected = {"{{ linux_wheel }}": "linux_x64", "{{ macos_wheel }}": "macos_x64"}
    wheels = sorted(Path("dist").glob("*.whl"))
    platforms = [wheel.stem.rsplit("-", 1)[1] for wheel in wheels]
    assert sorted(platforms) == sorted(expected), f"wheels for {platforms}"
    for wheel, platform in zip(wheels, platforms):
        assert wheel.stem.split("-")[2:4] == ["py3", "none"], wheel.name
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            info = next(n for n in names if n.endswith(".dist-info/WHEEL"))
            meta = zf.read(info).decode()
        assert f"Tag: py3-none-{platform}\n" in meta, meta
        assert "Root-Is-Purelib: false\n" in meta, meta
        binaries = [n for n in names if n.endswith("/hycs_std")]
        want = [f"stilt/transport/hysplit/bin/{expected[platform]}/hycs_std"]
        assert binaries == want, f"{wheel.name} holds {binaries}"
        print(f"{wheel.name}: {binaries[0]}")
    (sdist,) = Path("dist").glob("*.tar.gz")
    with tarfile.open(sdist) as tf:
        binaries = [n for n in tf.getnames() if n.endswith("/hycs_std")]
    assert not binaries, f"{sdist.name} holds {binaries}"
    print(f"{sdist.name}: no binary")

# Draft CHANGELOG entries from the commits since the last release
changelog:
    @uv run git-cliff --unreleased --strip all

# Print the version setuptools-scm computes from git
version:
    @uv run python -m setuptools_scm

# Tag and push release VERSION (e.g. `just release 0.1.0a23`); CI publishes it
release version:
    #!/usr/bin/env bash
    set -euo pipefail
    v="$1"
    test -z "$(git status --porcelain)" || { echo "Working tree is not clean." >&2; exit 1; }
    test "$(git branch --show-current)" = main || { echo "Release from main." >&2; exit 1; }
    git fetch --quiet --tags origin main
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" || { echo "main is not in sync with origin/main." >&2; exit 1; }
    grep -q "^## \[$v\]" CHANGELOG.md || { echo "CHANGELOG.md has no '## [$v]' section." >&2; exit 1; }
    # PEP 440: the version must be in normal form and newer than every v* tag,
    # or installers would not see it as the latest release.
    uv run --no-sync python - "$v" <<'PY'
    import subprocess
    import sys

    from packaging.version import InvalidVersion, Version

    new = Version(sys.argv[1])
    if str(new) != sys.argv[1]:
        sys.exit(f"{sys.argv[1]} normalizes to {new}; release it as {new}.")
    tags = subprocess.run(["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True).stdout.split()
    old = []
    for tag in tags:
        try:
            old.append(Version(tag[1:]))
        except InvalidVersion:
            pass
    if old and new <= max(old):
        sys.exit(f"{new} is not newer than the latest release, v{max(old)}.")
    PY
    git tag --annotate "v$v" --message "PYSTILT $v"
    git push origin "v$v"
    echo "Pushed v$v; the Publish workflow builds and releases it."

# Remove build artifacts and caches
clean:
    rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache .pyrefly_cache .import_linter_cache
    rm -rf .coverage coverage.xml junit.xml htmlcov docs/_build docs/reference/_api
    find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
