# Justfile for PYSTILT

# Show available commands
list:
    @just --list

# Install dependencies with uv
install:
	@echo "Installing dependencies with uv..."
	uv sync --group dev

# Build HTML documentation using Sphinx
build-docs:
	@echo "Building HTML documentation..."
	rm -rf docs/_build/
	MPLCONFIGDIR=/tmp/pystilt-mplconfig uv run sphinx-build -M html docs docs/_build

# Clean up build artifacts and cache files
clean:
	@echo "Cleaning up generated files..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf junit.xml
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

# Run pre-commit hooks on all files
pre-commit:
	@echo "Running pre-commit on all files..."
	uv run pre-commit run --all-files

# Run linting, type checking, and tests
quality-check:
	@echo "Running quality checks..."
	@echo "Linting with ruff..."
	uv run ruff check src/stilt
	@echo "Type checking with pyright..."
	uv run pyright src/stilt
	@echo "Checking import contracts..."
	uv run lint-imports
	just test

# Run ruff fixes and formatting
ruff:
	@echo "Running ruff fixes and formatting..."
	uv run ruff check --fix src/stilt
	uv run ruff format src/stilt

# Run tests with pytest
test:
	@echo "Running tests..."
	uv run pytest -v

# Wheel platform tag for each bundled HYSPLIT build. linux_x64/hycs_std is
# statically linked (no glibc symbol versions) and needs Linux 3.2 or newer,
# which every manylinux2014 (glibc 2.17) system has. macos_x64/hycs_std
# declares a minimum of macOS 10.16, which is macOS 11 (LC_BUILD_VERSION).
linux_wheel := "manylinux_2_17_x86_64"
macos_wheel := "macosx_11_0_x86_64"

# Build the sdist and one wheel per bundled HYSPLIT build into dist/
dist:
    rm -rf build dist
    uv build --sdist
    uv build --wheel -C--build-option=--plat-name={{linux_wheel}}
    uv build --wheel -C--build-option=--plat-name={{macos_wheel}}

# Check dist/: each wheel holds only its own hycs_std, the sdist none
check-dist:
    #!/usr/bin/env python3
    import tarfile
    import zipfile
    from pathlib import Path

    expected = {"{{linux_wheel}}": "linux_x64", "{{macos_wheel}}": "macos_x64"}
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
