"""
Helpers for the docs pages that run real PYSTILT code when the docs are built.

The pages run on the HRRR files the tests cache: 14 and 15 January 2021,
cropped around Salt Lake. A page that works with a project on disk starts from
a small sample project, which is built once per docs build (about 40 s) and
copied into a temporary folder for each page that uses it.

A page uses these from a block that is run but not shown (``:suppress:``)::

    import docs_examples

    docs_cwd = docs_examples.temp_workdir(sample=True)

and returns to the docs folder at its end with ``os.chdir(docs_cwd)``.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]

#: Where the sample project is built. ``just build-docs`` clears ``docs/_build``,
#: so every clean build makes a new one; ``just docs-serve`` reuses it.
SAMPLE = DOCS / "_build" / "sample_project"

#: The sample project's receptors: one site, the four hours before 10:00 UTC.
SAMPLE_HOURS = (6, 7, 8, 9)

_temporary_folders: list[Path] = []


def met() -> dict:
    """Return the met config of the files the tests cache."""
    default = DOCS.parent / "tests" / "met_cache"
    directory = Path(os.environ.get("STILT_TEST_MET_DIR", default)).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} is missing. The docs run on the met files the tests "
            "cache: see 'Examples in the docs' in AGENTS.md."
        )
    return {"directory": str(directory), "file_format": "%Y%m%d_%H", "file_tres": "6h"}


def sample_project() -> Path:
    """Return the folder of the sample project, building it the first time."""
    import stilt

    project = SAMPLE / "my_project"
    marker = SAMPLE / "complete"
    if not marker.exists():
        shutil.rmtree(SAMPLE, ignore_errors=True)
        receptors = [
            stilt.PointReceptor(
                time=f"2021-01-15 {hour:02d}:00",
                longitude=-111.848,
                latitude=40.766,
                altitude=10,
            )
            for hour in SAMPLE_HOURS
        ]
        grid = stilt.Grid(
            xmin=-113.0, xmax=-110.5, ymin=40.0, ymax=42.0, xres=0.01, yres=0.01
        )
        stilt.Project.init(
            str(project),
            receptors=receptors,
            mets={"hrrr": met()},
            variants={"hrrr": {}},
            n_hours=-24,
            numpar=200,
            grid=grid,
        ).run()
        marker.touch()
    return project


def temp_workdir(sample: bool = False) -> str:
    """
    Change to a new temporary folder and return the folder you left.

    With ``sample``, the new folder holds a copy of the sample project as
    ``my_project``, so a page can open it the way a reader would.
    """
    previous = os.getcwd()
    work = Path(tempfile.mkdtemp())
    _temporary_folders.append(work)
    if sample:
        shutil.copytree(sample_project(), work / "my_project")
    os.chdir(work)
    return previous


@atexit.register
def _remove_temporary_folders() -> None:
    for folder in _temporary_folders:
        shutil.rmtree(folder, ignore_errors=True)
