"""
The footprint and the map run end to end in a runtraj project, through runtraj's own worker.

:func:`runtraj.testing.run_stage` makes a project of one release whose
transport model hands back the particles given here, runs it as ``runtraj
run`` does, and checks what a project relies on: the stage's file is where
completion looks for it, a second run makes nothing again, and the settings
it records read back. A change in runtraj that breaks either stage fails
here, not in a user's run.
"""

from __future__ import annotations

import base64
import datetime as dt
import json

from runtraj.testing import run_stage

from stilt.footprint import read_footprint

from ..fixtures.factories import make_release
from ..fixtures.particles import finished, random_walk

GRID = {
    "xmin": -113.0,
    "xmax": -110.5,
    "ymin": 40.0,
    "ymax": 42.0,
    "xres": 0.01,
    "yres": 0.01,
}


def _walk():
    """Return a release and its particles, a seeded random walk back from it."""
    release = make_release(dt.datetime(2024, 1, 15, 18))  # where the walk starts
    return release, finished(random_walk(), release)


def test_the_footprint_runs_in_a_project(tmp_path):
    release, particles = _walk()

    sim = run_stage(
        "stilt.footprint.FootprintStage",
        particles,
        release,
        tmp_path,
        settings={"grid": GRID},
    )

    foot = read_footprint(sim.path("footprint"))
    assert foot is not None
    assert float(foot.sum()) > 0


def test_the_map_runs_after_the_footprint_in_a_project(tmp_path):
    release, particles = _walk()

    sim = run_stage(
        "stilt.footprint.MapStage",
        particles,
        release,
        tmp_path,
        others={"footprint": {"class": "stilt.footprint.FootprintStage", "grid": GRID}},
    )

    entry = json.loads(sim.path("map").read_text())
    assert entry["release"] == str(release.id)
    head, data = entry["image"].split(",", 1)
    assert head == "data:image/png;base64"
    assert base64.b64decode(data).startswith(b"\x89PNG")
