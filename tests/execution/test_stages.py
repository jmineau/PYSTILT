"""The worker's stage loop, held with two toy stages that import nothing from the footprint."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
import yaml
from runtraj.execution import worker
from runtraj.execution.worker import run_release

from stilt import Project

from ..fixtures import stages as toy
from ..fixtures.factories import make_met_config, make_release
from ..fixtures.stages import chain


@pytest.fixture(autouse=True)
def runs():
    """The calls of the toy stages' ``run``, emptied for each test."""
    toy.RUNS.clear()
    toy.PATHS.clear()
    return toy.RUNS


@pytest.fixture
def release():
    return make_release(dt.datetime(2023, 1, 1, 12), -111.85, 40.77)


@pytest.fixture
def model_runs(monkeypatch):
    """Replace the transport model run with one that writes three particle rows, and list the variants it ran."""
    ran: list[str] = []

    def fake(sim, *, workdir, keep_workdir=False, timeout=None):
        ran.append(sim.variant.name)
        particles = pd.DataFrame(
            {
                "age": [-60.0, -120.0, -180.0],
                "particle": [1.0, 1.0, 1.0],
                "lon": [-111.9] * 3,
                "lat": [40.7] * 3,
                "zagl": [10.0] * 3,
                "foot": [1e-5] * 3,
            }
        )
        particles["time"] = pd.Timestamp(sim.release.time) + pd.to_timedelta(
            particles["age"], unit="min"
        )
        sim.output.write_particles(sim.variant, sim.release, particles, [])
        return particles

    monkeypatch.setattr(worker, "run_particles", fake)
    return ran


def _project(
    tmp_path, release, variants=None, *, stages=None, output=None, **count
) -> Project:
    """Make a project of one release with both toy stages (or *stages*), whose model run is faked."""
    return Project.init(
        tmp_path / "proj",
        releases=[release],
        mets={"hrrr": make_met_config(tmp_path / "met")},
        models={"hysplit": {"n_hours": -24}},
        stages=chain(**count) if stages is None else stages,
        variants=variants or {"hrrr": {}},
        **({} if output is None else {"output": output}),
    )


def _run(project, release, **kwargs) -> list[str]:
    return run_release(
        project, str(release.id), compute_root=project.directory / "scratch", **kwargs
    )


# -- the order, and what runs --------------------------------------------------


# -- failures ------------------------------------------------------------------


# -- status and completion ------------------------------------------------------


def test_a_tree_this_layout_did_not_write_is_listed_as_it_records_itself(
    tmp_path, release, model_runs
):
    project = _project(tmp_path, release)
    _run(project, release)
    old = project.output.directory / "footprints" / "settings=hrrr-abc123"
    old.mkdir(parents=True)
    record = {"name": "hrrr", "hash": "abc123def", "particles": "hrrr-000000"}
    (old / "_settings.yaml").write_text(yaml.safe_dump({**record, "settings": {}}))

    output = Project(project.directory).output

    assert output.kinds() == ["particles", "count", "double", "footprints"]
    assert output.hashes("footprints") == {"hrrr-abc123": "abc123def"}
    folders = Project(project.directory).folders()
    row = folders[folders["kind"] == "footprints"].iloc[0]
    assert (row["variant"], row["differs"]) == (
        "",
        "the variant has no footprints stage",
    )


def test_a_variant_without_stages_needs_no_particles_hash_to_say_so(tmp_path, release):
    """Asking about a stage a variant does not use reads neither the met nor the model's build."""
    from runtraj.config import Variant
    from runtraj.output import Output
    from runtraj.transport.hysplit import HysplitConfig, MetConfig

    variant = Variant(
        name="hrrr",
        met="hrrr",
        met_config=MetConfig(directory=str(tmp_path / "no_such_met")),
        transport=HysplitConfig(n_hours=-24),
        model="hysplit",
    )
    with pytest.raises(FileNotFoundError):
        _ = variant.particles_hash  # the met's files are not here
    assert variant.stage_hashes == {}
    output = Output(tmp_path / "output")
    assert output.path("footprint", variant, str(release.id)) is None
    assert output.present("footprint", variant) == frozenset()
    assert output.folder("footprint", variant) is None
