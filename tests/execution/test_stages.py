"""The worker's stage loop, held with two toy stages that import nothing from the footprint."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
import yaml

from stilt import Project
from stilt.execution import worker
from stilt.execution.worker import run_release, run_stage
from stilt.output import completed

from ..fixtures import stages as toy
from ..fixtures.factories import make_met_config, make_release
from ..fixtures.stages import COUNT, DOUBLE, ECHO, chain


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


def _text(project, release, variant, stage) -> str:
    return project.simulation(str(release.id), variant).path(stage).read_text()


# -- the order, and what runs --------------------------------------------------


def test_stages_run_after_the_particles_in_the_order_they_read(
    tmp_path, release, model_runs, runs
):
    project = _project(tmp_path, release, scale=2)

    assert _run(project, release) == []

    rid = str(release.id)
    assert model_runs == ["hrrr"]
    assert runs == [("count", rid, "hrrr"), ("double", rid, "hrrr")]
    assert _text(project, release, "hrrr", "count") == "6"  # three rows, scale 2
    assert _text(project, release, "hrrr", "double") == "12"
    sim = project.simulation(rid, "hrrr")
    assert sim.has("particles") and sim.has("count") and sim.has("double")
    assert sim.is_complete and sim.failure is None


def test_each_stage_file_is_in_a_tree_named_for_its_stage(
    tmp_path, release, model_runs
):
    project = _project(tmp_path, release)
    _run(project, release)

    sim = project.simulation(str(release.id), "hrrr")
    output = project.output.directory
    variant = sim.variant
    assert sim.path("count") == (
        output
        / "count"
        / f"settings=hrrr-{variant.stage_hashes['count'][:6]}"
        / "date=2023-01-01"
        / f"{release.id}.txt"
    )
    assert sim.path("double").parent.parent.parent == output / "double"
    assert project.output.kinds() == ["particles", "count", "double"]
    # Nothing is left of the temporary name a stage writes under.
    assert [p.name for p in sim.path("count").parent.iterdir()] == [f"{release.id}.txt"]


def test_a_stage_folder_records_what_it_was_made_from(tmp_path, release, model_runs):
    project = _project(tmp_path, release)
    _run(project, release)

    variant = project.variants["hrrr"]
    folder = project.output.folder("double", variant)
    record = yaml.safe_load((folder / "_settings.yaml").read_text())
    assert record["stage"] == "double" and record["class"] == DOUBLE
    assert record["hash"] == variant.stage_hashes["double"]
    assert record["settings"] == {"add": 0}
    assert record["reads"] == {
        "particles": {
            "folder": project.output.folder("particles", variant).name[9:],
            "hash": variant.particles_hash,
        },
        "count": {
            "folder": project.output.folder("count", variant).name[9:],
            "hash": variant.stage_hashes["count"],
        },
    }
    # A new Output finds both folders again from their records.
    again = Project(project.directory)
    assert again.output.folder("double", again.variants["hrrr"]) == folder


def test_variants_with_equal_settings_share_a_stage_file_made_once(
    tmp_path, release, model_runs, runs
):
    variants = {
        "hrrr": {},
        "twin": {},  # every setting equal: the same three files
        "plus": {"stages": {"double": {"add": 1}}},  # the same count, its own double
        "zi08": {"ziscale": 0.8},  # other particles: its own of everything
    }
    project = _project(tmp_path, release, variants)

    assert _run(project, release) == []

    rid = str(release.id)
    assert model_runs == ["hrrr", "zi08"]
    assert runs == [
        ("count", rid, "hrrr"),
        ("double", rid, "hrrr"),
        ("double", rid, "plus"),
        ("count", rid, "zi08"),
        ("double", rid, "zi08"),
    ]
    assert _text(project, release, "twin", "double") == "6"
    assert _text(project, release, "plus", "double") == "7"
    assert project.incomplete().empty


def test_existing_stage_files_are_kept(tmp_path, release, model_runs, runs):
    project = _project(tmp_path, release)
    _run(project, release)
    runs.clear()
    model_runs.clear()

    assert _run(project, release) == []

    assert model_runs == [] and runs == []


def test_a_stage_is_made_again_when_what_it_reads_was(
    tmp_path, release, model_runs, runs
):
    project = _project(tmp_path, release)
    _run(project, release)
    sim = project.simulation(str(release.id), "hrrr")
    sim.path("double").write_text("stale")
    sim.path("count").unlink()
    runs.clear()
    model_runs.clear()

    assert _run(project, release) == []

    # The count was missing, so the double that reads it is made again too.
    assert model_runs == []
    assert [stage for stage, _, _ in runs] == ["count", "double"]
    assert sim.path("double").read_text() == "6"


def test_new_particles_remake_every_stage_file(tmp_path, release, model_runs, runs):
    project = _project(tmp_path, release)
    _run(project, release)
    sim = project.simulation(str(release.id), "hrrr")
    sim.path("particles").unlink()
    runs.clear()

    assert _run(project, release) == []
    assert [stage for stage, _, _ in runs] == ["count", "double"]

    runs.clear()
    assert _run(project, release, skip_existing=False) == []
    assert [stage for stage, _, _ in runs] == ["count", "double"]


def test_a_variant_without_a_stage_is_complete_without_its_file(
    tmp_path, release, model_runs, runs
):
    variants = {
        "hrrr": {"stages": {"double": None}},
        "plain": {"stages": {"count": None, "double": None}},
    }
    project = _project(tmp_path, release, variants)

    assert _run(project, release) == []

    assert [stage for stage, _, _ in runs] == ["count"]
    plain = project.simulation(str(release.id), "plain")
    assert plain.is_complete and plain.path("count") is None and not plain.has("count")
    assert project.incomplete().empty


# -- failures ------------------------------------------------------------------


def test_a_stage_that_fails_is_recorded_and_leaves_no_file(
    tmp_path, release, model_runs, runs
):
    project = _project(tmp_path, release, fail=True)

    problems = _run(project, release)

    assert problems == [
        "hrrr failed during count (RuntimeError): the count was told to fail"
    ]
    sim = project.simulation(str(release.id), "hrrr")
    # The half-written file was under a temporary name, and is gone.
    assert sim.has("particles") and not sim.has("count")
    assert list(sim.path("count").parent.iterdir()) == []
    # The stage that reads it did not run, and has nothing recorded of its own.
    assert [stage for stage, _, _ in runs] == ["count"]
    failure = sim.failure
    assert failure["stage"] == "count" and failure["reason"] == "RuntimeError"
    assert "the count was told to fail" in failure["traceback"]
    assert project.output.failure("double", sim.variant, release.id) is None
    assert not sim.is_complete


def test_a_stage_writes_under_a_hidden_name_that_ends_in_its_suffix(
    tmp_path, release, model_runs
):
    """A writer may choose its format by the extension, or append one that is missing."""
    project = _project(tmp_path, release)
    _run(project, release)

    handed = toy.PATHS[0]
    assert handed.startswith(f".{release.id}.") and handed.endswith(".txt")
    # A file still under such a name, as a killed run leaves, is no result.
    sim = project.simulation(str(release.id), "hrrr")
    folder = sim.path("count").parent
    sim.path("count").rename(folder / handed)
    assert project.output.present("count", sim.variant) == frozenset()
    assert not sim.has("count") and not sim.is_complete


def test_a_stage_that_fails_to_remake_its_file_leaves_no_old_one(
    tmp_path, release, model_runs, monkeypatch
):
    """A file made from what has since been made again must not count as complete."""
    project = _project(tmp_path, release)
    _run(project, release)
    sim = project.simulation(str(release.id), "hrrr")
    assert sim.is_complete
    sim.path("count").unlink()  # the count is made again, so the double is too

    def fails(self, config, sim, path):
        raise RuntimeError("no double today")

    monkeypatch.setattr(toy.DoubleStage, "run", fails)
    problems = _run(project, release)

    assert problems == ["hrrr failed during double (RuntimeError): no double today"]
    assert sim.has("count") and not sim.has("double")
    assert not sim.is_complete and sim.failure["stage"] == "double"
    assert project.status()["state"].tolist() == ["failed"]


def test_on_an_object_store_a_stage_that_fails_leaves_no_file(
    tmp_path, release, model_runs
):
    """A store has no temporary name to discard, so what a failed stage wrote is removed."""
    pytest.importorskip("upath")
    import uuid

    store = f"memory://pystilt-stages-{uuid.uuid4().hex}/output"
    project = _project(tmp_path, release, output=store, fail=True)

    problems = _run(project, release)

    assert problems == [
        "hrrr failed during count (RuntimeError): the count was told to fail"
    ]
    sim = project.simulation(str(release.id), "hrrr")
    assert [f"{release.id}.txt"] == toy.PATHS  # the stage wrote the file's own path
    assert sim.has("particles") and not sim.has("count")
    assert not sim.is_complete and sim.failure["stage"] == "count"
    assert project.status()["state"].tolist() == ["failed"]


def test_on_an_object_store_the_stages_run_and_complete(tmp_path, release, model_runs):
    pytest.importorskip("upath")
    import uuid

    store = f"memory://pystilt-stages-{uuid.uuid4().hex}/output"
    project = _project(tmp_path, release, output=store, scale=2)

    assert _run(project, release) == []

    assert _text(project, release, "hrrr", "double") == "12"
    assert project.status()["state"].tolist() == ["complete"]


def test_a_failure_is_cleared_when_the_stage_succeeds(
    tmp_path, release, model_runs, runs
):
    project = _project(tmp_path, release, fail=True)
    _run(project, release)
    sim = project.simulation(str(release.id), "hrrr")
    assert sim.failure is not None

    # The same settings folder, with a stage that now works.
    toy.CountStage.run, broken = _working_run, toy.CountStage.run
    try:
        assert _run(project, release) == []
    finally:
        toy.CountStage.run = broken
    assert sim.is_complete and sim.failure is None


def _working_run(self, config, sim, path):
    path.write_text("3")


def test_a_stage_that_writes_no_file_fails(tmp_path, release, model_runs):
    project = _project(tmp_path, release, silent=True)

    problems = _run(project, release)

    assert len(problems) == 1
    assert "The count stage returned without writing its file" in problems[0]
    sim = project.simulation(str(release.id), "hrrr")
    assert sim.failure["stage"] == "count" and not sim.has("count")


def test_a_stage_that_failed_is_named_though_an_earlier_one_is_only_unfinished(
    tmp_path, release, model_runs
):
    """Stages that do not read one another: the one with a record explains the simulation."""
    stages = {
        "count": {"class": COUNT, "stale": True},  # a connection error: no record
        "echo": {"class": ECHO, "fail": True},  # a real failure, recorded
    }
    project = _project(tmp_path, release, stages=stages)

    problems = _run(project, release)

    assert len(problems) == 2
    assert problems[0].startswith("hrrr did not finish during count (ConnectionError)")
    assert (
        problems[1]
        == "hrrr failed during echo (RuntimeError): the echo was told to fail"
    )
    sim = project.simulation(str(release.id), "hrrr")
    assert not sim.has("count") and not sim.has("echo")
    assert sim.failure["stage"] == "echo"
    status = project.status().iloc[0]
    assert (status["state"], status["stage"], status["reason"]) == (
        "failed",
        "echo",
        "RuntimeError",
    )


def test_a_stage_whose_input_is_missing_has_not_failed(tmp_path, release, model_runs):
    """An old record of a stage does not explain a simulation whose earlier stage is missing."""
    project = _project(tmp_path, release)
    _run(project, release)
    sim = project.simulation(str(release.id), "hrrr")
    record = {"stage": "double", "reason": "Old", "message": "from an earlier run"}
    project.output.record_failure("double", sim.variant, release.id, record)
    sim.path("double").unlink()
    assert sim.failure == record  # the count is there, so the double could have failed
    sim.path("count").unlink()
    assert sim.failure is None  # nothing to read: the double did not run
    assert project.status()["state"].tolist() == ["pending"]


def test_failed_particles_run_no_stage(tmp_path, release, monkeypatch, runs):
    project = _project(tmp_path, release)

    def fails(sim, **kwargs):
        raise ValueError("no met")

    monkeypatch.setattr(worker, "run_particles", fails)

    assert _run(project, release) == [
        "hrrr failed during particles (ValueError): no met"
    ]
    assert runs == []
    assert project.simulation(str(release.id), "hrrr").failure["stage"] == "particles"


def test_run_stage_refuses_a_stage_the_variant_does_not_use(
    tmp_path, release, model_runs
):
    project = _project(tmp_path, release, {"hrrr": {"stages": {"double": None}}})
    sim = project.simulation(str(release.id), "hrrr")
    with pytest.raises(ValueError, match="Variant 'hrrr' has no double stage"):
        run_stage(sim, "double")


# -- status and completion ------------------------------------------------------


def test_status_has_a_column_for_each_stage_and_says_which_failed(
    tmp_path, release, model_runs
):
    variants = {
        "hrrr": {},
        "broken": {"ziscale": 0.8, "stages": {"count": {"fail": True}}},
        "counted": {"ziscale": 0.9, "stages": {"double": None}},
        "plain": {"ziscale": 0.7, "stages": {"count": None, "double": None}},
    }
    project = _project(tmp_path, release, variants)
    _run(project, release)

    status = project.status().set_index("variant")

    columns = list(status.columns)
    assert columns[-7:] == [
        "particles",
        "count",
        "double",
        "state",
        "stage",
        "reason",
        "message",
    ]
    assert status["particles"].tolist() == [True, True, True, True]
    assert status.loc["hrrr", ["count", "double"]].tolist() == [True, True]
    assert status.loc["broken", ["count", "double"]].tolist() == [False, False]
    # A stage a variant does not use is neither there nor missing.
    assert status.loc["counted", "count"] and pd.isna(status.loc["counted", "double"])
    assert status.loc["plain", ["count", "double"]].isna().all()
    assert status["state"].tolist() == ["complete", "failed", "complete", "complete"]
    assert status.loc["broken", ["stage", "reason"]].tolist() == [
        "count",
        "RuntimeError",
    ]
    assert status.loc["hrrr", ["stage", "reason", "message"]].isna().all()
    assert list(project.incomplete()["variant"]) == ["broken"]


def test_a_project_without_stages_has_no_stage_column(tmp_path, release, model_runs):
    project = Project.init(
        tmp_path / "proj",
        releases=[release],
        mets={"hrrr": make_met_config(tmp_path / "met")},
        models={"hysplit": {"n_hours": -24}},
        variants={"hrrr": {}},
    )
    _run(project, release)

    status = project.status()

    assert list(status.columns[-5:]) == [
        "particles",
        "state",
        "stage",
        "reason",
        "message",
    ]
    assert status["state"].tolist() == ["complete"]


def test_completion_is_the_same_rule_from_listings_and_for_one_simulation(
    tmp_path, release, model_runs
):
    project = _project(tmp_path, release)
    sim = project.simulation(str(release.id), "hrrr")
    rid, variant, output = str(release.id), sim.variant, project.output
    assert output.complete(variant) == frozenset() and not sim.is_complete

    _run(project, release)
    assert output.complete(variant) == {rid} and sim.is_complete
    assert output.complete(variant) == completed(
        output.present("particles", variant),
        output.present("count", variant),
        output.present("double", variant),
    )

    sim.path("double").unlink()
    assert output.complete(variant) == frozenset() and not sim.is_complete
    assert output.present("count", variant) == {rid}
    assert project.status()["state"].tolist() == ["pending"]


def test_the_folders_table_lists_each_stage_tree(tmp_path, release, model_runs):
    project = _project(
        tmp_path, release, {"hrrr": {}, "tripled": {"stages": {"count": {"scale": 3}}}}
    )
    _run(project, release)

    folders = project.folders()

    assert list(folders["kind"]) == ["particles", "count", "count", "double", "double"]
    assert list(folders["files"]) == [1, 1, 1, 1, 1]
    count = folders[folders["kind"] == "count"].set_index("name")
    assert count.loc["hrrr", "variant"] == "hrrr"
    assert count.loc["tripled", "variant"] == "tripled"
    # A variant renamed in the config no longer uses the folders of its old settings.
    config = project.config_path.read_text().replace("scale: 3", "scale: 4")
    project.config_path.write_text(config)
    changed = Project(project.directory).folders()
    old = changed[(changed["kind"] == "count") & (changed["name"] == "tripled")]
    assert list(old["variant"]) == [""]
    assert list(old["differs"]) == ["scale: 3 (config: 4)"]
    old_double = changed[(changed["kind"] == "double") & (changed["name"] == "tripled")]
    assert list(old_double["differs"]) == ["made from other particles"]


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


def test_a_stage_folder_whose_class_is_not_installed_keeps_its_record(
    tmp_path, release, model_runs
):
    project = _project(tmp_path, release)
    _run(project, release)
    variant = project.variants["hrrr"]
    path = project.output.folder("count", variant) / "_settings.yaml"
    record = yaml.safe_load(path.read_text())
    record["class"] = "nopkg_for_this_test.stages.Count"
    path.write_text(yaml.safe_dump(record, sort_keys=False))

    output = Project(project.directory).output
    with pytest.warns(UserWarning, match="could not be read .*kept as written"):
        hashes = output.hashes("count")

    # As written, the settings hash as they did, so the folder is still found.
    assert list(hashes.values()) == [variant.stage_hashes["count"]]


def test_logs_and_scratch_are_not_kinds(tmp_path, release, model_runs):
    project = _project(tmp_path, release)
    _run(project, release)
    with pytest.raises(ValueError, match="kind is 'particles' or a stage's name"):
        project.output.hashes("logs")


def test_the_footprints_old_kind_is_refused_by_name(tmp_path, release, model_runs):
    """Code written for the earlier name gets a message, not an empty answer."""
    project = _project(tmp_path, release)
    _run(project, release)
    sim = project.simulation(str(release.id), "hrrr")
    message = "The footprints' kind is 'footprint', the name of their stage"
    with pytest.raises(ValueError, match=message):
        project.output.present("footprints", sim.variant)
    with pytest.raises(ValueError, match=message):
        project.output.table("footprints", sim.variant)
    with pytest.raises(ValueError, match=message):
        sim.path("footprints")
    with pytest.raises(ValueError, match=message):
        sim.has("footprints")
    # Any other name is taken for a stage this variant does not use.
    assert sim.path("footprint") is None and not sim.has("footprint")


def test_a_variant_without_stages_needs_no_particles_hash_to_say_so(tmp_path, release):
    """Asking about a stage a variant does not use reads neither the met nor the model's build."""
    from stilt.config import Variant
    from stilt.output import Output
    from stilt.transport.hysplit import HysplitConfig, MetConfig

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
