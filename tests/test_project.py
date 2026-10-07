"""Tests for stilt.project: the project directory, its tables, its results, and running it."""

import datetime as dt
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.config import ProjectConfig
from stilt.execution.config import ExecutionConfig
from stilt.execution.runner import resolve_compute_root
from stilt.output import Output
from stilt.project import Project
from stilt.receptors import PointReceptor
from stilt.spatial import Grid
from stilt.transport.hysplit import MetConfig
from stilt.transport.hysplit.driver import winderrtf

from .fixtures.factories import make_met_config, make_receptor
from .fixtures.footprints import as_footprint
from .fixtures.particles import finished

matplotlib.use("Agg")

_XYERR = {
    "siguverr": 1.0,
    "tluverr": 60.0,
    "zcoruverr": 100.0,
    "horcoruverr": 10.0,
}

_GRID = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _met(tmp_path) -> MetConfig:
    return make_met_config(tmp_path / "met")


def _config(tmp_path, include_footprint=True, **overrides) -> ProjectConfig:
    """Minimal ProjectConfig with one met and (optionally) a footprint grid."""
    overrides.setdefault("grid", _GRID if include_footprint else None)
    return ProjectConfig(
        mets={"hrrr": _met(tmp_path)}, **{"variants": {"hrrr": {}}, **overrides}
    )


def _project(tmp_path, receptors=None, name="proj", **overrides) -> Project:
    """Make a project in ``tmp_path / name`` with the minimal config."""
    return Project.init(
        tmp_path / name, config=_config(tmp_path, **overrides), receptors=receptors
    )


def _receptor(hour: int, longitude: float = -111.85, **attrs) -> PointReceptor:
    return make_receptor(dt.datetime(2023, 1, 1, hour), longitude, **attrs)


def _particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-60.0, -120.0],
            "particle": [1.0, 1.0],
            "lon": [-113.9, -113.5],
            "lat": [39.7, 39.6],
            "zagl": [10.0, 20.0],
            "foot": [1e-5, 2e-5],
            "dens": [1.2, 1.2],
            "samt": [1.0, 1.0],
            "sigw": [0.1, 0.1],
            "tlgr": [10.0, 10.0],
            "mlht": [500.0, 500.0],
        }
    )


def _write_trajectory(
    project: Project, receptor, variant="hrrr", realization=None
) -> Path:
    """Write a small particle file for one simulation into the output directory."""
    sim = project.simulation(receptor.id, variant, realization)
    particles = finished(_particles(), sim.receptor, sim.transport)
    return sim.output.write_particles(
        sim.variant, sim.receptor, particles, [], realization
    )


def _write_footprint(
    project: Project, receptor, variant="hrrr", *, empty=False
) -> Path:
    """Write a footprint (or an empty one) for one simulation into the output directory."""
    sim = project.simulation(receptor.id, variant)
    assert sim.variant.footprint is not None
    if empty:
        return sim.output.write_empty_footprint(sim.variant, sim.receptor)
    grid = sim.variant.footprint.grid
    assert grid is not None
    x_axis, y_axis = grid.axes
    data = xr.DataArray(
        np.random.default_rng(0).random((1, len(y_axis), len(x_axis))),
        dims=("time", "lat", "lon"),
        coords={"time": [sim.receptor.time], "lat": y_axis, "lon": x_axis},
    )
    foot = as_footprint(data, sim.receptor, sim.variant.footprint, sim.variant.name)
    return sim.output.write_footprint(sim.variant, foot)


def _pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(frame["receptor"], frame["variant"], strict=True))


# ---------------------------------------------------------------------------
# Making and opening a project
# ---------------------------------------------------------------------------


def test_init_writes_the_settings_given_and_the_receptors(tmp_path, point_receptor):
    project = Project.init(
        tmp_path / "proj",
        receptors=[point_receptor],
        mets={"hrrr": _met(tmp_path)},
        n_hours=-6,
        variants={"hrrr": {}},
    )

    text = (tmp_path / "proj" / "config.yaml").read_text()
    assert "n_hours: -6" in text and "numpar" not in text  # only what was given
    assert (tmp_path / "proj" / "receptors.csv").exists()
    assert project.config.n_hours == -6
    assert list(project.receptors["receptor"]) == [point_receptor.id]
    assert Project(tmp_path / "proj").config == project.config


def test_an_output_given_when_opening_replaces_the_configs(tmp_path, point_receptor):
    project = _project(tmp_path, [point_receptor])
    assert project.output.directory == project.directory / "output"

    elsewhere = Project(project.directory, output=tmp_path / "bucket")

    assert elsewhere.output.directory == tmp_path / "bucket"
    _write_trajectory(elsewhere, point_receptor)
    assert elsewhere.simulation(point_receptor.id, "hrrr").has_particles
    assert not project.simulation(point_receptor.id, "hrrr").has_particles


def test_init_takes_the_receptors_second(tmp_path, point_receptor):
    # The quickstart's order: a project directory, then its receptors.
    project = Project.init(
        tmp_path / "proj",
        [point_receptor],
        mets={"hrrr": _met(tmp_path)},
        variants={"hrrr": {}},
    )

    assert list(project.receptors["receptor"]) == [point_receptor.id]


def test_init_refuses_a_project_that_has_a_config(tmp_path):
    _project(tmp_path)
    text = (tmp_path / "proj" / "config.yaml").read_text()

    with pytest.raises(FileExistsError, match="already has a config.yaml"):
        Project.init(tmp_path / "proj", config=_config(tmp_path, numpar=10))
    assert (tmp_path / "proj" / "config.yaml").read_text() == text


def test_init_takes_a_config_or_keywords_not_both(tmp_path):
    with pytest.raises(TypeError, match="not both"):
        Project.init(tmp_path / "proj", config=_config(tmp_path), n_hours=-6)
    assert not (tmp_path / "proj").exists()


def test_init_writes_the_commented_starter_config(tmp_path):
    from stilt.config import STARTER_CONFIG

    project = Project.init(tmp_path / "proj", starter=True)

    assert project.config_path.read_text() == STARTER_CONFIG
    assert list(project.variants) == ["hrrr"]
    assert project.receptors.empty
    with pytest.raises(TypeError, match="starter"):
        Project.init(tmp_path / "other", starter=True, numpar=10)
    with pytest.raises(FileExistsError):
        Project.init(tmp_path / "proj", starter=True)


def test_init_copies_receptors_from_a_csv(tmp_path):
    csv = tmp_path / "mine.csv"
    csv.write_text(
        "time,longitude,latitude,altitude,site\n"
        "2023-01-01 12:00:00,-111.85,40.77,5,WBB\n"
    )
    project = Project.init(tmp_path / "proj", config=_config(tmp_path), receptors=csv)

    assert project.receptors["site"].tolist() == ["WBB"]


def test_opening_reads_nothing_until_asked(tmp_path):
    project = Project(tmp_path / "nothing")

    assert project.directory == (tmp_path / "nothing").resolve()
    assert project.name == "nothing"
    assert repr(project) == f"Project({str(project.directory)!r})"
    with pytest.raises(FileNotFoundError, match="Project.init"):
        _ = project.config
    assert not (tmp_path / "nothing").exists()


def test_a_relative_path_is_taken_from_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROJECTS", str(tmp_path / "env"))
    assert Project("rel").directory == (tmp_path / "rel").resolve()
    assert Project("$PROJECTS/p").directory == (tmp_path / "env" / "p").resolve()


def test_views_are_cached(tmp_path):
    project = _project(tmp_path)

    assert project.config is project.config
    assert project.receptors is project.receptors
    assert project._simulations is project._simulations
    assert project.variants is project.variants
    assert project.plot is project.plot


# ---------------------------------------------------------------------------
# Output directory and scratch
# ---------------------------------------------------------------------------


def test_output_defaults_to_the_projects_output_directory(tmp_path):
    project = _project(tmp_path)

    assert isinstance(project.output, Output)
    assert project.output.directory == tmp_path / "proj" / "output"
    assert not project.output.directory.exists()  # nothing is made by looking


def test_output_is_relative_to_the_project_unless_absolute(tmp_path):
    relative = _project(tmp_path, name="a", output="../results")
    absolute = _project(tmp_path, name="b", output=str(tmp_path / "abs"))

    assert relative.output.directory == (tmp_path / "results").resolve()
    assert absolute.output.directory == tmp_path / "abs"


def test_output_can_be_shared_between_projects(tmp_path, point_receptor):
    shared = str(tmp_path / "shared")
    a = _project(tmp_path, [point_receptor], name="a", output=shared)
    b = _project(tmp_path, [point_receptor], name="b", output=shared)

    _write_trajectory(a, point_receptor)

    rid = point_receptor.id
    assert a.output.directory == b.output.directory
    assert b.simulation(rid, "hrrr").has_particles
    assert (
        a.simulation(rid, "hrrr").particles_path
        == b.simulation(rid, "hrrr").particles_path
    )


def test_compute_root_defaults_under_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.delenv("PYSTILT_COMPUTE_ROOT", raising=False)

    expected = (tmp_path / "tmp" / "pystilt" / "proj").resolve()
    assert resolve_compute_root(_project(tmp_path)) == expected


def test_default_compute_root_is_resolved_like_an_explicit_one(tmp_path, monkeypatch):
    """A TMPDIR behind a symlink (as on macOS) gives the path a pool worker gets."""
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "link"))
    monkeypatch.delenv("PYSTILT_COMPUTE_ROOT", raising=False)
    project = _project(tmp_path)

    default = resolve_compute_root(project)
    assert default == real.resolve() / "pystilt" / "proj"
    assert resolve_compute_root(project, str(default)) == default


def test_compute_root_from_the_environment_and_explicit_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSTILT_COMPUTE_ROOT", str(tmp_path / "scratch"))
    project = _project(tmp_path)

    assert resolve_compute_root(project) == (tmp_path / "scratch").resolve()
    explicit = resolve_compute_root(project, tmp_path / "explicit")
    assert explicit == (tmp_path / "explicit").resolve()


def test_an_empty_compute_root_variable_is_unset(tmp_path, monkeypatch):
    """PYSTILT_COMPUTE_ROOT= means the default, not the current directory."""
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("PYSTILT_COMPUTE_ROOT", "")

    expected = (tmp_path / "tmp" / "pystilt" / "proj").resolve()
    assert resolve_compute_root(_project(tmp_path)) == expected


# ---------------------------------------------------------------------------
# Receptors
# ---------------------------------------------------------------------------


def test_receptors_are_a_table_with_the_files_labels(tmp_path):
    a, b = _receptor(12, site="WBB"), _receptor(13, longitude=-112.0, site="UOU")
    project = _project(tmp_path, [a, b])

    frame = project.receptors

    assert list(frame.columns) == ["receptor", "time", "kind", "location", "site"]
    assert frame["receptor"].tolist() == [a.id, b.id]
    assert frame["kind"].tolist() == ["point", "point"]
    assert frame["location"].tolist() == [a.location_id, b.location_id]
    assert str(frame["time"].dtype).startswith("datetime64")
    assert frame["site"].tolist() == ["WBB", "UOU"]
    assert project.receptor(b.id) == b
    with pytest.raises(KeyError, match="No receptor"):
        project.receptor("202301011200_0_0_0")


def test_a_project_without_receptors_is_empty(tmp_path):
    project = _project(tmp_path)

    assert project.receptors.empty
    assert list(project.receptors.columns) == ["receptor", "time", "kind", "location"]
    assert len(project.simulations) == 0
    assert project.status().empty
    assert project.run().empty


def test_add_receptors_appends_only_new_ones_and_refreshes_the_views(
    tmp_path, point_receptor, column_receptor, multipoint_receptor
):
    project = _project(tmp_path, [point_receptor])
    assert len(project.simulations) == 1

    added = project.add_receptors(
        [point_receptor, column_receptor, multipoint_receptor]
    )

    assert added == [column_receptor.id, multipoint_receptor.id]
    assert project.receptors["kind"].tolist() == ["point", "column", "multipoint"]
    assert len(project.simulations) == 3
    assert project.receptor(multipoint_receptor.id) == multipoint_receptor
    reopened = Project(project.directory).receptors["receptor"].tolist()
    assert reopened == project.receptors["receptor"].tolist()
    assert project.add_receptors(point_receptor) == []


def test_add_receptors_appends_in_the_files_own_columns(
    tmp_path, point_receptor, column_receptor
):
    """A hand-written file keeps its columns, its r_idx values, and extra columns."""
    project = _project(tmp_path)
    text = (
        "r_idx,time,lati,long,zagl,scene\n"
        "1155,2023-01-01 12:00:00,40.77,-111.85,5.0,A\n"
    )
    project.receptors_path.write_text(text)

    added = project.add_receptors([point_receptor, column_receptor])

    assert added == [column_receptor.id]
    stored = project.receptors_path.read_text()
    assert stored.startswith(text)  # the original bytes are untouched
    lines = stored.splitlines()
    assert lines[2].split(",")[0] == "1156" and lines[2].endswith(",")


def test_add_receptors_refuses_a_group_without_r_idx(tmp_path, column_receptor):
    project = _project(tmp_path)
    project.receptors_path.write_text("time,lati,long,zagl\n")
    with pytest.raises(ValueError, match="r_idx"):
        project.add_receptors([column_receptor])


def test_receptors_that_would_share_result_files_are_refused(tmp_path):
    """Two receptors with one id would overwrite each other's results."""
    from stilt.receptors import MultiPointReceptor

    def slant(alt):
        return MultiPointReceptor(
            time="2023-01-01 12:00",
            longitudes=[-111.85, -111.86],
            latitudes=[40.77, 40.78],
            altitudes=[alt, 500.0],
        )

    a, b = slant(10.001), slant(10.004)
    assert a.id == b.id and a != b
    with pytest.raises(ValueError, match="share the id"):
        _project(tmp_path, [a, b], name="both")

    project = _project(tmp_path, [a])
    with pytest.raises(ValueError, match="share the id"):
        project.add_receptors([b])
    assert project.add_receptors([slant(10.001)]) == []  # the same receptor again


def test_add_receptors_takes_receptors_or_a_csv(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(TypeError, match="Receptor"):
        project.add_receptors([1, 2])


# ---------------------------------------------------------------------------
# Variants and simulations
# ---------------------------------------------------------------------------


def test_variants_default_to_one_per_met(tmp_path):
    project = _project(tmp_path)

    assert list(project.variants) == ["hrrr"]
    assert project.variants["hrrr"].met == "hrrr"


def test_simulations_are_receptors_times_variants(tmp_path):
    a, b = _receptor(12, site="WBB"), _receptor(13, site="UOU")
    project = _project(
        tmp_path,
        [a, b],
        krand=4,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )

    sims = project.simulations

    assert list(sims.columns[:4]) == ["receptor", "variant", "realization", "model"]
    assert set(sims["model"]) == {"hysplit"}
    assert _pairs(sims) == [
        (a.id, "hrrr"),
        (a.id, "err"),
        (a.id, "err"),
        (b.id, "hrrr"),
        (b.id, "err"),
        (b.id, "err"),
    ]
    assert sims["realization"].tolist()[:3] == [pd.NA, 0, 1]
    assert set(sims.columns) >= {"time", "kind", "location", "site"}


def test_simulations_select_with_pandas(tmp_path):
    a = _receptor(12, site="WBB")
    b = _receptor(13, site="UOU")
    c = _receptor(14, site="WBB")
    project = _project(
        tmp_path, [a, b, c], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    sims = project.simulations

    wbb = sims[(sims.site == "WBB") & (sims.variant == "zi08")]
    assert _pairs(wbb) == [(a.id, "zi08"), (c.id, "zi08")]
    later = sims[sims.time.between("2023-01-01 13:00", "2023-01-01 14:00")]
    assert sorted(set(later["receptor"])) == sorted([b.id, c.id])
    assert len(project.status(wbb)) == 2


def test_simulation_handles_carry_the_variant_settings(tmp_path, point_receptor):
    project = _project(
        tmp_path,
        [point_receptor],
        krand=4,
        variants={
            "hrrr": {},
            "err": {**_XYERR, "realizations": 2, "grid": None},
            "zi08": {"ziscale": 0.8},
            "s2": {"smooth_factor": 2},
        },
    )
    rid = point_receptor.id

    base = project.simulation(rid, "hrrr")
    err = project.simulation(rid, "err", 1)
    zi = project.simulation(rid, "zi08")
    s2 = project.simulation(rid, "s2")

    assert winderrtf(base.variant.transport) == 0
    assert winderrtf(err.transport) == 1
    assert err.transport.seed == err.variant.transport.seed  # krand 4: no seed
    with pytest.raises(ValueError, match="realizations"):
        project.simulation(rid, "err")
    with pytest.raises(ValueError, match="realizations"):
        project.simulation(rid, "hrrr", 0)
    assert err.variant.footprint is None
    assert zi.variant.transport.ziscale == 0.8
    assert zi.variant.footprint == base.variant.footprint
    assert s2.variant.footprint is not None and s2.variant.footprint.smooth_factor == 2
    assert s2.variant.transport == base.variant.transport  # shares hrrr's particles
    assert project.simulation(rid, "hrrr") == base  # a value, not a handle
    with pytest.raises(KeyError, match="No variant"):
        project.simulation(rid, "nope")


def test_variants_with_equal_transport_settings_share_a_run(tmp_path, point_receptor):
    project = _project(
        tmp_path,
        [point_receptor],
        variants={"hrrr": {}, "s2": {"smooth_factor": 2}, "zi08": {"ziscale": 0.8}},
    )

    hrrr, s2, zi = (project.variants[v] for v in ("hrrr", "s2", "zi08"))
    assert hrrr.particles_hash == s2.particles_hash != zi.particles_hash

    _write_trajectory(project, point_receptor)
    assert project.simulation(point_receptor.id, "s2").has_particles
    assert not project.simulation(point_receptor.id, "zi08").has_particles
    assert len(project.output.hashes("particles")) == 1


def test_folders_say_which_the_config_no_longer_uses(tmp_path, point_receptor):
    project = _project(
        tmp_path, [point_receptor], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    for variant in ("hrrr", "zi08"):
        _write_trajectory(project, point_receptor, variant)
        _write_footprint(project, point_receptor, variant)
    assert (project.folders()["variant"] != "").all()

    # The user drops zi08 and changes hrrr's smoothing in config.yaml.
    _config(tmp_path, variants={"hrrr": {"smooth_factor": 0.5}}).to_yaml(
        project.config_path
    )
    folders = Project(project.directory).folders()
    stale = folders[folders["variant"] == ""]
    by_kind = {
        k: sorted(stale.loc[stale.kind == k, "name"])
        for k in ("particles", "footprints")
    }
    assert by_kind == {"particles": ["zi08"], "footprints": ["hrrr", "zi08"]}
    # Nothing was deleted.
    assert project.simulation(point_receptor.id, "zi08").is_complete


# ---------------------------------------------------------------------------
# Completion and status
# ---------------------------------------------------------------------------


def test_incomplete_follows_each_variant_outputs(tmp_path, point_receptor):
    project = _project(
        tmp_path,
        [point_receptor],
        variants={"hrrr": {}, "traj": {"grid": None}, "s2": {"smooth_factor": 2}},
    )

    assert project.incomplete()["variant"].tolist() == [
        "hrrr",
        "traj",
        "s2",
    ]

    _write_trajectory(project, point_receptor)  # shared by hrrr, traj, and s2
    assert project.incomplete()["variant"].tolist() == ["hrrr", "s2"]

    _write_footprint(project, point_receptor)
    _write_footprint(project, point_receptor, "s2", empty=True)
    assert project.incomplete().empty


def test_status_marks_outputs_a_variant_does_not_produce(tmp_path, point_receptor):
    project = _project(
        tmp_path, [point_receptor], variants={"hrrr": {}, "traj": {"grid": None}}
    )
    _write_trajectory(project, point_receptor)

    status = project.status()

    assert list(status.columns[-6:]) == [
        "particles",
        "footprint",
        "state",
        "step",
        "reason",
        "message",
    ]
    assert status["reason"].isna().all()
    by_variant = status.set_index("variant")
    assert by_variant.loc["hrrr", "particles"] == True  # noqa: E712
    assert by_variant.loc["hrrr", "footprint"] == False  # noqa: E712
    assert pd.isna(by_variant.loc["traj", "footprint"])
    assert by_variant.loc["traj", "state"] == "complete"
    assert by_variant.loc["hrrr", "state"] == "pending"


def test_status_tells_an_interrupted_run_from_one_never_run(tmp_path):
    """A log with no particles and no failure record is a run that was cut off."""
    a, b, c = _receptor(12), _receptor(13), _receptor(14)
    project = _project(tmp_path, [a, b, c])
    variant = project.variants["hrrr"]
    project.output.write_log(variant, a.id, "started\n")
    project.output.write_log(variant, b.id, "HYSPLIT said no\n")
    project.output.record_failure("particles", variant, b.id, {"reason": "X"})

    status = project.status()

    assert status["state"].tolist() == ["interrupted", "failed", "pending"]
    assert project.incomplete()["receptor"].tolist() == [a.id, b.id, c.id]


def test_status_counts_an_empty_footprint_as_complete(tmp_path):
    a, b = _receptor(12), _receptor(13)
    project = _project(tmp_path, [a, b])
    assert project.status()["state"].tolist() == ["pending", "pending"]

    _write_trajectory(project, a)
    _write_footprint(project, a)
    _write_trajectory(project, b)
    _write_footprint(project, b, empty=True)
    status = project.status()
    assert status["state"].tolist() == ["complete", "complete"]
    empty = project.simulation(b.id, "hrrr")
    assert empty.has_footprint and empty.footprint is None


def test_status_opens_no_result_file_and_builds_no_receptor(tmp_path, monkeypatch):
    """#141: on a large project, opening files or building receptors per row took minutes."""
    import pyarrow.parquet as pq

    a, b = _receptor(12), _receptor(13)
    project = _project(tmp_path, [a, b])
    _write_trajectory(project, a)
    _write_footprint(project, a)
    _write_footprint(project, b, empty=True)
    sims = project.simulations

    def refuse(*args, **kwargs):
        raise AssertionError("status() opened a result file")

    for name in ("read_schema", "read_metadata", "read_table", "ParquetFile"):
        monkeypatch.setattr(pq, name, refuse)
    monkeypatch.setattr(Project, "_receptors", None)

    assert project.status(sims)["state"].tolist() == ["complete", "pending"]


def _mixed_state_project(tmp_path):
    """Six receptors under three variants, in every state a simulation can be in."""
    receptors = [_receptor(h) for h in range(10, 16)]
    project = _project(
        tmp_path,
        receptors,
        variants={
            "hrrr": {},
            "smooth": {"smooth_factor": 2},  # shares hrrr's particles
            "zi08": {"ziscale": 0.8, "grid": None},  # particles only
        },
    )
    a, b, c, d, e, _ = receptors
    _write_trajectory(project, a)  # particles, no footprint
    _write_trajectory(project, b)
    _write_footprint(project, b)  # complete for hrrr, not for smooth
    _write_trajectory(project, c)
    _write_footprint(project, c, empty=True)  # an empty footprint counts
    _write_footprint(project, c, "smooth")
    _write_trajectory(project, d, "zi08")  # complete: zi08 makes no footprint
    _write_trajectory(project, e)
    _write_footprint(project, e)
    _write_footprint(project, e, "smooth")
    _write_trajectory(project, e, "zi08")  # complete under every variant
    return project


def test_status_and_incomplete_in_every_state(tmp_path):
    """Each simulation's results and state, read from the folder listings."""
    project = _mixed_state_project(tmp_path)
    status = project.status()
    by_pair = {
        (f"{t.hour}", v): (state, particles, footprint)
        for t, v, state, particles, footprint in zip(
            status.time,
            status.variant,
            status.state,
            status.particles,
            status.footprint,
            strict=True,
        )
    }
    na = pd.NA
    # By receptor hour: 10 is a, 11 b, ..., 15 f.
    expected = {
        "10": [
            ("pending", True, False),
            ("pending", True, False),
            ("pending", False, na),
        ],
        "11": [
            ("complete", True, True),
            ("pending", True, False),
            ("pending", False, na),
        ],
        "12": [
            ("complete", True, True),
            ("complete", True, True),
            ("pending", False, na),
        ],
        "13": [
            ("pending", False, False),
            ("pending", False, False),
            ("complete", True, na),
        ],
        "14": [
            ("complete", True, True),
            ("complete", True, True),
            ("complete", True, na),
        ],
        "15": [
            ("pending", False, False),
            ("pending", False, False),
            ("pending", False, na),
        ],
    }
    for hour, rows in expected.items():
        for variant, want in zip(("hrrr", "smooth", "zi08"), rows, strict=True):
            state, particles, footprint = by_pair[(hour, variant)]
            got = (state, particles, footprint if footprint is not na else na)
            assert got[:2] == want[:2], (hour, variant)
            assert (got[2] is na) == (want[2] is na) and (
                got[2] is na or got[2] == want[2]
            ), (hour, variant)

    incomplete = project.incomplete()
    assert len(incomplete) == int((status.state != "complete").sum())
    smooth = project.simulations[project.simulations.variant == "smooth"]
    assert [t.hour for t in project.incomplete(smooth)["time"]] == [10, 11, 13, 15]


def test_incomplete_of_a_project_with_no_results_is_everything(tmp_path):
    project = _project(tmp_path, [_receptor(12)])
    assert _pairs(project.incomplete()) == _pairs(project.simulations)
    assert not project.output.directory.exists()  # looking creates nothing


# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------


def test_particles_of_a_selection_is_one_table(tmp_path):
    a, b = _receptor(12), _receptor(13)
    project = _project(tmp_path, [a, b], variants={"hrrr": {}, "fine": {"grid": None}})
    assert project.particles().empty

    path = _write_trajectory(project, a)

    folder = project.output.folder("particles", project.variants["hrrr"])
    assert path.parent == folder / "date=2023-01-01"
    particles = project.particles()
    assert list(particles.columns[:2]) == ["receptor", "variant"]
    one = project.simulation(a.id, "hrrr").particles
    # Both variants share the run, so each gets its own copy of its rows.
    assert particles["variant"].value_counts().to_dict() == {
        "hrrr": len(one),
        "fine": len(one),
    }
    assert set(particles["receptor"]) == {a.id}
    hrrr = particles[particles.variant == "hrrr"].drop(columns=["receptor", "variant"])
    pd.testing.assert_frame_equal(
        hrrr.reset_index(drop=True)[one.columns], one, check_dtype=False
    )
    sims = project.simulations
    assert project.particles(sims[sims.receptor == b.id]).empty


def test_footprints_open_one_variant_as_a_dataset(tmp_path, monkeypatch):
    done, empty, missing = _receptor(12), _receptor(13), _receptor(14)
    project = _project(
        tmp_path,
        [done, empty, missing],
        variants={"hrrr": {}, "traj": {"grid": None}},
    )
    _write_footprint(project, done)
    _write_footprint(project, empty, empty=True)
    _write_trajectory(project, done, "traj")

    foot = project.simulation(done.id, "hrrr").footprint
    # Footprints are read from the folder listing; no receptor is built.
    sims = project.simulations
    monkeypatch.setattr(Project, "_receptors", None)
    ds = project.footprints(sims[sims.variant == "hrrr"])

    assert isinstance(ds, xr.Dataset)
    assert list(ds.receptor.values) == [done.id]
    assert ds.attrs["empty"] == [empty.id]
    assert ds.attrs["missing"] == [missing.id]
    np.testing.assert_allclose(
        ds.foot.sel(receptor=done.id, hour=0).values,
        foot.isel(time=0).values,
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="no grid"):
        project.footprints(sims[sims.variant == "traj"])
    with pytest.raises(ValueError, match="has a footprint yet"):
        project.footprints(
            sims[(sims.receptor == missing.id) & (sims.variant == "hrrr")]
        )


def test_footprints_take_one_variant(tmp_path, point_receptor):
    project = _project(
        tmp_path, [point_receptor], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    for variant in ("hrrr", "zi08"):
        _write_footprint(project, point_receptor, variant)
    sims = project.simulations

    with pytest.raises(ValueError, match="one variant"):
        project.footprints()
    ds = project.footprints(sims[sims.variant == "zi08"])
    assert ds.attrs["stilt_name"] == "zi08"
    assert list(ds.receptor.values) == [point_receptor.id]


def test_jacobian_of_a_variant(tmp_path):
    a, b, c = _receptor(12), _receptor(13), _receptor(14)
    project = _project(
        tmp_path,
        [a, b, c],
        variants={"hrrr": {}, "zi08": {"ziscale": 0.8}, "traj": {"grid": None}},
    )
    _write_footprint(project, a)
    _write_footprint(project, b, empty=True)
    target = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.5, yres=0.5)
    bins = pd.IntervalIndex.from_breaks(
        pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="12h"),
        closed="left",
    )

    sims = project.simulations
    hrrr = sims[sims.variant == "hrrr"]
    H = project.jacobian(hrrr, target, bins)

    assert list(H.receptors) == [a.id]
    assert H.empty == [b.id]
    assert H.missing == [c.id]
    assert H.data.shape == (1, len(bins) * len(target.index))
    expected = project.simulation(a.id, "hrrr").footprint.stilt.aggregate(target, bins)
    np.testing.assert_allclose(
        H.to_frame().iloc[0].to_numpy().reshape(len(bins), -1).T,
        expected.to_numpy(),
        rtol=1e-6,
    )
    some = project.jacobian(hrrr[hrrr.receptor.isin([b.id, a.id])], target, bins)
    assert list(some.receptors) == [a.id]
    right_closed = pd.IntervalIndex.from_breaks(
        bins.left.append(bins.right[-1:]), closed="right"
    )
    with pytest.raises(ValueError, match="closed on the left"):
        project.jacobian(hrrr, target, right_closed)
    with pytest.raises(ValueError, match="no footprints yet"):
        project.jacobian(sims[sims.variant == "zi08"], target, bins)
    with pytest.raises(ValueError, match="no grid"):
        project.jacobian(sims[sims.variant == "traj"], target, bins)
    with pytest.raises(ValueError, match="one variant"):
        project.jacobian(sims, target, bins)


def test_jacobian_in_batches_and_threads_is_the_same_matrix(tmp_path):
    receptors = [_receptor(h) for h in range(8, 16)]
    project = _project(tmp_path, receptors, execution={"cpus": 3})
    for r in receptors[:-1]:
        _write_footprint(project, r)
    _write_footprint(project, receptors[-1], empty=True)
    target = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.5, yres=0.5)
    bins = pd.IntervalIndex.from_breaks(
        pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="3h"),
        closed="left",
    )
    whole = project.jacobian(project.simulations, target, bins, workers=1)
    batched = project.jacobian(project.simulations, target, bins, workers=3, batch=2)

    assert (
        list(batched.receptors)
        == list(whole.receptors)
        == [r.id for r in receptors[:-1]]
    )
    assert batched.empty == whole.empty == [receptors[-1].id]
    assert (batched.data != whole.data).nnz == 0
    expected = [
        project.simulation(r.id, "hrrr").footprint.stilt.aggregate(target, bins)
        for r in receptors[:-1]
    ]
    np.testing.assert_allclose(
        batched.data.toarray(),
        np.stack([e.to_numpy().T.ravel() for e in expected]),
        rtol=1e-6,
    )


def test_jacobian_lists_each_date_folder_once(tmp_path, monkeypatch):
    import stilt.output as output_module

    a = _receptor(12)
    project = _project(tmp_path, [a])
    _write_footprint(project, a)
    target = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.5, yres=0.5)
    bins = pd.IntervalIndex.from_breaks(
        pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="12h"),
        closed="left",
    )
    listed: list[str] = []
    real_scandir = output_module.os.scandir

    def scandir(path):
        listed.append(Path(path).name)
        return real_scandir(path)

    monkeypatch.setattr(output_module.os, "scandir", scandir)
    H = project.jacobian(project.simulations, target, bins)

    assert list(H.receptors) == [a.id]
    assert listed.count("date=2023-01-01") == 1


def test_plot_availability_returns_axes(tmp_path, point_receptor):
    import matplotlib.pyplot as plt

    project = _project(tmp_path, [point_receptor])

    ax = project.plot.availability()

    assert ax is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# run() and submit()
# ---------------------------------------------------------------------------


@pytest.fixture
def ran(monkeypatch):
    """Record what a local run hands to the workers, and run nothing."""
    calls: list[dict] = []

    def run_receptors(project, receptor_ids, **kwargs):
        calls.append({"project": project, "ids": list(receptor_ids), **kwargs})

    monkeypatch.setattr("stilt.execution.worker.run_receptors", run_receptors)
    return calls


def test_run_hands_the_incomplete_receptors_to_the_workers(tmp_path, ran):
    done, todo = _receptor(12), _receptor(13)
    project = _project(
        tmp_path, [done, todo], include_footprint=False, execution={"cpus": 3}
    )
    _write_trajectory(project, done)

    status = project.run()

    assert list(status["receptor"]) == [todo.id]  # what ran, and where it stands
    assert list(status["state"]) == ["pending"]  # the fake ran nothing
    [call] = ran
    assert call["ids"] == [todo.id]
    assert call["project"] is project
    assert call["execution"] == project.config.execution
    assert call["execution"].cpus == 3
    # The runner works out the scratch directory before handing it over.
    assert call["compute_root"] == resolve_compute_root(project)
    assert call["skip_existing"] is True


def test_run_without_skip_runs_every_receptor_once(tmp_path, ran):
    done, todo = _receptor(12), _receptor(13)
    project = _project(
        tmp_path, [done, todo], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    _write_trajectory(project, done)

    project.run(skip_existing=False, compute_root=tmp_path / "scratch")

    [call] = ran
    assert call["ids"] == [
        done.id,
        todo.id,
    ]  # each receptor once, whatever its variants
    assert call["skip_existing"] is False
    assert call["compute_root"] == tmp_path / "scratch"


def test_run_takes_execution_settings_in_place_of_the_configs(tmp_path, ran):
    project = _project(tmp_path, [_receptor(12)], execution={"cpus": 3})

    override = ExecutionConfig(cpus=8, timeout=600, keep_scratch=True)
    project.run(execution=override)

    # The whole override reaches the workers, not only its cpus.
    assert ran[0]["execution"] == override


def test_run_with_nothing_to_do_starts_nothing(tmp_path, ran, point_receptor):
    project = _project(tmp_path, [point_receptor])
    _write_trajectory(project, point_receptor)
    _write_footprint(project, point_receptor, empty=True)

    assert project.run().empty
    assert ran == []


def test_run_after_adding_a_variant_runs_the_receptor_again(
    tmp_path, ran, point_receptor
):
    """A finished project grows a variant: only that variant is incomplete."""
    project = _project(tmp_path, [point_receptor])
    _write_trajectory(project, point_receptor)
    _write_footprint(project, point_receptor)
    assert project.run().empty

    # The user adds a variant to config.yaml.
    _config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}).to_yaml(
        project.config_path
    )
    grown = Project(project.directory)
    assert _pairs(grown.incomplete()) == [(point_receptor.id, "zi08")]

    grown.run()

    assert ran[0]["ids"] == [point_receptor.id]


def test_run_finds_a_missing_realization(tmp_path, ran, point_receptor):
    project = _project(
        tmp_path,
        [point_receptor],
        grid=None,
        krand=4,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )
    _write_trajectory(project, point_receptor)
    _write_trajectory(project, point_receptor, "err", 0)

    project.run()

    assert ran[0]["ids"] == [point_receptor.id]
    missing = project.incomplete()
    assert _pairs(missing) == [(point_receptor.id, "err")]
    assert missing["realization"].tolist() == [1]


def test_submit_needs_slurm(tmp_path, point_receptor):
    project = _project(tmp_path, [point_receptor])
    with pytest.raises(ValueError, match="slurm"):
        project.submit()


def test_run_on_slurm_submits_and_waits(tmp_path, monkeypatch, point_receptor):
    project = _project(
        tmp_path, [point_receptor], execution={"backend": "slurm", "time": "01:00:00"}
    )
    waited = []
    submitted = []

    def submit(project, pending, execution, skip_existing, compute_root):
        submitted.append(
            {"pending": pending, "execution": execution, "compute_root": compute_root}
        )
        return "777", "kingspeak"

    def wait(job_id, *, cluster=None):
        waited.append((job_id, cluster))

    monkeypatch.setattr("stilt.execution.runner._submit", submit)
    monkeypatch.setattr("stilt.execution.runner.wait", wait)

    status = project.run(compute_root="/s")
    assert waited == [("777", "kingspeak")]
    assert submitted[0]["pending"] == [point_receptor.id]
    assert submitted[0]["compute_root"] == "/s"
    assert submitted[0]["execution"].backend == "slurm"
    # The status of what was submitted; nothing ran here.
    assert list(status["receptor"]) == [point_receptor.id]
    assert list(status["state"]) == ["pending"]


def test_receptors_are_built_only_when_asked_for(tmp_path, monkeypatch):
    import stilt.project

    a, b = _receptor(12, site="WBB"), _receptor(13, site="UOU")
    _project(tmp_path, [a, b])
    project = Project(tmp_path / "proj")
    built = []
    build = stilt.project.receptors_from_rows

    def counting(rows):
        receptors = build(rows)
        built.extend(r.id for r in receptors)
        return receptors

    monkeypatch.setattr(stilt.project, "receptors_from_rows", counting)

    assert len(project.receptors) == 2
    project.status()
    assert built == []  # listing, selecting, and status build nothing

    assert project.receptor(b.id) == b
    assert built == [b.id]


def test_a_bad_receptors_file_fails_when_the_project_is_read(tmp_path):
    project = _project(tmp_path)
    project.receptors_path.write_text(
        "time,longitude,latitude,altitude\n2023-01-01 12:00:00,-111.85,95.0,5\n"
    )

    with pytest.raises(ValueError, match="latitude must be within"):
        _ = Project(project.directory).receptors


# ---------------------------------------------------------------------------
# A selection of simulations (#107)
# ---------------------------------------------------------------------------


def test_simulations_is_a_dataframe_and_a_copy(tmp_path):
    project = _project(tmp_path, [_receptor(12)])
    sims = project.simulations
    assert isinstance(sims, pd.DataFrame)
    sims["mine"] = 1  # the user's copy
    assert "mine" not in project.simulations.columns


def test_a_selection_may_be_a_frame_a_mask_or_another_table(tmp_path):
    import pyarrow as pa

    a, b = _receptor(12, site="WBB"), _receptor(13, site="UOU")
    project = _project(
        tmp_path, [a, b], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    _write_trajectory(project, a, "zi08")
    sims = project.simulations
    expected = [(a.id, "zi08")]

    wbb = sims[(sims.site == "WBB") & (sims.variant == "zi08")]
    mask = (sims.site == "WBB") & (sims.variant == "zi08")
    table = pa.table({"receptor": [a.id], "variant": ["zi08"]})
    for sel in (wbb, mask, mask.to_numpy(), table):
        st = project.status(sel)
        assert _pairs(st) == expected
        assert st["particles"].tolist() == [True]
        assert st["site"].tolist() == ["WBB"]  # the rest of the row, from the project
        assert _pairs(project.incomplete(sel)) == expected  # no footprint yet
    polars = pytest.importorskip("polars")
    st = project.status(polars.DataFrame({"receptor": [a.id], "variant": ["zi08"]}))
    assert _pairs(st) == expected


def test_a_pandas_selection_keeps_its_own_columns(tmp_path):
    a, b = _receptor(12), _receptor(13)
    project = _project(tmp_path, [a, b])
    mine = pd.DataFrame({"receptor": [b.id], "obs": [1.9]})
    merged = project.simulations.merge(mine, on="receptor")

    st = project.status(merged)
    assert _pairs(st) == [(b.id, "hrrr")]
    assert st.obs.tolist() == [1.9]


def test_a_selection_needs_receptor_and_variant_columns(tmp_path):
    project = _project(tmp_path, [_receptor(12)])
    with pytest.raises(ValueError, match="'receptor' and 'variant'"):
        project.status(project.receptors)
    unknown = pd.DataFrame({"receptor": [_receptor(12).id], "variant": ["nope"]})
    with pytest.raises(KeyError, match="nope"):
        project.status(unknown)
    with pytest.raises(KeyError, match="No receptor"):
        project.receptor("202301011200_-1_1_1")


def test_relative_paths_start_from_the_project_not_the_working_directory(
    tmp_path, monkeypatch, point_receptor
):
    monkeypatch.setenv("MET_ROOT", str(tmp_path / "archive"))
    (tmp_path / "real_output").mkdir()
    (tmp_path / "output_link").symlink_to(tmp_path / "real_output")
    hrrr = {"file_format": "%Y%m%d_%H", "file_tres": "1h"}
    Project.init(
        tmp_path / "proj",
        receptors=[point_receptor],
        mets={
            "local": {**hrrr, "directory": "met"},
            "shared": {**hrrr, "directory": "$MET_ROOT/hrrr"},
        },
        output=str(tmp_path / "output_link"),
        variants={"local": {}, "shared": {}},
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    project = Project(tmp_path / "proj")
    variants = project.variants
    assert variants["local"].met_config.directory == project.directory / "met"
    assert variants["shared"].met_config.directory == (
        (tmp_path / "archive/hrrr").resolve()
    )
    # Reached through a link, the output is the directory it points to.
    assert project.output == Output((tmp_path / "real_output").resolve())
    # A simulation finds relative file names in its settings from the project.
    assert project.simulation(point_receptor.id, "local").directory == project.directory


def test_init_refuses_a_variant_with_bad_settings_and_writes_nothing(tmp_path):
    """A variant's transport settings are checked before config.yaml is written."""
    met = make_met_config(str(tmp_path / "met"))
    with pytest.raises(ValueError, match="numpr"):
        Project.init(
            tmp_path / "proj", mets={"hrrr": met}, variants={"hrrr": {"numpr": 10}}
        )
    assert not (tmp_path / "proj" / "config.yaml").exists()


def test_status_says_why_the_failed_simulations_failed(tmp_path, point_receptor):
    other = PointReceptor(
        time="2023-07-15 19:00", longitude=-111.848, latitude=40.766, altitude=10
    )
    project = _project(
        tmp_path, [point_receptor, other], variants={"traj": {"grid": None}}
    )
    sim = project.simulation(str(point_receptor.id), "traj")
    sim.output.record_failure(
        "particles",
        sim.variant,
        sim.receptor.id,
        {"step": "particles", "reason": "MET_COVERAGE", "message": "m"},
    )

    status = project.status()
    failed = status[status.state == "failed"]

    assert failed[["receptor", "variant", "step", "reason"]].to_dict("records") == [
        {
            "receptor": str(point_receptor.id),
            "variant": "traj",
            "step": "particles",
            "reason": "MET_COVERAGE",
        }
    ]
    by_receptor = status.set_index("receptor")
    assert by_receptor.loc[str(other.id), "state"] == "pending"
    assert pd.isna(by_receptor.loc[str(other.id), "reason"])


# ---------------------------------------------------------------------------
# Input tables
# ---------------------------------------------------------------------------


def test_add_table_keeps_a_receptor_table_and_adds_only_new_receptors(tmp_path):
    from stilt.transforms import averaging_kernel_table

    a, b = _receptor(12), _receptor(13)
    project = _project(tmp_path, [a, b])
    first = averaging_kernel_table([a], levels=[0.0, 3000.0], values=[[1.0, 1.0]])
    both = averaging_kernel_table(
        [a, b], levels=[0.0, 3000.0], values=[[0.2, 0.2], [0.5, 0.5]]
    )

    path = project.add_table("kernels", first)
    assert path == project.directory / "tables" / "kernels.parquet"
    project.add_table("kernels", both)  # a's rows are kept as they were
    project.add_table("kernels", both)  # and adding them again changes nothing

    held = pd.read_parquet(path)
    assert list(held.receptor) == [a.id, a.id, b.id, b.id]
    assert list(held.value) == [1.0, 1.0, 0.5, 0.5]
    with pytest.raises(ValueError, match="columns"):
        project.add_table("kernels", pd.DataFrame({"receptor": [b.id]}))
    with pytest.raises(ValueError, match="table name"):
        project.add_table("../kernels", first)
