"""Tests for stilt.project: the project directory, its tables, its results, and running it."""

import datetime as dt
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.config import ExecutionConfig, Grid, MetConfig, ProjectConfig
from stilt.execution import resolve_compute_root
from stilt.footprint.io import _describe
from stilt.output import Output
from stilt.project import Project, Simulations
from stilt.receptors import PointReceptor
from stilt.simulation import SimID
from stilt.transport.hysplit.driver import winderrtf
from stilt.transport.hysplit.model import finish_particles

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
    return MetConfig(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h"
    )


def _config(tmp_path, include_footprint=True, **overrides) -> ProjectConfig:
    """Minimal ProjectConfig with one met and (optionally) a footprint grid."""
    overrides.setdefault("grid", _GRID if include_footprint else None)
    return ProjectConfig(mets={"hrrr": _met(tmp_path)}, **overrides)


def _project(tmp_path, receptors=None, name="proj", **overrides) -> Project:
    """Make a project in ``tmp_path / name`` with the minimal config."""
    return Project.init(
        tmp_path / name, config=_config(tmp_path, **overrides), receptors=receptors
    )


def _receptor(hour: int, longitude: float = -111.85, **attrs) -> PointReceptor:
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, hour),
        longitude=longitude,
        latitude=40.77,
        altitude=5.0,
        attrs=attrs,
    )


def _particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-60.0, -120.0],
            "indx": [1.0, 1.0],
            "long": [-113.9, -113.5],
            "lati": [39.7, 39.6],
            "zagl": [10.0, 20.0],
            "foot": [1e-5, 2e-5],
            "dens": [1.2, 1.2],
            "samt": [1.0, 1.0],
            "sigw": [0.1, 0.1],
            "tlgr": [10.0, 10.0],
            "mlht": [500.0, 500.0],
        }
    )


def _write_trajectory(project: Project, receptor, variant="hrrr") -> Path:
    """Write a small particle file for one simulation into the output directory."""
    sim = project.simulation(receptor.id, variant)
    particles = finish_particles(_particles(), sim.receptor, sim.variant.transport)
    folder = sim.output.particles(sim.variant)
    return folder.write(sim.receptor, particles, [])


def _write_footprint(
    project: Project, receptor, variant="hrrr", *, empty=False
) -> Path:
    """Write a footprint (or an empty one) for one simulation into the output directory."""
    sim = project.simulation(receptor.id, variant)
    assert sim.variant.footprint is not None
    folder = sim.output.particles(sim.variant)
    feet = folder.footprints(sim.variant.footprint, name=sim.variant.name)
    if empty:
        return feet.write_empty(sim.receptor, "outside_domain", name=sim.variant.name)
    grid = sim.variant.footprint.grid
    assert grid is not None
    x_axis, y_axis = grid.axes
    data = xr.DataArray(
        np.random.rand(1, len(y_axis), len(x_axis)),
        dims=("time", "lat", "lon"),
        coords={"time": [sim.receptor.time], "lat": y_axis, "lon": x_axis},
    )
    foot = _describe(data, sim.receptor, sim.variant.footprint, sim.variant.name)
    return feet.write(foot)


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
    )

    text = (tmp_path / "proj" / "config.yaml").read_text()
    assert "n_hours: -6" in text and "numpar" not in text  # only what was given
    assert (tmp_path / "proj" / "receptors.csv").exists()
    assert project.config.n_hours == -6
    assert list(project.receptors["receptor"]) == [point_receptor.id]
    assert Project(tmp_path / "proj").config == project.config


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
    assert project.simulations is project.simulations
    assert project.variants is project.variants
    assert project.plot is project.plot


# ---------------------------------------------------------------------------
# Output directory and scratch
# ---------------------------------------------------------------------------


def test_output_defaults_to_the_projects_output_directory(tmp_path):
    project = _project(tmp_path)

    assert isinstance(project.output, Output)
    assert project.output.path == tmp_path / "proj" / "output"
    assert not project.output.path.exists()  # nothing is made by looking


def test_output_is_relative_to_the_project_unless_absolute(tmp_path):
    relative = _project(tmp_path, name="a", output="../results")
    absolute = _project(tmp_path, name="b", output=str(tmp_path / "abs"))

    assert relative.output.path == (tmp_path / "results").resolve()
    assert absolute.output.path == tmp_path / "abs"


def test_output_can_be_shared_between_projects(tmp_path, point_receptor):
    shared = str(tmp_path / "shared")
    a = _project(tmp_path, [point_receptor], name="a", output=shared)
    b = _project(tmp_path, [point_receptor], name="b", output=shared)

    _write_trajectory(a, point_receptor)

    rid = point_receptor.id
    assert a.output.path == b.output.path
    assert b.simulation(rid, "hrrr").has_particles
    assert (
        a.simulation(rid, "hrrr")._particle_set
        == b.simulation(rid, "hrrr")._particle_set
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
    assert project.simulations.status().empty
    assert project.run() == []


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

    assert list(sims.frame.columns[:4]) == ["receptor", "variant", "group", "model"]
    assert set(sims["model"]) == {"hysplit"}
    assert _pairs(sims) == [
        (a.id, "hrrr"),
        (a.id, "err-0"),
        (a.id, "err-1"),
        (b.id, "hrrr"),
        (b.id, "err-0"),
        (b.id, "err-1"),
    ]
    assert sims["group"].tolist()[:3] == ["hrrr", "err", "err"]
    assert set(sims.frame.columns) >= {"time", "kind", "location", "site"}


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
    assert len(wbb.status()) == 2


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
    err = project.simulation(rid, "err-1")
    zi = project.simulation(rid, "zi08")
    s2 = project.simulation(rid, "s2")

    assert winderrtf(base.variant.transport) == 0
    assert winderrtf(err.variant.transport) == 1
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
    assert len(project.output.particle_sets()) == 1


def test_unreferenced_lists_output_folders_the_config_no_longer_uses(
    tmp_path, point_receptor
):
    project = _project(
        tmp_path, [point_receptor], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    for variant in ("hrrr", "zi08"):
        _write_trajectory(project, point_receptor, variant)
        _write_footprint(project, point_receptor, variant)
    assert project.unreferenced() == {"particles": [], "footprints": []}

    # The user drops zi08 and changes hrrr's smoothing in config.yaml.
    _config(tmp_path, variants={"hrrr": {"smooth_factor": 0.5}}).to_yaml(
        project.config_path
    )
    stale = Project(project.directory).unreferenced()
    assert [k.split("-")[0] for k in stale["particles"]] == ["zi08"]
    assert sorted(k.rsplit("-", 1)[0] for k in stale["footprints"]) == ["hrrr", "zi08"]
    # Nothing was deleted.
    assert project.simulation(point_receptor.id, "zi08").is_complete()


# ---------------------------------------------------------------------------
# Completion and status
# ---------------------------------------------------------------------------


def test_incomplete_follows_each_variant_outputs(tmp_path, point_receptor):
    project = _project(
        tmp_path,
        [point_receptor],
        variants={"hrrr": {}, "traj": {"grid": None}, "s2": {"smooth_factor": 2}},
    )

    assert project.simulations.incomplete()["variant"].tolist() == [
        "hrrr",
        "traj",
        "s2",
    ]

    _write_trajectory(project, point_receptor)  # shared by hrrr, traj, and s2
    assert project.simulations.incomplete()["variant"].tolist() == ["hrrr", "s2"]

    _write_footprint(project, point_receptor)
    _write_footprint(project, point_receptor, "s2", empty=True)
    assert project.simulations.incomplete().frame.empty


def test_status_marks_outputs_a_variant_does_not_produce(tmp_path, point_receptor):
    project = _project(
        tmp_path, [point_receptor], variants={"hrrr": {}, "traj": {"grid": None}}
    )
    _write_trajectory(project, point_receptor)

    status = project.simulations.status()

    assert list(status.columns[-4:]) == ["particles", "footprint", "empty", "complete"]
    by_variant = status.set_index("variant")
    assert by_variant.loc["hrrr", "particles"] == True  # noqa: E712
    assert by_variant.loc["hrrr", "footprint"] == False  # noqa: E712
    assert by_variant.loc["hrrr", "empty"] == False  # noqa: E712
    assert pd.isna(by_variant.loc["traj", "footprint"])
    assert pd.isna(by_variant.loc["traj", "empty"])
    assert by_variant.loc["traj", "complete"] == True  # noqa: E712
    assert by_variant.loc["hrrr", "complete"] == False  # noqa: E712


def test_status_counts_an_empty_footprint_as_complete(tmp_path):
    a, b = _receptor(12), _receptor(13)
    project = _project(tmp_path, [a, b])
    assert project.simulations.status()["complete"].tolist() == [False, False]

    _write_trajectory(project, a)
    _write_footprint(project, a)
    _write_trajectory(project, b)
    _write_footprint(project, b, empty=True)
    status = project.simulations.status()
    assert status["complete"].tolist() == [True, True]
    assert status["empty"].tolist() == [False, True]


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


def test_incomplete_and_status_agree_with_is_complete(tmp_path):
    """The folder listing must give `Simulation.is_complete()`'s answer."""
    project = _mixed_state_project(tmp_path)
    sims = project.simulations
    handles = [project.simulation(r, v) for r, v in _pairs(sims)]

    expected = [sim.id for sim in handles if not sim.is_complete()]
    assert 0 < len(expected) < len(handles)
    assert [
        SimID(r, v) for r, v in _pairs(project.simulations.incomplete())
    ] == expected

    status = project.simulations.status()
    assert status["complete"].tolist() == [sim.is_complete() for sim in handles]
    assert status["particles"].tolist() == [sim.has_particles for sim in handles]
    for row, sim in zip(status.itertuples(), handles, strict=True):
        if sim.makes_footprint:
            assert row.footprint == sim.has_footprint
            assert row.empty == (sim.empty_reason is not None)
        else:
            assert pd.isna(row.footprint) and pd.isna(row.empty)

    smooth = sims[sims.variant == "smooth"]
    assert _pairs(smooth.incomplete()) == [
        (sim.receptor.id, "smooth")
        for sim in handles
        if sim.variant.name == "smooth" and not sim.is_complete()
    ]


def test_incomplete_of_a_project_with_no_results_is_everything(tmp_path):
    project = _project(tmp_path, [_receptor(12)])
    assert _pairs(project.simulations.incomplete()) == _pairs(project.simulations)
    assert not project.output.path.exists()  # looking creates nothing


# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------


def test_load_particles_of_a_selection_is_one_table(tmp_path):
    a, b = _receptor(12), _receptor(13)
    project = _project(tmp_path, [a, b], variants={"hrrr": {}, "fine": {"grid": None}})
    assert project.simulations.load_particles().empty

    path = _write_trajectory(project, a)

    assert path.parent == project.output.particle_sets()[0].path / "date=2023-01-01"
    particles = project.simulations.load_particles()
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
    assert sims[sims.receptor == b.id].load_particles().empty


def test_load_footprints_skips_empty_footprints_and_trajectory_only_variants(
    tmp_path,
):
    done, empty, missing = _receptor(12), _receptor(13), _receptor(14)
    project = _project(
        tmp_path,
        [done, empty, missing],
        variants={"hrrr": {}, "traj": {"grid": None}},
    )
    _write_footprint(project, done)
    _write_footprint(project, empty, empty=True)
    _write_trajectory(project, done, "traj")

    loaded = project.simulations.load_footprints()

    assert list(loaded) == [SimID(done.id, "hrrr")]
    assert isinstance(loaded[SimID(done.id, "hrrr")], xr.DataArray)


def test_load_footprints_by_variant(tmp_path, point_receptor):
    project = _project(
        tmp_path, [point_receptor], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    for variant in ("hrrr", "zi08"):
        _write_footprint(project, point_receptor, variant)
    sims = project.simulations

    assert len(project.simulations.load_footprints()) == 2
    [(sid, foot)] = sims[sims.variant == "zi08"].load_footprints().items()
    assert sid == SimID(point_receptor.id, "zi08")
    assert foot.stilt.name == "zi08"


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
    H = hrrr.jacobian(target, bins)

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
    some = hrrr[hrrr.receptor.isin([b.id, a.id])].jacobian(target, bins)
    assert list(some.receptors) == [a.id]
    right_closed = pd.IntervalIndex.from_breaks(
        bins.left.append(bins.right[-1:]), closed="right"
    )
    with pytest.raises(ValueError, match="closed on the left"):
        hrrr.jacobian(target, right_closed)
    with pytest.raises(ValueError, match="no footprints yet"):
        sims[sims.variant == "zi08"].jacobian(target, bins)
    with pytest.raises(ValueError, match="no grid"):
        sims[sims.variant == "traj"].jacobian(target, bins)
    with pytest.raises(ValueError, match="one variant"):
        sims.jacobian(target, bins)


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
        return [f"result-{rid}" for rid in receptor_ids]

    monkeypatch.setattr("stilt.execution.worker.run_receptors", run_receptors)
    return calls


def test_run_hands_the_incomplete_receptors_to_the_workers(tmp_path, ran):
    done, todo = _receptor(12), _receptor(13)
    project = _project(
        tmp_path, [done, todo], include_footprint=False, execution={"cpus": 3}
    )
    _write_trajectory(project, done)

    results = project.run()

    assert results == [f"result-{todo.id}"]
    [call] = ran
    assert call["ids"] == [todo.id]
    assert call["project"] is project
    assert call["n_cores"] == 3
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

    project.run(execution=ExecutionConfig(cpus=8))

    assert ran[0]["n_cores"] == 8


def test_run_with_nothing_to_do_starts_nothing(tmp_path, ran, point_receptor):
    project = _project(tmp_path, [point_receptor])
    _write_trajectory(project, point_receptor)
    _write_footprint(project, point_receptor, empty=True)

    assert project.run() == []
    assert ran == []


def test_run_after_adding_a_variant_runs_the_receptor_again(
    tmp_path, ran, point_receptor
):
    """A finished project grows a variant: only that variant is incomplete."""
    project = _project(tmp_path, [point_receptor])
    _write_trajectory(project, point_receptor)
    _write_footprint(project, point_receptor)
    assert project.run() == []

    # The user adds a variant to config.yaml.
    _config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}).to_yaml(
        project.config_path
    )
    grown = Project(project.directory)
    assert _pairs(grown.simulations.incomplete()) == [(point_receptor.id, "zi08")]

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
    _write_trajectory(project, point_receptor, "err-0")

    project.run()

    assert ran[0]["ids"] == [point_receptor.id]
    assert _pairs(project.simulations.incomplete()) == [(point_receptor.id, "err-1")]


def test_submit_needs_slurm(tmp_path, point_receptor):
    project = _project(tmp_path, [point_receptor])
    with pytest.raises(ValueError, match="slurm"):
        project.submit()


def test_run_on_slurm_submits_and_waits(tmp_path, monkeypatch, point_receptor):
    project = _project(tmp_path, [point_receptor], execution={"backend": "slurm"})

    class _Job:
        def wait(self):
            pass

        def exception(self):
            return None

        def result(self):
            return ["done"]

    submitted = []

    def submit(project, **kwargs):
        submitted.append(kwargs)
        return [_Job(), _Job()]

    monkeypatch.setattr("stilt.execution.runner.submit", submit)

    assert project.run(compute_root="/s") == ["done", "done"]
    assert submitted[0]["compute_root"] == "/s"
    assert submitted[0]["execution"].backend == "slurm"


def test_receptors_are_built_only_when_asked_for(tmp_path):
    a, b = _receptor(12, site="WBB"), _receptor(13, site="UOU")
    _project(tmp_path, [a, b])
    project = Project(tmp_path / "proj")

    assert len(project.receptors) == 2
    project.simulations.status()
    assert project._built == {}  # listing, selecting, and status build nothing

    assert project.receptor(b.id) == b
    assert list(project._built) == [b.id]
    assert project.receptor(b.id) is project.receptor(b.id)


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


def test_a_selection_takes_columns_and_masks_and_yields_simulations(tmp_path):
    a, b = _receptor(12, site="WBB"), _receptor(13, site="UOU")
    project = _project(
        tmp_path, [a, b], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    sims = project.simulations

    assert isinstance(sims, Simulations)
    assert len(sims) == 4
    pd.testing.assert_series_equal(sims.variant, sims.frame["variant"])
    pd.testing.assert_series_equal(sims["site"], sims.frame["site"])

    wbb = sims[(sims.site == "WBB") & (sims.variant == "zi08")]
    assert isinstance(wbb, Simulations)
    [sim] = list(wbb)
    assert sim == project.simulation(a.id, "zi08")
    assert [s.id for s in sims] == [SimID(r, v) for r, v in _pairs(sims)]


def test_a_selection_points_to_frame_for_other_pandas(tmp_path):
    project = _project(tmp_path, [_receptor(12)])
    sims = project.simulations
    with pytest.raises(AttributeError, match="sims.frame"):
        sims.groupby("variant")
    assert sims.frame.groupby("variant").size().to_dict() == {"hrrr": 1}


def test_a_selection_from_a_table_made_with_pandas(tmp_path):
    a, b = _receptor(12), _receptor(13)
    project = _project(tmp_path, [a, b])
    mine = pd.DataFrame({"receptor": [b.id], "obs": [1.9]})
    merged = project.simulations.frame.merge(mine, on="receptor")

    back = Simulations(project, merged)
    assert [s.id for s in back] == [SimID(b.id, "hrrr")]
    assert back.obs.tolist() == [1.9]
    with pytest.raises(ValueError, match="'receptor' and 'variant'"):
        Simulations(project, project.receptors)
