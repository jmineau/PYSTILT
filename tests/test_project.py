"""Tests for runtraj.project: the project directory, its tables, its results, and running it."""

import datetime as dt
import doctest
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from runtraj.config import ProjectConfig
from runtraj.execution.config import ExecutionConfig
from runtraj.execution.runner import resolve_compute_root
from runtraj.models.hysplit import MetConfig
from runtraj.models.hysplit.driver import winderrtf
from runtraj.output import Output
from runtraj.releases import ColumnRelease, PointRelease

from stilt import Project, transport_error
from stilt.footprint.grid import Grid
from stilt.footprint.stage import store_empty_footprint, store_footprint
from stilt.project import STILT_R_HYSPLIT
from tests.fixtures.factories import blocks

from .fixtures.factories import make_hysplit_dir, make_met_config, make_release
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
    """
    Minimal ProjectConfig with one met and (optionally) a footprint grid.

    Its ``hysplit`` block has STILT-R's values, as ``Project.init`` writes
    them, unless *overrides* give the ``models`` blocks.
    """
    overrides.setdefault("grid", _GRID if include_footprint else None)
    if "models" not in overrides:
        overrides = {**STILT_R_HYSPLIT, **overrides}
    return ProjectConfig(
        **blocks(
            mets={"hrrr": _met(tmp_path)}, **{"variants": {"hrrr": {}}, **overrides}
        )
    )


def _project(tmp_path, releases=None, name="proj", **overrides) -> Project:
    """Make a project in ``tmp_path / name`` with the minimal config."""
    return Project.init(
        tmp_path / name, config=_config(tmp_path, **overrides), releases=releases
    )


def _release(hour: int, longitude: float = -111.85, **attrs) -> PointRelease:
    return make_release(dt.datetime(2023, 1, 1, hour), longitude, **attrs)


def _particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [-60.0, -120.0],
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
    project: Project, release, variant="hrrr", realization=None
) -> Path:
    """Write a small particle file for one simulation into the output directory."""
    sim = project.simulation(release.id, variant, realization)
    particles = finished(_particles(), sim.release)
    return sim.output.write_particles(
        sim.variant, sim.release, particles, [], realization
    )


def _write_footprint(
    project: Project, release, variant="hrrr", realization=None, *, empty=False
) -> Path:
    """Write a footprint (or an empty one) for one simulation into the output directory."""
    sim = project.simulation(release.id, variant, realization)
    assert sim.variant.stages.get("footprint") is not None
    if empty:
        return store_empty_footprint(sim.output, sim.variant, sim.release, realization)
    grid = sim.variant.stages.get("footprint").grid
    assert grid is not None
    x_axis, y_axis = grid.axes
    data = xr.DataArray(
        # Each realization gets its own values.
        np.random.default_rng(realization or 0).random((1, len(y_axis), len(x_axis))),
        dims=("time", "lat", "lon"),
        coords={"time": [sim.release.time], "lat": y_axis, "lon": x_axis},
    )
    foot = as_footprint(
        data, sim.release, sim.variant.stages.get("footprint"), sim.variant.name
    )
    return store_footprint(sim.output, sim.variant, foot, realization)


def _pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(frame["release"], frame["variant"], strict=True))


# ---------------------------------------------------------------------------
# Making and opening a project
# ---------------------------------------------------------------------------


def test_init_writes_the_settings_given_and_the_releases(tmp_path, point_release):
    project = Project.init(
        tmp_path / "proj",
        releases=[point_release],
        mets={"hrrr": _met(tmp_path)},
        models={"hysplit": {"n_hours": -6}},
        variants={"hrrr": {}},
    )

    text = (tmp_path / "proj" / "config.yaml").read_text()
    assert "n_hours: -6" in text and "kmix0" not in text  # no other defaults
    assert (tmp_path / "proj" / "releases.csv").exists()
    assert project.config.models == {"hysplit": {"n_hours": -6, **STILT_R_HYSPLIT}}
    assert project.variants["hrrr"].transport.n_hours == -6
    assert list(project.releases["release"]) == [point_release.id]
    assert Project(tmp_path / "proj").config == project.config


def test_an_output_given_when_opening_replaces_the_configs(tmp_path, point_release):
    project = _project(tmp_path, [point_release])
    assert project.output.directory == project.directory / "output"

    elsewhere = Project(project.directory, output=tmp_path / "bucket")

    assert elsewhere.output.directory == tmp_path / "bucket"
    _write_trajectory(elsewhere, point_release)
    assert elsewhere.simulation(point_release.id, "hrrr").has_particles
    assert not project.simulation(point_release.id, "hrrr").has_particles


def test_init_writes_stilt_rs_hysplit_values_where_none_are_given(tmp_path):
    project = Project.init(
        tmp_path / "proj",
        mets={"hrrr": _met(tmp_path)},
        models={"hysplit": {"n_hours": -24, "numpar": 200, "kmixd": None}},
        variants={"hrrr": {}},
    )

    hysplit = project.config.models["hysplit"]
    assert hysplit == {
        "n_hours": -24,
        "numpar": 200,
        "kmixd": None,
        **{k: v for k, v in STILT_R_HYSPLIT.items() if k not in ("numpar", "kmixd")},
    }
    transport = project.variants["hrrr"].transport
    assert (transport.numpar, transport.kmixd, transport.krand) == (200, None, 4)


def test_init_writes_stilt_rs_values_when_no_model_is_given(tmp_path):
    # Without models:, the project runs HYSPLIT, so the values go in its block.
    config = ProjectConfig(
        mets={"hrrr": _met(tmp_path)},
        variants={"hrrr": {"n_hours": -24}},
    )

    project = Project.init(tmp_path / "proj", config=config)

    assert project.config.models == {"hysplit": STILT_R_HYSPLIT}
    assert project.variants["hrrr"].transport.capemin == -1.0


def test_init_refuses_a_project_that_has_a_config(tmp_path):
    _project(tmp_path)
    text = (tmp_path / "proj" / "config.yaml").read_text()

    with pytest.raises(FileExistsError, match="already has a config.yaml"):
        Project.init(tmp_path / "proj", config=_config(tmp_path, numpar=10))
    assert (tmp_path / "proj" / "config.yaml").read_text() == text


def test_init_takes_a_config_or_keywords_not_both(tmp_path):
    with pytest.raises(TypeError, match="not both"):
        Project.init(tmp_path / "proj", config=_config(tmp_path), **blocks(n_hours=-6))
    assert not (tmp_path / "proj").exists()


def test_init_writes_the_commented_starter_config(tmp_path):
    from stilt.project import STARTER_CONFIG

    project = Project.init(tmp_path / "proj", starter=True)

    assert project.config_path.read_text() == STARTER_CONFIG
    assert list(project.variants) == ["hrrr"]
    assert project.releases.empty
    with pytest.raises(TypeError, match="starter"):
        Project.init(tmp_path / "other", starter=True, **blocks(numpar=10))
    with pytest.raises(FileExistsError):
        Project.init(tmp_path / "proj", starter=True)


def test_init_copies_releases_from_a_csv(tmp_path):
    csv = tmp_path / "mine.csv"
    csv.write_text(
        "time,longitude,latitude,altitude,site\n"
        "2023-01-01 12:00:00,-111.85,40.77,5,WBB\n"
    )
    project = Project.init(tmp_path / "proj", config=_config(tmp_path), releases=csv)

    assert project.releases["site"].tolist() == ["WBB"]


@pytest.mark.parametrize("empty", [[], "header.csv"])
def test_init_refuses_releases_that_hold_none_and_writes_nothing(tmp_path, empty):
    """An empty list or a header-only file made an empty project, with no warning (#440)."""
    (tmp_path / "header.csv").write_text("time,longitude,latitude,altitude\n")
    releases = tmp_path / empty if isinstance(empty, str) else empty

    with pytest.raises(ValueError, match="No releases were given"):
        Project.init(tmp_path / "proj", releases, config=_config(tmp_path))
    assert not (tmp_path / "proj").exists()


def test_views_are_cached(tmp_path):
    project = _project(tmp_path)

    assert project.config is project.config
    assert project.releases is project.releases
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


def test_output_can_be_shared_between_projects(tmp_path, point_release):
    shared = str(tmp_path / "shared")
    a = _project(tmp_path, [point_release], name="a", output=shared)
    b = _project(tmp_path, [point_release], name="b", output=shared)

    _write_trajectory(a, point_release)

    rid = point_release.id
    assert a.output.directory == b.output.directory
    assert b.simulation(rid, "hrrr").has_particles
    assert (
        a.simulation(rid, "hrrr").particles_path
        == b.simulation(rid, "hrrr").particles_path
    )


def test_compute_root_defaults_under_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.delenv("RUNTRAJ_COMPUTE_ROOT", raising=False)

    expected = (tmp_path / "tmp" / "runtraj" / "proj").resolve()
    assert resolve_compute_root(_project(tmp_path)) == expected


def test_default_compute_root_is_resolved_like_an_explicit_one(tmp_path, monkeypatch):
    """A TMPDIR behind a symlink (as on macOS) gives the path a pool worker gets."""
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "link"))
    monkeypatch.delenv("RUNTRAJ_COMPUTE_ROOT", raising=False)
    project = _project(tmp_path)

    default = resolve_compute_root(project)
    assert default == real.resolve() / "runtraj" / "proj"
    assert resolve_compute_root(project, str(default)) == default


def test_compute_root_from_the_environment_and_explicit_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNTRAJ_COMPUTE_ROOT", str(tmp_path / "scratch"))
    project = _project(tmp_path)

    assert resolve_compute_root(project) == (tmp_path / "scratch").resolve()
    explicit = resolve_compute_root(project, tmp_path / "explicit")
    assert explicit == (tmp_path / "explicit").resolve()


def test_an_empty_compute_root_variable_is_unset(tmp_path, monkeypatch):
    """RUNTRAJ_COMPUTE_ROOT= means the default, not the current directory."""
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("RUNTRAJ_COMPUTE_ROOT", "")

    expected = (tmp_path / "tmp" / "runtraj" / "proj").resolve()
    assert resolve_compute_root(_project(tmp_path)) == expected


# ---------------------------------------------------------------------------
# Releases
# ---------------------------------------------------------------------------


def test_releases_are_a_table_with_the_files_labels(tmp_path):
    a, b = _release(12, site="WBB"), _release(13, longitude=-112.0, site="UOU")
    project = _project(tmp_path, [a, b])

    frame = project.releases

    assert list(frame.columns) == ["release", "time", "kind", "location", "site"]
    assert frame["release"].tolist() == [a.id, b.id]
    assert frame["kind"].tolist() == ["point", "point"]
    assert frame["location"].tolist() == [a.location_id, b.location_id]
    assert str(frame["time"].dtype).startswith("datetime64")
    assert frame["site"].tolist() == ["WBB", "UOU"]
    assert project.release(b.id) == b
    with pytest.raises(KeyError, match="No release"):
        project.release("202301011200_0_0_0")


def test_a_project_without_releases_is_empty(tmp_path):
    project = _project(tmp_path)

    assert project.releases.empty
    assert list(project.releases.columns) == ["release", "time", "kind", "location"]
    assert len(project.simulations) == 0
    assert project.status().empty
    assert project.run().empty


def test_add_releases_appends_only_new_ones_and_refreshes_the_views(
    tmp_path, point_release, column_release, multipoint_release
):
    project = _project(tmp_path, [point_release])
    assert len(project.simulations) == 1

    added = project.add_releases([point_release, column_release, multipoint_release])

    assert added == [column_release.id, multipoint_release.id]
    assert project.releases["kind"].tolist() == ["point", "column", "multipoint"]
    assert len(project.simulations) == 3
    assert project.release(multipoint_release.id) == multipoint_release
    reopened = Project(project.directory).releases["release"].tolist()
    assert reopened == project.releases["release"].tolist()
    assert project.add_releases(point_release) == []


def test_add_releases_appends_in_the_files_own_columns(tmp_path, column_release):
    """A hand-written file keeps its columns, its group values, and extra columns."""
    project = _project(tmp_path)
    text = (
        "group,time,lati,long,zagl,scene\n"
        "1155,2023-01-01 12:00:00,40.77,-111.85,5.0,A\n"
    )
    project.releases_path.write_text(text)

    added = project.add_releases([column_release])

    assert added == [column_release.id]
    stored = project.releases_path.read_text()
    assert stored.startswith(text)  # the original bytes are untouched
    lines = stored.splitlines()
    assert lines[2].split(",")[0] == "1156" and lines[2].endswith(",")


def test_add_releases_refuses_a_group_without_a_group_column(tmp_path, column_release):
    project = _project(tmp_path)
    project.releases_path.write_text("time,lati,long,zagl\n")
    with pytest.raises(ValueError, match="group"):
        project.add_releases([column_release])


def test_add_releases_refuses_a_file_with_the_old_group_column(tmp_path, point_release):
    project = _project(tmp_path)
    project.releases_path.write_text("r_idx,time,lati,long,zagl\n")
    with pytest.raises(ValueError, match="Rename it to 'group'"):
        project.add_releases([point_release])


def test_a_project_with_the_old_receptors_csv_is_refused(tmp_path):
    """Nothing reads ``receptors.csv`` any more, so it must not pass for no releases."""
    project = _project(tmp_path)
    project.releases_path.unlink(missing_ok=True)
    (project.directory / "receptors.csv").write_text("r_idx,time,lati,long,zagl\n")
    with pytest.raises(ValueError, match="rename the file"):
        Project(project.directory)
    # releases.csv wins once it exists.
    project.releases_path.write_text("time,lati,long,zagl\n")
    assert Project(project.directory).simulations.empty


def test_releases_that_would_share_result_files_are_refused(tmp_path):
    """Two releases with one id would overwrite each other's results."""
    from runtraj.releases import MultiPointRelease

    def slant(alt):
        return MultiPointRelease(
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
        project.add_releases([b])
    assert project.add_releases([slant(10.001)]) == []  # the same release again


def test_add_releases_takes_releases_or_a_csv(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(TypeError, match="Release"):
        project.add_releases([1, 2])


# ---------------------------------------------------------------------------
# Variants and simulations
# ---------------------------------------------------------------------------


def test_variants_default_to_one_per_met(tmp_path):
    project = _project(tmp_path)

    assert list(project.variants) == ["hrrr"]
    assert project.variants["hrrr"].met == "hrrr"


def test_simulations_are_releases_times_variants(tmp_path):
    a, b = _release(12, site="WBB"), _release(13, site="UOU")
    project = _project(
        tmp_path,
        [a, b],
        krand=4,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )

    sims = project.simulations

    assert list(sims.columns[:4]) == ["release", "variant", "realization", "model"]
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
    a = _release(12, site="WBB")
    b = _release(13, site="UOU")
    c = _release(14, site="WBB")
    project = _project(
        tmp_path, [a, b, c], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    sims = project.simulations

    wbb = sims[(sims.site == "WBB") & (sims.variant == "zi08")]
    assert _pairs(wbb) == [(a.id, "zi08"), (c.id, "zi08")]
    later = sims[sims.time.between("2023-01-01 13:00", "2023-01-01 14:00")]
    assert sorted(set(later["release"])) == sorted([b.id, c.id])
    assert len(project.status(wbb)) == 2


def test_simulation_handles_carry_the_variant_settings(tmp_path, point_release):
    project = _project(
        tmp_path,
        [point_release],
        krand=4,
        variants={
            "hrrr": {},
            "err": {**_XYERR, "realizations": 2, "grid": None},
            "zi08": {"ziscale": 0.8},
            "s2": {"smooth_factor": 2},
        },
    )
    rid = point_release.id

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
    assert err.variant.stages.get("footprint") is None
    assert zi.variant.transport.ziscale == 0.8
    assert zi.variant.stages.get("footprint") == base.variant.stages.get("footprint")
    assert (
        s2.variant.stages.get("footprint") is not None
        and s2.variant.stages.get("footprint").smooth_factor == 2
    )
    assert s2.variant.transport == base.variant.transport  # shares hrrr's particles
    assert project.simulation(rid, "hrrr") == base  # a value, not a handle
    with pytest.raises(KeyError, match="No variant"):
        project.simulation(rid, "nope")


def test_variants_with_equal_transport_settings_share_a_run(tmp_path, point_release):
    project = _project(
        tmp_path,
        [point_release],
        variants={"hrrr": {}, "s2": {"smooth_factor": 2}, "zi08": {"ziscale": 0.8}},
    )

    hrrr, s2, zi = (project.variants[v] for v in ("hrrr", "s2", "zi08"))
    assert hrrr.particles_hash == s2.particles_hash != zi.particles_hash

    _write_trajectory(project, point_release)
    assert project.simulation(point_release.id, "s2").has_particles
    assert not project.simulation(point_release.id, "zi08").has_particles
    assert len(project.output.hashes("particles")) == 1


def _with_unreadable_met(project, tmp_path, **overrides):
    """Open *project*'s directory again with a met whose directory does not exist."""
    config = ProjectConfig(
        **blocks(
            mets={"hrrr": MetConfig(directory=tmp_path / "no-such-met")},
            **{"variants": {"hrrr": {}}, "grid": _GRID, **STILT_R_HYSPLIT, **overrides},
        )
    )
    config.to_yaml(project.config_path)
    return Project(project.directory)


def test_status_without_a_readable_met_is_pending_when_no_folder_can_hold_results(
    tmp_path, point_release
):
    """The weather product is the only part of the hash that needs the met (#377)."""
    project = _project(tmp_path, [point_release], variants={"hrrr": {"ziscale": 0.8}})
    _write_trajectory(project, point_release)

    # The settings differ from the folder's in more than the met: nothing is there.
    unread = _with_unreadable_met(project, tmp_path)
    status = unread.status()
    assert list(status["state"]) == ["pending"]
    assert list(status["particles"]) == [False]
    assert list(status["footprint"]) == [False]
    assert unread.incomplete().shape[0] == 1
    assert list(unread.folders()["variant"]) == [""]
    assert "ziscale" in unread.folders().loc[0, "differs"]


def test_status_without_a_readable_met_raises_when_a_folder_may_hold_results(
    tmp_path, point_release
):
    """Same settings but the met's: the folder, whatever it is named, may be this variant's."""
    project = _project(tmp_path, [point_release])
    sim = project.simulation(point_release.id, "hrrr")
    particles = finished(_particles(), sim.release)
    renamed = replace(sim.variant, name="renamed")
    sim.output.write_particles(renamed, sim.release, particles, [])

    unread = _with_unreadable_met(project, tmp_path)

    with pytest.raises(FileNotFoundError, match="Met 'hrrr'"):
        unread.status()
    with pytest.raises(FileNotFoundError, match="Met 'hrrr'"):
        unread.folders()


def test_status_without_a_readable_met_and_an_empty_output_is_pending(
    tmp_path, point_release
):
    project = _project(tmp_path, [point_release])

    unread = _with_unreadable_met(project, tmp_path)

    assert list(unread.status()["state"]) == ["pending"]
    assert unread.folders().empty


def _without_hycs_std(project, monkeypatch, tmp_path):
    """Open *project*'s directory again on a machine with no hycs_std, as one that only reports status."""
    monkeypatch.setenv("PATH", str(tmp_path / "no-hysplit"))
    return Project(project.directory)


def test_status_without_hycs_std_finds_the_folder_the_variant_made(
    tmp_path, point_release, monkeypatch
):
    """The build is the only part of the hash that needs hycs_std; the variant's name settles the rest."""
    project = _project(tmp_path, [point_release])
    _write_trajectory(project, point_release)
    _write_footprint(project, point_release)

    status = _without_hycs_std(project, monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="hycs_std"):
        _ = status.variants["hrrr"].particles_hash
    assert list(status.status()["state"]) == ["complete"]
    assert list(status.folders()["variant"]) == ["hrrr", "hrrr"]


def test_status_without_hycs_std_leaves_a_folder_another_variant_made(
    tmp_path, point_release, monkeypatch
):
    project = _project(tmp_path, [point_release])
    sim = project.simulation(point_release.id, "hrrr")
    particles = finished(_particles(), sim.release)
    other = replace(sim.variant, name="other")
    sim.output.write_particles(other, sim.release, particles, [])

    status = _without_hycs_std(project, monkeypatch, tmp_path)
    assert list(status.status()["state"]) == ["pending"]
    assert list(status.folders()["variant"]) == [""]


def test_status_without_hycs_std_raises_when_the_variant_ran_two_builds(
    tmp_path, point_release, monkeypatch
):
    """Two folders of one variant whose settings differ only in hycs_std: which is this machine's is unknown."""
    project = _project(tmp_path, [point_release])
    _write_trajectory(project, point_release)
    build = make_hysplit_dir(tmp_path / "rebuilt", body="# rebuilt\nexit 2\n")
    _config(tmp_path, directory=build).to_yaml(project.config_path)
    _write_trajectory(Project(project.directory), point_release)
    assert len(project.output.hashes("particles")) == 2
    _config(tmp_path).to_yaml(project.config_path)  # back to the hycs_std on PATH

    status = _without_hycs_std(project, monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="hycs_std"):
        status.status()


def test_folders_say_which_the_config_no_longer_uses(tmp_path, point_release):
    project = _project(
        tmp_path, [point_release], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    for variant in ("hrrr", "zi08"):
        _write_trajectory(project, point_release, variant)
        _write_footprint(project, point_release, variant)
    assert (project.folders()["variant"] != "").all()

    # The user drops zi08 and changes hrrr's smoothing in config.yaml.
    _config(tmp_path, variants={"hrrr": {"smooth_factor": 0.5}}).to_yaml(
        project.config_path
    )
    folders = Project(project.directory).folders()
    stale = folders[folders["variant"] == ""]
    by_kind = {
        k: sorted(stale.loc[stale.kind == k, "name"])
        for k in ("particles", "footprint")
    }
    assert by_kind == {"particles": ["zi08"], "footprint": ["hrrr", "zi08"]}
    # Nothing was deleted.
    assert project.simulation(point_release.id, "zi08").is_complete


# ---------------------------------------------------------------------------
# Completion and status
# ---------------------------------------------------------------------------


def test_incomplete_follows_each_variant_outputs(tmp_path, point_release):
    project = _project(
        tmp_path,
        [point_release],
        variants={"hrrr": {}, "traj": {"grid": None}, "s2": {"smooth_factor": 2}},
    )

    assert project.incomplete()["variant"].tolist() == [
        "hrrr",
        "traj",
        "s2",
    ]

    _write_trajectory(project, point_release)  # shared by hrrr, traj, and s2
    assert project.incomplete()["variant"].tolist() == ["hrrr", "s2"]

    _write_footprint(project, point_release)
    _write_footprint(project, point_release, "s2", empty=True)
    assert project.incomplete().empty


def test_status_marks_outputs_a_variant_does_not_produce(tmp_path, point_release):
    project = _project(
        tmp_path, [point_release], variants={"hrrr": {}, "traj": {"grid": None}}
    )
    _write_trajectory(project, point_release)

    status = project.status()

    assert list(status.columns[-6:]) == [
        "particles",
        "footprint",
        "state",
        "stage",
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
    a, b, c = _release(12), _release(13), _release(14)
    project = _project(tmp_path, [a, b, c])
    variant = project.variants["hrrr"]
    project.output.write_log(variant, a.id, "started\n")
    project.output.write_log(variant, b.id, "HYSPLIT said no\n")
    project.output.record_failure("particles", variant, b.id, {"reason": "X"})

    status = project.status()

    assert status["state"].tolist() == ["interrupted", "failed", "pending"]
    assert project.incomplete()["release"].tolist() == [a.id, b.id, c.id]


def test_status_counts_an_empty_footprint_as_complete(tmp_path):
    a, b = _release(12), _release(13)
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


def test_status_opens_no_result_file_and_builds_no_release(tmp_path, monkeypatch):
    """#141: on a large project, opening files or building releases per row took minutes."""
    import pyarrow.parquet as pq

    a, b = _release(12), _release(13)
    project = _project(tmp_path, [a, b])
    _write_trajectory(project, a)
    _write_footprint(project, a)
    _write_footprint(project, b, empty=True)
    sims = project.simulations

    def refuse(*args, **kwargs):
        raise AssertionError("status() opened a result file")

    for name in ("read_schema", "read_metadata", "read_table", "ParquetFile"):
        monkeypatch.setattr(pq, name, refuse)
    monkeypatch.setattr(Project, "_releases", None)

    assert project.status(sims)["state"].tolist() == ["complete", "pending"]


def _mixed_state_project(tmp_path):
    """Six releases under three variants, in every state a simulation can be in."""
    releases = [_release(h) for h in range(10, 16)]
    project = _project(
        tmp_path,
        releases,
        variants={
            "hrrr": {},
            "smooth": {"smooth_factor": 2},  # shares hrrr's particles
            "zi08": {"ziscale": 0.8, "grid": None},  # particles only
        },
    )
    a, b, c, d, e, _ = releases
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
    # By release hour: 10 is a, 11 b, ..., 15 f.
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
    project = _project(tmp_path, [_release(12)])
    assert _pairs(project.incomplete()) == _pairs(project.simulations)
    assert not project.output.directory.exists()  # looking creates nothing


# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------


def test_particles_of_a_selection_is_one_table(tmp_path):
    a, b = _release(12), _release(13)
    project = _project(tmp_path, [a, b], variants={"hrrr": {}, "fine": {"grid": None}})
    assert project.particles().empty

    path = _write_trajectory(project, a)

    folder = project.output.folder("particles", project.variants["hrrr"])
    assert path.parent == folder / "date=2023-01-01"
    particles = project.particles()
    assert list(particles.columns[:2]) == ["release", "variant"]
    one = project.simulation(a.id, "hrrr").particles
    # Both variants share the run, so each gets its own copy of its rows.
    assert particles["variant"].value_counts().to_dict() == {
        "hrrr": len(one),
        "fine": len(one),
    }
    assert set(particles["release"]) == {a.id}
    hrrr = particles[particles.variant == "hrrr"].drop(columns=["release", "variant"])
    pd.testing.assert_frame_equal(
        hrrr.reset_index(drop=True)[one.columns], one, check_dtype=False
    )
    sims = project.simulations
    assert project.particles(sims[sims.release == b.id]).empty


def test_particles_come_in_the_order_of_the_selection(tmp_path):
    """Python hashes strings differently in each process; the rows must not follow it."""
    releases = [_release(hour) for hour in range(8, 20)]
    project = _project(tmp_path, releases)
    for release in releases:
        _write_trajectory(project, release)
    sims = project.simulations

    for selected in (sims, sims.iloc[::-1], sims.iloc[[5, 2, 9, 0, 11, 7]]):
        got = project.particles(selected, ["particle"])
        assert list(dict.fromkeys(got["release"])) == list(selected["release"])
        # Each release's rows stay together, in the file's order.
        assert got["particle"].tolist() == [1.0] * len(got)


def test_particles_name_their_variant_and_realization(tmp_path):
    release = _release(12)
    project = _project(
        tmp_path,
        [release],
        grid=None,
        krand=4,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )
    _write_trajectory(project, release)
    for k in (1, 0):
        _write_trajectory(project, release, "err", k)

    got = project.particles(columns=["particle"])

    assert got["realization"].dtype == "Int64"
    assert got["variant"].tolist() == ["hrrr"] * 2 + ["err"] * 4
    assert got["realization"].isna().tolist() == [True] * 2 + [False] * 4
    assert got["realization"].tolist()[2:] == [0, 0, 1, 1]


def test_particles_reads_only_the_columns_asked_for(tmp_path):
    """A day of a site was 12 GB with every column (#191)."""
    a = _release(12)
    project = _project(tmp_path, [a])
    _write_trajectory(project, a)
    every = project.particles()

    some = project.particles(columns=["lon", "lat", "foot"])
    assert list(some.columns) == [
        "release",
        "variant",
        "realization",
        "lon",
        "lat",
        "foot",
    ]
    pd.testing.assert_frame_equal(some, every[list(some.columns)])

    timed = project.particles(columns=["time", "foot"])
    assert list(timed.columns) == [
        "release",
        "variant",
        "realization",
        "time",
        "foot",
    ]
    assert (timed["time"] == every["time"]).all()

    with pytest.raises(ValueError, match="No column 'height'"):
        project.particles(columns=["height"])


def _write_spread_particles(
    project: Project, release, variant, realization, spread, seed
) -> None:
    """Write 300 particles scattered *spread* degrees around the release, with the same footprint value each."""
    sim = project.simulation(release.id, variant, realization)
    rng = np.random.default_rng(seed)
    n = 300
    particles = pd.DataFrame(
        {
            "age": np.full(n, -60.0),
            "particle": np.arange(1, n + 1, dtype=float),
            "lon": release.longitude + rng.normal(0, spread, n),
            "lat": release.latitude + rng.normal(0, spread, n),
            "zagl": np.full(n, 10.0),
            "foot": np.full(n, 1e-3),
        }
    )
    particles = finished(particles, sim.release)
    sim.output.write_particles(sim.variant, sim.release, particles, [], realization)


def _run_docstring_examples(func, **names):
    """Run the ``>>>`` examples in *func*'s docstring with *names* defined; return what they defined."""
    (test,) = doctest.DocTestFinder().find(func, globs=names)
    assert test.examples, f"{func.__name__} has no docstring examples"
    for example in test.examples:  # the docstring skips them for pytest's doctest run
        example.options.pop(doctest.SKIP, None)
    runner = doctest.DocTestRunner()
    runner.run(test, clear_globs=False)  # pytest shows what a failure printed
    assert runner.failures == 0
    return test.globs


def test_transport_error_docstring_recipe_gives_the_realizations_of_sim_transport_error(
    tmp_path,
):
    """The docstring's recipe passes one table per realization, as ``sim.transport_error`` does (#239)."""
    a, b = _release(12), _release(13)
    project = _project(
        tmp_path,
        [a, b],
        grid=None,
        krand=4,
        variants={"hrrr": {}, "hrrr-err": {**_XYERR, "realizations": 4}},
    )
    # Every release and realization scatters its particles differently, so
    # merging any of them changes the answer.
    for i, release in enumerate((a, b)):
        _write_spread_particles(project, release, "hrrr", None, 0.05, 10 * i)
        for k in range(4):
            spread = 0.08 * (1 + i) + 0.04 * k
            _write_spread_particles(project, release, "hrrr-err", k, spread, 10 * i + k)
    lat, lon = np.linspace(39.5, 42.0, 101), np.linspace(-113.0, -110.5, 101)
    flux = xr.DataArray(
        np.add.outer(np.sin(7 * lat), np.cos(5 * lon)) + 2,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    sims = project.simulations
    assert len(sims[sims.variant == "hrrr-err"]) == 8

    for release in (a, b):
        rid = release.id
        names = _run_docstring_examples(
            transport_error, project=project, rid=rid, flux=flux
        )
        sim = project.simulation(rid, "hrrr")
        realizations = [project.simulation(rid, "hrrr-err", k) for k in range(4)]
        expected = sim.transport_error(realizations, flux)

        assert names["result"].realizations == expected.realizations == 4
        assert names["result"].variance == pytest.approx(expected.variance)
        assert names["result"].enhancement_perturbed == pytest.approx(
            expected.enhancement_perturbed
        )

        # The old recipe: one variant is one table, so one realization.
        ensemble = sims[(sims.variant == "hrrr-err") & (sims.release == rid)]
        merged = transport_error(
            sim.particles,
            [rows for _, rows in project.particles(ensemble).groupby("variant")],
            flux,
            release=sim.release,
        )
        assert merged.realizations == 1
        assert merged.variance != pytest.approx(expected.variance, rel=0.2)


def _write_aged_particles(
    project: Project, release, variant, realization, spread, seed
):
    """Write 300 particles scattered *spread* degrees around the release, each of a random age."""
    sim = project.simulation(release.id, variant, realization)
    rng = np.random.default_rng(seed)
    n = 300
    particles = pd.DataFrame(
        {
            "age": -rng.integers(1, 12, n) * 60.0,
            "particle": np.arange(1, n + 1, dtype=float),
            "lon": release.longitude + rng.normal(0, spread, n),
            "lat": release.latitude + rng.normal(0, spread, n),
            "zagl": np.full(n, 10.0),
            "foot": np.full(n, 1e-3),
        }
    )
    if hasattr(release, "bottom"):
        # A column release starts its particles at heights only its age = 0
        # rows give; a point release has the one height it was written with.
        start = particles.assign(
            age=0.0,
            lon=release.longitude,
            lat=release.latitude,
            zagl=release.bottom + (release.top - release.bottom) * rng.random(n),
            foot=0.0,
        )
        particles = pd.concat([start, particles], ignore_index=True)
    particles = finished(particles, sim.release)
    sim.output.write_particles(sim.variant, sim.release, particles, [], realization)


def _flux() -> xr.DataArray:
    lat, lon = np.linspace(39.5, 42.0, 101), np.linspace(-113.0, -110.5, 101)
    return xr.DataArray(
        np.add.outer(np.sin(7 * lat), np.cos(5 * lon)) + 2,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )


_ERROR_COLUMNS = [
    "enhancement",
    "enhancement_perturbed",
    "background",
    "variance",
    "noise",
    "realizations",
]


@pytest.mark.parametrize("kind", ["point", "column"])
@pytest.mark.parametrize("realizations", [None, 2])
def test_transport_error_of_a_selection_is_sim_transport_error_per_release(
    tmp_path, kind, realizations
):
    """Each row is ``sim.transport_error`` of its release, with the realizations it has run."""

    def release(hour):
        time = dt.datetime(2023, 1, 1, hour)
        if kind == "point":
            return make_release(time)
        return ColumnRelease(
            time=time, longitude=-111.85, latitude=40.77, bottom=5.0, top=500.0
        )

    a, b, c, d = (release(h) for h in (12, 13, 14, 15))
    error = {**_XYERR, "grid": None}
    if realizations is not None:
        error["realizations"] = realizations
    lifetime = {"kind": "first_order_lifetime", "lifetime_hours": 2.0}
    project = _project(
        tmp_path,
        [a, b, c, d],
        hnf_plume=False,
        # The lifetime weights the particles of hrrr and not those of
        # hrrr-err, so using the wrong variant's transforms changes the answer.
        variants={"hrrr": {"transforms": [lifetime]}, "hrrr-err": error},
    )
    ks = project.variants["hrrr-err"].realization_numbers
    # Every release and realization scatters its particles differently, so
    # pairing the wrong release or realization changes the answer.
    for i, r in enumerate((a, b, c)):
        _write_aged_particles(project, r, "hrrr", None, 0.05 + 0.01 * i, 10 * i)
    for i, r in enumerate((a, b, d)):
        # b has only its first realization yet.
        for k in ks[:1] if r is b else ks:
            spread = 0.08 * (1 + i) + 0.04 * (k or 0)
            seed = 10 * i + (k or 0) + 100
            _write_aged_particles(project, r, "hrrr-err", k, spread, seed)
    flux = _flux()

    sims = project.simulations
    sel = sims[sims.variant == "hrrr"].iloc[::-1]  # d, c, b, a
    errors = project.transport_error(sel, "hrrr-err", flux, length_scale=200.0)

    assert list(errors.columns) == list(sel.columns) + _ERROR_COLUMNS
    pd.testing.assert_frame_equal(errors[list(sel.columns)], sel)
    rows = errors.set_index("release")
    for r in (a, b):
        sim = project.simulation(r.id, "hrrr")
        ran = ks[:1] if r is b else ks
        expected = sim.transport_error(
            [project.simulation(r.id, "hrrr-err", k) for k in ran],
            flux,
            length_scale=200.0,
        )
        got = rows.loc[r.id]
        assert got["realizations"] == expected.realizations == len(ran)
        for column in _ERROR_COLUMNS[:-1]:
            assert got[column] == pytest.approx(getattr(expected, column), rel=1e-12)
    # c has no error run and d no unperturbed run.
    for r in (c, d):
        assert rows.loc[r.id, "realizations"] == 0
        assert rows.loc[r.id, _ERROR_COLUMNS[:-1]].isna().all()
    assert errors["realizations"].dtype == np.int64

    in_threads = project.transport_error(
        sel, "hrrr-err", flux, length_scale=200.0, workers=3
    )
    pd.testing.assert_frame_equal(in_threads, errors)


def test_transport_error_takes_the_unperturbed_run_and_an_error_variant(tmp_path):
    a = _release(12)
    project = _project(
        tmp_path,
        [a],
        grid=None,
        variants={
            "hrrr": {},
            "hrrr-err": {**_XYERR, "realizations": 2},
            "zi08": {"ziscale": 0.8},
        },
    )
    sims, flux = project.simulations, _flux()
    hrrr = sims[sims.variant == "hrrr"]

    with pytest.raises(ValueError, match="one variant"):
        project.transport_error(sims, "hrrr-err", flux)
    with pytest.raises(ValueError, match="ensemble"):
        one = sims[(sims.variant == "hrrr-err") & (sims.realization == 0)]
        project.transport_error(one, "zi08", flux)
    with pytest.raises(KeyError, match="No variant 'hrrr-error'"):
        project.transport_error(hrrr, "hrrr-error", flux)
    with pytest.raises(ValueError, match="itself"):
        project.transport_error(hrrr, "hrrr", flux)

    # Nothing has run: every row is NaN.
    errors = project.transport_error(hrrr, "hrrr-err", flux)
    assert errors["realizations"].tolist() == [0]
    assert errors["variance"].isna().all()


def test_footprints_open_one_variant_as_a_dataset(tmp_path, monkeypatch):
    done, empty, missing = _release(12), _release(13), _release(14)
    project = _project(
        tmp_path,
        [done, empty, missing],
        variants={"hrrr": {}, "traj": {"grid": None}},
    )
    _write_footprint(project, done)
    _write_footprint(project, empty, empty=True)
    _write_trajectory(project, done, "traj")

    foot = project.simulation(done.id, "hrrr").footprint
    # Footprints are read from the folder listing; no release is built.
    sims = project.simulations
    monkeypatch.setattr(Project, "_releases", None)
    ds = project.footprints(sims[sims.variant == "hrrr"])

    assert isinstance(ds, xr.Dataset)
    assert list(ds.release.values) == [done.id]
    assert ds.attrs["empty"] == [empty.id]
    assert ds.attrs["missing"] == [missing.id]
    np.testing.assert_allclose(
        ds.foot.sel(release=done.id, hour=0).values,
        foot.isel(time=0).values,
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="no grid"):
        project.footprints(sims[sims.variant == "traj"])
    with pytest.raises(ValueError, match="has a footprint yet"):
        project.footprints(
            sims[(sims.release == missing.id) & (sims.variant == "hrrr")]
        )


def test_footprints_take_one_variant(tmp_path, point_release):
    project = _project(
        tmp_path, [point_release], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    for variant in ("hrrr", "zi08"):
        _write_footprint(project, point_release, variant)
    sims = project.simulations

    with pytest.raises(ValueError, match="one variant"):
        project.footprints()
    ds = project.footprints(sims[sims.variant == "zi08"])
    assert ds.attrs["stilt_name"] == "zi08"
    assert list(ds.release.values) == [point_release.id]


def test_jacobian_of_a_variant(tmp_path):
    a, b, c = _release(12), _release(13), _release(14)
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

    assert list(H.releases) == [a.id]
    assert H.empty == [b.id]
    assert H.missing == [c.id]
    assert H.data.shape == (1, len(bins) * len(target.index))
    expected = project.simulation(a.id, "hrrr").footprint.stilt.aggregate(target, bins)
    np.testing.assert_allclose(
        H.to_frame().iloc[0].to_numpy().reshape(len(bins), -1).T,
        expected.to_numpy(),
        rtol=1e-6,
    )
    some = project.jacobian(hrrr[hrrr.release.isin([b.id, a.id])], target, bins)
    assert list(some.releases) == [a.id]
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


def test_jacobian_of_an_ensemble_is_one_realization_at_a_time(tmp_path):
    a, b = _release(12), _release(13)
    project = _project(
        tmp_path,
        [a, b],
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 3}},
    )
    for k in range(3):
        _write_footprint(project, a, "err", k)
    _write_footprint(project, b, "err", 0)
    target = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.5, yres=0.5)
    bins = pd.IntervalIndex.from_breaks(
        pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="12h"),
        closed="left",
    )
    sims = project.simulations
    ens = sims[sims.variant == "err"]

    # Several realizations: the error names the column and says to pick one.
    for several in (ens, ens[ens.release == a.id]):
        with pytest.raises(ValueError, match="realization column") as error:
            project.jacobian(several, target, bins)
        assert "ens[ens.realization == 0]" in str(error.value)
        assert "[0, 1, 2]" in str(error.value)
    with pytest.raises(ValueError, match="realization column"):
        project.footprints(ens)

    # The documented loop: one Jacobian per realization, from that realization's files.
    H = {k: project.jacobian(ens[ens.realization == k], target, bins) for k in range(3)}
    assert [list(H[k].releases) for k in range(3)] == [[a.id, b.id], [a.id], [a.id]]
    assert H[1].missing == [b.id]
    for k in range(3):
        expected = project.simulation(a.id, "err", k).footprint.stilt.aggregate(
            target, bins
        )
        np.testing.assert_allclose(
            H[k].to_frame().iloc[0].to_numpy().reshape(len(bins), -1).T,
            expected.to_numpy(),
            rtol=1e-6,
        )
    assert (H[0].data[:1] != H[1].data).nnz > 0


def test_jacobian_in_threads_is_the_same_matrix(tmp_path):
    releases = [_release(h) for h in range(8, 16)]
    project = _project(tmp_path, releases, execution={"local": {"cpus": 3}})
    for r in releases[:-1]:
        _write_footprint(project, r)
    _write_footprint(project, releases[-1], empty=True)
    target = Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.5, yres=0.5)
    bins = pd.IntervalIndex.from_breaks(
        pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="3h"),
        closed="left",
    )
    whole = project.jacobian(project.simulations, target, bins, workers=1)
    batched = project.jacobian(project.simulations, target, bins, workers=3)

    assert (
        list(batched.releases) == list(whole.releases) == [r.id for r in releases[:-1]]
    )
    assert batched.empty == whole.empty == [releases[-1].id]
    assert (batched.data != whole.data).nnz == 0
    expected = [
        project.simulation(r.id, "hrrr").footprint.stilt.aggregate(target, bins)
        for r in releases[:-1]
    ]
    np.testing.assert_allclose(
        batched.data.toarray(),
        np.stack([e.to_numpy().T.ravel() for e in expected]),
        rtol=1e-6,
    )


def test_jacobian_lists_each_date_folder_once(tmp_path, monkeypatch):
    import runtraj.output as output_module

    a = _release(12)
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

    assert list(H.releases) == [a.id]
    assert listed.count("date=2023-01-01") == 1


def test_plot_availability_returns_axes(tmp_path, point_release):
    import matplotlib.pyplot as plt

    project = _project(tmp_path, [point_release])

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

    def run_releases(project, release_ids, **kwargs):
        calls.append({"project": project, "ids": list(release_ids), **kwargs})

    monkeypatch.setattr("runtraj.execution.worker.run_releases", run_releases)
    return calls


def test_run_hands_the_incomplete_releases_to_the_workers(tmp_path, ran):
    done, todo = _release(12), _release(13)
    project = _project(
        tmp_path,
        [done, todo],
        include_footprint=False,
        execution={"local": {"cpus": 3}},
    )
    _write_trajectory(project, done)

    status = project.run()

    assert list(status["release"]) == [todo.id]  # what ran, and where it stands
    assert list(status["state"]) == ["pending"]  # the fake ran nothing
    [call] = ran
    assert call["ids"] == [todo.id]
    assert call["project"] is project
    assert call["execution"] == project.config.execution
    assert call["execution"].cpus == 3
    # The runner works out the scratch directory before handing it over.
    assert call["compute_root"] == resolve_compute_root(project)
    assert call["skip_existing"] is True


def test_run_without_skip_runs_every_release_once(tmp_path, ran):
    done, todo = _release(12), _release(13)
    project = _project(
        tmp_path, [done, todo], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    _write_trajectory(project, done)

    project.run(skip_existing=False, compute_root=tmp_path / "scratch")

    [call] = ran
    assert call["ids"] == [
        done.id,
        todo.id,
    ]  # each release once, whatever its variants
    assert call["skip_existing"] is False
    assert call["compute_root"] == tmp_path / "scratch"


def test_run_takes_execution_settings_in_place_of_the_configs(tmp_path, ran):
    project = _project(tmp_path, [_release(12)], execution={"local": {"cpus": 3}})

    override = ExecutionConfig(local={"cpus": 8}, timeout=600, keep_workdir=True)
    project.run(execution=override)

    # The whole override reaches the workers, not only its cpus.
    assert ran[0]["execution"] == override


def test_run_with_nothing_to_do_starts_nothing(tmp_path, ran, point_release):
    project = _project(tmp_path, [point_release])
    _write_trajectory(project, point_release)
    _write_footprint(project, point_release, empty=True)

    assert project.run().empty
    assert ran == []


def test_run_after_adding_a_variant_runs_the_release_again(
    tmp_path, ran, point_release
):
    """A finished project grows a variant: only that variant is incomplete."""
    project = _project(tmp_path, [point_release])
    _write_trajectory(project, point_release)
    _write_footprint(project, point_release)
    assert project.run().empty

    # The user adds a variant to config.yaml.
    _config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}).to_yaml(
        project.config_path
    )
    grown = Project(project.directory)
    assert _pairs(grown.incomplete()) == [(point_release.id, "zi08")]

    grown.run()

    assert ran[0]["ids"] == [point_release.id]


def test_run_finds_a_missing_realization(tmp_path, ran, point_release):
    project = _project(
        tmp_path,
        [point_release],
        grid=None,
        krand=4,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )
    _write_trajectory(project, point_release)
    _write_trajectory(project, point_release, "err", 0)

    project.run()

    assert ran[0]["ids"] == [point_release.id]
    missing = project.incomplete()
    assert _pairs(missing) == [(point_release.id, "err")]
    assert missing["realization"].tolist() == [1]


def test_submit_needs_slurm(tmp_path, point_release):
    project = _project(tmp_path, [point_release])
    with pytest.raises(ValueError, match="slurm"):
        project.submit()


def test_releases_are_built_only_when_asked_for(tmp_path, monkeypatch):
    import runtraj.project

    a, b = _release(12, site="WBB"), _release(13, site="UOU")
    _project(tmp_path, [a, b])
    project = Project(tmp_path / "proj")
    built = []
    build = runtraj.project.releases_from_rows

    def counting(rows):
        releases = build(rows)
        built.extend(r.id for r in releases)
        return releases

    monkeypatch.setattr(runtraj.project, "releases_from_rows", counting)

    assert len(project.releases) == 2
    project.status()
    assert built == []  # listing, selecting, and status build nothing

    assert project.release(b.id) == b
    assert built == [b.id]


def test_a_bad_releases_file_fails_when_the_project_is_read(tmp_path):
    project = _project(tmp_path)
    project.releases_path.write_text(
        "time,longitude,latitude,altitude\n2023-01-01 12:00:00,-111.85,95.0,5\n"
    )

    with pytest.raises(ValueError, match="latitude must be a number within"):
        _ = Project(project.directory).releases


# ---------------------------------------------------------------------------
# A selection of simulations (#107)
# ---------------------------------------------------------------------------


def test_simulations_is_a_dataframe_and_a_copy(tmp_path):
    project = _project(tmp_path, [_release(12)])
    sims = project.simulations
    assert isinstance(sims, pd.DataFrame)
    sims["mine"] = 1  # the user's copy
    assert "mine" not in project.simulations.columns


def test_a_selection_may_be_a_frame_a_mask_or_another_table(tmp_path):
    import pyarrow as pa

    a, b = _release(12, site="WBB"), _release(13, site="UOU")
    project = _project(
        tmp_path, [a, b], variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    _write_trajectory(project, a, "zi08")
    sims = project.simulations
    expected = [(a.id, "zi08")]

    wbb = sims[(sims.site == "WBB") & (sims.variant == "zi08")]
    mask = (sims.site == "WBB") & (sims.variant == "zi08")
    table = pa.table({"release": [a.id], "variant": ["zi08"]})
    for sel in (wbb, mask, mask.to_numpy(), table):
        st = project.status(sel)
        assert _pairs(st) == expected
        assert st["particles"].tolist() == [True]
        assert st["site"].tolist() == ["WBB"]  # the rest of the row, from the project
        assert _pairs(project.incomplete(sel)) == expected  # no footprint yet
    polars = pytest.importorskip("polars")
    st = project.status(polars.DataFrame({"release": [a.id], "variant": ["zi08"]}))
    assert _pairs(st) == expected


def test_a_pandas_selection_keeps_its_own_columns(tmp_path):
    a, b = _release(12), _release(13)
    project = _project(tmp_path, [a, b])
    mine = pd.DataFrame({"release": [b.id], "obs": [1.9]})
    merged = project.simulations.merge(mine, on="release")

    st = project.status(merged)
    assert _pairs(st) == [(b.id, "hrrr")]
    assert st.obs.tolist() == [1.9]


def test_a_selection_needs_release_and_variant_columns(tmp_path):
    project = _project(tmp_path, [_release(12)])
    with pytest.raises(ValueError, match="'release' and 'variant'"):
        project.status(project.releases)
    unknown = pd.DataFrame({"release": [_release(12).id], "variant": ["nope"]})
    with pytest.raises(KeyError, match="nope"):
        project.status(unknown)
    with pytest.raises(KeyError, match="No release"):
        project.release("202301011200_-1_1_1")


def test_status_says_why_the_failed_simulations_failed(tmp_path, point_release):
    other = PointRelease(
        time="2023-07-15 19:00", longitude=-111.848, latitude=40.766, altitude=10
    )
    project = _project(
        tmp_path, [point_release, other], variants={"traj": {"grid": None}}
    )
    sim = project.simulation(str(point_release.id), "traj")
    sim.output.record_failure(
        "particles",
        sim.variant,
        sim.release.id,
        {"stage": "particles", "reason": "MET_COVERAGE", "message": "m"},
    )

    status = project.status()
    failed = status[status.state == "failed"]

    assert failed[["release", "variant", "stage", "reason"]].to_dict("records") == [
        {
            "release": str(point_release.id),
            "variant": "traj",
            "stage": "particles",
            "reason": "MET_COVERAGE",
        }
    ]
    by_release = status.set_index("release")
    assert by_release.loc[str(other.id), "state"] == "pending"
    assert pd.isna(by_release.loc[str(other.id), "reason"])


# ---------------------------------------------------------------------------
# Input tables
# ---------------------------------------------------------------------------


def test_add_table_keeps_a_release_table_and_adds_only_new_releases(tmp_path):
    from runtraj.transforms import averaging_kernel_table

    a, b = _release(12), _release(13)
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
    assert list(held.release) == [a.id, a.id, b.id, b.id]
    assert list(held.value) == [1.0, 1.0, 0.5, 0.5]
    with pytest.raises(ValueError, match="columns"):
        project.add_table("kernels", pd.DataFrame({"release": [b.id]}))
    with pytest.raises(ValueError, match="table name"):
        project.add_table("../kernels", first)


def test_a_table_with_the_old_receptor_column_says_it_is_now_release(tmp_path):
    a = _release(12)
    project = _project(tmp_path, [a])
    kernels = pd.DataFrame(
        {"receptor": [a.id, a.id], "level": [0.0, 3000.0], "value": [1.0, 1.0]}
    )
    project.add_table("kernels", kernels)  # a table of the old form, as it was kept
    new = kernels.rename(columns={"receptor": "release"})
    with pytest.raises(ValueError, match="now named 'release'"):
        project.add_table("kernels", new)


def test_a_selection_with_the_old_receptor_column_says_it_is_now_release(tmp_path):
    a = _release(12)
    project = _project(tmp_path, [a])
    old = project.simulations.rename(columns={"release": "receptor"})
    with pytest.raises(ValueError, match="now named 'release'"):
        project.status(old)
