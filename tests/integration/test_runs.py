"""
Integration tests for PYSTILT end-to-end execution.

These tests run real HYSPLIT against real met files and are collected by
default.  They are automatically skipped when the stilt-tutorials met data
is not present (the ``met_dir`` fixture handles this).

Run integration tests explicitly:
    pytest tests/ -v -m integration

Skip them:
    pytest tests/ -m "not integration"
"""

import numpy as np
import pandas as pd
import pytest
from runtraj.config import ProjectConfig
from runtraj.identity import transport_from_settings
from runtraj.models.hysplit.driver import winderrtf
from runtraj.particles import particles_metadata

from stilt import Project
from stilt.footprint.targets import Mesh
from stilt.project import STILT_R_HYSPLIT
from tests.fixtures.factories import blocks

from ..conftest import integration
from .conftest import reference_met

_XYERR = {"siguverr": 2.0, "tluverr": 60.0, "zcoruverr": 500.0, "horcoruverr": 40.0}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_id(release, variant: str = "hrrr") -> tuple[str, str]:
    return (release.id, variant)


def _incomplete(project: Project) -> list[tuple[str, str]]:
    rows = project.incomplete()
    return list(zip(rows["release"], rows["variant"], strict=True))


def _with(
    config: ProjectConfig,
    *,
    hysplit: dict | None = None,
    footprint: dict | None = None,
    **updates,
) -> ProjectConfig:
    """
    A validated copy of *config* with *updates* applied.

    *hysplit* changes settings of the ``hysplit`` block under ``models``,
    and *footprint* of the ``footprint`` block under ``stages``.
    """
    data = config.model_dump()
    if hysplit:
        data["models"]["hysplit"] = {**data["models"]["hysplit"], **hysplit}
    if footprint:
        data["stages"]["footprint"] = {**data["stages"]["footprint"], **footprint}
    return ProjectConfig.model_validate({**data, **updates})


# ---------------------------------------------------------------------------
# Point release - trajectory only
# ---------------------------------------------------------------------------


@integration
def test_the_functions_make_what_a_project_stores(tmp_path, wbb_release, wbb_config):
    """run_trajectories and calc_footprint give a seeded project run's particles and footprint."""
    import stilt

    config = _with(wbb_config, hysplit={"krand": 2, "seed": 7})
    project = Project.init(tmp_path / "p", config=config, releases=[wbb_release])
    project.run()
    sim = project.simulation(*_sim_id(wbb_release))

    # Project.init wrote STILT-R's HYSPLIT values; run_trajectories takes HYSPLIT's own.
    settings = {**STILT_R_HYSPLIT, "n_hours": -6, "numpar": 100, "krand": 2, "seed": 7}
    particles = stilt.run_trajectories(wbb_release, config.mets["hrrr"], **settings)
    stored = sim.particles
    columns = [c for c in particles.columns if c in stored.columns]
    pd.testing.assert_frame_equal(
        particles[columns].reset_index(drop=True),
        stored[columns].reset_index(drop=True),
        check_dtype=False,
    )
    grid = project.variants["hrrr"].stages.get("footprint").grid
    foot = stilt.calc_footprint(particles, wbb_release, grid)
    assert sim.footprint is not None
    np.testing.assert_allclose(foot.values, sim.footprint.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# Point release - trajectory + footprint
# ---------------------------------------------------------------------------


@integration
def test_footprint(tmp_path, wbb_release, wbb_config):
    """Full run produces a readable footprint NetCDF with (time, lat, lon) dims."""
    model = Project.init(
        tmp_path / "footprint",
        config=wbb_config,
        releases=[wbb_release],
    )
    model.run()

    sim = model.simulation(*_sim_id(wbb_release))
    assert sim.footprint_path is not None and sim.footprint_path.exists()
    assert sim.has_footprint
    assert sim.footprint is not None
    assert {"time", "lat", "lon"} <= set(sim.footprint.dims)
    assert float(sim.footprint.sum()) > 0


@integration
def test_empty_footprint(tmp_path, wbb_release, wbb_config):
    """A grid the particles never reach leaves a marker with the reason, no NetCDF."""
    far_grid = {
        "xmin": -100.0,
        "xmax": -99.0,
        "ymin": 30.0,
        "ymax": 31.0,
        "xres": 0.1,
        "yres": 0.1,
    }
    model = Project.init(
        tmp_path / "empty",
        config=_with(wbb_config, footprint={"grid": far_grid}),
        releases=[wbb_release],
    )
    model.run()

    sim = model.simulation(*_sim_id(wbb_release))
    assert sim.is_complete
    assert sim.has_particles
    assert sim.has_footprint
    assert sim.footprint is None
    ds = model.footprints()
    assert ds.sizes["release"] == 0
    assert ds.attrs["empty"] == [wbb_release.id]
    # An empty footprint is complete.
    assert model.status()["state"].tolist() == ["complete"]

    # A rerun has nothing to do and does not touch the empty record.
    before = sim.footprint_path.stat().st_mtime_ns
    model.run()
    assert sim.footprint_path.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# Failure path - missing met files
# ---------------------------------------------------------------------------


@integration
def test_failure_missing_met(tmp_path, wbb_release, traj_only_config, met_dir):
    """A simulation whose met files are missing fails alone, naming the hours."""
    # The met holds a summer file only, so the winter release's hours have none.
    summer_only = tmp_path / "summer_only"
    summer_only.mkdir()
    summer = next(met_dir.glob("20210715_00-05*"))
    (summer_only / summer.name).symlink_to(summer)

    bad_config = _with(
        traj_only_config,
        mets={
            "hrrr": {
                "directory": summer_only,
            }
        },
    )
    model = Project.init(
        tmp_path / "fail_missing_met",
        config=bad_config,
        releases=[wbb_release],
    )
    # run() must not raise: a per-simulation failure is captured, not fatal.
    model.run()

    sim = model.simulation(*_sim_id(wbb_release))
    # The particles are absent (incomplete), and the worker recorded why.
    assert not sim.has_particles
    assert sim.failure is not None
    assert sim.failure["reason"] == "MET_COVERAGE"
    assert "No met file" in sim.failure["message"]
    status = model.status()
    assert status["state"].tolist() == ["failed"]
    assert status["reason"].tolist() == ["MET_COVERAGE"]


@integration
def test_particles_that_leave_the_met_domain_complete_the_run(
    tmp_path, traj_only_config
):
    """A January night at WBB empties the test met's 4 by 3 degree crop in about 11 hours (#189)."""
    from runtraj.releases import PointRelease

    # Unseeded, the last particle left after 10 to 12 hours in most runs, but
    # one of the 100 stayed all 24 hours in some (#529), so the turbulence is
    # seeded: the same particles every run, and every one leaves.
    release = PointRelease(
        time="2021-01-15 06:00", longitude=-111.848, latitude=40.766, altitude=10
    )
    config = _with(traj_only_config, hysplit={"n_hours": -24, "krand": 2, "seed": 7})
    project = Project.init(tmp_path / "domain_exit", config=config, releases=[release])
    project.run()

    sim = project.simulation(*_sim_id(release))
    assert sim.is_complete
    assert sim.failure is None
    hours = sim.particles["age"].abs().max() / 60
    assert hours < 24  # every particle left the met's domain first


def _project_on_met_cut_short(tmp_path, wbb_release, traj_only_config, met_dir):
    """A 12 h project whose 18-23 Z met file on the 14th holds only its first hour."""
    cut_met = tmp_path / "met"
    cut_met.mkdir()
    for path in met_dir.iterdir():
        (cut_met / path.name).symlink_to(path)
    # 18-23 Z on the 14th is 12 to 7 h before the release. Its six hourly
    # periods are the same size, so the first sixth is the 18 Z period alone.
    damaged = next(cut_met.glob("20210114_18-23*"))
    data = damaged.read_bytes()
    period = len(data) // 6
    assert data[period + 14 : period + 18] == b"INDX"
    damaged.unlink()
    damaged.write_bytes(data[:period])
    config = _with(
        traj_only_config,
        hysplit={"n_hours": -12},
        mets={
            "hrrr": {
                "directory": cut_met,
            }
        },
    )
    return Project.init(
        tmp_path / "met_cut_short", config=config, releases=[wbb_release]
    )


def _skip_the_met_checks(monkeypatch, tmp_path) -> None:
    """Give HYSPLIT the run's files, the one cut short too, as if nothing had looked at them first."""
    from runtraj.models.hysplit import Met

    blocks = ("20210114_18-23*", "20210115_00-05*", "20210115_06-11*")
    files = [next((tmp_path / "met").glob(block)).resolve() for block in blocks]
    monkeypatch.setattr(Met, "files_for", lambda self, window: files)
    monkeypatch.setattr(Met, "check", lambda self, *args, **kwargs: None)


@integration
def test_failure_met_cut_short(tmp_path, wbb_release, traj_only_config, met_dir):
    """A met file cut after its first hour fails the run before HYSPLIT starts, naming the gap (#169, #189)."""
    project = _project_on_met_cut_short(
        tmp_path, wbb_release, traj_only_config, met_dir
    )
    project.run()

    sim = project.simulation(*_sim_id(wbb_release))
    assert not sim.has_particles
    assert sim.failure is not None and sim.failure["reason"] == "MET_COVERAGE"
    # The hours no file holds: the cut file still has 18:00.
    assert "for 2021-01-14 19:00 to 2021-01-14 23:00" in sim.failure["message"]


@integration
def test_a_run_whose_met_runs_out_fails_on_any_hysplit_build(
    tmp_path, wbb_release, traj_only_config, met_dir, monkeypatch
):
    """
    Without the checks before the run, a run whose met runs out fails (#189, #213).

    Depending on the HYSPLIT build and the turbulence draws, HYSPLIT either
    warns ``no more meteorology`` or stops every particle and reports a
    complete run, which the driver fails by where the particles ended. Which
    one a seed gives differs between builds, so the test asks only for what
    both give. The unit tests of runtraj's driver hold the two messages.
    """
    config = _with(traj_only_config, hysplit={"krand": 2, "seed": 7})
    project = _project_on_met_cut_short(tmp_path, wbb_release, config, met_dir)
    _skip_the_met_checks(monkeypatch, tmp_path)
    project.run()

    sim = project.simulation(*_sim_id(wbb_release))
    assert not sim.has_particles
    assert sim.failure is not None and sim.failure["reason"] == "MET_COVERAGE"
    message = sim.failure["message"]
    assert "no more meteorology" in message or "the met ran out" in message
    assert sim.log_path is not None
    assert "Only one time period of meteo data" in sim.log_path.read_text()


# ---------------------------------------------------------------------------
# Idempotency - skip_existing=True
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Column and multipoint releases
# ---------------------------------------------------------------------------


@integration
def test_column(tmp_path, wbb_column_release, wbb_config):
    """Column release produces trajectory and footprint; its id spells out the column."""
    model = Project.init(
        tmp_path / "column",
        config=wbb_config,
        releases=[wbb_column_release],
    )
    model.run()

    sid = _sim_id(wbb_column_release)
    r = wbb_column_release
    assert sid[0].endswith(f"_X{r.bottom:g}-{r.top:g}"), (
        f"Expected a column release id, got {sid}"
    )

    sim = model.simulation(*sid)
    assert sim.has_particles
    assert sim.has_footprint


@integration
def test_multipoint(tmp_path, wbb_multipoint_release, multipoint_config):
    """Multipoint release (3 locations) produces trajectory and footprint."""
    model = Project.init(
        tmp_path / "multipoint",
        config=multipoint_config,
        releases=[wbb_multipoint_release],
    )
    model.run()

    sid = _sim_id(wbb_multipoint_release)
    assert "multi_" in sid[0], f"Expected a multipoint release id, got {sid}"

    sim = model.simulation(*sid)
    assert sim.has_particles
    assert sim.has_footprint


# ---------------------------------------------------------------------------
# A second footprint from the same particles
# ---------------------------------------------------------------------------


@integration
def test_footprint_only_variant_rasterizes_the_same_particles(
    tmp_path, wbb_release, multifoot_config
):
    """A variant that changes only the grid shares hrrr's particles and runs no HYSPLIT."""
    model = Project.init(
        tmp_path / "multifoot",
        config=multifoot_config,
        releases=[wbb_release],
    )
    model.run()

    fine = model.simulation(*_sim_id(wbb_release))
    coarse = model.simulation(*_sim_id(wbb_release, "coarse"))

    assert fine.has_footprint and coarse.has_footprint
    assert coarse.particles_path == fine.particles_path  # one set of particles
    assert len(model.output.hashes("particles")) == 1
    assert len(model.output.hashes("footprint")) == 2
    assert coarse.footprint is not None and fine.footprint is not None
    assert coarse.footprint.stilt.grid.xres == 0.05
    assert fine.footprint.stilt.grid.xres == 0.01
    # Same particles, so the same total sensitivity inside the shared domain order.
    assert float(coarse.footprint.sum()) > 0


@integration
def test_adding_a_footprint_only_variant_runs_no_hysplit(
    tmp_path, wbb_release, wbb_config, wbb_grid
):
    """A finished project grows a footprint-only variant; the particles are reused."""
    project = tmp_path / "grow"
    first = Project.init(project, config=wbb_config, releases=[wbb_release])
    first.run()
    base = first.simulation(*_sim_id(wbb_release))
    traj_mtime = base.particles_path.stat().st_mtime
    log_before = base.log_path.read_text()

    # The user adds a variant to config.yaml.
    s2 = {"stages": {"footprint": {"smooth_factor": 2}}}
    _with(wbb_config, variants={"hrrr": {}, "s2": s2}).to_yaml(project / "config.yaml")
    grown = Project(project)
    assert _incomplete(grown) == [_sim_id(wbb_release, "s2")]
    grown.run()

    assert base.particles_path.stat().st_mtime == traj_mtime
    assert base.log_path.read_text() == log_before
    assert grown.incomplete().empty


@integration
def test_pbl_height_is_the_footprints_and_shares_the_run(
    tmp_path, wbb_release, wbb_config
):
    """Variants that differ only in how PYSTILT computes foot share one HYSPLIT run."""
    richardson = {
        "stages": {"footprint": {"pbl_height": "richardson", "pbl_height_min": 150}}
    }
    config = _with(wbb_config, variants={"hrrr": {}, "rich": richardson})
    project = Project.init(tmp_path / "pbl", config=config, releases=[wbb_release])
    project.run()

    own = project.simulation(*_sim_id(wbb_release))
    computed = project.simulation(*_sim_id(wbb_release, "rich"))
    assert project.incomplete().empty
    assert computed.particles_path == own.particles_path
    assert len(project.output.hashes("footprint")) == 2
    # The particle file holds HYSPLIT's foot; the computed one is stored nowhere.
    stored = pd.read_parquet(own.particles_path, columns=["foot"])["foot"]
    assert stored.equals(own.particles["foot"])
    assert own.footprint is not None and computed.footprint is not None
    assert float(abs(own.footprint - computed.footprint).sum()) > 0
    # The stored footprint is the one the simulation makes again.
    again = computed.calc_footprint()
    assert again is not None
    assert float(again.sum()) == pytest.approx(
        float(computed.footprint.sum()), rel=1e-6
    )


# ---------------------------------------------------------------------------
# CLI - stilt run via CliRunner
# ---------------------------------------------------------------------------


@integration
def test_cli_run(tmp_path, wbb_config, wbb_release):
    """CLI `stilt run` produces trajectory and footprint artifacts."""
    from runtraj.cli import app
    from typer.testing import CliRunner

    project_dir = tmp_path / "cli_project"
    project_dir.mkdir()

    wbb_config.to_yaml(project_dir / "config.yaml")

    (project_dir / "releases.csv").write_text(
        "time,lati,long,zagl\n2021-01-15 06:00:00,40.5,-112.0,5.0\n"
    )

    result = CliRunner().invoke(app, ["run", str(project_dir)])

    assert result.exit_code == 0, (
        f"stilt run exited {result.exit_code}\nOutput:\n{result.output}"
    )
    assert "complete=1" in result.output

    sim = Project(project_dir).simulation(*_sim_id(wbb_release))
    assert sim.has_particles, "CLI run: no particles"
    assert sim.has_footprint, "CLI run: no footprint"
    assert (project_dir / "output" / "particles").is_dir()


# ---------------------------------------------------------------------------
# Wind-error variants
# ---------------------------------------------------------------------------


@integration
def test_error_variant(tmp_path, wbb_release, traj_only_config):
    """A variant with WINDERR fields is its own HYSPLIT run with perturbed winds."""
    config = _with(
        traj_only_config,
        execution={"keep_workdir": True},
        variants={"hrrr": {}, "hrrr-err": _XYERR},
    )
    model = Project.init(tmp_path / "error", config=config, releases=[wbb_release])
    model.run()

    main = model.simulation(*_sim_id(wbb_release))
    err = model.simulation(*_sim_id(wbb_release, "hrrr-err"))
    err_scratch = err.kept_workdir
    main_scratch = main.kept_workdir

    assert (err_scratch / "WINDERR").exists()
    assert not (main_scratch / "WINDERR").exists()
    assert "winderrtf=1" in (err_scratch / "SETUP.CFG").read_text().lower()

    main_traj = main.particles
    error_traj = err.particles
    assert len(main_traj) > 0 and len(error_traj) > 0
    assert set(main_traj.columns) == set(error_traj.columns)
    assert (
        not main_traj["lon"]
        .reset_index(drop=True)
        .equals(error_traj["lon"].reset_index(drop=True))
    ), "Error trajectory identical to main — wind perturbation had no effect"
    settings = particles_metadata(err.particles_path).settings
    assert winderrtf(transport_from_settings(settings)) == 1


@integration
def test_error_realizations(tmp_path, wbb_release, traj_only_config):
    """Two realizations run, differ from each other, and resume one at a time."""
    config = _with(
        traj_only_config,
        hysplit={"krand": 4},  # HYSPLIT seeds each run from the clock
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )
    model = Project.init(
        tmp_path / "realizations", config=config, releases=[wbb_release]
    )
    model.run()

    sims = model.simulations
    sims = sims[sims.variant == "err"]
    assert sims["realization"].tolist() == [0, 1]
    assert model.incomplete(sims).empty

    particles = model.particles(sims)
    e0, e1 = (particles[particles.realization == k] for k in (0, 1))
    s0 = e0.groupby("particle")["foot"].sum()
    s1 = e1.groupby("particle")["foot"].sum().reindex(s0.index)
    assert not np.allclose(s0.to_numpy(), s1.to_numpy())

    # Resume: drop one realization; only it reruns.
    main = model.simulation(*_sim_id(wbb_release))
    err0 = model.simulation(*_sim_id(wbb_release, "err"), 0)
    err1 = model.simulation(*_sim_id(wbb_release, "err"), 1)
    main_bytes = main.particles_path.read_bytes()
    err0_bytes = err0.particles_path.read_bytes()
    err1.particles_path.unlink()
    assert _incomplete(model) == [err1.id[:2]]

    model.run(skip_existing=True)

    assert main.particles_path.read_bytes() == main_bytes
    assert err0.particles_path.read_bytes() == err0_bytes
    assert err1.particles_path.exists()


@integration
def test_seeded_error_realizations_differ_and_reproduce(
    tmp_path, wbb_release, traj_only_config
):
    """krand=2 with a seed: realizations differ and a rerun is bit-identical."""
    config = _with(
        traj_only_config,
        hysplit={"krand": 2, "seed": 7},
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )

    def run(project):
        model = Project.init(project, config=config, releases=[wbb_release])
        model.run()
        assert model.incomplete().empty
        return model

    a = run(tmp_path / "a")
    sims = a.simulations
    particles = a.particles(sims[sims.variant == "err"])
    e0, e1 = (particles[particles.realization == k] for k in (0, 1))
    s0 = e0.groupby("particle")["foot"].sum()
    s1 = e1.groupby("particle")["foot"].sum().reindex(s0.index)
    assert not np.allclose(s0.to_numpy(), s1.to_numpy())
    main = a.simulation(*_sim_id(wbb_release)).particles
    s_main = main.groupby("particle")["foot"].sum().reindex(s0.index)
    assert not np.allclose(s_main.to_numpy(), s0.to_numpy())

    b = run(tmp_path / "b")
    for variant, k in (("hrrr", None), ("err", 0), ("err", 1)):
        pd.testing.assert_frame_equal(
            a.simulation(*_sim_id(wbb_release, variant), k).particles,
            b.simulation(*_sim_id(wbb_release, variant), k).particles,
        )


# ---------------------------------------------------------------------------
# Geometry-derived footprint config
# ---------------------------------------------------------------------------


@integration
def test_geometry_footprint(tmp_path, wbb_release, met_dir):
    """A footprint named by geometry derives its raster, runs, and aggregates."""
    from ..fixtures.r_stilt_reference import (
        REFERENCE_KRAND,
        REFERENCE_SEED,
    )

    # Half-degree windows so particles stay in the derived raster for hours.
    spec = {
        "kind": "windows",
        "coords": [(-112.0, 40.5), (-112.6, 40.9)],
        "size": 0.5,
        "ids": ["wbb", "nw"],
    }
    config = ProjectConfig(
        **blocks(
            mets={"hrrr": reference_met(met_dir)},
            n_hours=-6,
            numpar=100,
            krand=REFERENCE_KRAND,
            seed=REFERENCE_SEED,
            geometry=spec,
            cells_per_target=10,
            variants={"hrrr": {}},
        )
    )
    variant = next(iter(config.resolve().values()))
    fc = variant.stages.get("footprint")
    assert fc is not None
    assert fc.grid.xres == fc.grid.yres == 0.05  # 0.5 / 10
    assert variant.stages["footprint"].geometry_hash

    model = Project.init(tmp_path / "geom", config=config, releases=[wbb_release])
    model.run()

    sim = model.simulation(*_sim_id(wbb_release))
    assert sim.has_footprint
    foot = sim.footprint
    assert foot is not None
    assert foot.stilt.config.grid == fc.grid
    assert foot.stilt.config.geometry == fc.geometry
    assert foot.stilt.geometry_hash == variant.stages["footprint"].geometry_hash

    assert fc.geometry is not None
    mesh = Mesh.from_spec(fc.geometry)
    r_time = pd.Timestamp(wbb_release.time)
    bins = pd.interval_range(
        start=r_time - pd.Timedelta(hours=6), end=r_time, freq="1h", closed="left"
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        agg = foot.stilt.aggregate(mesh, bins)
    assert agg.index.tolist() == ["wbb", "nw"]
    assert agg.loc["wbb"].sum() > 0  # the release sits inside its own window
    assert agg.to_numpy().sum() <= float(foot.sum()) + 1e-12


# ---------------------------------------------------------------------------
# Forward run (n_hours > 0)
# ---------------------------------------------------------------------------


@integration
def test_forward_run(tmp_path, met_dir, wbb_grid):
    """A forward simulation runs end to end and carries a forward time axis."""
    from runtraj.releases import PointRelease

    from ..fixtures.r_stilt_reference import (
        REFERENCE_ALTITUDE,
        REFERENCE_KRAND,
        REFERENCE_LATITUDE,
        REFERENCE_LONGITUDE,
        REFERENCE_SEED,
        REFERENCE_SUMMER_TIME,
    )

    # The 2021-07-15 06:00-11:59 block holds the whole +3 h window.
    release = PointRelease(
        time=REFERENCE_SUMMER_TIME,
        longitude=REFERENCE_LONGITUDE,
        latitude=REFERENCE_LATITUDE,
        altitude=REFERENCE_ALTITUDE,
    )
    config = ProjectConfig(
        **blocks(
            mets={"hrrr": reference_met(met_dir)},
            n_hours=3,
            numpar=100,
            krand=REFERENCE_KRAND,
            seed=REFERENCE_SEED,
            hnf_plume=True,
            grid=wbb_grid,
            variants={"hrrr": {}},
        )
    )

    model = Project.init(tmp_path / "forward", config=config, releases=[release])
    model.run()

    sim = model.simulation(*_sim_id(release))
    assert sim.has_particles, f"no trajectory for {sim.id}"
    assert sim.particles is not None

    particles = sim.particles
    assert len(particles) > 0
    # HYSPLIT reports elapsed minutes signed by run direction
    assert (particles["age"] >= 0).all()
    assert particles["age"].max() > 0
    # Stored uncorrected: the footprint applies the near-field correction.
    assert "foot_no_hnf_dilution" not in particles.columns

    start, stop = release.time, release.time + pd.Timedelta(hours=3)

    assert sim.has_footprint, f"no footprint for {sim.id}"
    foot = sim.footprint
    assert foot is not None
    times = pd.DatetimeIndex(foot["time"].values)
    # hourly layers running forward, inside the run's window
    assert times.is_monotonic_increasing
    assert times.min() >= pd.Timestamp(start)
    assert times.max() <= pd.Timestamp(stop)
    assert set(times.to_series().diff().dropna()) == {pd.Timedelta(hours=1)}
    assert float(foot.sum()) > 0
