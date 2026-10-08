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

from stilt.config import ProjectConfig
from stilt.execution.runner import resolve_compute_root
from stilt.footprint.targets import Mesh
from stilt.identity import transport_from_settings
from stilt.particles import particles_metadata
from stilt.project import Project
from stilt.transport.hysplit.driver import winderrtf

from ..conftest import integration
from .conftest import reference_met

_XYERR = {"siguverr": 2.0, "tluverr": 60.0, "zcoruverr": 500.0, "horcoruverr": 40.0}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_id(receptor, variant: str = "hrrr") -> tuple[str, str]:
    return (receptor.id, variant)


def _incomplete(project: Project) -> list[tuple[str, str]]:
    rows = project.incomplete()
    return list(zip(rows["receptor"], rows["variant"], strict=True))


def _with(config: ProjectConfig, **updates) -> ProjectConfig:
    """A validated copy of *config* with *updates* applied."""
    return ProjectConfig.model_validate({**config.model_dump(), **updates})


# ---------------------------------------------------------------------------
# Point receptor - trajectory only
# ---------------------------------------------------------------------------


@integration
def test_particles(tmp_path, wbb_receptor, traj_only_config):
    """HYSPLIT runs and produces a non-empty trajectory parquet."""
    model = Project.init(
        tmp_path / "trajectory",
        config=traj_only_config,
        receptors=[wbb_receptor],
    )
    model.run()

    sid = _sim_id(wbb_receptor)
    assert list(model.simulations["receptor"]) == [sid[0]]
    sim = model.simulation(*sid)
    assert sim.has_particles
    assert sim.particles_path is not None
    assert sim.particles_path.parent.name == "date=2021-01-15"
    assert sim.particles_path.parent.parent.name.startswith("settings=hrrr-")
    assert len(pd.read_parquet(sim.particles_path)) > 0, "Particle file is empty"
    assert sim.particles is not None and len(sim.particles) > 0
    assert not (resolve_compute_root(model) / sid[0] / sid[1]).exists(), (
        "the scratch working directory is removed"
    )

    log_text = sim.log
    for phrase in ("FATAL ERROR", "Segmentation fault", "hycs_std: not found"):
        assert phrase not in log_text, f"Fatal phrase in log: {phrase!r}"


@integration
def test_the_functions_make_what_a_project_stores(tmp_path, wbb_receptor, wbb_config):
    """run_trajectories and calc_footprint give a seeded project run's particles and footprint."""
    import stilt

    config = _with(wbb_config, krand=2, seed=7)
    project = Project.init(tmp_path / "p", config=config, receptors=[wbb_receptor])
    project.run()
    sim = project.simulation(*_sim_id(wbb_receptor))

    particles = stilt.run_trajectories(
        wbb_receptor, config.mets["hrrr"], n_hours=-6, numpar=100, krand=2, seed=7
    )
    stored = sim.particles
    columns = [c for c in particles.columns if c in stored.columns]
    pd.testing.assert_frame_equal(
        particles[columns].reset_index(drop=True),
        stored[columns].reset_index(drop=True),
        check_dtype=False,
    )
    assert config.footprint.grid is not None
    foot = stilt.calc_footprint(particles, wbb_receptor, config.footprint.grid)
    assert sim.footprint is not None
    np.testing.assert_allclose(foot.values, sim.footprint.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# Point receptor - trajectory + footprint
# ---------------------------------------------------------------------------


@integration
def test_footprint(tmp_path, wbb_receptor, wbb_config):
    """Full run produces a readable footprint NetCDF with (time, lat, lon) dims."""
    model = Project.init(
        tmp_path / "footprint",
        config=wbb_config,
        receptors=[wbb_receptor],
    )
    model.run()

    sim = model.simulation(*_sim_id(wbb_receptor))
    assert sim.footprint_path is not None and sim.footprint_path.exists()
    assert sim.has_footprint
    assert sim.footprint is not None
    assert {"time", "lat", "lon"} <= set(sim.footprint.dims)
    assert float(sim.footprint.sum()) > 0


@integration
def test_empty_footprint(tmp_path, wbb_receptor, wbb_config):
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
        config=_with(wbb_config, grid=far_grid),
        receptors=[wbb_receptor],
    )
    model.run()

    sim = model.simulation(*_sim_id(wbb_receptor))
    assert sim.is_complete
    assert sim.has_particles
    assert sim.has_footprint
    assert sim.footprint is None
    ds = model.footprints()
    assert ds.sizes["receptor"] == 0
    assert ds.attrs["empty"] == [wbb_receptor.id]
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
def test_failure_missing_met(tmp_path, wbb_receptor, traj_only_config, met_dir):
    """A simulation whose met files are missing fails alone, naming the hours."""
    # The met holds a summer file only, so the winter receptor's hours have none.
    summer_only = tmp_path / "summer_only"
    summer_only.mkdir()
    summer = next(met_dir.glob("20210715_00-05*"))
    (summer_only / summer.name).symlink_to(summer)
    met = traj_only_config.mets["hrrr"]

    bad_config = _with(
        traj_only_config,
        mets={
            "hrrr": {
                "directory": summer_only,
                "file_format": met["file_format"],
                "file_tres": met["file_tres"],
            }
        },
    )
    model = Project.init(
        tmp_path / "fail_missing_met",
        config=bad_config,
        receptors=[wbb_receptor],
    )
    # run() must not raise: a per-simulation failure is captured, not fatal.
    model.run()

    sim = model.simulation(*_sim_id(wbb_receptor))
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
    from stilt.receptors import PointReceptor

    # The last particle left after 10 to 12 hours in 24 runs. From the
    # reference receptor, 30 km south-west, one of the 100 particles stayed
    # all 24 hours in about a third of runs.
    receptor = PointReceptor(
        time="2021-01-15 06:00", longitude=-111.848, latitude=40.766, altitude=10
    )
    config = _with(traj_only_config, n_hours=-24)
    project = Project.init(
        tmp_path / "domain_exit", config=config, receptors=[receptor]
    )
    project.run()

    sim = project.simulation(*_sim_id(receptor))
    assert sim.is_complete
    assert sim.failure is None
    hours = sim.particles["time"].abs().max() / 60
    assert hours < 24  # every particle left the met's domain first


@integration
def test_failure_met_cut_short(tmp_path, wbb_receptor, traj_only_config, met_dir):
    """A met file cut to one time period stops the particles early, which fails the run (#169, #189)."""
    cut_met = tmp_path / "met"
    cut_met.mkdir()
    for path in met_dir.iterdir():
        (cut_met / path.name).symlink_to(path)
    # 18-23 Z on the 14th is 12 to 7 h before the receptor. Its six hourly
    # periods are the same size, so the first sixth is the 18 Z period alone.
    damaged = next(cut_met.glob("20210114_18-23*"))
    data = damaged.read_bytes()
    period = len(data) // 6
    assert data[period + 14 : period + 18] == b"INDX"
    damaged.unlink()
    damaged.write_bytes(data[:period])

    met = traj_only_config.mets["hrrr"]
    config = _with(
        traj_only_config,
        n_hours=-12,
        mets={
            "hrrr": {
                "directory": cut_met,
                "file_format": met["file_format"],
                "file_tres": met["file_tres"],
            }
        },
    )
    project = Project.init(
        tmp_path / "met_cut_short", config=config, receptors=[wbb_receptor]
    )
    project.run()

    sim = project.simulation(*_sim_id(wbb_receptor))
    assert not sim.has_particles
    assert sim.failure is not None and sim.failure["reason"] == "MET_COVERAGE"
    assert "no more meteorology" in sim.failure["message"]
    assert sim.log_path is not None
    assert "Only one time period of meteo data" in sim.log_path.read_text()


# ---------------------------------------------------------------------------
# Idempotency - skip_existing=True
# ---------------------------------------------------------------------------


@integration
def test_idempotency(tmp_path, wbb_receptor, traj_only_config):
    """Second run with skip_existing=True does not overwrite existing output."""
    model = Project.init(
        tmp_path / "idempotency",
        config=traj_only_config,
        receptors=[wbb_receptor],
    )

    model.run()
    sim = model.simulation(*_sim_id(wbb_receptor))
    assert sim.has_particles
    mtime_before = sim.particles_path.stat().st_mtime

    model.run()  # skip_existing=True is the default
    assert sim.particles_path.stat().st_mtime == mtime_before, (
        "Parquet was overwritten on second run"
    )


# ---------------------------------------------------------------------------
# Column and multipoint receptors
# ---------------------------------------------------------------------------


@integration
def test_column(tmp_path, wbb_column_receptor, wbb_config):
    """Column receptor produces trajectory and footprint; its id spells out the column."""
    model = Project.init(
        tmp_path / "column",
        config=wbb_config,
        receptors=[wbb_column_receptor],
    )
    model.run()

    sid = _sim_id(wbb_column_receptor)
    r = wbb_column_receptor
    assert sid[0].endswith(f"_X{r.bottom:g}-{r.top:g}"), (
        f"Expected a column receptor id, got {sid}"
    )

    sim = model.simulation(*sid)
    assert sim.has_particles
    assert sim.has_footprint


@integration
def test_multipoint(tmp_path, wbb_multipoint_receptor, multipoint_config):
    """Multipoint receptor (3 locations) produces trajectory and footprint."""
    model = Project.init(
        tmp_path / "multipoint",
        config=multipoint_config,
        receptors=[wbb_multipoint_receptor],
    )
    model.run()

    sid = _sim_id(wbb_multipoint_receptor)
    assert "multi_" in sid[0], f"Expected a multipoint receptor id, got {sid}"

    sim = model.simulation(*sid)
    assert sim.has_particles
    assert sim.has_footprint


# ---------------------------------------------------------------------------
# A second footprint from the same particles
# ---------------------------------------------------------------------------


@integration
def test_footprint_only_variant_rasterizes_the_same_particles(
    tmp_path, wbb_receptor, multifoot_config
):
    """A variant that changes only the grid shares hrrr's particles and runs no HYSPLIT."""
    model = Project.init(
        tmp_path / "multifoot",
        config=multifoot_config,
        receptors=[wbb_receptor],
    )
    model.run()

    fine = model.simulation(*_sim_id(wbb_receptor))
    coarse = model.simulation(*_sim_id(wbb_receptor, "coarse"))

    assert fine.has_footprint and coarse.has_footprint
    assert coarse.particles_path == fine.particles_path  # one set of particles
    assert len(model.output.hashes("particles")) == 1
    assert len(model.output.hashes("footprints")) == 2
    assert coarse.footprint is not None and fine.footprint is not None
    assert coarse.footprint.stilt.grid.xres == 0.05
    assert fine.footprint.stilt.grid.xres == 0.01
    # Same particles, so the same total sensitivity inside the shared domain order.
    assert float(coarse.footprint.sum()) > 0


@integration
def test_adding_a_footprint_only_variant_runs_no_hysplit(
    tmp_path, wbb_receptor, wbb_config, wbb_grid
):
    """A finished project grows a footprint-only variant; the particles are reused."""
    project = tmp_path / "grow"
    first = Project.init(project, config=wbb_config, receptors=[wbb_receptor])
    first.run()
    base = first.simulation(*_sim_id(wbb_receptor))
    traj_mtime = base.particles_path.stat().st_mtime
    log_before = base.log_path.read_text()

    # The user adds a variant to config.yaml.
    _with(wbb_config, variants={"hrrr": {}, "s2": {"smooth_factor": 2}}).to_yaml(
        project / "config.yaml"
    )
    grown = Project(project)
    assert _incomplete(grown) == [_sim_id(wbb_receptor, "s2")]
    grown.run()

    assert base.particles_path.stat().st_mtime == traj_mtime
    assert base.log_path.read_text() == log_before
    assert grown.incomplete().empty


# ---------------------------------------------------------------------------
# CLI - stilt run via CliRunner
# ---------------------------------------------------------------------------


@integration
def test_cli_run(tmp_path, wbb_config, wbb_receptor):
    """CLI `stilt run` produces trajectory and footprint artifacts."""
    from typer.testing import CliRunner

    from stilt.cli import app

    project_dir = tmp_path / "cli_project"
    project_dir.mkdir()

    wbb_config.to_yaml(project_dir / "config.yaml")

    (project_dir / "receptors.csv").write_text(
        "time,lati,long,zagl\n2021-01-15 06:00:00,40.5,-112.0,5.0\n"
    )

    result = CliRunner().invoke(app, ["run", str(project_dir)])

    assert result.exit_code == 0, (
        f"stilt run exited {result.exit_code}\nOutput:\n{result.output}"
    )
    assert "complete=1" in result.output

    sim = Project(project_dir).simulation(*_sim_id(wbb_receptor))
    assert sim.has_particles, "CLI run: no particles"
    assert sim.has_footprint, "CLI run: no footprint"
    assert (project_dir / "output" / "particles").is_dir()


@integration
def test_cli_tasks_split_the_receptors(tmp_path, traj_only_config):
    """Two `stilt run --task` shares run every receptor once; a rerun has nothing to do."""
    from typer.testing import CliRunner

    from stilt.cli import app

    project_dir = tmp_path / "tasks"
    project_dir.mkdir()
    traj_only_config.to_yaml(project_dir / "config.yaml")
    (project_dir / "receptors.csv").write_text(
        "time,lati,long,zagl\n"
        "2021-01-15 06:00:00,40.5,-112.0,5.0\n"
        "2021-01-15 06:00:00,40.6,-111.9,5.0\n"
        "2021-01-15 06:00:00,40.7,-111.8,5.0\n"
    )
    project = Project(project_dir)
    every = list(dict.fromkeys(project.simulations["receptor"]))

    first = CliRunner().invoke(app, ["run", str(project_dir), "--task", "0/2"])
    assert first.exit_code == 0, first.output
    status = project.status()
    assert set(status.loc[status.state == "complete", "receptor"]) == {
        every[0],
        every[2],
    }

    second = CliRunner().invoke(app, ["run", str(project_dir), "--task", "1/2"])
    assert second.exit_code == 0, second.output
    assert set(project.status()["state"]) == {"complete"}

    again = CliRunner().invoke(app, ["run", str(project_dir), "--task", "0/2"])
    assert again.exit_code == 0, again.output
    assert "This run:  total=0" in again.output


# ---------------------------------------------------------------------------
# Wind-error variants
# ---------------------------------------------------------------------------


@integration
def test_error_variant(tmp_path, wbb_receptor, traj_only_config):
    """A variant with WINDERR fields is its own HYSPLIT run with perturbed winds."""
    config = _with(
        traj_only_config,
        execution={"keep_scratch": True},
        variants={"hrrr": {}, "hrrr-err": _XYERR},
    )
    model = Project.init(tmp_path / "error", config=config, receptors=[wbb_receptor])
    model.run()

    main = model.simulation(*_sim_id(wbb_receptor))
    err = model.simulation(*_sim_id(wbb_receptor, "hrrr-err"))
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
def test_error_realizations(tmp_path, wbb_receptor, traj_only_config):
    """Two realizations run, differ from each other, and resume one at a time."""
    config = _with(
        traj_only_config,
        krand=4,  # HYSPLIT seeds each run from the clock
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )
    model = Project.init(
        tmp_path / "realizations", config=config, receptors=[wbb_receptor]
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
    main = model.simulation(*_sim_id(wbb_receptor))
    err0 = model.simulation(*_sim_id(wbb_receptor, "err"), 0)
    err1 = model.simulation(*_sim_id(wbb_receptor, "err"), 1)
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
    tmp_path, wbb_receptor, traj_only_config
):
    """krand=2 with a seed: realizations differ and a rerun is bit-identical."""
    config = _with(
        traj_only_config,
        krand=2,
        seed=7,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )

    def run(project):
        model = Project.init(project, config=config, receptors=[wbb_receptor])
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
    main = a.simulation(*_sim_id(wbb_receptor)).particles
    s_main = main.groupby("particle")["foot"].sum().reindex(s0.index)
    assert not np.allclose(s_main.to_numpy(), s0.to_numpy())

    b = run(tmp_path / "b")
    for variant, k in (("hrrr", None), ("err", 0), ("err", 1)):
        pd.testing.assert_frame_equal(
            a.simulation(*_sim_id(wbb_receptor, variant), k).particles,
            b.simulation(*_sim_id(wbb_receptor, variant), k).particles,
        )


# ---------------------------------------------------------------------------
# Geometry-derived footprint config
# ---------------------------------------------------------------------------


@integration
def test_geometry_footprint(tmp_path, wbb_receptor, met_dir):
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
        mets={"hrrr": reference_met(met_dir)},
        n_hours=-6,
        numpar=100,
        krand=REFERENCE_KRAND,
        seed=REFERENCE_SEED,
        geometry=spec,
        cells_per_target=10,
        variants={"hrrr": {}},
    )
    variant = next(iter(config.resolve().values()))
    fc = variant.footprint
    assert fc is not None
    assert fc.grid.xres == fc.grid.yres == 0.05  # 0.5 / 10
    assert variant.geometry_hash

    model = Project.init(tmp_path / "geom", config=config, receptors=[wbb_receptor])
    model.run()

    sim = model.simulation(*_sim_id(wbb_receptor))
    assert sim.has_footprint
    foot = sim.footprint
    assert foot is not None
    assert foot.stilt.config.grid == fc.grid
    assert foot.stilt.config.geometry == fc.geometry
    assert foot.stilt.geometry_hash == variant.geometry_hash

    assert fc.geometry is not None
    mesh = Mesh.from_spec(fc.geometry)
    r_time = pd.Timestamp(wbb_receptor.time)
    bins = pd.interval_range(
        start=r_time - pd.Timedelta(hours=6), end=r_time, freq="1h", closed="left"
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        agg = foot.stilt.aggregate(mesh, bins)
    assert agg.index.tolist() == ["wbb", "nw"]
    assert agg.loc["wbb"].sum() > 0  # the receptor sits inside its own window
    assert agg.to_numpy().sum() <= float(foot.sum()) + 1e-12


# ---------------------------------------------------------------------------
# Forward run (n_hours > 0)
# ---------------------------------------------------------------------------


@integration
def test_forward_run(tmp_path, met_dir, wbb_grid):
    """A forward simulation runs end to end and carries a forward time axis."""
    from stilt.receptors import PointReceptor

    from ..fixtures.r_stilt_reference import (
        REFERENCE_ALTITUDE,
        REFERENCE_KRAND,
        REFERENCE_LATITUDE,
        REFERENCE_LONGITUDE,
        REFERENCE_SEED,
        REFERENCE_SUMMER_TIME,
    )

    # The 2021-07-15 06:00-11:59 block holds the whole +3 h window.
    receptor = PointReceptor(
        time=REFERENCE_SUMMER_TIME,
        longitude=REFERENCE_LONGITUDE,
        latitude=REFERENCE_LATITUDE,
        altitude=REFERENCE_ALTITUDE,
    )
    config = ProjectConfig(
        mets={"hrrr": reference_met(met_dir)},
        n_hours=3,
        numpar=100,
        krand=REFERENCE_KRAND,
        seed=REFERENCE_SEED,
        hnf_plume=True,  # exercises calc_plume_dilution on a forward track
        grid=wbb_grid,
        variants={"hrrr": {}},
    )

    model = Project.init(tmp_path / "forward", config=config, receptors=[receptor])
    model.run()

    sim = model.simulation(*_sim_id(receptor))
    assert sim.has_particles, f"no trajectory for {sim.id}"
    assert sim.particles is not None

    particles = sim.particles
    assert len(particles) > 0
    # HYSPLIT reports elapsed minutes signed by run direction
    assert (particles["time"] >= 0).all()
    assert particles["time"].max() > 0
    assert "foot_no_hnf_dilution" in particles.columns

    start, stop = receptor.time, receptor.time + pd.Timedelta(hours=3)

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
