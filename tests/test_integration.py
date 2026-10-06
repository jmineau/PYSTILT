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
from stilt.execution import resolve_compute_root
from stilt.footprint.targets import Mesh
from stilt.identity import transport_from_settings
from stilt.meteorology import MetConfig
from stilt.particles import particles_metadata
from stilt.project import Project
from stilt.simulation import SimID
from stilt.transport.hysplit.driver import winderrtf
from stilt.variants import resolve

from .conftest import integration

_XYERR = {"siguverr": 2.0, "tluverr": 60.0, "zcoruverr": 500.0, "horcoruverr": 40.0}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_id(receptor, variant: str = "hrrr") -> SimID:
    return SimID(receptor.id, variant)


def _incomplete(project: Project) -> list[SimID]:
    rows = project.simulations.incomplete()
    return [SimID(r, v) for r, v in zip(rows["receptor"], rows["variant"], strict=True)]


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
    assert list(model.simulations["receptor"]) == [sid.receptor]
    sim = model.simulation(*sid)
    assert sim.has_particles
    assert sim.particles_path is not None
    assert sim.particles_path.parent.name == "date=2021-01-15"
    assert sim.particles_path.parent.parent.name.startswith("settings=hrrr-")
    assert len(pd.read_parquet(sim.particles_path)) > 0, "Particle file is empty"
    assert sim.particles is not None and len(sim.particles) > 0
    assert not (resolve_compute_root(model) / sim.id).exists(), (
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
    assert sim.is_complete()
    assert sim.has_particles
    assert sim.empty_reason == "outside_domain"
    assert sim.footprint is None
    assert model.simulations.load_footprints() == {}
    # An empty footprint is complete; sim.empty_reason says why it is empty.
    assert model.simulations.status()["state"].tolist() == ["complete"]

    # A rerun has nothing to do and does not touch the empty record.
    before = sim.footprint_path.stat().st_mtime_ns
    model.run()
    assert sim.footprint_path.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# Failure path - missing met files
# ---------------------------------------------------------------------------


@integration
def test_failure_missing_met(tmp_path, wbb_receptor, traj_only_config):
    """Simulation fails gracefully when the met directory is empty."""
    empty_met = tmp_path / "empty_met"
    empty_met.mkdir()

    bad_config = traj_only_config.model_copy(
        update={
            "mets": {
                "hrrr": MetConfig(
                    directory=empty_met,
                    file_format="%Y%m%d.%Hz.hrrra",
                    file_tres="6h",
                )
            }
        }
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
    assert sim.failure["reason"] == "MISSING_MET_FILES"
    status = model.simulations.status()
    assert status["state"].tolist() == ["failed"]
    assert status["reason"].tolist() == ["MISSING_MET_FILES"]


@integration
def test_failure_met_cut_short(tmp_path, wbb_receptor, traj_only_config, met_dir):
    """A met file cut to one time period fails the run that needs it."""
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
                "file_format": met.file_format,
                "file_tres": met.file_tres,
            }
        },
    )
    project = Project.init(
        tmp_path / "met_cut_short", config=config, receptors=[wbb_receptor]
    )
    project.run()

    sim = project.simulation(*_sim_id(wbb_receptor))
    assert not sim.has_particles
    assert sim.failure is not None and sim.failure["reason"] == "MET_TRUNCATED"
    assert sim.log_path is not None
    log = sim.log_path.read_text()
    assert "Only one time period of meteo data" in log
    assert "Meteorology ends early" in log


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
    assert sid.receptor.endswith(f"_X{r.bottom:g}-{r.top:g}"), (
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
    assert "multi_" in sid.receptor, f"Expected a multipoint receptor id, got {sid}"

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
    assert coarse._particle_set == fine._particle_set  # one set of particles
    assert len(model.output.particle_sets()) == 1
    made_from_fine = [
        f
        for f in model.output.footprint_sets()
        if f.particles_key == fine._particle_set.key
    ]
    assert {f.name for f in made_from_fine} == {"hrrr", "coarse"}
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
    assert grown.simulations.incomplete().frame.empty


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
    assert "completed=1" in result.output

    sim = Project(project_dir).simulation(*_sim_id(wbb_receptor))
    assert sim.has_particles, "CLI run: no particles"
    assert sim.has_footprint, "CLI run: no footprint"
    assert (project_dir / "output" / "particles").is_dir()


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
    rid = str(wbb_receptor.id)
    err_scratch = err._particle_set.scratch_path(rid)
    main_scratch = main._particle_set.scratch_path(rid)

    assert (err_scratch / "WINDERR").exists()
    assert not (main_scratch / "WINDERR").exists()
    assert "winderrtf=1" in (err_scratch / "SETUP.CFG").read_text().lower()

    main_traj = main.particles
    error_traj = err.particles
    assert len(main_traj) > 0 and len(error_traj) > 0
    assert set(main_traj.columns) == set(error_traj.columns)
    assert (
        not main_traj["long"]
        .reset_index(drop=True)
        .equals(error_traj["long"].reset_index(drop=True))
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

    sims = model.simulations[model.simulations.group == "err"]
    assert sims["variant"].tolist() == ["err-0", "err-1"]
    assert sims.incomplete().frame.empty

    particles = sims.load_particles()
    e0, e1 = (particles[particles.variant == v] for v in ("err-0", "err-1"))
    s0 = e0.groupby("indx")["foot"].sum()
    s1 = e1.groupby("indx")["foot"].sum().reindex(s0.index)
    assert not np.allclose(s0.to_numpy(), s1.to_numpy())

    # Resume: drop one realization; only it reruns.
    main = model.simulation(*_sim_id(wbb_receptor))
    err0 = model.simulation(*_sim_id(wbb_receptor, "err-0"))
    err1 = model.simulation(*_sim_id(wbb_receptor, "err-1"))
    main_bytes = main.particles_path.read_bytes()
    err0_bytes = err0.particles_path.read_bytes()
    err1.particles_path.unlink()
    assert _incomplete(model) == [err1.id]

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
        assert model.simulations.incomplete().frame.empty
        return model

    a = run(tmp_path / "a")
    errs = a.simulations[a.simulations.group == "err"]
    particles = errs.load_particles()
    e0, e1 = (particles[particles.variant == v] for v in ("err-0", "err-1"))
    s0 = e0.groupby("indx")["foot"].sum()
    s1 = e1.groupby("indx")["foot"].sum().reindex(s0.index)
    assert not np.allclose(s0.to_numpy(), s1.to_numpy())
    main = a.simulation(*_sim_id(wbb_receptor)).particles
    s_main = main.groupby("indx")["foot"].sum().reindex(s0.index)
    assert not np.allclose(s_main.to_numpy(), s0.to_numpy())

    b = run(tmp_path / "b")
    for variant in ("hrrr", "err-0", "err-1"):
        pd.testing.assert_frame_equal(
            a.simulation(*_sim_id(wbb_receptor, variant)).particles,
            b.simulation(*_sim_id(wbb_receptor, variant)).particles,
        )


# ---------------------------------------------------------------------------
# Geometry-derived footprint config
# ---------------------------------------------------------------------------


@integration
def test_geometry_footprint(tmp_path, wbb_receptor, met_dir):
    """A footprint named by geometry derives its raster, runs, and aggregates."""

    from .fixtures.r_stilt_reference import (
        REFERENCE_KRAND,
        REFERENCE_MET_FILE_FORMAT,
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
        mets={
            "hrrr": {
                "directory": met_dir,
                "file_format": REFERENCE_MET_FILE_FORMAT,
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
        krand=REFERENCE_KRAND,
        seed=REFERENCE_SEED,
        geometry=spec,
        cells_per_target=10,
    )
    variant = next(iter(resolve(config).values()))
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

    from .fixtures.r_stilt_reference import (
        REFERENCE_ALTITUDE,
        REFERENCE_KRAND,
        REFERENCE_LATITUDE,
        REFERENCE_LONGITUDE,
        REFERENCE_MET_FILE_FORMAT,
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
        mets={
            "hrrr": {
                "directory": met_dir,
                "file_format": REFERENCE_MET_FILE_FORMAT,
                "file_tres": "6h",
            }
        },
        n_hours=3,
        numpar=100,
        krand=REFERENCE_KRAND,
        seed=REFERENCE_SEED,
        hnf_plume=True,  # exercises calc_plume_dilution on a forward track
        grid=wbb_grid,
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

    start, stop = sim.time_range
    assert start == receptor.time
    assert stop == receptor.time + pd.Timedelta(hours=3)

    assert sim.has_footprint, f"no footprint for {sim.id}"
    foot = sim.footprint
    assert foot is not None
    times = pd.DatetimeIndex(foot["time"].values)
    # hourly layers running forward, inside the window time_range reports
    assert times.is_monotonic_increasing
    assert times.min() >= pd.Timestamp(start)
    assert times.max() <= pd.Timestamp(stop)
    assert set(times.to_series().diff().dropna()) == {pd.Timedelta(hours=1)}
    assert float(foot.sum()) > 0
