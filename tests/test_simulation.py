"""Tests for stilt.simulation: the Simulation value object."""

import pandas as pd
import pytest
import xarray as xr

from stilt.config import Variant
from stilt.execution.worker import make_footprint
from stilt.footprint.config import FootprintConfig
from stilt.output import Output
from stilt.particles import particles_metadata
from stilt.simulation import Simulation
from stilt.spatial import Grid
from stilt.transforms import FirstOrderLifetime, transform_kind
from stilt.transport.hysplit import MetConfig

from .fixtures.factories import make_met_config, make_variant
from .fixtures.particles import finished

GRID = Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)
FOOT = FootprintConfig(grid=GRID, time_integrate=True, smooth_factor=0.0)


def _met_config(tmp_path, **kwargs) -> MetConfig:
    return make_met_config(tmp_path / "met", **kwargs)


def _variant(
    tmp_path, name="hrrr", footprint: FootprintConfig | None = None, **overrides
) -> Variant:
    """A resolved variant with the test transport defaults and an optional footprint."""
    return make_variant(
        name,
        met_config=_met_config(tmp_path),
        footprint=footprint,
        **{"numpar": 10, "hnf_plume": False, **overrides},
    )


def _sim(
    tmp_path, receptor, *, footprint=None, variant="hrrr", directory=None, **overrides
) -> Simulation:
    """A simulation whose results live in ``tmp_path/output``."""
    return Simulation(
        receptor,
        _variant(tmp_path, variant, footprint=footprint, **overrides),
        Output(tmp_path / "output"),
        directory,
    )


def _trajectories(receptor, params, foot: float = 1e-5) -> pd.DataFrame:
    particles = pd.DataFrame(
        {
            "age": [-60],
            "particle": [1],
            "lon": [-111.9],
            "lat": [40.7],
            "zagl": [10.0],
            "foot": [foot],
        }
    )
    return finished(particles, receptor, params)


def _write_particles(sim: Simulation) -> pd.DataFrame:
    """Put particles for *sim* in the output directory and return them."""
    particles = _trajectories(sim.receptor, sim.variant.transport)
    sim.output.write_particles(sim.variant, sim.receptor, particles, [])
    return particles


# ---------------------------------------------------------------------------
# A value
# ---------------------------------------------------------------------------


def test_simulation_is_a_frozen_value(point_receptor, tmp_path):
    a = _sim(tmp_path, point_receptor, footprint=FOOT)
    b = _sim(tmp_path, point_receptor, footprint=FOOT)
    assert a == b and hash(a) == hash(b)
    assert a != _sim(tmp_path, point_receptor, footprint=FOOT, variant="other")
    with pytest.raises(AttributeError):
        a.receptor = point_receptor  # type: ignore[misc]
    assert a.id == (point_receptor.id, "hrrr", None)
    assert str(a) == f"{point_receptor.id}/hrrr"
    assert a.variant.transport.numpar == 10
    assert a.variant.footprint == FOOT
    assert not (tmp_path / "output").exists()  # building one creates nothing


def test_paths_are_none_until_the_run_exists(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    assert sim.particles_path is None
    assert sim.footprint_path is None
    assert sim.log_path is None
    assert not sim.has_particles and not sim.has_footprint
    assert not sim.is_complete
    assert sim.failure is None
    with pytest.raises(FileNotFoundError):
        _ = sim.particles
    with pytest.raises(FileNotFoundError):
        _ = sim.footprint

    rid = str(point_receptor.id)
    sim.output.write_log(sim.variant, rid, "")  # makes the particles folder
    assert sim.particles_path == sim.output.path("particles", sim.variant, rid)
    assert sim.particles_path is not None and not sim.particles_path.exists()
    assert sim.log_path == sim.output.log_path(sim.variant, rid)
    assert sim.footprint_path is None  # no footprint folder yet
    sim.output.record_failure("footprints", sim.variant, rid, {})
    assert sim.footprint_path == sim.output.path("footprints", sim.variant, rid)


def test_log_property_raises_when_missing(point_receptor, tmp_path):
    with pytest.raises(FileNotFoundError):
        _ = _sim(tmp_path, point_receptor).log


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------


def test_reads_particles_written_under_the_same_settings(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    written = _write_particles(sim)

    again = _sim(tmp_path, point_receptor)  # a fresh value
    assert again.has_particles and again.is_complete
    assert len(again.particles) == len(written)
    meta = particles_metadata(again.particles_path)
    assert meta.receptor == point_receptor
    assert meta.settings["numpar"] == 10
    assert meta.met_files == []
    assert "datetime" in again.particles.columns
    assert again.particles is again.particles  # kept once read


def test_results_read_before_the_run_are_read_again_after_it(point_receptor, tmp_path):
    """A read that finds nothing keeps nothing, so the same object sees the run land."""
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    with pytest.raises(FileNotFoundError):
        _ = sim.particles
    with pytest.raises(FileNotFoundError):
        _ = sim.footprint

    traj = _write_particles(sim)
    assert sim.particles is not None
    with pytest.raises(FileNotFoundError):
        _ = sim.footprint

    make_footprint(sim, traj)
    assert sim.footprint is not None
    assert sim.footprint is sim.footprint  # kept once read


def test_particles_only_variant_has_no_footprint(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    _write_particles(sim)
    assert sim.footprint is None  # final: the variant makes none


def test_variants_with_equal_transport_share_particles(point_receptor, tmp_path):
    fine = _sim(tmp_path, point_receptor, footprint=FOOT)
    coarse = _sim(
        tmp_path,
        point_receptor,
        variant="coarse",
        footprint=FOOT.model_copy(
            update={"grid": GRID.model_copy(update={"xres": 0.5, "yres": 0.5})}
        ),
    )
    _write_particles(fine)
    assert coarse.particles_path == fine.particles_path
    assert coarse.has_particles  # written under fine's variant
    assert fine.footprint_path is None and coarse.footprint_path is None


def test_completion_needs_particles_and_the_footprint_when_configured(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    traj = _write_particles(sim)
    assert sim.has_particles and not sim.is_complete
    make_footprint(sim, traj)
    assert sim.is_complete
    assert sim.footprint_path is not None
    assert sim.footprint_path.parent.parent.name.startswith("settings=hrrr-")


def test_particles_only_variant_is_complete_with_particles(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert sim.variant.footprint is None
    _write_particles(sim)
    assert sim.is_complete


def test_written_footprint_reads_back(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    traj = _write_particles(sim)
    foot = make_footprint(sim, traj)
    assert foot is not None and foot.stilt.name == "hrrr"

    again = _sim(tmp_path, point_receptor, footprint=FOOT)
    back = again.footprint
    assert isinstance(back, xr.DataArray)
    assert back.stilt.receptor == point_receptor and back.stilt.grid == GRID
    xr.testing.assert_allclose(back, foot.astype("float32").astype("float64"))
    assert again.is_complete and again.failure is None


def test_empty_footprint_is_recorded_with_its_reason(point_receptor, tmp_path):
    far = FOOT.model_copy(
        update={"grid": Grid(xmin=0, xmax=1, ymin=0, ymax=1, xres=0.1, yres=0.1)}
    )
    sim = _sim(tmp_path, point_receptor, footprint=far)
    traj = _write_particles(sim)

    assert make_footprint(sim, traj) is None

    assert sim.footprint is None
    assert sim.has_footprint and sim.is_complete
    assert sim.is_complete and sim.failure is None


def test_failure_reads_the_record_for_the_missing_step(point_receptor, tmp_path):
    grid = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.1, yres=0.1)
    sim = _sim(tmp_path, point_receptor, footprint=FootprintConfig(grid=grid))
    particles = {"step": "particles", "reason": "MET_COVERAGE"}
    footprint = {"step": "footprint", "reason": "ValueError"}
    sim.output.record_failure("particles", sim.variant, sim.receptor.id, particles)
    sim.output.record_failure("footprints", sim.variant, sim.receptor.id, footprint)
    # The particles are missing, so the particles step is why.
    assert sim.failure == particles
    # Once they exist, a stale particles record does not count; the footprint's does.
    _write_particles(sim)
    assert sim.failure == footprint


def test_a_log_alone_is_not_a_failure(point_receptor, tmp_path):
    """A simulation that shares a log with one that ran is not failed for it."""
    sim = _sim(tmp_path, point_receptor)
    sim.output.write_log(sim.variant, sim.receptor.id, "hycs_std ran\n")
    assert sim.failure is None
    assert sim.log.startswith("hycs_std")


# ---------------------------------------------------------------------------
# Calculating a footprint without writing it
# ---------------------------------------------------------------------------


def test_calc_footprint_uses_the_variant_settings_and_writes_nothing(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _write_particles(sim)
    foot = sim.calc_footprint()
    assert foot is not None and foot.stilt.config == FOOT and foot.stilt.name == "hrrr"
    assert not sim.has_footprint
    with pytest.raises(FileNotFoundError):
        _ = sim.footprint


def test_calc_footprint_requires_a_grid_and_particles(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    with pytest.raises(FileNotFoundError, match="no particles"):
        sim.calc_footprint(grid=GRID)
    _write_particles(sim)
    with pytest.raises(TypeError, match="no footprint settings"):
        sim.calc_footprint()


def test_calc_footprint_replaces_the_settings_given(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _write_particles(sim)

    class Halve:
        def apply(self, particles, receptor=None, directory=None):
            return particles.assign(foot=particles["foot"] * 0.5)

    base = sim.calc_footprint()
    halved = sim.calc_footprint(transforms=[Halve()])
    assert float(halved.sum()) == pytest.approx(0.5 * float(base.sum()))
    coarse = sim.calc_footprint(grid=GRID.model_copy(update={"xres": 0.5, "yres": 0.5}))
    assert coarse.stilt.grid.xres == 0.5
    assert coarse.stilt.config.smooth_factor == FOOT.smooth_factor
    smooth = sim.calc_footprint(smooth_factor=0.0)
    assert smooth.stilt.config.smooth_factor == 0.0
    assert smooth.stilt.grid == GRID
    decay = sim.calc_footprint(transforms=[FirstOrderLifetime(lifetime_hours=1.0)])
    assert [transform_kind(t) for t in decay.stilt.config.transforms] == [
        "first_order_lifetime"
    ]


def test_calc_footprint_returns_none_when_nothing_reaches_the_grid(
    point_receptor, tmp_path
):
    far = FOOT.model_copy(
        update={"grid": Grid(xmin=0, xmax=1, ymin=0, ymax=1, xres=0.1, yres=0.1)}
    )
    sim = _sim(tmp_path, point_receptor, footprint=far)
    _write_particles(sim)
    assert sim.calc_footprint() is None


def test_calc_footprint_uses_the_receptor_kernel_from_a_project_table(
    point_receptor, tmp_path
):
    from stilt.transforms import AveragingKernel

    rid = str(point_receptor.id)
    table = pd.DataFrame(
        {"receptor": [rid, rid], "level": [0.0, 3000.0], "value": [0.5, 0.5]}
    )
    table.to_parquet(tmp_path / "kernels.parquet")
    config = FOOT.model_copy(
        update={"transforms": [AveragingKernel(table="kernels.parquet")]}
    )
    sim = _sim(tmp_path, point_receptor, footprint=config, directory=tmp_path)
    with_height = _trajectories(point_receptor, sim.variant.transport)
    with_height["release_height"] = (
        10.0  # the kernel weights particles by release height
    )
    sim.output.write_particles(sim.variant, point_receptor, with_height, [])
    plain = _sim(tmp_path, point_receptor, footprint=FOOT, variant="plain")
    # Same transport settings: the particles are shared.
    assert plain.particles_path == sim.particles_path

    weighted = sim.calc_footprint()  # the table is found in sim.directory
    base = plain.calc_footprint()
    assert float(weighted.sum()) == pytest.approx(0.5 * float(base.sum()))


# ---------------------------------------------------------------------------
# What the particles give beyond the footprint
# ---------------------------------------------------------------------------


def test_background_and_transport_error_weight_the_particles_like_the_footprint(
    point_receptor, tmp_path
):
    import numpy as np

    from stilt.particles import background, transport_error

    lifetime = [FirstOrderLifetime(lifetime_hours=1.0)]
    footprint = FOOT.model_copy(update={"transforms": lifetime})
    sim = _sim(tmp_path, point_receptor, footprint=footprint)
    # A second run of the receptor stands in for a wind-error variant.
    err = _sim(
        tmp_path, point_receptor, footprint=footprint, variant="err", ziscale=0.8
    )
    _write_particles(sim)
    _write_particles(err)
    particles = sim.particles  # as read back from the file

    field = pd.Series([400.0], index=pd.Index([1], name="particle"))
    got = sim.background(field)
    direct = background(particles, field, transforms=lifetime, receptor=point_receptor)
    assert got.value == direct.value
    pd.testing.assert_series_equal(got.weights, direct.weights)
    assert got.value < 400.0  # the lifetime decay weights the endpoint

    flux = xr.DataArray(
        np.ones((3, 3)),
        dims=("lat", "lon"),
        coords={"lat": [40.6, 40.7, 40.8], "lon": [-112.0, -111.9, -111.8]},
    )
    one = sim.transport_error(err, flux, noise_splits=0)
    many = sim.transport_error([err, err], flux, noise_splits=0)
    error = transport_error(
        particles,
        [err.particles],
        flux,
        transforms=lifetime,
        receptor=point_receptor,
        noise_splits=0,
    )
    np.testing.assert_equal(
        (one.variance, one.enhancement), (error.variance, error.enhancement)
    )
    assert many.realizations == 2
