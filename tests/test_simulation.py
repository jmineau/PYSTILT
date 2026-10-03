"""Tests for stilt.simulation: SimID and the Simulation value object."""

import datetime as dt

import pandas as pd
import pytest
import xarray as xr

from stilt.config import (
    FootprintConfig,
    Grid,
    MetConfig,
)
from stilt.execution import make_footprint
from stilt.output import Output
from stilt.particles import particles_metadata, prepare
from stilt.simulation import SimID, Simulation
from stilt.transforms import FirstOrderLifetime, TransformContext, transform_kind
from stilt.transport import ModelInfo
from stilt.transport.hysplit import HysplitConfig
from stilt.transport.hysplit.model import finish_particles
from stilt.variants import Variant

GRID = Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)
FOOT = FootprintConfig(grid=GRID, time_integrate=True, smooth_factor=0.0)


def _met_config(tmp_path, **kwargs) -> MetConfig:
    return MetConfig(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h", **kwargs
    )


def _variant(
    tmp_path, name="hrrr", footprint: FootprintConfig | None = None, **overrides
) -> Variant:
    """A resolved variant with the test transport defaults and an optional footprint."""
    params = HysplitConfig(
        **{"n_hours": -24, "numpar": 10, "hnf_plume": False, **overrides}
    )
    return Variant(
        name=name,
        group=name,
        met="hrrr",
        met_config=_met_config(tmp_path),
        transport=params,
        model=ModelInfo(version="v5.1.0"),
        footprint=footprint,
    )


def _sim(
    tmp_path, receptor, *, footprint=None, variant="hrrr", **overrides
) -> Simulation:
    """A simulation whose results live in ``tmp_path/output``."""
    return Simulation(
        receptor,
        _variant(tmp_path, variant, footprint=footprint, **overrides),
        Output(tmp_path / "output"),
    )


def _trajectories(receptor, params, foot: float = 1e-5) -> pd.DataFrame:
    particles = pd.DataFrame(
        {
            "time": [-60],
            "indx": [1],
            "long": [-111.9],
            "lati": [40.7],
            "zagl": [10.0],
            "foot": [foot],
        }
    )
    return prepare(finish_particles(particles, receptor, params), receptor)


def _write_particles(sim: Simulation) -> pd.DataFrame:
    """Put particles for *sim* in the output directory and return them."""
    particles = _trajectories(sim.receptor, sim.variant.transport)
    folder = sim.output.particles(sim.variant)
    folder.write(sim.receptor, particles, sim.variant.transport, [])
    return particles


def _context(sim: Simulation, directory=None) -> TransformContext:
    return TransformContext(
        receptor=sim.receptor, variant=sim.variant.name, directory=directory
    )


# ---------------------------------------------------------------------------
# SimID
# ---------------------------------------------------------------------------


def test_simid_is_a_receptor_and_a_variant(point_receptor):
    sid = SimID(point_receptor.id, "hrrr")
    assert sid.receptor == point_receptor.id
    assert sid.variant == "hrrr"
    assert str(sid) == f"{point_receptor.id}/hrrr"


def test_simid_parses_its_string_and_tuple_forms(point_receptor):
    sid = SimID(point_receptor.id, "hrrr")
    assert SimID.parse(str(sid)) == sid
    assert SimID.parse((str(point_receptor.id), "hrrr")) == sid
    assert SimID.parse(sid) is sid


@pytest.mark.parametrize("bad", ["nohash", "id/", "/hrrr"])
def test_simid_parse_rejects_malformed_ids(bad):
    with pytest.raises(ValueError):
        SimID.parse(bad)


def test_simid_is_pathlike(point_receptor, tmp_path):
    sid = SimID(point_receptor.id, "hrrr")
    assert tmp_path / sid == tmp_path / str(point_receptor.id) / "hrrr"


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
    assert a.id == SimID(point_receptor.id, "hrrr")
    assert a.variant.transport.numpar == 10
    assert a.variant.footprint == FOOT
    assert not (tmp_path / "output").exists()  # building one creates nothing


def test_paths_are_none_until_the_run_exists(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    assert sim._particle_set is None and sim._footprint_set is None
    assert sim.particles_path is None
    assert sim.footprint_path is None
    assert sim.log_path is None
    assert not sim.has_particles and not sim.has_footprint
    assert not sim.is_complete()
    assert sim.outcome is None
    with pytest.raises(FileNotFoundError):
        _ = sim.particles
    with pytest.raises(FileNotFoundError):
        _ = sim.footprint

    run = sim.output.particles(sim.variant)
    rid = str(point_receptor.id)
    assert sim._particle_set == run
    assert sim.particles_path == run.file(rid)
    assert sim.log_path == run.log_path(rid)
    assert sim.footprint_path is None  # no footprint folder yet
    feet = run.footprints(FOOT, name="hrrr")
    assert sim._footprint_set == feet
    assert sim.footprint_path == feet.file(rid)


def test_simulation_time_range_backward(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, n_hours=-6)
    start, stop = sim.time_range
    assert stop == point_receptor.time
    assert start == point_receptor.time - dt.timedelta(hours=6)


def test_simulation_time_range_forward(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, n_hours=6)
    start, stop = sim.time_range
    assert start == point_receptor.time
    assert stop == point_receptor.time + dt.timedelta(hours=6)


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
    assert again.has_particles and again.is_complete()
    assert len(again.particles) == len(written)
    meta = particles_metadata(again.particles_path)
    assert meta.receptor == point_receptor
    assert meta.params.numpar == 10
    assert again.met_files == []
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

    make_footprint(sim, traj, context=_context(sim))
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
    assert coarse._particle_set == fine._particle_set
    assert coarse.has_particles  # written under fine's variant
    assert (
        fine._footprint_set is None and coarse._footprint_set is None
    )  # no footprints yet


def test_completion_needs_particles_and_the_footprint_when_configured(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    traj = _write_particles(sim)
    assert sim.has_particles and not sim.is_complete()
    make_footprint(sim, traj, context=_context(sim))
    assert sim.is_complete()
    assert sim._footprint_set is not None and sim._footprint_set.name == "hrrr"


def test_particles_only_variant_is_complete_with_particles(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert not sim.makes_footprint
    _write_particles(sim)
    assert sim.is_complete()


def test_written_footprint_reads_back(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    traj = _write_particles(sim)
    foot = make_footprint(sim, traj, context=_context(sim))
    assert foot is not None and foot.stilt.name == "hrrr"

    again = _sim(tmp_path, point_receptor, footprint=FOOT)
    back = again.footprint
    assert isinstance(back, xr.DataArray)
    assert back.stilt.receptor == point_receptor and back.stilt.grid == GRID
    xr.testing.assert_allclose(back, foot.astype("float32").astype("float64"))
    assert again.outcome == "complete"


def test_empty_footprint_is_recorded_with_its_reason(point_receptor, tmp_path):
    far = FOOT.model_copy(
        update={"grid": Grid(xmin=0, xmax=1, ymin=0, ymax=1, xres=0.1, yres=0.1)}
    )
    sim = _sim(tmp_path, point_receptor, footprint=far)
    traj = _write_particles(sim)

    assert make_footprint(sim, traj, context=_context(sim)) is None

    assert sim.footprint is None
    assert sim.has_footprint and sim.is_complete()
    assert sim.empty_reason == "outside_domain"
    assert sim.outcome == "complete"


def test_outcome_reads_a_failure_from_the_log(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    run = sim.output.particles(sim.variant)
    run.write_log(
        sim.receptor.id, "Insufficient number of meteorological files found\n"
    )
    assert sim.outcome == "failed:MISSING_MET_FILES"
    assert sim.log.startswith("Insufficient")


# ---------------------------------------------------------------------------
# Calculating a footprint without writing
# ---------------------------------------------------------------------------


def test_generate_footprint_uses_the_variant_settings_and_writes_nothing(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _write_particles(sim)
    foot = sim.generate_footprint()
    assert foot is not None and foot.stilt.config == FOOT and foot.stilt.name == "hrrr"
    assert not sim.has_footprint
    with pytest.raises(FileNotFoundError):
        _ = sim.footprint


def test_generate_footprint_requires_settings_and_particles(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    with pytest.raises(FileNotFoundError, match="no particles"):
        sim.generate_footprint(FOOT)
    _write_particles(sim)
    with pytest.raises(TypeError, match="no footprint settings"):
        sim.generate_footprint()


def test_generate_footprint_takes_other_settings_and_extra_transforms(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _write_particles(sim)

    class Halve:
        def apply(self, particles, context):
            return particles.assign(foot=particles["foot"] * 0.5)

    base = sim.generate_footprint()
    halved = sim.generate_footprint(transforms=[Halve()])
    assert float(halved.sum()) == pytest.approx(0.5 * float(base.sum()))
    coarse = sim.generate_footprint(
        FOOT.model_copy(
            update={"grid": GRID.model_copy(update={"xres": 0.5, "yres": 0.5})}
        )
    )
    assert coarse.stilt.grid.xres == 0.5
    decay = sim.generate_footprint(transforms=[FirstOrderLifetime(lifetime_hours=1.0)])
    assert [transform_kind(t) for t in decay.stilt.config.transforms] == [
        "first_order_lifetime"
    ]


def test_generate_footprint_returns_none_when_nothing_reaches_the_grid(
    point_receptor, tmp_path
):
    far = FOOT.model_copy(
        update={"grid": Grid(xmin=0, xmax=1, ymin=0, ymax=1, xres=0.1, yres=0.1)}
    )
    sim = _sim(tmp_path, point_receptor, footprint=far)
    _write_particles(sim)
    assert sim.generate_footprint() is None


def test_generate_footprint_uses_the_receptor_kernel_from_a_project_table(
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
    sim = _sim(tmp_path, point_receptor, footprint=config)
    with_height = _trajectories(point_receptor, sim.variant.transport)
    with_height["xhgt"] = 10.0  # the kernel weights particles by release height
    folder = sim.output.particles(sim.variant)
    folder.write(point_receptor, with_height, sim.variant.transport, [])
    plain = _sim(tmp_path, point_receptor, footprint=FOOT, variant="plain")
    assert (
        plain._particle_set == sim._particle_set
    )  # same transport settings: the particles are shared

    weighted = sim.generate_footprint(context=_context(sim, directory=tmp_path))
    base = plain.generate_footprint()
    assert float(weighted.sum()) == pytest.approx(0.5 * float(base.sum()))
