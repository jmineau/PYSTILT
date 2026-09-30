"""Tests for stilt.simulation: SimID, VariantOutput, and Simulation."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from stilt.config import (
    FootprintConfig,
    Grid,
    MetConfig,
    STILTParams,
    VariantConfig,
)
from stilt.errors import HYSPLITTimeoutError
from stilt.footprint import Footprint
from stilt.hysplit.driver import HYSPLITResult
from stilt.meteorology import MetStream
from stilt.output import Output
from stilt.simulation import SimID, Simulation, VariantOutput
from stilt.trajectory import Trajectories
from stilt.transforms import FirstOrderLifetime, transform_kind

GRID = Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)
FOOT = FootprintConfig(grid=GRID, time_integrate=True, smooth_factor=0.0)


def _params(**kwargs) -> STILTParams:
    data = {"n_hours": -24, "numpar": 10, "hnf_plume": False}
    data.update(kwargs)
    return STILTParams(**data)


def _variant(
    name="hrrr", footprint: FootprintConfig | None = None, **overrides
) -> VariantConfig:
    """A resolved variant with the test transport defaults and an optional footprint."""
    data = {"n_hours": -24, "numpar": 10, "hnf_plume": False, **overrides}
    if footprint is not None:
        data.update(footprint.model_dump())
    return VariantConfig(name=name, group=name, met="hrrr", **data)


def _met_config(tmp_path, **kwargs) -> MetConfig:
    return MetConfig(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h", **kwargs
    )


def _sim(
    tmp_path,
    receptor,
    *,
    footprint=None,
    variant="hrrr",
    met_kwargs=None,
    keep_scratch=False,
    **param_overrides,
) -> Simulation:
    """A simulation writing to ``tmp_path/output`` and running in ``tmp_path/scratch``."""
    config = _variant(variant, footprint=footprint, **param_overrides)
    met_config = _met_config(tmp_path, **(met_kwargs or {}))
    outputs = VariantOutput(
        Output(tmp_path / "output"),
        variant,
        config.transport_settings(met_config),
        config.footprint,
    )
    return Simulation(
        receptor,
        config,
        met=MetStream("hrrr", met_config),
        outputs=outputs,
        directory=tmp_path / "scratch" / SimID(receptor.id, variant),
        project_dir=tmp_path,
        keep_scratch=keep_scratch,
    )


def _particles_df(foot: float = 1e-5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-60],
            "indx": [1],
            "long": [-111.9],
            "lati": [40.7],
            "zagl": [10.0],
            "foot": [foot],
        }
    )


def _trajectories(receptor, foot: float = 1e-5) -> Trajectories:
    return Trajectories.from_particles(
        particles=_particles_df(foot), receptor=receptor, params=_params(), met_files=[]
    )


class _FakeMet:
    id = "hrrr"

    def required_files(self, **kwargs):
        return []

    def stage_files_for_simulation(self, **kwargs):
        return []


def _fake_runner_returning(particles, *, fail: Exception | None = None):
    """A stand-in HYSPLITDriver that writes ``stilt.log`` into its directory and returns *particles*."""

    class _Runner:
        def __init__(self, *, directory, receptor, params, met_files):
            self.directory = Path(directory)
            self.params = params

        def prepare(self):
            (self.directory / "CONTROL").write_text("control")

        def execute(self, timeout, rm_dat):
            log = self.directory / "stilt.log"
            log.write_text("hysplit ran\n")
            if fail is not None:
                raise fail
            return HYSPLITResult(particles=particles, log_path=log)

    return _Runner


@pytest.fixture
def fake_hysplit(monkeypatch):
    """Replace the HYSPLIT driver with one that returns a small particle table."""
    monkeypatch.setattr(
        "stilt.simulation.HYSPLITDriver", _fake_runner_returning(_particles_df())
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
# VariantOutput
# ---------------------------------------------------------------------------


def test_variant_output_finds_nothing_until_a_run_exists(tmp_path):
    config = _variant(footprint=FOOT)
    out = Output(tmp_path / "output")
    vo = VariantOutput(
        out, "hrrr", config.transport_settings(_met_config(tmp_path)), FOOT
    )
    assert vo.run is None and vo.footprints is None
    assert not (tmp_path / "output").exists()  # looking creates nothing

    run = vo.ensure_run()
    assert vo.run == run and run.name == "hrrr"
    assert vo.footprints is None  # the folder exists only once written to
    feet = vo.ensure_footprints()
    assert feet is not None and vo.footprints == feet
    assert feet.name == "hrrr" and feet.run == run

    # A second VariantOutput for the same settings finds the same folders.
    again = VariantOutput(out, "hrrr", vo.settings, FOOT)
    assert again.run == run and again.footprints == feet


def test_variant_output_without_grid_has_no_footprints(tmp_path):
    config = _variant()
    vo = VariantOutput(
        Output(tmp_path / "output"),
        "hrrr",
        config.transport_settings(_met_config(tmp_path)),
        None,
    )
    vo.ensure_run()
    assert vo.footprints is None and vo.ensure_footprints() is None


# ---------------------------------------------------------------------------
# Construction and paths
# ---------------------------------------------------------------------------


def test_construction_creates_nothing(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    assert not sim.directory.exists()
    assert not (tmp_path / "output").exists()
    assert sim.id == SimID(point_receptor.id, "hrrr")
    assert sim.variant == "hrrr"
    assert sim.params.numpar == 10
    assert sim.footprint_config == FOOT
    assert sim.met_dir == sim.directory / "met"


def test_paths_are_none_until_the_run_exists(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    assert sim.trajectories_path is None
    assert sim.footprint_path is None
    assert sim.log_path is None
    assert not sim.has_trajectory and not sim.has_footprint
    assert not sim.is_complete()
    assert sim.outcome is None

    run = sim.outputs.ensure_run()
    rid = str(point_receptor.id)
    assert sim.trajectories_path == run.particles_path(rid)
    assert sim.log_path == run.log_path(rid)
    assert sim.footprint_path is None  # no footprint folder yet
    feet = sim.outputs.ensure_footprints()
    assert sim.footprint_path == feet.footprint_path(rid)


def test_simulation_time_range_backward(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, n_hours=-6)
    start, stop = sim.time_range
    assert sim.is_backward
    assert stop == point_receptor.time
    assert start == point_receptor.time - dt.timedelta(hours=6)


def test_simulation_time_range_forward(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, n_hours=6)
    start, stop = sim.time_range
    assert not sim.is_backward
    assert start == point_receptor.time
    assert stop == point_receptor.time + dt.timedelta(hours=6)


def test_meteorology_subgrid_requires_bounds(point_receptor, tmp_path):
    with pytest.raises(ValueError, match="subgrid_bounds"):
        _sim(tmp_path, point_receptor, met_kwargs={"subgrid_enable": True})


# ---------------------------------------------------------------------------
# Running HYSPLIT
# ---------------------------------------------------------------------------


def test_run_trajectories_writes_particles_and_log_and_clears_scratch(
    point_receptor, tmp_path, fake_hysplit
):
    sim = _sim(tmp_path, point_receptor)
    sim.met = _FakeMet()

    sim.run_trajectories(write=True)

    assert sim.has_trajectory
    assert sim.trajectories is not None and len(sim.trajectories.data) == 1
    assert sim.log == "hysplit ran\n"
    assert not sim.directory.exists()  # scratch is gone after success
    assert not sim.run.scratch_path(str(point_receptor.id)).exists()
    assert sim.outcome == "complete"

    # A fresh simulation reads the particles back from the output directory.
    again = _sim(tmp_path, point_receptor)
    assert again.has_trajectory
    assert again.trajectories.receptor == point_receptor
    assert again.trajectories.params == sim.params
    assert "datetime" in again.trajectories.data.columns


def test_run_trajectories_without_write_keeps_the_output_directory_clean(
    point_receptor, tmp_path, fake_hysplit
):
    sim = _sim(tmp_path, point_receptor)
    sim.met = _FakeMet()
    sim.run_trajectories(write=False)
    assert sim.trajectories is not None
    assert not sim.has_trajectory
    assert sim.log == "hysplit ran\n"  # the log is always kept


def test_keep_scratch_copies_the_working_directory(
    point_receptor, tmp_path, fake_hysplit
):
    sim = _sim(tmp_path, point_receptor, keep_scratch=True)
    sim.met = _FakeMet()
    sim.run_trajectories(write=True)
    kept = sim.run.scratch_path(str(point_receptor.id))
    assert (kept / "CONTROL").read_text() == "control"
    assert not sim.directory.exists()


def test_a_failed_run_keeps_its_scratch_and_log(point_receptor, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "stilt.simulation.HYSPLITDriver",
        _fake_runner_returning(_particles_df(), fail=HYSPLITTimeoutError("too slow")),
    )
    sim = _sim(tmp_path, point_receptor)
    sim.met = _FakeMet()

    with pytest.raises(HYSPLITTimeoutError):
        sim.run_trajectories(write=True)

    assert not sim.has_trajectory
    assert sim.log == "hysplit ran\n"
    kept = sim.run.scratch_path(str(point_receptor.id))
    assert (kept / "CONTROL").exists() and (kept / "stilt.log").exists()
    assert not sim.directory.exists()
    assert sim.outcome == "failed:UNKNOWN"


def test_perturbed_params_are_stored_with_the_particles(
    point_receptor, tmp_path, fake_hysplit
):
    sim = _sim(
        tmp_path,
        point_receptor,
        siguverr=1.0,
        tluverr=60.0,
        zcoruverr=100.0,
        horcoruverr=10.0,
    )
    sim.met = _FakeMet()
    sim.run_trajectories(write=True)
    assert (
        _sim(
            tmp_path,
            point_receptor,
            siguverr=1.0,
            tluverr=60.0,
            zcoruverr=100.0,
            horcoruverr=10.0,
        ).trajectories.params.winderrtf
        == 1
    )


def test_log_property_raises_when_missing(point_receptor, tmp_path):
    with pytest.raises(FileNotFoundError):
        _ = _sim(tmp_path, point_receptor).log


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------


def _with_particles(tmp_path, receptor, **kwargs) -> Simulation:
    """A simulation with a footprint whose particles are already in the output directory."""
    sim = _sim(tmp_path, receptor, footprint=FOOT, **kwargs)
    sim.outputs.ensure_run().write_particles(_trajectories(receptor))
    return sim


def test_generate_footprint_uses_the_variant_settings_and_writes_them(
    point_receptor, tmp_path
):
    sim = _with_particles(tmp_path, point_receptor)
    foot = sim.generate_footprint(write=True)
    assert foot is not None and foot.name == "hrrr"
    assert foot.config == FOOT
    assert sim.footprint is foot
    assert sim.has_footprint and sim.is_complete()
    assert sim.footprints.name == "hrrr"

    again = _sim(tmp_path, point_receptor, footprint=FOOT)
    assert again.footprint is not None
    assert again.footprint.data.shape == foot.data.shape


def test_generate_footprint_requires_settings(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    sim.outputs.ensure_run().write_particles(_trajectories(point_receptor))
    with pytest.raises(TypeError, match="no footprint settings"):
        sim.generate_footprint()


def test_ad_hoc_settings_get_their_own_folder_and_leave_the_variant_alone(
    point_receptor, tmp_path
):
    sim = _with_particles(tmp_path, point_receptor)
    own = sim.generate_footprint(write=True)
    coarse = FOOT.model_copy(
        update={"grid": GRID.model_copy(update={"xres": 0.5, "yres": 0.5})}
    )

    other = sim.generate_footprint(coarse, write=True)

    assert other is not None and other.grid.xres == 0.5
    assert sim.footprint is own  # untouched (#65)
    names = {f.name: f for f in sim.run.footprint_sets()}
    assert len(names) == 1 and set(names) == {"hrrr"}
    folders = sim.run.footprint_sets()
    assert len(folders) == 2  # both named hrrr, different hashes
    assert {f.config.grid.xres for f in folders} == {0.1, 0.5}


def test_generate_footprint_applies_extra_python_transforms(point_receptor, tmp_path):
    sim = _with_particles(tmp_path, point_receptor)

    class Halve:
        def apply(self, particles, context):
            return particles.assign(foot=particles["foot"] * 0.5)

    base = sim.generate_footprint()
    halved = sim.generate_footprint(transforms=[Halve()])
    assert float(halved.data.sum()) == pytest.approx(0.5 * float(base.data.sum()))
    assert (
        sim.footprint is base
    )  # extra transforms do not replace the variant's footprint


def test_generate_footprint_records_the_transforms_it_applied(point_receptor, tmp_path):
    sim = _with_particles(tmp_path, point_receptor)
    decay = FirstOrderLifetime(lifetime_hours=1.0)
    foot = sim.generate_footprint(transforms=[decay])
    assert [transform_kind(t) for t in foot.config.transforms] == [
        "first_order_lifetime"
    ]


def test_generate_footprint_autoruns_hysplit_when_particles_are_missing(
    point_receptor, tmp_path, fake_hysplit
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    sim.met = _FakeMet()
    foot = sim.generate_footprint(write=True)
    assert foot is not None
    assert sim.has_trajectory and sim.has_footprint


def test_empty_footprint_is_recorded_with_its_reason(point_receptor, tmp_path):
    far = FOOT.model_copy(
        update={"grid": Grid(xmin=0, xmax=1, ymin=0, ymax=1, xres=0.1, yres=0.1)}
    )
    sim = _sim(tmp_path, point_receptor, footprint=far)
    sim.outputs.ensure_run().write_particles(_trajectories(point_receptor))

    assert sim.generate_footprint(write=True) is None

    assert sim.footprint is None
    assert sim.has_footprint and sim.is_complete()
    assert sim.empty_reason == "outside_domain"
    assert sim.outcome == "complete"


def test_transform_context_carries_receptor_variant_and_directory(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, variant="zi08")
    ctx = sim.transform_context()
    assert ctx.receptor is point_receptor
    assert ctx.variant == "zi08"
    assert ctx.directory == tmp_path


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
    with_height = _trajectories(point_receptor)
    with_height.data["xhgt"] = 10.0  # the kernel weights particles by release height
    sim = _sim(tmp_path, point_receptor, footprint=config)
    sim.outputs.ensure_run().write_particles(with_height)
    plain = _sim(tmp_path, point_receptor, footprint=FOOT, variant="plain")
    assert plain.run == sim.run  # same transport settings: the particles are shared

    weighted = sim.generate_footprint()
    base = plain.generate_footprint()
    assert float(weighted.data.sum()) == pytest.approx(0.5 * float(base.data.sum()))


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


def test_completion_needs_particles_and_the_footprint_when_configured(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    assert not sim.is_complete()
    sim.outputs.ensure_run().write_particles(_trajectories(point_receptor))
    assert sim.has_trajectory and not sim.is_complete()
    sim.generate_footprint(write=True)
    assert sim.is_complete()


def test_particles_only_variant_is_complete_with_particles(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert not sim.makes_footprint
    sim.outputs.ensure_run().write_particles(_trajectories(point_receptor))
    assert sim.is_complete()


def test_variants_with_equal_transport_share_particles(point_receptor, tmp_path):
    fine = _with_particles(tmp_path, point_receptor)
    coarse = _sim(
        tmp_path,
        point_receptor,
        variant="coarse",
        footprint=FOOT.model_copy(
            update={"grid": GRID.model_copy(update={"xres": 0.5, "yres": 0.5})}
        ),
    )
    assert coarse.run == fine.run
    assert coarse.has_trajectory  # written under fine's variant
    assert fine.generate_footprint(write=True) is not None
    assert coarse.generate_footprint(write=True) is not None
    assert {f.name for f in fine.run.footprint_sets()} == {"hrrr", "coarse"}
    assert fine.footprints != coarse.footprints


def test_footprint_from_a_written_netcdf_matches(point_receptor, tmp_path):
    """Sanity: what the output directory hands back is a Footprint with the receptor and grid."""
    sim = _with_particles(tmp_path, point_receptor)
    foot = sim.generate_footprint(write=True)
    back = _sim(tmp_path, point_receptor, footprint=FOOT).footprint
    assert isinstance(back, Footprint)
    assert back.receptor == point_receptor and back.grid == GRID
    xr.testing.assert_allclose(back.data, foot.data.astype("float32").astype("float64"))
