"""Tests for stilt.simulation (SimID and Simulation behavior)."""

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

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
from stilt.errors import EmptyFootprintError, HYSPLITTimeoutError
from stilt.footprint import Footprint
from stilt.hysplit.driver import HYSPLITResult
from stilt.meteorology import MetStream
from stilt.simulation import SimID, Simulation
from stilt.store import LocalStore
from stilt.trajectory import Trajectories
from stilt.transforms import FirstOrderLifetime

GRID = Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)
FOOT = FootprintConfig(grid=GRID, time_integrate=True, smooth_factor=0.0)


def _params(**kwargs) -> STILTParams:
    data = {"n_hours": -24, "numpar": 10, "hnf_plume": False}
    data.update(kwargs)
    return STILTParams(**data)


def _variant(
    name="hrrr",
    footprint: FootprintConfig | None = None,
    derived_from=None,
    **overrides,
) -> VariantConfig:
    """A resolved variant with the test transport defaults and an optional footprint."""
    data = {"n_hours": -24, "numpar": 10, "hnf_plume": False, **overrides}
    if footprint is not None:
        data.update(footprint.model_dump())
    return VariantConfig(
        name=name, group=name, met="hrrr", derived_from=derived_from, **data
    )


def _met(tmp_path, **kwargs) -> MetStream:
    mc = MetConfig(
        directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="1h", **kwargs
    )
    return MetStream(
        "hrrr",
        directory=mc.directory,
        file_format=mc.file_format,
        file_tres=mc.file_tres,
        n_min=mc.n_min,
        source_type=mc.source,
        source_kwargs=mc.source_kwargs,
        backend=mc.backend,
        subgrid_enable=mc.subgrid_enable,
        subgrid_bounds=mc.subgrid_bounds,
        subgrid_buffer=mc.subgrid_buffer,
        subgrid_levels=mc.subgrid_levels,
        subgrid_dir=mc.subgrid_dir,
    )


def _sim(
    tmp_path,
    receptor,
    *,
    footprint=None,
    variant="hrrr",
    store=None,
    met_kwargs=None,
    mkdir=True,
    **param_overrides,
) -> Simulation:
    sim_dir = tmp_path / "simulations" / "by-id" / SimID(receptor.id, variant)
    if mkdir:
        sim_dir.mkdir(parents=True, exist_ok=True)
    return Simulation(
        receptor,
        _variant(variant, footprint=footprint, **param_overrides),
        met=_met(tmp_path, **(met_kwargs or {})),
        directory=sim_dir,
        store=store,
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


def _column_particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, -60.0, -120.0],
            "indx": [1, 1, 1],
            "long": [-111.86, -111.86, -111.86],
            "lati": [40.76, 40.76, 40.76],
            "zagl": [50.0, 50.0, 50.0],
            "foot": [1.0, 1.0, 1.0],
        }
    )


def _write_trajectory(path: Path, receptor, root: Path, foot: float = 1e-5) -> None:
    traj = Trajectories.from_particles(
        particles=_particles_df(foot),
        receptor=receptor,
        params=_params(),
        met_files=[root / "metfile"],
    )
    traj.to_parquet(path)


class _FakeMet:
    id = "hrrr"

    def required_files(self, **kwargs):
        return []

    def stage_files_for_simulation(self, **kwargs):
        return []


def _fake_runner_returning(particles):
    """A stand-in HYSPLITDriver that writes ``stilt.log`` and returns *particles*."""

    class _FakeRunner:
        def __init__(self, *, directory, **kwargs):
            self.log_path = Path(directory) / "stilt.log"

        def prepare(self):
            return None

        def execute(self, timeout, rm_dat):
            self.log_path.write_text("ok")
            return HYSPLITResult(particles=particles, log_path=self.log_path)

    return _FakeRunner


# ---------------------------------------------------------------------------
# SimID
# ---------------------------------------------------------------------------


def test_simid_is_a_receptor_and_a_variant(point_receptor):
    sid = SimID(point_receptor.id, "hrrr")
    assert sid.receptor == point_receptor.id
    assert sid.variant == "hrrr"
    assert str(sid) == f"{point_receptor.id}/hrrr"


def test_simid_parses_its_string_and_tuple_forms(point_receptor):
    value = f"{point_receptor.id}/hrrr-zi08"
    sid = SimID.parse(value)
    assert sid == SimID(point_receptor.id, "hrrr-zi08")
    assert sid.receptor.time == pd.Timestamp(point_receptor.time)
    assert SimID.parse((str(point_receptor.id), "hrrr-zi08")) == sid
    assert SimID.parse(sid) is sid


@pytest.mark.parametrize("bad", ["20230101", "202301011200_-111.85_40.77_5", "a/"])
def test_simid_parse_rejects_malformed_ids(bad):
    with pytest.raises(ValueError):
        SimID.parse(bad)


def test_simid_is_pathlike(point_receptor, tmp_path):
    sid = SimID(point_receptor.id, "hrrr")
    p = tmp_path / "simulations" / "by-id" / sid
    assert p.parent.name == str(point_receptor.id)
    assert p.name == "hrrr"


# ---------------------------------------------------------------------------
# Construction and layout
# ---------------------------------------------------------------------------


def test_construction_does_not_create_directory(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, mkdir=False)

    assert not sim.directory.exists()
    assert not sim.has_trajectory
    assert sim.outcome is None


def test_simulation_id_is_receptor_and_variant_name(point_receptor, tmp_path):
    sim = Simulation(point_receptor, _variant("zi08"), met=_met(tmp_path))
    assert sim.id == SimID(point_receptor.id, "zi08")
    assert sim.directory.parts[-2:] == (str(point_receptor.id), "zi08")
    assert sim.params.numpar == 10 and sim.footprint_config is None


def test_a_simulation_needs_a_met_stream_unless_derived(point_receptor, tmp_path):
    with pytest.raises(ValueError, match="needs a met stream"):
        Simulation(point_receptor, _variant())


def test_output_paths_live_in_the_variant_directory(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    rid = str(point_receptor.id)

    assert sim.trajectories_path == sim.directory / f"{rid}_traj.parquet"
    assert sim.footprint_path == sim.directory / f"{rid}_foot.nc"
    assert sim.empty_footprint_path == sim.directory / f"{rid}_foot.empty"
    assert sim.met_dir == sim.directory / "met"


def test_key_prefix_and_key(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, variant="hrrr-err-1")
    rid = str(point_receptor.id)
    prefix = f"simulations/by-id/{rid}/hrrr-err-1"

    assert sim.key_prefix == prefix
    assert sim.key(sim.trajectories_path) == f"{prefix}/{rid}_traj.parquet"
    assert sim.key(sim.log_path) == f"{prefix}/stilt.log"
    assert sim.key(sim.footprint_path) == f"{prefix}/{rid}_foot.nc"
    assert sim.key(sim.empty_footprint_path) == f"{prefix}/{rid}_foot.empty"
    # Only the basename matters: an out-of-tree path maps onto this sim's prefix.
    assert sim.key("elsewhere/stilt.log") == f"{prefix}/stilt.log"


def test_resolve_prefers_local_then_store_then_none(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    sim = _sim(tmp_path / "cache", point_receptor, store=LocalStore(storage_root))

    assert sim.resolve(sim.log_path) is None

    remote_log = storage_root / sim.key_prefix / "stilt.log"
    remote_log.parent.mkdir(parents=True)
    remote_log.write_text("remote")
    assert sim.resolve(sim.log_path) == remote_log

    sim.log_path.write_text("local")
    assert sim.resolve(sim.log_path) == sim.log_path


def test_simulation_time_range_backward(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, n_hours=-24)
    start, stop = sim.time_range
    assert stop == point_receptor.time
    assert start < stop


def test_simulation_time_range_forward(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, n_hours=24)
    start, stop = sim.time_range
    assert start == point_receptor.time
    assert stop - start == dt.timedelta(hours=24)


# ---------------------------------------------------------------------------
# Meteorology
# ---------------------------------------------------------------------------


def test_meteorology_subgrid_requires_bounds(point_receptor, tmp_path):
    """subgrid_enable=True without bounds raises a ValidationError at config time."""
    with pytest.raises(Exception, match="subgrid_bounds is required"):
        _sim(tmp_path, point_receptor, met_kwargs={"subgrid_enable": True})


def test_meteorology_subgrid_enable_accepts_bool(point_receptor, tmp_path):
    from stilt.config.spatial import Bounds

    sim = _sim(
        tmp_path,
        point_receptor,
        met_kwargs={
            "subgrid_enable": True,
            "subgrid_bounds": Bounds(xmin=-114, xmax=-110, ymin=39, ymax=42),
        },
    )
    assert sim.met is not None
    assert sim.met.subgrid_enable is True


def test_simulation_met_files_stage_into_the_variant_directory(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor)
    assert sim.met is not None
    sim.met.directory.mkdir(parents=True, exist_ok=True)
    source = sim.met.directory / point_receptor.time.strftime(sim.met.file_format)
    source.touch()

    staged = sim.met_files

    assert staged == [sim.met_dir / source.name]
    assert staged[0].exists()
    assert sim.source_met_files == [source]


# ---------------------------------------------------------------------------
# Running HYSPLIT
# ---------------------------------------------------------------------------


def test_run_trajectories_uses_source_met_files_in_metadata(
    monkeypatch, point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor)
    source_dir = tmp_path / "archive" / "hrrr"
    source_dir.mkdir(parents=True)
    source_file = source_dir / point_receptor.time.strftime("%Y%m%d_%H")
    source_file.touch()
    sim.met = MetStream(
        "hrrr", directory=source_dir, file_format="%Y%m%d_%H", file_tres="1h"
    )
    seen: dict[str, list[Path]] = {}
    runner = _fake_runner_returning(_particles_df())

    class _Recording(runner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            seen["runner_met_files"] = kwargs["met_files"]

    def _fake_from_particles(particles, *, receptor, params, met_files):
        seen["traj_met_files"] = met_files
        return Trajectories(
            receptor=receptor,
            params=params,
            met_files=met_files,
            data=particles.assign(datetime=pd.to_datetime(point_receptor.time)),
        )

    monkeypatch.setattr("stilt.simulation.HYSPLITDriver", _Recording)
    monkeypatch.setattr(
        "stilt.simulation.Trajectories.from_particles", _fake_from_particles
    )

    sim.run_trajectories(timeout=1, rm_dat=False, write=False)

    assert seen["runner_met_files"] == [sim.met_dir / source_file.name]
    assert seen["traj_met_files"] == [source_file]


def test_run_trajectories_timeout_maps_to_domain_error(
    monkeypatch, point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor)

    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        def prepare(self):
            return None

        def execute(self, timeout, rm_dat):
            raise HYSPLITTimeoutError("boom")

    monkeypatch.setattr("stilt.simulation.HYSPLITDriver", _FakeRunner)
    monkeypatch.setattr(sim, "met", _FakeMet())

    with pytest.raises(HYSPLITTimeoutError):
        sim.run_trajectories(timeout=1, rm_dat=False)


def test_run_trajectories_creates_directory_and_writes(
    monkeypatch, point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, mkdir=False)
    monkeypatch.setattr(
        "stilt.simulation.HYSPLITDriver", _fake_runner_returning(_particles_df())
    )
    monkeypatch.setattr(sim, "met", _FakeMet())

    sim.run_trajectories(timeout=1, rm_dat=False, write=True)

    assert sim.directory.is_dir()
    assert sim.trajectories_path.exists()
    assert sim.has_trajectory
    assert sim.log_path.read_text() == "ok"


def test_perturbed_params_are_stored_with_the_trajectory(
    monkeypatch, point_receptor, tmp_path
):
    """A trajectory's own params say whether it was perturbed; no flag needed."""
    err = dict(siguverr=1.0, tluverr=60.0, zcoruverr=500.0, horcoruverr=40.0)
    sim = _sim(tmp_path, point_receptor, variant="hrrr-err", **err)
    monkeypatch.setattr(
        "stilt.simulation.HYSPLITDriver", _fake_runner_returning(_particles_df())
    )
    monkeypatch.setattr(sim, "met", _FakeMet())

    sim.run_trajectories(write=True)

    loaded = Trajectories.from_parquet(sim.trajectories_path)
    assert loaded.params.winderrtf == 1
    assert loaded.params.siguverr == 1.0


def test_derived_simulation_refuses_to_run_hysplit(point_receptor, tmp_path):
    parent = _sim(tmp_path, point_receptor)
    derived = Simulation(
        point_receptor,
        _variant("hrrr-s2", footprint=FOOT, derived_from="hrrr"),
        parent=parent,
    )
    with pytest.raises(ValueError, match="derived"):
        derived.run_trajectories()


# ---------------------------------------------------------------------------
# Loading outputs
# ---------------------------------------------------------------------------


def test_log_property_raises_when_missing(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    with pytest.raises(FileNotFoundError):
        _ = sim.log


def test_simulation_log_loads_from_store(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    sim = _sim(tmp_path / "cache", point_receptor, store=LocalStore(storage_root))
    remote = storage_root / sim.key(sim.log_path)
    remote.parent.mkdir(parents=True)
    remote.write_text("from the store")

    assert sim.log == "from the store"


def test_trajectories_none_when_no_parquet(point_receptor, tmp_path):
    assert _sim(tmp_path, point_receptor).trajectories is None


def test_trajectories_load_from_store(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    sim = _sim(tmp_path / "cache", point_receptor, store=LocalStore(storage_root))
    remote = storage_root / sim.key(sim.trajectories_path)
    remote.parent.mkdir(parents=True)
    _write_trajectory(remote, point_receptor, storage_root)

    assert sim.trajectories is not None
    assert sim.trajectories.receptor.id == point_receptor.id


def test_footprint_none_when_not_present(point_receptor, tmp_path):
    assert _sim(tmp_path, point_receptor).footprint is None


def test_footprint_loads_from_store(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    sim = _sim(tmp_path / "cache", point_receptor, store=LocalStore(storage_root))
    foot_path = storage_root / sim.key(sim.footprint_path)
    foot_path.parent.mkdir(parents=True)
    data = xr.DataArray(
        [[[1.0]]],
        coords={
            "time": [pd.Timestamp(point_receptor.time)],
            "lat": [40.77],
            "lon": [-111.85],
        },
        dims=("time", "lat", "lon"),
    )
    config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-113.0, ymin=39.0, ymax=40.0, xres=0.1, yres=0.1)
    )
    Footprint(point_receptor, config, data, name="hrrr").to_netcdf(foot_path)

    foot = sim.footprint
    assert foot is not None
    assert foot.name == "hrrr"


def test_status_reads_outputs_and_log_from_the_store(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    sim = _sim(tmp_path / "cache", point_receptor, store=LocalStore(storage_root))
    remote = storage_root / sim.key(sim.trajectories_path)
    remote.parent.mkdir(parents=True)
    _write_trajectory(remote, point_receptor, storage_root)
    assert sim.outcome == "complete"

    remote.unlink()
    (remote.parent / "stilt.log").write_text(
        "Insufficient number of meteorological files found"
    )
    assert "MISSING_MET_FILES" in str(sim.outcome)


def test_derived_simulation_reads_its_parent_trajectory(point_receptor, tmp_path):
    parent = _sim(tmp_path, point_receptor)
    _write_trajectory(parent.trajectories_path, point_receptor, tmp_path)
    derived = Simulation(
        point_receptor,
        _variant("hrrr-s2", footprint=FOOT, derived_from="hrrr"),
        parent=parent,
        directory=tmp_path
        / "simulations"
        / "by-id"
        / SimID(point_receptor.id, "hrrr-s2"),
    )

    assert derived.is_derived
    assert derived.trajectories_path == parent.trajectories_path
    assert derived.has_trajectory
    assert derived.trajectories is parent.trajectories
    assert derived.footprint_path.parent == derived.directory
    assert derived._met_stream() is parent.met


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------


def _sim_with_particles(tmp_path, receptor, **kwargs) -> Simulation:
    sim = _sim(tmp_path, receptor, footprint=FOOT, **kwargs)
    sim._trajectories = Trajectories.from_particles(
        particles=_column_particles(),
        receptor=receptor,
        params=_params(),
        met_files=[tmp_path / "metfile"],
    )
    return sim


def test_generate_footprint_uses_the_simulation_footprint_by_default(
    point_receptor, tmp_path
):
    sim = _sim_with_particles(tmp_path, point_receptor)

    foot = sim.generate_footprint(write=True)

    assert float(foot.data.sum()) > 0
    assert foot.name == "hrrr"
    assert sim.footprint_path.exists()
    assert sim.footprint is foot


def test_generate_footprint_requires_settings(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    with pytest.raises(TypeError, match="no footprint settings"):
        sim.generate_footprint()


def test_generate_footprint_accepts_ad_hoc_settings(point_receptor, tmp_path):
    sim = _sim_with_particles(tmp_path, point_receptor)
    base = sim.generate_footprint()
    transformed = sim.generate_footprint(
        FOOT.model_copy(update={"transforms": [FirstOrderLifetime(lifetime_hours=1.0)]})
    )
    assert float(transformed.data.sum()) < float(base.data.sum())
    assert not sim.footprint_path.exists()  # nothing written without write=True


class HalvingTransform:
    """Python-side transform that halves ``foot`` and records its context."""

    def __init__(self):
        self.context = None

    def apply(self, particles, context):
        self.context = context
        out = particles.copy()
        out["foot"] = out["foot"] * 0.5
        return out


def test_generate_footprint_applies_extra_python_transforms(point_receptor, tmp_path):
    sim = _sim_with_particles(tmp_path, point_receptor, variant="hrrr-ak")
    halve = HalvingTransform()

    base = sim.generate_footprint()
    halved = sim.generate_footprint(transforms=[halve])

    assert float(base.data.sum()) > 0
    assert float(halved.data.sum()) == pytest.approx(0.5 * float(base.data.sum()))
    assert halve.context is not None
    assert halve.context.receptor is sim.receptor
    assert halve.context.variant == "hrrr-ak"


def test_generate_footprint_applies_dotted_path_config_transforms(
    point_receptor, tmp_path
):
    sim = _sim_with_particles(tmp_path, point_receptor)
    dotted = FootprintConfig(
        grid=GRID,
        time_integrate=True,
        smooth_factor=0.0,
        transforms=[{"kind": f"{__name__}.{HalvingTransform.__name__}"}],
    )
    assert isinstance(dotted.transforms[0], HalvingTransform)

    base = sim.generate_footprint()
    halved = sim.generate_footprint(dotted)

    assert float(halved.data.sum()) == pytest.approx(0.5 * float(base.data.sum()))
    assert dotted.transforms[0].context.variant == "hrrr"


def test_generate_footprint_autoruns_trajectories_when_missing(
    point_receptor, tmp_path, monkeypatch
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    run_calls: list[bool] = []

    def _fake_run_trajectories(*, write: bool = False, **kwargs) -> None:
        run_calls.append(write)
        sim._trajectories = Trajectories.from_particles(
            particles=_column_particles(),
            receptor=point_receptor,
            params=_params(),
            met_files=[tmp_path / "metfile"],
        )

    monkeypatch.setattr(sim, "run_trajectories", _fake_run_trajectories)

    foot = sim.generate_footprint()

    assert foot is not None
    assert run_calls == [False]
    assert sim.trajectories is not None


def test_generate_footprint_uses_the_receptor_kernel_from_a_project_table(
    column_receptor, tmp_path
):
    """Two receptors, one config, one table: each footprint gets its own kernel."""
    from stilt.receptors import ColumnReceptor
    from stilt.transforms import AveragingKernel, averaging_kernel_table

    store = LocalStore(tmp_path)
    other = ColumnReceptor(
        column_receptor.time,
        column_receptor.longitude + 0.1,
        column_receptor.latitude,
        column_receptor.bottom,
        column_receptor.top,
    )
    averaging_kernel_table(
        [column_receptor, other],
        levels=[0.0, 3000.0],
        values=[[1.0, 1.0], [0.25, 0.25]],
    ).to_parquet(tmp_path / "kernels.parquet")
    config = FOOT.model_copy(
        update={"transforms": [AveragingKernel(table="kernels.parquet")]}
    )

    sums = {}
    for receptor in (column_receptor, other):
        sim = _sim(tmp_path, receptor, store=store, footprint=config)
        particles = pd.DataFrame(
            {
                "time": [0.0, -60.0, 0.0, -60.0],
                "indx": [1, 1, 2, 2],
                "long": [receptor.longitude] * 4,
                "lati": [receptor.latitude] * 4,
                "zagl": [500.0, 500.0, 2500.0, 2500.0],
                "xhgt": [500.0, 500.0, 2500.0, 2500.0],
                "foot": [1.0, 1.0, 1.0, 1.0],
            }
        )
        sim._trajectories = Trajectories.from_particles(
            particles=particles,
            receptor=receptor,
            params=_params(),
            met_files=[tmp_path / "metfile"],
        )
        sums[receptor.id] = float(sim.generate_footprint().data.sum())

    assert sums[column_receptor.id] > 0
    assert sums[other.id] == pytest.approx(0.25 * sums[column_receptor.id])


def test_transform_context_carries_receptor_variant_and_store(point_receptor, tmp_path):
    store = LocalStore(tmp_path)
    sim = _sim(tmp_path, point_receptor, variant="hrrr-err-2", store=store)

    ctx = sim.transform_context()

    assert ctx.receptor is point_receptor
    assert ctx.variant == "hrrr-err-2"
    assert ctx.store is store
    assert _sim(tmp_path, point_receptor).transform_context().store is None


# ---------------------------------------------------------------------------
# Expected outputs and completion
# ---------------------------------------------------------------------------


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_what_a_simulation_produces(point_receptor, tmp_path):
    traj_only = _sim(tmp_path, point_receptor)
    assert not traj_only.is_derived and not traj_only.makes_footprint
    both = _sim(tmp_path, point_receptor, footprint=FOOT)
    assert not both.is_derived and both.makes_footprint


def test_is_complete_needs_every_expected_output(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    assert not sim.is_complete()
    _touch(sim.trajectories_path)
    assert not sim.is_complete()
    _touch(sim.footprint_path)
    assert sim.is_complete()


def test_completion_requires_the_footprint_when_one_is_configured(
    point_receptor, tmp_path
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _touch(sim.trajectories_path)
    assert not sim.is_complete()

    _touch(sim.footprint_path)
    assert sim.is_complete()


def test_completion_requires_the_trajectory_even_when_the_footprint_exists(
    point_receptor, tmp_path
):
    """A lost trajectory makes the simulation incomplete; the worker backfills it."""
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _touch(sim.footprint_path)
    assert sim.has_footprint and not sim.has_trajectory
    assert not sim.is_complete()


def test_empty_footprint_marker_counts_complete(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _touch(sim.trajectories_path)
    sim.write_empty_footprint_marker("outside_domain")
    assert sim.is_complete()
    assert sim.empty_reason == "outside_domain"
    assert sim.footprint is None


def test_has_footprint_falls_back_to_store(point_receptor, tmp_path):
    storage_root = tmp_path / "remote"
    sim = _sim(tmp_path / "cache", point_receptor, store=LocalStore(storage_root))
    assert not sim.has_footprint
    _touch(storage_root / sim.key(sim.footprint_path))
    assert sim.has_footprint


def test_derived_simulation_needs_only_its_footprint(point_receptor, tmp_path):
    parent = _sim(tmp_path, point_receptor)
    derived = Simulation(
        point_receptor,
        _variant("hrrr-s2", footprint=FOOT, derived_from="hrrr"),
        parent=parent,
        directory=tmp_path / "d",
    )
    assert derived.is_derived and derived.makes_footprint
    _touch(derived.footprint_path)
    assert derived.is_complete()


def test_write_and_clear_empty_footprint_marker(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    assert sim.empty_reason is None
    marker = sim.write_empty_footprint_marker("no_particles")

    assert marker == sim.empty_footprint_path
    assert marker.name.endswith("_foot.empty")
    assert marker.read_text() == "no_particles\n"
    assert sim.has_footprint
    assert sim.empty_reason == "no_particles"

    sim.clear_empty_footprint_marker()
    assert not marker.exists()
    assert not sim.has_footprint
    assert sim.empty_reason is None
    # Clearing twice is harmless.
    sim.clear_empty_footprint_marker()


def test_empty_marker_without_reason_reads_unknown(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    sim.empty_footprint_path.parent.mkdir(parents=True, exist_ok=True)
    sim.empty_footprint_path.touch()
    assert sim.empty_reason == "unknown"


def _stub_trajectories(sim, receptor):
    sim._trajectories = SimpleNamespace(
        data=pd.DataFrame({"indx": [1], "foot": [1.0]}), receptor=receptor
    )


def test_generate_footprint_writes_marker_and_removes_stale_netcdf(
    point_receptor, tmp_path, monkeypatch
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _stub_trajectories(sim, point_receptor)
    _touch(sim.footprint_path)

    def raise_empty(*args, **kwargs):
        raise EmptyFootprintError("outside_domain")

    monkeypatch.setattr(Footprint, "calculate", raise_empty)

    assert sim.generate_footprint(write=True) is None
    assert sim.empty_reason == "outside_domain"
    assert not sim.footprint_path.exists()
    assert sim.footprint is None
    assert sim.is_complete() or not sim.has_trajectory


def test_generate_footprint_clears_stale_marker(point_receptor, tmp_path, monkeypatch):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _stub_trajectories(sim, point_receptor)
    sim.write_empty_footprint_marker("outside_domain")
    written: list[Path] = []
    foot = SimpleNamespace(to_netcdf=lambda path: written.append(path))
    monkeypatch.setattr(Footprint, "calculate", lambda *a, **k: foot)

    assert sim.generate_footprint(write=True) is foot
    assert written == [sim.footprint_path]
    assert sim.empty_reason is None


def test_generate_footprint_without_write_leaves_files_alone(
    point_receptor, tmp_path, monkeypatch
):
    sim = _sim(tmp_path, point_receptor, footprint=FOOT)
    _stub_trajectories(sim, point_receptor)
    _touch(sim.footprint_path)

    def raise_empty(*args, **kwargs):
        raise EmptyFootprintError("no_particles")

    monkeypatch.setattr(Footprint, "calculate", raise_empty)

    assert sim.generate_footprint() is None
    assert sim.footprint_path.exists()
    assert sim.empty_reason is None


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


def _write_all_outputs(sim):
    sim.directory.mkdir(parents=True, exist_ok=True)
    sim.log_path.write_text("log")
    sim.trajectories_path.write_bytes(b"traj")
    sim.footprint_path.write_bytes(b"foot")


def test_publish_copies_outputs_into_store(point_receptor, tmp_path):
    storage_root = tmp_path / "output"
    sim = _sim(tmp_path / "compute", point_receptor, store=LocalStore(storage_root))
    _write_all_outputs(sim)
    sim.met_dir.mkdir()
    (sim.met_dir / "staged").write_text("met")

    sim.publish()

    published = storage_root / sim.key_prefix
    assert (published / "stilt.log").read_text() == "log"
    assert (published / sim.trajectories_path.name).read_bytes() == b"traj"
    assert (published / sim.footprint_path.name).read_bytes() == b"foot"
    assert not (published / "met").exists()  # staged met is compute-local
    assert not list(published.glob("*.tmp"))
    assert sorted(p.name for p in storage_root.iterdir()) == ["simulations"]


def test_publish_skips_missing_outputs(point_receptor, tmp_path):
    storage_root = tmp_path / "output"
    sim = _sim(tmp_path / "compute", point_receptor, store=LocalStore(storage_root))
    sim.trajectories_path.write_bytes(b"traj")

    sim.publish()

    published = storage_root / sim.key_prefix
    assert (published / sim.trajectories_path.name).read_bytes() == b"traj"
    assert not (published / "stilt.log").exists()
    assert not (published / sim.footprint_path.name).exists()


def test_publish_derived_does_not_copy_the_parent_trajectory(point_receptor, tmp_path):
    storage_root = tmp_path / "output"
    store = LocalStore(storage_root)
    parent = _sim(tmp_path / "compute", point_receptor, store=store)
    parent.trajectories_path.write_bytes(b"traj")
    derived = Simulation(
        point_receptor,
        _variant("hrrr-s2", footprint=FOOT, derived_from="hrrr"),
        parent=parent,
        directory=tmp_path
        / "compute"
        / "simulations"
        / "by-id"
        / SimID(point_receptor.id, "hrrr-s2"),
        store=store,
    )
    _touch(derived.footprint_path)

    derived.publish()

    published = storage_root / derived.key_prefix
    assert (published / derived.footprint_path.name).exists()
    assert not (published / parent.trajectories_path.name).exists()


def test_publish_noop_when_store_root_contains_sim_directory(point_receptor, tmp_path):
    """When the store's location for the sim *is* its directory, nothing is copied."""
    sim = _sim(tmp_path, point_receptor, store=LocalStore(tmp_path))
    assert sim.directory == LocalStore(tmp_path).local_path(sim.key_prefix)
    _write_all_outputs(sim)
    before = {
        p.name: p.stat().st_mtime_ns for p in sim.directory.iterdir() if p.is_file()
    }

    sim.publish()

    after = {
        p.name: p.stat().st_mtime_ns for p in sim.directory.iterdir() if p.is_file()
    }
    assert after == before
    assert sim.trajectories_path.read_bytes() == b"traj"


def test_publish_noop_without_store(point_receptor, tmp_path):
    sim = _sim(tmp_path, point_receptor)
    _write_all_outputs(sim)
    sim.publish()  # must not raise
    assert sim.trajectories_path.read_bytes() == b"traj"


def test_publish_without_directory_is_noop(point_receptor, tmp_path):
    storage_root = tmp_path / "output"
    sim = _sim(
        tmp_path / "compute",
        point_receptor,
        store=LocalStore(storage_root),
        mkdir=False,
    )

    sim.publish()

    assert not (storage_root / sim.key_prefix).exists()
