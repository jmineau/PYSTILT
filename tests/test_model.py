"""Tests for stilt.model: project inputs, variants, simulation collections, run()."""

import datetime as dt
import uuid
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.collections import (
    FOOTPRINT,
    TRAJECTORY,
    OutputCollection,
    SimulationCollection,
)
from stilt.config import Grid, MetConfig, ModelConfig, RuntimeSettings
from stilt.errors import ConfigChangedError, ConfigValidationError
from stilt.execution import LocalHandle, SlurmExecutor
from stilt.footprint import Footprint
from stilt.model import Model
from stilt.project import CONFIG_KEY, RECEPTORS_KEY
from stilt.receptors import PointReceptor
from stilt.simulation import SimID, Simulation
from stilt.trajectory import Trajectories

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


def _config(tmp_path, include_footprint=True, **overrides) -> ModelConfig:
    """Minimal ModelConfig with one met stream and (optionally) a footprint grid."""
    overrides.setdefault("grid", _GRID if include_footprint else None)
    return ModelConfig(mets={"hrrr": _met(tmp_path)}, **overrides)


def _receptor(hour: int, longitude: float = -111.85) -> PointReceptor:
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, hour),
        longitude=longitude,
        latitude=40.77,
        altitude=5.0,
    )


def _rid(receptor) -> str:
    return str(receptor.id)


def _sid(receptor, variant="hrrr") -> SimID:
    return SimID(receptor.id, variant)


def _write_trajectory(model: Model, sid) -> Path:
    """Write a stub trajectory output."""
    sim = model.simulation(sid)
    sim.directory.mkdir(parents=True, exist_ok=True)
    sim.trajectories_path.write_bytes(b"stub")
    return sim.trajectories_path


def _write_footprint(model: Model, sid, *, empty=False) -> Path:
    """Write a stub footprint output (or an empty marker)."""
    sim = model.simulation(sid)
    sim.directory.mkdir(parents=True, exist_ok=True)
    if empty:
        return sim.write_empty_footprint_marker()
    sim.footprint_path.write_bytes(b"stub")
    return sim.footprint_path


def _particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-60, -120],
            "indx": [1, 1],
            "long": [-111.9, -112.0],
            "lati": [40.7, 40.6],
            "zagl": [10.0, 20.0],
            "foot": [1e-5, 2e-5],
            "dens": [1.2, 1.2],
            "samt": [1.0, 1.0],
            "sigw": [0.1, 0.1],
            "tlgr": [10.0, 10.0],
            "mlht": [500.0, 500.0],
        }
    )


def _write_real_trajectory(model: Model, sid) -> Path:
    """Write a loadable trajectory parquet built from stub particles."""
    sim = model.simulation(sid)
    traj = Trajectories.from_particles(
        _particles(), receptor=sim.receptor, params=sim.params, met_files=[]
    )
    traj.to_parquet(sim.trajectories_path)
    return sim.trajectories_path


def _write_real_footprint(model: Model, sid) -> Path:
    """Write a loadable footprint netCDF."""
    sim = model.simulation(sid)
    assert sim.footprint_config is not None
    lons = np.array([-113.95, -113.85])
    lats = np.array([39.05, 39.15])
    data = xr.DataArray(
        np.random.rand(1, len(lats), len(lons)),
        dims=("time", "lat", "lon"),
        coords={"time": [sim.receptor.time], "lat": lats, "lon": lons},
    )
    foot = Footprint(
        receptor=sim.receptor, config=sim.footprint_config, data=data, name=sim.variant
    )
    foot.to_netcdf(sim.footprint_path)
    return sim.footprint_path


def _memory_project() -> str:
    """Unique memory:// root (the memory filesystem is process-global)."""
    return f"memory://bucket/{uuid.uuid4().hex}"


class _CapturingExecutor:
    """Fake executor that records start() calls without running workers."""

    dispatch = "push"

    def __init__(self, handle=None):
        self.handle = handle if handle is not None else LocalHandle()
        self.start_calls: list[dict] = []

    def start(self, pending: list[str], **kwargs):
        self.start_calls.append({"pending": list(pending), **kwargs})
        return self.handle

    @property
    def was_started(self) -> bool:
        return len(self.start_calls) > 0


class _CountingHandle:
    job_id = "fake"
    detached = False
    done = True

    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


# ---------------------------------------------------------------------------
# Receptors
# ---------------------------------------------------------------------------


def test_receptors_from_csv(tmp_path):
    csv = tmp_path / "in.csv"
    csv.write_text(
        "time,latitude,longitude,altitude\n2023-01-01 12:00:00,40.77,-111.85,5.0\n"
    )

    model = Model(project=tmp_path, receptors=csv)

    assert len(model.receptors) == 1
    assert model.receptors[0].latitude == pytest.approx(40.77)
    assert model.receptors.source_path == csv


def test_model_accepts_empty_receptor_list(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    assert len(model.receptors) == 0
    assert list(model.receptors) == []
    assert model.register() == []


def test_receptors_support_lookup_by_receptor_id(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    sid = _sid(point_receptor)

    assert model.receptors[sid.receptor] == point_receptor
    assert sid.receptor in model.receptors
    with pytest.raises(KeyError):
        model.receptors["nope"]


def test_receptors_are_normalized_once(tmp_path):
    model = Model(project=tmp_path, receptors=[_receptor(12)])

    assert model.receptors is model.receptors


def test_receptors_must_be_receptor_objects_or_a_path(tmp_path):
    model = Model(
        project=tmp_path, receptors=[("2023-01-01 12:00", -111.85, 40.77, 5.0)]
    )
    with pytest.raises(TypeError, match="Receptor"):
        len(model.receptors)


def test_receptors_raise_when_nothing_available(tmp_path):
    model = Model(project=tmp_path, receptors=None)

    with pytest.raises(FileNotFoundError, match="No receptors available"):
        len(model.receptors)


def test_receptors_default_to_project_receptors_csv(tmp_path, point_receptor):
    Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    ).register()

    model = Model(project=tmp_path)

    assert len(model.receptors) == 1
    assert model.receptors[0].id == point_receptor.id


# ---------------------------------------------------------------------------
# Config, name, repr, compute_root, queue
# ---------------------------------------------------------------------------


def test_config_raises_when_no_yaml(tmp_path):
    model = Model(project=tmp_path)

    with pytest.raises(FileNotFoundError, match="config.yaml"):
        _ = model.config


def test_config_loads_from_project_yaml(tmp_path):
    _config(tmp_path, n_hours=-12).to_yaml(tmp_path / CONFIG_KEY)

    model = Model(project=tmp_path)

    assert isinstance(model.config, ModelConfig)
    assert model.config.n_hours == -12
    assert model.config.mets["hrrr"].directory == tmp_path / "met"


def test_model_kwargs_build_config(tmp_path):
    model = Model(
        project=tmp_path,
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        n_hours=-12,
    )

    assert isinstance(model.config, ModelConfig)
    assert model.config.n_hours == -12
    assert pd.to_timedelta(model.config.mets["hrrr"].file_tres) == pd.Timedelta("1h")


def test_model_config_and_kwargs_raises(tmp_path):
    with pytest.raises(TypeError, match="Cannot pass both"):
        Model(project=tmp_path, config=_config(tmp_path), n_hours=-12)


def test_met_config_subgrid_requires_bounds(tmp_path):
    with pytest.raises(Exception, match="subgrid_bounds is required"):
        ModelConfig(
            mets={
                "hrrr": MetConfig(
                    directory=tmp_path / "met",
                    file_format="%Y%m%d_%H",
                    file_tres="1h",
                    subgrid_enable=True,
                )
            },
        )


def test_model_repr(tmp_path):
    model = Model(project=tmp_path / "my_project")

    assert model.project.name == "my_project"
    assert repr(model) == f"Model(project={str(tmp_path / 'my_project')!r})"
    assert model.project.root == str(tmp_path / "my_project")


def test_compute_root_defaults_to_project_simulations_dir(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path))

    assert model.compute_root == tmp_path / "simulations" / "by-id"
    assert model.compute_root == model.project.simulations_dir


def test_compute_root_for_cloud_project_lives_under_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))

    model = Model(project="memory://bucket/proj", config=_config(tmp_path))

    assert model.compute_root == tmp_path / "tmp" / "pystilt" / "proj"


def test_compute_root_from_runtime_settings(tmp_path):
    runtime = RuntimeSettings(compute_root=tmp_path / "scratch")

    model = Model(
        project=tmp_path / "project", config=_config(tmp_path), runtime=runtime
    )

    assert model.compute_root == (tmp_path / "scratch").resolve()


def test_compute_root_explicit_override_wins(tmp_path, point_receptor):
    runtime = RuntimeSettings(compute_root=tmp_path / "runtime-scratch")

    model = Model(
        project=tmp_path / "project",
        config=_config(tmp_path),
        receptors=[point_receptor],
        compute_root=tmp_path / "explicit",
        runtime=runtime,
    )

    assert model.compute_root == (tmp_path / "explicit").resolve()
    sim_id = _sid(point_receptor)
    assert (
        model.simulation(sim_id).directory == (tmp_path / "explicit").resolve() / sim_id
    )


def test_queue_is_none_without_db_url(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), runtime=RuntimeSettings())

    assert model.queue is None


def test_queue_resolves_from_runtime_db_url(tmp_path, monkeypatch):
    captured: list[RuntimeSettings] = []
    fake_queue = object()
    monkeypatch.setattr(
        "stilt.model.resolve_queue",
        lambda runtime: captured.append(runtime) or fake_queue,
    )
    runtime = RuntimeSettings(db_url="postgresql://runtime-db/pystilt")

    model = Model(project=tmp_path, config=_config(tmp_path), runtime=runtime)

    assert model.queue is fake_queue
    assert model.queue is fake_queue  # cached
    assert captured == [runtime]


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------


def test_variants_default_to_one_per_met(tmp_path):
    config = ModelConfig(mets={"hrrr": _met(tmp_path), "nam": _met(tmp_path)})
    model = Model(project=tmp_path, config=config)

    assert list(model.variants) == ["hrrr", "nam"]
    assert model.variants["nam"].met == "nam"


def test_simulation_handles_carry_the_variant_settings(tmp_path, point_receptor):
    config = _config(
        tmp_path,
        krand=4,
        variants={
            "hrrr": {},
            "err": {**_XYERR, "realizations": 2, "grid": None},
            "zi08": {"ziscale": 0.8},
            "s2": {"from": "hrrr", "smooth_factor": 2},
        },
    )
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])

    base = model.simulation(_sid(point_receptor))
    err = model.simulation(_sid(point_receptor, "err-1"))
    zi = model.simulation(_sid(point_receptor, "zi08"))
    s2 = model.simulation(_sid(point_receptor, "s2"))

    assert base.params.winderrtf == 0
    assert err.params.winderrtf == 1
    assert err.footprint_config is None
    assert zi.params.ziscale == 0.8
    assert zi.footprint_config == base.footprint_config
    assert s2.parent is base
    assert s2.met is None
    assert s2.footprint_config is not None and s2.footprint_config.smooth_factor == 2
    assert s2.directory == model.compute_root / _sid(point_receptor, "s2")
    assert model.simulation(str(_sid(point_receptor))) is base  # cached


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


def test_register_writes_config_and_receptors_to_project(tmp_path, point_receptor):
    project_dir = tmp_path / "project"
    model = Model(
        project=project_dir, config=_config(tmp_path), receptors=[point_receptor]
    )

    ids = model.register()

    assert ids == [_rid(point_receptor)]
    assert (project_dir / CONFIG_KEY).exists()
    assert (project_dir / RECEPTORS_KEY).exists()

    clone = Model(project=project_dir)
    assert len(clone.receptors) == 1
    assert clone.receptors[0].id == point_receptor.id
    assert clone.config.mets["hrrr"].directory == tmp_path / "met"


def test_register_copies_source_csv_byte_for_byte(tmp_path):
    original = (
        "time,lati,long,zagl\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0\n"
        "2023-01-01 13:00:00,40.78,-111.86,5.0\n"
    )
    csv = tmp_path / "inputs" / "my_receptors.csv"
    csv.parent.mkdir()
    csv.write_text(original)
    project_dir = tmp_path / "project"

    model = Model(project=project_dir, config=_config(tmp_path), receptors=csv)

    assert len(model.register()) == 2
    assert (project_dir / RECEPTORS_KEY).read_text() == original


def test_register_preserves_project_receptors_csv(tmp_path):
    """Registering the project's own receptors.csv must not rewrite the file."""
    _config(tmp_path).to_yaml(tmp_path / CONFIG_KEY)
    original = (
        "time,lati,long,zagl\n"
        "2023-01-01 12:00:00,40.77,-111.85,5.0\n"
        "2023-01-01 13:00:00,40.78,-111.86,5.0\n"
    )
    csv = tmp_path / RECEPTORS_KEY
    csv.write_text(original)

    assert len(Model(project=tmp_path).register()) == 2
    assert csv.read_text() == original


def test_register_appends_in_memory_receptors_to_existing_csv(tmp_path):
    """In-memory receptors are appended to a receptors.csv already in the project."""
    rec_a, rec_b = _receptor(12), _receptor(13)
    Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a]).register()

    ids = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[rec_b]
    ).register()

    assert ids == [_rid(rec_b)]
    assert [r.id for r in Model(project=tmp_path).receptors] == [rec_a.id, rec_b.id]


def test_register_explicit_batch_merges_with_existing_receptors(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a])

    assert model.register() == [_rid(rec_a)]
    assert model.register(receptors=[rec_b]) == [_rid(rec_b)]

    expected = [_sid(rec_a), _sid(rec_b)]
    assert model.simulations.keys() == expected
    assert model.simulations[_sid(rec_b)].id == _sid(rec_b)
    assert model.receptors[_rid(rec_b)] == rec_b
    assert Model(project=tmp_path).simulations.keys() == expected


def test_register_explicit_batch_dedupes_by_receptor_id(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    model.register()

    assert model.register(receptors=[point_receptor]) == [_rid(point_receptor)]
    assert len(model.receptors) == 1
    assert len(Model(project=tmp_path).receptors) == 1


def test_register_refreshes_receptors_after_explicit_batch(tmp_path, point_receptor):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])
    assert len(model.receptors) == 0  # materialize the empty cache first

    [rid] = model.register(receptors=[point_receptor])

    assert model.receptors[rid] == point_receptor
    assert model.simulations[(rid, "hrrr")].receptor == point_receptor


def test_register_seeds_queue_with_receptor_ids(tmp_path, point_receptor, monkeypatch):
    class _FakeQueue:
        def __init__(self):
            self.registered: list[list[str]] = []

        def register(self, ids):
            self.registered.append(list(ids))

    queue = _FakeQueue()
    monkeypatch.setattr("stilt.model.resolve_queue", lambda runtime: queue)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
        runtime=RuntimeSettings(db_url="postgresql://runtime-db/pystilt"),
    )

    ids = model.register()

    assert queue.registered == [ids] == [[_rid(point_receptor)]]


def test_register_on_cloud_project_round_trips_inputs(tmp_path, point_receptor):
    root = _memory_project()
    model = Model(project=root, config=_config(tmp_path), receptors=[point_receptor])

    model.register()

    assert model.project.store.exists(CONFIG_KEY)
    assert model.project.store.exists(RECEPTORS_KEY)

    clone = Model(project=root)
    assert clone.project.is_cloud
    assert len(clone.receptors) == 1
    assert clone.receptors[0].id == point_receptor.id
    assert clone.config.mets["hrrr"].directory == tmp_path / "met"
    assert clone.simulations.keys() == [_sid(point_receptor)]


def test_register_refuses_to_change_a_registered_variant(tmp_path, point_receptor):
    Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {}}),
        receptors=[point_receptor],
    ).register()
    changed = _config(tmp_path, variants={"hrrr": {"ziscale": 0.8}})

    with pytest.raises(ConfigChangedError, match="hrrr: ziscale"):
        Model(project=tmp_path, config=changed).register()
    assert Model(project=tmp_path).variants["hrrr"].ziscale == 1.0  # not overwritten


def test_register_refuses_a_config_yaml_edited_in_place(tmp_path, point_receptor):
    """The record, not config.yaml, is what a changed setting is compared with."""
    Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    ).register()
    text = (tmp_path / CONFIG_KEY).read_text()
    (tmp_path / CONFIG_KEY).write_text(text + "ziscale: 0.8\n")

    with pytest.raises(ConfigChangedError, match="hrrr: ziscale"):
        Model(project=tmp_path).register()  # the CLI path: config from the project


def test_register_refuses_a_changed_met(tmp_path, point_receptor):
    Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    ).register()
    moved = _config(tmp_path)
    moved.mets["hrrr"] = moved.mets["hrrr"].model_copy(
        update={"directory": tmp_path / "x"}
    )
    Model(project=tmp_path, config=moved).register()  # where the files are is cosmetic

    subgridded = _config(tmp_path)
    subgridded.mets["hrrr"] = subgridded.mets["hrrr"].model_copy(
        update={"subgrid_enable": True}
    )
    with pytest.raises(ConfigChangedError, match="met hrrr: subgrid_enable"):
        Model(project=tmp_path, config=subgridded).register()


def test_register_never_rewrites_an_existing_config_yaml(tmp_path, point_receptor):
    (tmp_path / CONFIG_KEY).write_text(
        "# my notes\n"
        f"mets:\n  hrrr: {{directory: {tmp_path / 'met'}, file_format: '%Y%m%d_%H', file_tres: 6h}}\n"
        "numpar: 10\n"
    )
    before = (tmp_path / CONFIG_KEY).read_text()

    Model(project=tmp_path, receptors=[point_receptor]).register()
    Model(project=tmp_path).run(executor=_CapturingExecutor())

    assert (tmp_path / CONFIG_KEY).read_text() == before


def test_register_writes_a_short_config_and_a_full_record(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, numpar=50),
        receptors=[point_receptor],
    )
    model.register()

    text = (tmp_path / CONFIG_KEY).read_text()
    assert "numpar: 50" in text and "capemin" not in text
    record = model.project.load_record()
    assert record["variants"]["hrrr"]["numpar"] == 50
    assert "capemin" in record["variants"]["hrrr"]
    assert set(record["mets"]) == {"hrrr"}


def test_orphans_are_recorded_variants_the_config_dropped(tmp_path, point_receptor):
    Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}),
        receptors=[point_receptor],
    ).register()

    model = Model(project=tmp_path, config=_config(tmp_path, variants={"hrrr": {}}))
    assert model.orphans() == ["zi08"]
    model.register()
    assert model.orphans() == ["zi08"]  # registering never forgets a variant


def test_remove_deletes_a_variant_and_what_derives_from_it(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(
            tmp_path,
            variants={
                "hrrr": {},
                "hrrr-s2": {"from": "hrrr", "smooth_factor": 2.0},
                "zi08": {"ziscale": 0.8},
            },
        ),
        receptors=[point_receptor],
    )
    model.register()
    for variant in ("hrrr", "zi08"):
        _write_trajectory(model, _sid(point_receptor, variant))
        _write_footprint(model, _sid(point_receptor, variant))
    _write_footprint(model, _sid(point_receptor, "hrrr-s2"))
    assert model.simulations.incomplete().keys() == []

    deleted = model.remove("hrrr")

    assert deleted == [_sid(point_receptor, "hrrr"), _sid(point_receptor, "hrrr-s2")]
    assert not model.simulation(_sid(point_receptor, "hrrr")).directory.exists()
    assert not model.simulation(_sid(point_receptor, "hrrr-s2")).directory.exists()
    assert model.simulation(_sid(point_receptor, "zi08")).is_complete()
    assert set(model.project.load_record()["variants"]) == {"zi08"}
    # The variant is new again: changed settings under its name are accepted.
    Model(
        project=tmp_path,
        config=_config(
            tmp_path, variants={"hrrr": {"numpar": 7}, "zi08": {"ziscale": 0.8}}
        ),
    ).register()


def test_remove_takes_a_realization_group_or_an_orphan(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {}, "err": {"realizations": 2}}),
        receptors=[point_receptor],
    )
    model.register()
    for name in ("err-0", "err-1"):
        _write_trajectory(model, _sid(point_receptor, name))

    assert [s.variant for s in model.remove("err")] == ["err-0", "err-1"]

    trimmed = Model(project=tmp_path, config=_config(tmp_path, variants={"hrrr": {}}))
    assert trimmed.orphans() == []
    with pytest.raises(KeyError):
        trimmed.remove("err")


def test_register_refuses_a_changed_default_that_reaches_a_variant(tmp_path):
    Model(project=tmp_path, config=_config(tmp_path), receptors=[]).register()

    with pytest.raises(ConfigChangedError, match="numpar"):
        Model(project=tmp_path, config=_config(tmp_path, numpar=5000)).register()
    with pytest.raises(ConfigChangedError, match="grid"):
        Model(project=tmp_path, config=_config(tmp_path, grid=None)).register()


def test_register_allows_new_variants_and_cosmetic_changes(tmp_path, point_receptor):
    Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    ).register()

    grown = _config(
        tmp_path,
        timeout=5,
        execution={"backend": "local"},
        variants={"hrrr": {}, "zi08": {"ziscale": 0.8}},
    )
    Model(project=tmp_path, config=grown).register()

    assert list(Model(project=tmp_path).variants) == ["hrrr", "zi08"]


# ---------------------------------------------------------------------------
# simulations: the receptors × variants selection
# ---------------------------------------------------------------------------


def test_simulations_mapping_protocol(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    sid_a, sid_b = _sid(rec_a), _sid(rec_b)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    sims = model.simulations

    assert isinstance(sims, SimulationCollection)
    assert sims.keys() == [sid_a, sid_b]
    assert len(sims) == 2
    assert sid_a in sims
    assert str(sid_a) in sims
    assert (str(rec_a.id), "hrrr") in sims
    assert _sid(_receptor(14)) not in sims
    assert _sid(rec_a, "nam") not in sims
    assert "not a sim id" not in sims
    assert 42 not in sims

    sim = sims[sid_a]
    assert isinstance(sim, Simulation)
    assert sims[str(sid_a)] is sim  # cached on the model
    assert sim.directory == model.compute_root / sid_a
    assert not sim.directory.exists()  # building a handle has no side effects
    assert [s.id for s in sims] == sims.keys()
    with pytest.raises(KeyError):
        sims[_sid(rec_a, "nam")]


def test_simulations_span_receptors_times_variants(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    config = _config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}})
    model = Model(project=tmp_path, config=config, receptors=[rec_a, rec_b])

    assert model.simulations.keys() == [
        _sid(rec_a),
        _sid(rec_a, "zi08"),
        _sid(rec_b),
        _sid(rec_b, "zi08"),
    ]
    assert model.simulations.receptors == [_rid(rec_a), _rid(rec_b)]
    assert model.simulations.variants == ["hrrr", "zi08"]


def test_simulations_empty_when_no_receptors(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    assert model.simulations.keys() == []
    assert len(model.simulations.incomplete()) == 0
    assert model.simulations.status().empty


def test_sel_by_variant_and_realization_group(tmp_path, point_receptor):
    config = _config(
        tmp_path,
        krand=4,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 3}},
    )
    sims = Model(
        project=tmp_path, config=config, receptors=[point_receptor]
    ).simulations

    assert sims.sel(variant="err").variants == ["err-0", "err-1", "err-2"]
    assert sims.sel(variant="err-1").variants == ["err-1"]
    assert sims.sel(variant=["hrrr", "err-2"]).variants == ["hrrr", "err-2"]
    with pytest.raises(KeyError):
        sims.sel(variant="nope")


def test_sel_by_receptor_time_location_and_predicate(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13, longitude=-111.90)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    sims = model.simulations

    assert sims.sel(receptor=_rid(rec_b)).keys() == [_sid(rec_b)]
    assert sims.sel(time=slice("2023-01-01 12:30", None)).keys() == [_sid(rec_b)]
    assert sims.sel(time=("2023-01-01 11:00", "2023-01-01 12:00")).keys() == [
        _sid(rec_a)
    ]
    assert sims.sel(time="2023-01-01 13:00").keys() == [_sid(rec_b)]
    assert sims.sel(location=rec_b.location_id).keys() == [_sid(rec_b)]
    assert sims.sel(where=lambda r: r.longitude < -111.87).keys() == [_sid(rec_b)]
    # Selections compose.
    assert sims.sel(location=rec_a.location_id).sel(receptor=_rid(rec_b)).keys() == []


def test_receptor_collection_sel(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13, longitude=-111.90)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])

    assert [
        r.id for r in model.receptors.sel(time=slice(None, "2023-01-01 12:00"))
    ] == [rec_a.id]
    assert [r.id for r in model.receptors.sel(location=[rec_b.location_id])] == [
        rec_b.id
    ]


def test_incomplete_follows_each_variant_outputs(tmp_path, point_receptor):
    config = _config(
        tmp_path,
        variants={
            "hrrr": {},
            "traj": {"grid": None},
            "s2": {"from": "hrrr", "smooth_factor": 2},
        },
    )
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    sims = model.simulations

    assert sims.incomplete().variants == ["hrrr", "traj", "s2"]

    _write_trajectory(model, _sid(point_receptor))
    _write_trajectory(model, _sid(point_receptor, "traj"))
    assert sims.incomplete().variants == ["hrrr", "s2"]

    _write_footprint(model, _sid(point_receptor))
    _write_footprint(model, _sid(point_receptor, "s2"), empty=True)
    assert sims.incomplete().keys() == []
    # incomplete() is itself a selection
    assert isinstance(sims.incomplete(), SimulationCollection)


def test_status_frame_marks_outputs_a_variant_does_not_produce(
    tmp_path, point_receptor
):
    config = _config(
        tmp_path,
        variants={
            "hrrr": {},
            "traj": {"grid": None},
            "s2": {"from": "hrrr", "smooth_factor": 2},
        },
    )
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    _write_trajectory(model, _sid(point_receptor))

    status = model.simulations.status().set_index("variant")

    assert list(status.columns) == ["receptor", TRAJECTORY, FOOTPRINT, "complete"]
    assert status.loc["hrrr", TRAJECTORY] == True  # noqa: E712
    assert status.loc["hrrr", FOOTPRINT] == False  # noqa: E712
    assert pd.isna(status.loc["traj", FOOTPRINT])
    assert pd.isna(status.loc["s2", TRAJECTORY])
    assert not status["complete"].any()


# ---------------------------------------------------------------------------
# outputs over a selection
# ---------------------------------------------------------------------------


def test_trajectories_paths_load_and_missing(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    trajectories = model.simulations.trajectories

    assert isinstance(trajectories, OutputCollection)
    assert trajectories.paths() == {}

    path = _write_real_trajectory(model, _sid(rec_a))

    assert path == (
        tmp_path
        / "simulations"
        / "by-id"
        / _rid(rec_a)
        / "hrrr"
        / f"{_rid(rec_a)}_traj.parquet"
    )
    assert trajectories.paths() == {_sid(rec_a): path}
    [loaded] = trajectories.load().values()
    assert isinstance(loaded, Trajectories)
    assert loaded.receptor.id == rec_a.id
    assert trajectories.missing().keys() == [_sid(rec_b)]
    assert model.simulations.sel(time="2023-01-01 13:00").trajectories.paths() == {}


def test_trajectories_exclude_derived_variants(tmp_path, point_receptor):
    config = _config(
        tmp_path, variants={"hrrr": {}, "s2": {"from": "hrrr", "smooth_factor": 2}}
    )
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    _write_trajectory(model, _sid(point_receptor))

    assert len(model.simulations.trajectories) == 1
    assert model.simulations.trajectories.paths() == {
        _sid(point_receptor): model.simulation(_sid(point_receptor)).trajectories_path
    }
    assert len(model.simulations.footprint) == 2


def test_footprint_paths_exclude_empty_markers(tmp_path):
    rec_done, rec_empty, rec_missing = _receptor(12), _receptor(13), _receptor(14)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_done, rec_empty, rec_missing],
    )
    foot_path = _write_footprint(model, _sid(rec_done))
    _write_footprint(model, _sid(rec_empty), empty=True)

    assert foot_path.name == f"{_rid(rec_done)}_foot.nc"
    assert model.simulations.footprint.paths() == {_sid(rec_done): foot_path}
    assert model.simulations.footprint.missing().keys() == [_sid(rec_missing)]


def test_footprint_excludes_trajectory_only_variants(tmp_path, point_receptor):
    config = _config(tmp_path, variants={"hrrr": {}, "traj": {"grid": None}})
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])

    assert model.simulations.footprint.missing().variants == ["hrrr"]
    assert model.simulations.sel(variant="traj").footprint.missing().keys() == []


def test_footprint_load_by_variant(tmp_path, point_receptor):
    config = _config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}})
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    for variant in ("hrrr", "zi08"):
        _write_real_footprint(model, _sid(point_receptor, variant))

    assert len(model.simulations.footprint.load()) == 2
    loaded = model.simulations.sel(variant="zi08").footprint.load()
    assert list(loaded) == [_sid(point_receptor, "zi08")]
    [foot] = loaded.values()
    assert isinstance(foot, Footprint)
    assert foot.name == "zi08"
    assert not foot.is_empty


def test_outputs_fall_back_to_project_store(tmp_path, point_receptor):
    project_dir = tmp_path / "proj"
    model = Model(
        project=project_dir,
        compute_root=tmp_path / "scratch",
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    sid = _sid(point_receptor)
    stored_dir = project_dir / "simulations" / "by-id" / sid
    stored_dir.mkdir(parents=True)
    traj = stored_dir / f"{_rid(point_receptor)}_traj.parquet"
    foot = stored_dir / f"{_rid(point_receptor)}_foot.nc"
    traj.write_bytes(b"stub")
    foot.write_bytes(b"stub")

    assert model.simulation(sid).directory == (tmp_path / "scratch").resolve() / sid
    assert model.simulations.trajectories.paths() == {sid: traj}
    assert model.simulations.footprint.paths() == {sid: foot}
    assert model.simulations.incomplete().keys() == []
    assert model.simulations[sid].outcome == "complete"


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


def test_status_is_the_simulation_table(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])
    assert list(model.status().columns) == [
        "receptor",
        "variant",
        "trajectory",
        "footprint",
        "complete",
    ]
    assert model.status().empty

    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    assert model.status()["complete"].tolist() == [False, False]

    _write_trajectory(model, _sid(rec_a))
    _write_footprint(model, _sid(rec_a))
    _write_trajectory(model, _sid(rec_b))
    _write_footprint(model, _sid(rec_b), empty=True)
    assert model.status()["complete"].tolist() == [True, True]


def test_sel_raises_for_unknown_receptor_or_variant(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {}, "err": {"realizations": 2}}),
        receptors=[point_receptor],
    )
    assert model.simulations.sel(variant="err").variants == ["err-0", "err-1"]
    assert (
        model.simulations.sel(time="2020-01-01").keys() == []
    )  # a filter may be empty
    with pytest.raises(KeyError, match="hrr"):
        model.simulations.sel(variant="hrr")
    with pytest.raises(KeyError, match="nope"):
        model.simulations.sel(receptor="nope")


# ---------------------------------------------------------------------------
# accessors and plotting
# ---------------------------------------------------------------------------


def test_accessors_are_cached(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path))

    assert model.receptors is model.receptors
    assert model.plot is model.plot
    assert model.variants is model.variants
    assert model.simulations is model.simulations


def test_plot_availability_returns_axes(tmp_path, point_receptor):
    from matplotlib.axes import Axes

    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )

    ax = model.plot.availability()

    assert isinstance(ax, Axes)
    assert ax.get_title() == "Simulation Availability"


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def _run_model(tmp_path, point_receptor, executor, skip_existing=True, wait=True):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    return model, model.run(executor=executor, skip_existing=skip_existing, wait=wait)


def test_run_dispatches_receptors_with_project_and_compute_root(
    tmp_path, point_receptor
):
    exc = _CapturingExecutor()

    model, handle = _run_model(tmp_path, point_receptor, exc)

    assert handle is exc.handle
    call = exc.start_calls[0]
    assert call["pending"] == [_rid(point_receptor)]
    assert call["project"] == str(tmp_path) == model.project.root
    assert call["compute_root"] == str(tmp_path / "simulations" / "by-id")
    assert call["skip_existing"] is True
    assert call.get("n_workers") is None


def test_run_forwards_explicit_compute_root(tmp_path, point_receptor):
    project_dir = tmp_path / "project"
    compute_root = tmp_path / "scratch"
    model = Model(
        project=project_dir,
        compute_root=compute_root,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=False)

    assert exc.start_calls[0]["project"] == str(project_dir)
    assert exc.start_calls[0]["compute_root"] == str(compute_root.resolve())


def test_run_propagates_skip_existing_false(tmp_path, point_receptor):
    exc = _CapturingExecutor()

    _run_model(tmp_path, point_receptor, exc, skip_existing=False)

    assert exc.start_calls[0]["skip_existing"] is False


def test_run_resolves_executor_from_config(tmp_path, point_receptor, monkeypatch):
    config = _config(tmp_path, execution={"backend": "local", "n_workers": 3})
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    exc = _CapturingExecutor()
    captured: list[dict] = []
    monkeypatch.setattr(
        "stilt.model.get_executor", lambda execution: captured.append(execution) or exc
    )

    handle = model.run(skip_existing=False, wait=False)

    assert handle is exc.handle
    assert captured == [{"backend": "local", "n_workers": 3}]


def test_run_registers_inputs_before_start(tmp_path, point_receptor):
    class _CheckingExecutor(_CapturingExecutor):
        def start(self, pending, **kwargs):
            root = Path(kwargs["project"])
            self.seen = ((root / CONFIG_KEY).exists(), (root / RECEPTORS_KEY).exists())
            return super().start(pending, **kwargs)

    exc = _CheckingExecutor()

    _run_model(tmp_path, point_receptor, exc)

    assert exc.seen == (True, True)
    assert len(Model(project=tmp_path).receptors) == 1


def test_run_refuses_a_changed_variant(tmp_path, point_receptor):
    _run_model(tmp_path, point_receptor, _CapturingExecutor())
    changed = Model(
        project=tmp_path,
        config=_config(tmp_path, ziscale=0.8),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()

    with pytest.raises(ConfigChangedError):
        changed.run(executor=exc)
    assert not exc.was_started


def test_run_skip_existing_omits_complete_receptors(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_trajectory(model, _sid(point_receptor))
    _write_footprint(model, _sid(point_receptor), empty=True)
    exc = _CapturingExecutor()

    handle = model.run(executor=exc, skip_existing=True)

    assert not exc.was_started
    assert isinstance(handle, LocalHandle)
    assert handle.done


def test_run_skip_existing_redispatches_missing_footprint(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_trajectory(model, _sid(point_receptor))
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert exc.start_calls[0]["pending"] == [_rid(point_receptor)]


def test_run_after_adding_a_variant_redispatches_the_receptor(tmp_path, point_receptor):
    """A finished project grows a variant: only that variant is incomplete."""
    first = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    first.register()
    _write_trajectory(first, _sid(point_receptor))
    _write_footprint(first, _sid(point_receptor))

    grown = Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}),
    )
    assert grown.simulations.incomplete().keys() == [_sid(point_receptor, "zi08")]
    exc = _CapturingExecutor()

    grown.run(executor=exc, skip_existing=True)

    assert exc.start_calls[0]["pending"] == [_rid(point_receptor)]


def test_run_skip_existing_redispatches_a_missing_realization(tmp_path, point_receptor):
    config = _config(
        tmp_path,
        grid=None,
        krand=4,
        variants={"hrrr": {}, "err": {**_XYERR, "realizations": 2}},
    )
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    _write_trajectory(model, _sid(point_receptor))
    _write_trajectory(model, _sid(point_receptor, "err-0"))
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert exc.start_calls[0]["pending"] == [_rid(point_receptor)]
    assert model.simulations.incomplete().keys() == [_sid(point_receptor, "err-1")]

    _write_trajectory(model, _sid(point_receptor, "err-1"))
    again = _CapturingExecutor()
    model.run(executor=again, skip_existing=True)
    assert not again.was_started


def test_run_skip_existing_dispatches_only_incomplete_receptors(tmp_path):
    rec_done, rec_todo = _receptor(12), _receptor(13)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[rec_done, rec_todo],
    )
    _write_trajectory(model, _sid(rec_done))
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert exc.start_calls[0]["pending"] == [_rid(rec_todo)]


def test_run_no_skip_dispatches_everything(tmp_path):
    rec_done, rec_todo = _receptor(12), _receptor(13)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[rec_done, rec_todo],
    )
    _write_trajectory(model, _sid(rec_done))
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=False)

    assert exc.start_calls[0]["pending"] == [_rid(rec_done), _rid(rec_todo)]
    assert exc.start_calls[0]["skip_existing"] is False


def test_run_returns_completed_handle_without_receptors(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])
    exc = _CapturingExecutor()

    handle = model.run(executor=exc, skip_existing=False)

    assert not exc.was_started
    assert isinstance(handle, LocalHandle)
    assert handle.done


def test_run_rejects_slurm_on_cloud_project(tmp_path, point_receptor):
    model = Model(
        project=_memory_project(), config=_config(tmp_path), receptors=[point_receptor]
    )

    with pytest.raises(ConfigValidationError, match="requires a local project"):
        model.run(executor=SlurmExecutor(n_workers=1), skip_existing=False)


def test_run_on_cloud_project_passes_uri_to_executor(tmp_path, point_receptor):
    root = _memory_project()
    model = Model(project=root, config=_config(tmp_path), receptors=[point_receptor])
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=False)

    assert exc.start_calls[0]["project"] == root
    assert exc.start_calls[0]["compute_root"] == str(model.compute_root)
    assert model.project.store.exists(CONFIG_KEY)
    assert model.project.store.exists(RECEPTORS_KEY)


def test_run_wait_calls_handle_wait(tmp_path, point_receptor):
    handle = _CountingHandle()
    exc = _CapturingExecutor(handle=handle)

    _, returned = _run_model(tmp_path, point_receptor, exc, wait=True)

    assert returned is handle
    assert handle.wait_calls == 1


def test_run_no_wait_returns_handle_without_waiting(tmp_path, point_receptor):
    handle = _CountingHandle()
    exc = _CapturingExecutor(handle=handle)

    _, returned = _run_model(tmp_path, point_receptor, exc, wait=False)

    assert returned is handle
    assert handle.wait_calls == 0


def test_remove_resolves_orphans_from_the_record(tmp_path, point_receptor):
    """rm works when config.yaml declares neither the variant nor its parent (#43)."""
    model = Model(
        project=tmp_path,
        config=_config(
            tmp_path,
            variants={"hrrr": {}, "s2": {"from": "hrrr", "smooth_factor": 2.0}},
        ),
        receptors=[point_receptor],
    )
    model.register()
    _write_trajectory(model, _sid(point_receptor, "hrrr"))
    _write_footprint(model, _sid(point_receptor, "s2"))

    later = Model(
        project=tmp_path, config=_config(tmp_path, variants={"zi08": {"ziscale": 0.8}})
    )
    assert later.orphans() == ["hrrr", "s2"]

    assert [s.variant for s in later.remove("s2")] == ["s2"]
    assert not (
        tmp_path / "simulations" / "by-id" / _rid(point_receptor) / "s2"
    ).exists()
    assert [s.variant for s in later.remove("hrrr")] == ["hrrr"]
    assert later.orphans() == []


def test_register_warns_about_orphans(tmp_path, point_receptor, caplog):
    Model(
        project=tmp_path,
        config=_config(tmp_path, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}),
        receptors=[point_receptor],
    ).register()

    with caplog.at_level("WARNING", logger="stilt.model"):
        Model(
            project=tmp_path, config=_config(tmp_path, variants={"hrrr": {}})
        ).register()
    assert "zi08" in caplog.text and "stilt rm" in caplog.text


def test_sel_where_uses_receptor_attrs(tmp_path):
    (tmp_path / RECEPTORS_KEY).write_text(
        "time,longitude,latitude,altitude,scene\n"
        "2023-01-01 12:00:00,-111.85,40.77,5.0,A\n"
        "2023-01-01 13:00:00,-111.85,40.77,5.0,B\n"
        "2023-01-01 14:00:00,-111.85,40.77,5.0,A\n"
    )
    model = Model(project=tmp_path, config=_config(tmp_path))

    scene_a = model.simulations.sel(where=lambda r: r.attrs["scene"] == "A")
    assert len(scene_a) == 2
    assert [str(s.id.receptor)[:12] for s in scene_a] == [
        "202301011200",
        "202301011400",
    ]
