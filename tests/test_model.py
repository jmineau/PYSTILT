"""Tests for stilt.model: project inputs, simulation collections, and run()."""

import datetime as dt
import uuid
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.config import FootprintConfig, Grid, MetConfig, ModelConfig, RuntimeSettings
from stilt.errors import ConfigValidationError
from stilt.execution import LocalHandle, SlurmExecutor
from stilt.footprint import Footprint
from stilt.model import Model, StatusCounts
from stilt.project import CONFIG_KEY, RECEPTORS_KEY
from stilt.receptors import PointReceptor
from stilt.simulation import ERROR_TRAJECTORY, TRAJECTORY, SimID, Simulation
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


def _config(tmp_path, include_footprint=True, error=False, **overrides) -> ModelConfig:
    """Minimal ModelConfig with one met stream and (optionally) one footprint."""
    footprints = {"slv": FootprintConfig(grid=_GRID)} if include_footprint else {}
    kwargs = dict(_XYERR) if error else {}
    kwargs.update(overrides)
    return ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        footprints=footprints,
        **kwargs,
    )


def _receptor(hour: int, longitude: float = -111.85) -> PointReceptor:
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, hour),
        longitude=longitude,
        latitude=40.77,
        altitude=5.0,
    )


def _sid(receptor, met="hrrr") -> str:
    return str(SimID.from_parts(met, receptor))


def _sim_dir(model: Model, sim_id: str) -> Path:
    """Create and return the simulation directory (Simulation no longer does)."""
    directory = model.simulation(sim_id).directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_trajectory(model: Model, sim_id: str, *, error: bool = False) -> Path:
    """Write a stub trajectory (or error-trajectory) output."""
    _sim_dir(model, sim_id)
    sim = model.simulation(sim_id)
    path = sim.error_trajectories_path if error else sim.trajectories_path
    path.write_bytes(b"stub")
    return path


def _write_footprint(model: Model, sim_id: str, name: str, *, empty=False) -> Path:
    """Write a stub footprint output (or an empty marker)."""
    _sim_dir(model, sim_id)
    sim = model.simulation(sim_id)
    if empty:
        return sim.write_empty_footprint_marker(name)
    path = sim.footprint_path(name)
    path.write_bytes(b"stub")
    return path


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


def _write_real_trajectory(model: Model, sim_id: str, *, error=False) -> Path:
    """Write a loadable trajectory parquet built from stub particles."""
    _sim_dir(model, sim_id)
    sim = model.simulation(sim_id)
    traj = Trajectories.from_particles(
        _particles(),
        receptor=sim.receptor,
        params=model.params,
        met_files=[],
        is_error=error,
    )
    path = sim.error_trajectories_path if error else sim.trajectories_path
    traj.to_parquet(path)
    return path


def _write_real_footprint(model: Model, sim_id: str, name: str) -> Path:
    """Write a loadable footprint netCDF."""
    _sim_dir(model, sim_id)
    sim = model.simulation(sim_id)
    lons = np.array([-113.95, -113.85])
    lats = np.array([39.05, 39.15])
    data = xr.DataArray(
        np.random.rand(1, len(lats), len(lons)),
        dims=("time", "lat", "lon"),
        coords={"time": [sim.receptor.time], "lat": lats, "lon": lons},
    )
    foot = Footprint(
        receptor=sim.receptor,
        config=model.config.footprints[name],
        data=data,
        name=name,
    )
    path = sim.footprint_path(name)
    foot.to_netcdf(path)
    return path


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


def test_receptors_from_single_tuple(tmp_path):
    model = Model(
        project=tmp_path, receptors=("2023-01-01 12:00:00", -111.85, 40.77, 5.0)
    )

    assert len(model.receptors) == 1
    assert isinstance(model.receptors[0], PointReceptor)
    assert model.receptors[0].latitude == pytest.approx(40.77)
    assert model.receptors[0].longitude == pytest.approx(-111.85)


def test_receptors_from_sequence_of_tuples(tmp_path):
    model = Model(
        project=tmp_path,
        receptors=[
            ("2023-01-01 12:00:00", -111.85, 40.77, 5.0),
            ("2023-01-01 13:00:00", -111.86, 40.78, 10.0),
        ],
    )

    assert len(model.receptors) == 2
    assert all(isinstance(r, PointReceptor) for r in model.receptors)
    assert f"{model.receptors[1].time:%Y%m%d%H%M}" == "202301011300"
    assert model.receptors[1].altitude == pytest.approx(10.0)


def test_model_accepts_empty_receptor_list(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    assert len(model.receptors) == 0
    assert list(model.receptors) == []
    assert model.register() == []


def test_receptors_support_lookup_by_receptor_id(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    sim_id = _sid(point_receptor)

    assert model.receptors[SimID(sim_id).receptor] == point_receptor
    assert SimID(sim_id).receptor in model.receptors
    with pytest.raises(KeyError):
        model.receptors["nope"]


def test_receptors_are_normalized_once(tmp_path):
    model = Model(
        project=tmp_path, receptors=("2023-01-01 12:00:00", -111.85, 40.77, 5.0)
    )

    assert model.receptors is model.receptors


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


def test_model_name_and_repr(tmp_path):
    model = Model(project=tmp_path / "my_project")

    assert model.name == "my_project"
    assert repr(model) == f"Model(project={str(tmp_path / 'my_project')!r})"
    assert model.project.root == str(tmp_path / "my_project")


def test_model_name_for_cloud_project():
    model = Model(project="memory://bucket/My_Project/")

    assert model.project.is_cloud
    assert model.name == "my-project"


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
# register()
# ---------------------------------------------------------------------------


def test_register_writes_config_and_receptors_to_project(tmp_path, point_receptor):
    project_dir = tmp_path / "project"
    model = Model(
        project=project_dir, config=_config(tmp_path), receptors=[point_receptor]
    )

    sim_ids = model.register()

    assert sim_ids == [_sid(point_receptor)]
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
    sim_ids = model.register()

    assert len(sim_ids) == 2
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

    model = Model(project=tmp_path)
    sim_ids = model.register()

    assert len(sim_ids) == 2
    assert csv.read_text() == original


def test_register_without_args_keeps_existing_receptors_csv(tmp_path):
    """In-memory receptors do not overwrite a receptors.csv already in the project."""
    rec_a, rec_b = _receptor(12), _receptor(13)
    Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a]).register()

    sim_ids = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[rec_b]
    ).register()

    assert sim_ids == [_sid(rec_b)]
    stored = Model(project=tmp_path).receptors
    assert [r.id for r in stored] == [rec_a.id]


def test_register_explicit_batch_merges_with_existing_receptors(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    sid_a, sid_b = _sid(rec_a), _sid(rec_b)

    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a])
    first = model.register()
    second = model.register(receptors=[rec_b])

    assert first == [sid_a]
    assert second == [sid_b]
    assert model.simulations.keys() == sorted([sid_a, sid_b])
    assert sid_a in model.simulations
    assert sid_b in model.simulations
    assert str(model.simulations[sid_a].id) == sid_a
    assert str(model.simulations[sid_b].id) == sid_b
    assert model.receptors[SimID(sid_b).receptor] == rec_b

    fresh = Model(project=tmp_path)
    assert fresh.simulations.keys() == sorted([sid_a, sid_b])
    assert len(fresh.receptors) == 2


def test_register_explicit_batch_dedupes_by_receptor_id(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    model.register()

    sim_ids = model.register(receptors=[point_receptor])

    assert sim_ids == [_sid(point_receptor)]
    assert len(model.receptors) == 1
    assert len(Model(project=tmp_path).receptors) == 1


def test_register_refreshes_receptors_after_explicit_batch(tmp_path, point_receptor):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])
    assert len(model.receptors) == 0  # materialize the empty cache first

    [sim_id] = model.register(receptors=[point_receptor])

    assert model.receptors[SimID(sim_id).receptor] == point_receptor
    assert str(model.simulations[sim_id].id) == sim_id


def test_register_seeds_queue_when_configured(tmp_path, point_receptor, monkeypatch):
    class _FakeQueue:
        def __init__(self):
            self.registered: list[list[str]] = []

        def register(self, sim_ids):
            self.registered.append(list(sim_ids))

    queue = _FakeQueue()
    monkeypatch.setattr("stilt.model.resolve_queue", lambda runtime: queue)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
        runtime=RuntimeSettings(db_url="postgresql://runtime-db/pystilt"),
    )

    sim_ids = model.register()

    assert queue.registered == [sim_ids]


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


# ---------------------------------------------------------------------------
# simulations collection
# ---------------------------------------------------------------------------


def test_simulations_mapping_protocol(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    sid_a, sid_b = _sid(rec_a), _sid(rec_b)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    sims = model.simulations

    assert sims.keys() == sorted([sid_a, sid_b])
    assert list(sims) == sims.keys()
    assert len(sims) == 2
    assert sid_a in sims
    assert "hrrr_202301011400_-111.85_40.77_5" not in sims
    assert "nam_202301011200_-111.85_40.77_5" not in sims
    assert "not a sim id" not in sims
    assert 42 not in sims

    sim = sims[sid_a]
    assert isinstance(sim, Simulation)
    assert sims[sid_a] is sim  # cached
    assert sim.directory == model.compute_root / sid_a
    assert not sim.directory.exists()  # building a handle has no side effects

    items = list(sims.items())
    assert [sid for sid, _ in items] == sims.keys()
    assert all(isinstance(s, Simulation) for _, s in items)
    assert [str(s.id) for s in sims.values()] == sims.keys()


def test_simulations_span_receptors_times_mets(tmp_path, point_receptor):
    config = _config(tmp_path)
    config.mets["nam"] = config.mets["hrrr"]
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])

    assert model.simulations.keys() == [
        _sid(point_receptor, "hrrr"),
        _sid(point_receptor, "nam"),
    ]
    assert model.simulations.ids(mets="nam") == [_sid(point_receptor, "nam")]
    assert model.simulations.ids(mets=["hrrr"]) == [_sid(point_receptor, "hrrr")]


def test_simulations_empty_when_no_receptors(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    assert model.simulations.keys() == []
    assert model.simulations.select() == []
    assert model.simulations.incomplete() == []


def test_simulations_ids_and_select_filter_by_time_and_location(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13, longitude=-111.90)
    sid_b = _sid(rec_b)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    time_range = (pd.Timestamp("2023-01-01 12:30"), pd.Timestamp("2023-01-01 13:30"))

    assert model.simulations.ids(time_range=time_range) == [sid_b]
    assert model.simulations.ids(location_ids={SimID(sid_b).location}) == [sid_b]
    assert (
        model.simulations.ids(
            time_range=time_range, location_ids={SimID(_sid(rec_a)).location}
        )
        == []
    )

    got = model.simulations.select(
        time_range=time_range, location_ids={SimID(sid_b).location}
    )
    assert len(got) == 1
    assert str(got[0].id) == sid_b


def test_simulations_ids_reject_unknown_met(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )

    with pytest.raises(ConfigValidationError, match="Unknown met name"):
        model.simulations.ids(mets="bogus")
    with pytest.raises(ConfigValidationError, match="Unknown met name"):
        model.trajectories.paths(mets="bogus")


def test_simulations_select_filters_by_footprint(tmp_path, point_receptor):
    sid = _sid(point_receptor)
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_trajectory(model, sid)

    assert model.simulations.select(footprint="slv") == []
    assert model.simulations.ids(footprint="slv") == []

    _write_footprint(model, sid, "slv")

    assert model.simulations.ids(footprint="slv") == [sid]
    result = model.simulations.select(footprint="slv")
    assert len(result) == 1
    assert str(result[0].id) == sid


def test_simulations_select_footprint_counts_empty_marker(tmp_path, point_receptor):
    sid = _sid(point_receptor)
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_footprint(model, sid, "slv", empty=True)

    assert model.simulations.ids(footprint="slv") == [sid]


def test_simulations_incomplete_trajectory_only(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[rec_a, rec_b],
    )
    _write_trajectory(model, _sid(rec_a))

    assert model.simulations.incomplete() == [_sid(rec_b)]


def test_simulations_incomplete_uses_configured_footprints(tmp_path):
    rec_complete, rec_missing = _receptor(12), _receptor(13)
    sid_complete, sid_missing = _sid(rec_complete), _sid(rec_missing)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_complete, rec_missing],
    )
    _write_trajectory(model, sid_complete)
    _write_footprint(model, sid_complete, "slv")
    _write_trajectory(model, sid_missing)

    assert model.simulations.incomplete() == [sid_missing]


def test_simulations_incomplete_treats_empty_marker_as_complete(tmp_path):
    rec_empty, rec_missing = _receptor(12), _receptor(13)
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[rec_empty, rec_missing]
    )
    _write_trajectory(model, _sid(rec_empty))
    _write_footprint(model, _sid(rec_empty), "slv", empty=True)

    assert model.simulations.incomplete() == [_sid(rec_missing)]


def test_simulations_incomplete_requires_error_trajectory_when_enabled(
    tmp_path, point_receptor
):
    sid = _sid(point_receptor)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False, error=True),
        receptors=[point_receptor],
    )
    assert model.params.error_enabled
    _write_trajectory(model, sid)

    assert model.simulations.incomplete() == [sid]
    assert model.simulations.missing(ERROR_TRAJECTORY) == [sid]

    _write_trajectory(model, sid, error=True)

    assert model.simulations.incomplete() == []
    assert model.simulations.missing(ERROR_TRAJECTORY) == []


def test_simulations_incomplete_honours_filters(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13, longitude=-111.90)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[rec_a, rec_b],
    )

    assert model.simulations.incomplete(location_ids={SimID(_sid(rec_b)).location}) == [
        _sid(rec_b)
    ]
    with pytest.raises(ConfigValidationError, match="Unknown met name"):
        model.simulations.incomplete(mets="bogus")


def test_simulations_missing_by_output_name(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    sid_a, sid_b = _sid(rec_a), _sid(rec_b)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    _write_trajectory(model, sid_a)
    _write_footprint(model, sid_b, "slv")

    assert model.simulations.missing(TRAJECTORY) == [sid_b]
    assert model.simulations.missing(ERROR_TRAJECTORY) == [sid_a, sid_b]
    assert model.simulations.missing("slv") == [sid_a]
    assert model.simulations.missing("other") == [sid_a, sid_b]


def test_simulations_paths_by_output_name(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    sid_a, sid_b = _sid(rec_a), _sid(rec_b)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    traj_a = _write_trajectory(model, sid_a)
    error_a = _write_trajectory(model, sid_a, error=True)
    foot_b = _write_footprint(model, sid_b, "slv")
    _write_footprint(model, sid_a, "slv", empty=True)

    assert model.simulations.paths(TRAJECTORY) == [traj_a]
    assert model.simulations.paths(ERROR_TRAJECTORY) == [error_a]
    assert model.simulations.paths("slv") == [foot_b]
    assert (
        model.simulations.paths(
            "slv", time_range=("2023-01-01 11:30", "2023-01-01 12:30")
        )
        == []
    )


# ---------------------------------------------------------------------------
# trajectories accessor
# ---------------------------------------------------------------------------


def test_trajectories_paths_main_and_error(tmp_path, point_receptor):
    sid = _sid(point_receptor)
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )

    assert model.trajectories.paths() == []
    assert model.trajectories.paths(error=True) == []

    traj_path = _write_trajectory(model, sid)
    error_path = _write_trajectory(model, sid, error=True)

    assert traj_path == tmp_path / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    assert model.trajectories.paths() == [traj_path]
    assert model.trajectories.paths(error=True) == [error_path]


def test_trajectories_paths_fall_back_to_project_store(tmp_path, point_receptor):
    sid = _sid(point_receptor)
    project_dir = tmp_path / "proj"
    model = Model(
        project=project_dir,
        compute_root=tmp_path / "scratch",
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    stored = project_dir / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(b"stub")

    assert model.simulation(sid).directory == (tmp_path / "scratch").resolve() / sid
    assert model.trajectories.paths() == [stored]
    assert model.trajectories.missing() == []
    assert model.simulations[sid].status == "complete"


def test_trajectories_load(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    _write_real_trajectory(model, _sid(rec_a))
    _write_real_trajectory(model, _sid(rec_a), error=True)
    _write_real_trajectory(model, _sid(rec_b))

    loaded = model.trajectories.load()

    assert len(loaded) == 2
    assert all(isinstance(t, Trajectories) for t in loaded)
    assert [t.receptor.id for t in loaded] == [rec_a.id, rec_b.id]
    assert not any(t.is_error for t in loaded)

    errors = model.trajectories.load(error=True)
    assert len(errors) == 1
    assert errors[0].is_error
    assert errors[0].receptor.id == rec_a.id

    by_time = model.trajectories.load(
        time_range=("2023-01-01 12:30", "2023-01-01 14:00")
    )
    assert [t.receptor.id for t in by_time] == [rec_b.id]


def test_trajectories_missing(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    _write_trajectory(model, _sid(rec_a))

    assert model.trajectories.missing() == [_sid(rec_b)]


# ---------------------------------------------------------------------------
# footprints accessor
# ---------------------------------------------------------------------------


def test_footprints_names_come_from_config(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path))

    assert model.footprints.names() == ["slv"]
    assert list(model.footprints) == ["slv"]
    assert len(model.footprints) == 1
    assert model.footprints["slv"].name == "slv"
    assert model.footprints["slv"] is model.footprints["slv"]

    empty = Model(project=tmp_path, config=_config(tmp_path, include_footprint=False))
    assert empty.footprints.names() == []


def test_footprints_paths_exclude_empty_markers(tmp_path):
    rec_complete, rec_empty, rec_missing = _receptor(12), _receptor(13), _receptor(14)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_complete, rec_empty, rec_missing],
    )
    foot_path = _write_footprint(model, _sid(rec_complete), "slv")
    _write_footprint(model, _sid(rec_empty), "slv", empty=True)

    assert foot_path == (
        tmp_path
        / "simulations"
        / "by-id"
        / _sid(rec_complete)
        / f"{_sid(rec_complete)}_slv_foot.nc"
    )
    assert model.footprints["slv"].paths() == [foot_path]
    assert model.footprints["slv"].missing() == [_sid(rec_missing)]


def test_footprints_paths_empty_when_nothing_written(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )

    assert model.footprints["slv"].paths() == []
    assert model.footprints["slv"].load() == []
    assert model.footprints["slv"].missing() == [_sid(point_receptor)]


def test_footprints_load(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])
    _write_real_footprint(model, _sid(rec_a), "slv")
    _write_footprint(model, _sid(rec_b), "slv", empty=True)

    loaded = model.footprints["slv"].load()

    assert len(loaded) == 1
    assert isinstance(loaded[0], Footprint)
    assert loaded[0].receptor.id == rec_a.id
    assert not loaded[0].is_empty


def test_footprints_load_filters_by_met(tmp_path, point_receptor):
    config = _config(tmp_path)
    config.mets["nam"] = config.mets["hrrr"]
    model = Model(project=tmp_path, config=config, receptors=[point_receptor])
    for met in ("hrrr", "nam"):
        _write_real_footprint(model, _sid(point_receptor, met), "slv")

    assert len(model.footprints["slv"].load()) == 2
    paths = model.footprints["slv"].paths(mets="nam")
    assert len(paths) == 1
    assert paths[0].name.startswith("nam_")
    assert len(model.footprints["slv"].load(mets="nam")) == 1


def test_footprints_paths_fall_back_to_project_store(tmp_path, point_receptor):
    sid = _sid(point_receptor)
    project_dir = tmp_path / "proj"
    model = Model(
        project=project_dir,
        compute_root=tmp_path / "scratch",
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    stored = project_dir / "simulations" / "by-id" / sid / f"{sid}_slv_foot.nc"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(b"stub")

    assert model.footprints["slv"].paths() == [stored]
    assert model.footprints["slv"].missing() == []


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


def test_status_counts_empty_project(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    assert model.status() == StatusCounts()
    assert model.status() == StatusCounts(total=0, completed=0, pending=0)


def test_status_counts_reflect_outputs_on_disk(tmp_path):
    rec_a, rec_b = _receptor(12), _receptor(13)
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[rec_a, rec_b])

    assert model.status() == StatusCounts(total=2, completed=0, pending=2)

    _write_trajectory(model, _sid(rec_a))
    assert model.status() == StatusCounts(total=2, completed=0, pending=2)

    _write_footprint(model, _sid(rec_a), "slv")
    assert model.status() == StatusCounts(total=2, completed=1, pending=1)

    _write_trajectory(model, _sid(rec_b))
    _write_footprint(model, _sid(rec_b), "slv", empty=True)
    assert model.status() == StatusCounts(total=2, completed=2, pending=0)


# ---------------------------------------------------------------------------
# accessors and plotting
# ---------------------------------------------------------------------------


def test_accessors_are_cached(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path))

    assert model.plot is model.plot
    assert model.trajectories is model.trajectories
    assert model.footprints is model.footprints
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


def test_run_dispatches_to_executor_with_project_and_compute_root(
    tmp_path, point_receptor
):
    exc = _CapturingExecutor()

    model, handle = _run_model(tmp_path, point_receptor, exc)

    assert exc.was_started
    assert handle is exc.handle
    call = exc.start_calls[0]
    assert call["pending"] == [_sid(point_receptor)]
    assert call["project"] == str(tmp_path)
    assert call["project"] == model.project.root
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


def test_run_skip_existing_defaults_to_config(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, skip_existing=False),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()

    model.run(executor=exc)

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
    assert exc.was_started
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


def test_run_skip_existing_omits_completed_trajectories(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[point_receptor],
    )
    _write_trajectory(model, _sid(point_receptor))
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert not exc.was_started


def test_run_skip_existing_omits_completed_footprints(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_trajectory(model, _sid(point_receptor))
    _write_footprint(model, _sid(point_receptor), "slv")
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert not exc.was_started


def test_run_skip_existing_treats_empty_marker_as_done(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_trajectory(model, _sid(point_receptor))
    _write_footprint(model, _sid(point_receptor), "slv", empty=True)
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert not exc.was_started


def test_run_skip_existing_redispatches_missing_footprint(tmp_path, point_receptor):
    model = Model(
        project=tmp_path, config=_config(tmp_path), receptors=[point_receptor]
    )
    _write_trajectory(model, _sid(point_receptor))
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert exc.was_started
    assert exc.start_calls[0]["pending"] == [_sid(point_receptor)]


def test_run_skip_existing_redispatches_missing_error_trajectory(
    tmp_path, point_receptor
):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False, error=True),
        receptors=[point_receptor],
    )
    _write_trajectory(model, _sid(point_receptor))
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert exc.was_started
    assert exc.start_calls[0]["pending"] == [_sid(point_receptor)]

    _write_trajectory(model, _sid(point_receptor), error=True)
    again = _CapturingExecutor()
    model.run(executor=again, skip_existing=True)
    assert not again.was_started


def test_run_skip_existing_dispatches_only_incomplete(tmp_path):
    rec_done, rec_todo = _receptor(12), _receptor(13)
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[rec_done, rec_todo],
    )
    _write_trajectory(model, _sid(rec_done))
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert exc.start_calls[0]["pending"] == [_sid(rec_todo)]


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

    assert exc.start_calls[0]["pending"] == [_sid(rec_done), _sid(rec_todo)]
    assert exc.start_calls[0]["skip_existing"] is False


def test_run_returns_completed_handle_when_nothing_to_do(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[point_receptor],
    )
    _write_trajectory(model, _sid(point_receptor))
    exc = _CapturingExecutor()

    handle = model.run(executor=exc, skip_existing=True)

    assert not exc.was_started
    assert isinstance(handle, LocalHandle)
    assert handle.done
    assert handle.wait() is None


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

    assert exc.was_started
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
