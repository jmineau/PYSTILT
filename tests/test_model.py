"""Tests for stilt.model with repository/service boundaries."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from stilt.artifacts import FsspecArtifactStore
from stilt.config import FootprintConfig, Grid, MetConfig, ModelConfig
from stilt.model import Model
from stilt.receptor import Receptor
from stilt.repositories import SQLiteRepository
from stilt.runtime import RuntimeSettings
from stilt.simulation import SimID

InMemoryRepository = SQLiteRepository.in_memory


def _config(tmp_path, point_receptor=None, include_footprint=True) -> ModelConfig:
    footprints = (
        {
            "slv": FootprintConfig(
                grid=Grid(
                    xmin=-114.0,
                    xmax=-113.0,
                    ymin=39.0,
                    ymax=40.0,
                    xres=0.1,
                    yres=0.1,
                )
            )
        }
        if include_footprint
        else {}
    )
    return ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
        footprints=footprints,
    )


def test_receptors_loaded_from_repository_when_not_explicit(tmp_path, point_receptor):
    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])

    model = Model(project=tmp_path, repository=repo, receptors=None)
    loaded = model.receptors

    assert len(loaded) == 1
    assert loaded[0].location_id == point_receptor.location_id


def test_get_simulations_filters_by_time_and_location(tmp_path):
    repo = InMemoryRepository(tmp_path)
    rec_a = Receptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_b = Receptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )

    sid_a = str(SimID.from_parts("hrrr", rec_a))
    sid_b = str(SimID.from_parts("hrrr", rec_b))
    repo.register_many([(sid_a, rec_a), (sid_b, rec_b)])

    model = Model(project=tmp_path, config=_config(tmp_path), repository=repo)
    got = model.get_simulations(
        time_range=(pd.Timestamp("2023-01-01 12:30"), pd.Timestamp("2023-01-01 13:30")),
        location_ids={SimID(sid_b).location_id},
    )

    assert len(got) == 1
    assert str(got[0].id) == sid_b


def test_get_simulation_ids_filters_by_time_and_location(tmp_path):
    repo = InMemoryRepository(tmp_path)
    rec_a = Receptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_b = Receptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )

    sid_a = str(SimID.from_parts("hrrr", rec_a))
    sid_b = str(SimID.from_parts("hrrr", rec_b))
    repo.register_many([(sid_a, rec_a), (sid_b, rec_b)])

    model = Model(project=tmp_path, config=_config(tmp_path), repository=repo)
    got = model.get_simulation_ids(
        time_range=(pd.Timestamp("2023-01-01 12:30"), pd.Timestamp("2023-01-01 13:30")),
        location_ids={SimID(sid_b).location_id},
    )

    assert got == [sid_b]


def test_simulations_mapping_len_and_contains(tmp_path, point_receptor):
    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])

    model = Model(project=tmp_path, config=_config(tmp_path), repository=repo)
    sims = model.simulations

    assert sid in sims
    assert len(sims) == 1


def test_simulations_mapping_getitem(tmp_path, point_receptor):
    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])

    model = Model(project=tmp_path, config=_config(tmp_path), repository=repo)
    from stilt.simulation import Simulation

    sim = model.simulations[sid]
    assert isinstance(sim, Simulation)


def test_simulations_mapping_uses_artifact_store_for_reload(
    tmp_path, point_receptor, monkeypatch
):
    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])

    output_root = tmp_path / "artifacts"
    traj_path = output_root / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_bytes(b"stub")

    class _FakeTraj:
        is_error = False

    model = Model(
        project=tmp_path / "cache",
        config=_config(tmp_path),
        repository=repo,
        artifact_store=FsspecArtifactStore(output_root),
    )

    monkeypatch.setattr(
        "stilt.simulation.Trajectories.from_parquet", lambda p: _FakeTraj()
    )
    assert model.simulations[sid].status == "complete"
    assert model.simulations[sid].trajectories is not None


def test_simulations_mapping_iter(tmp_path, point_receptor):
    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])

    model = Model(project=tmp_path, config=_config(tmp_path), repository=repo)
    all_ids = list(model.simulations)
    assert sid in all_ids


def test_simulations_mapping_values(tmp_path, point_receptor):
    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])

    model = Model(project=tmp_path, config=_config(tmp_path), repository=repo)
    vals = list(model.simulations.values())
    assert len(vals) == 1


def test_simulations_mapping_items(tmp_path, point_receptor):
    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])

    model = Model(project=tmp_path, config=_config(tmp_path), repository=repo)
    items = list(model.simulations.items())
    assert len(items) == 1
    assert items[0][0] == sid


def test_config_raises_when_no_yaml(tmp_path):
    model = Model(project=tmp_path)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        _ = model.config


def test_model_project_name_from_path(tmp_path):
    model = Model(project=tmp_path / "my_project")
    assert model.project == "my_project"


def test_receptors_from_csv(tmp_path):
    csv = tmp_path / "receptors.csv"
    csv.write_text(
        "time,latitude,longitude,altitude\n2023-01-01 12:00:00,40.77,-111.85,5.0\n"
    )

    model = Model(project=tmp_path, receptors=csv)
    assert len(model.receptors) == 1
    assert model.receptors[0].latitude == pytest.approx(40.77)


def test_receptors_from_single_tuple(tmp_path):
    model = Model(
        project=tmp_path, receptors=("2023-01-01 12:00:00", -111.85, 40.77, 5.0)
    )

    assert len(model.receptors) == 1
    assert isinstance(model.receptors[0], Receptor)
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
    assert all(isinstance(r, Receptor) for r in model.receptors)
    assert model.receptors[1].timestr == "202301011300"
    assert model.receptors[1].altitude == pytest.approx(10.0)


def test_model_loads_config_and_receptors_via_artifact_store(tmp_path):
    output_root = tmp_path / "remote"
    output_root.mkdir()
    cfg = _config(tmp_path)
    cfg.to_yaml(output_root / "config.yaml")
    (output_root / "receptors.csv").write_text(
        "time,latitude,longitude,altitude\n2023-01-01 12:00:00,40.77,-111.85,5.0\n"
    )

    model = Model(
        project=tmp_path / "cache",
        repository=InMemoryRepository(tmp_path / "cache"),
        artifact_store=FsspecArtifactStore(output_root),
    )

    assert model.config.mets["hrrr"].directory == tmp_path / "met"
    assert len(model.receptors) == 1


def test_model_uses_runtime_defaults_for_compute_root_and_cache_dir(tmp_path):
    runtime = RuntimeSettings(
        compute_root=tmp_path / "scratch",
        cache_dir=tmp_path / "cache",
    )

    model = Model(
        project=tmp_path / "project",
        config=_config(tmp_path),
        runtime=runtime,
    )

    assert model.compute_root == (tmp_path / "scratch").resolve()
    assert isinstance(model.artifact_store, FsspecArtifactStore)
    assert model.artifact_store._cache_dir == tmp_path / "cache"


def test_model_resolves_relative_met_dirs_from_runtime_archive(tmp_path):
    config = ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=Path("hrrr"),
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        }
    )
    runtime = RuntimeSettings(met_archive=tmp_path / "archive")

    model = Model(
        project=tmp_path / "project",
        config=config,
        runtime=runtime,
    )

    assert model.mets["hrrr"].directory == (tmp_path / "archive" / "hrrr").resolve()


def test_model_uses_runtime_db_url_for_cloud_output(tmp_path, monkeypatch):
    captured: list[str] = []
    runtime = RuntimeSettings(db_url="postgresql://runtime-db/pystilt")

    monkeypatch.setattr(
        "stilt.model.PostgreSQLRepository",
        lambda url: captured.append(url) or InMemoryRepository(tmp_path / "repo"),
    )

    model = Model(
        project=tmp_path / "project",
        output_dir="s3://bucket/project",
        config=_config(tmp_path),
        runtime=runtime,
        artifact_store=FsspecArtifactStore(tmp_path / "artifacts"),
    )

    assert model.repository is not None
    assert captured == ["postgresql://runtime-db/pystilt"]


def test_cloud_submit_bootstraps_config_and_receptors_to_storage(
    tmp_path, point_receptor
):
    output_root = tmp_path / "artifacts"
    repo = InMemoryRepository(tmp_path / "repo")
    artifact_store = FsspecArtifactStore(output_root)

    model = Model(
        project="s3://bucket/project",
        config=_config(tmp_path),
        receptors=[point_receptor],
        repository=repo,
        artifact_store=artifact_store,
    )

    model.submit()

    assert artifact_store.exists("config.yaml")
    assert artifact_store.exists("receptors.csv")

    clone = Model(
        project="s3://bucket/project",
        repository=repo,
        artifact_store=artifact_store,
    )
    assert len(clone.receptors) == 1
    assert clone.config.mets["hrrr"].directory == tmp_path / "met"


def test_local_submit_bootstraps_inputs_to_separate_output_root(
    tmp_path, point_receptor
):
    project_dir = tmp_path / "project"
    output_root = tmp_path / "output"

    model = Model(
        project=project_dir,
        output_dir=output_root,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )

    model.submit()

    assert (output_root / "config.yaml").exists()
    assert (output_root / "receptors.csv").exists()

    clone = Model(project=output_root)
    assert len(clone.receptors) == 1
    assert clone.config.mets["hrrr"].directory == tmp_path / "met"


def test_local_submit_bootstraps_inputs_to_project_root(tmp_path, point_receptor):
    project_dir = tmp_path / "project"

    model = Model(
        project=project_dir,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )

    model.submit()

    assert (project_dir / "config.yaml").exists()
    assert (project_dir / "receptors.csv").exists()

    clone = Model(project=project_dir)
    assert len(clone.receptors) == 1
    assert clone.config.mets["hrrr"].directory == tmp_path / "met"


def test_model_uses_output_dir_for_default_sqlite_repository(tmp_path):
    project_dir = tmp_path / "project"
    output_root = tmp_path / "output"

    model = Model(
        project=project_dir,
        output_dir=output_root,
        config=_config(tmp_path),
    )

    assert isinstance(model.repository, SQLiteRepository)
    assert model.repository._db_path == output_root / "simulations" / "state.sqlite"


def test_cloud_run_passes_project_uri_to_executor(tmp_path, point_receptor):
    repo = InMemoryRepository(tmp_path / "repo")
    artifact_store = FsspecArtifactStore(tmp_path / "artifacts")

    model = Model(
        project="s3://bucket/project",
        config=_config(tmp_path),
        receptors=[point_receptor],
        repository=repo,
        artifact_store=artifact_store,
    )

    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=False)

    assert exc.was_started
    assert exc.start_calls[0]["project"] == "s3://bucket/project"


def test_receptors_are_normalized_once(tmp_path):
    model = Model(
        project=tmp_path, receptors=("2023-01-01 12:00:00", -111.85, 40.77, 5.0)
    )

    first = model.receptors
    second = model.receptors

    assert first is second


def test_make_params_resolves_subgrid_enable(tmp_path, point_receptor):
    with pytest.raises(
        ValueError,
        match="subgrid_enable is not yet implemented",
    ):
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
    cfg = ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        },
    )
    with pytest.raises(TypeError, match="Cannot pass both"):
        Model(project=tmp_path, config=cfg, n_hours=-12)


def test_get_simulations_empty_when_no_sims(tmp_path):
    repo = InMemoryRepository(tmp_path)
    model = Model(project=tmp_path, config=_config(tmp_path), repository=repo)
    result = model.get_simulations()
    assert result == []


def test_get_simulations_filters_by_footprint(tmp_path, point_receptor):
    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)

    model = Model(project=tmp_path, config=_config(tmp_path), repository=repo)
    # No footprint registered → filtered out
    result = model.get_simulations(footprint="slv")
    assert result == []

    # Register footprint
    repo.mark_footprint_complete(sid, "slv")
    result = model.get_simulations(footprint="slv")
    assert len(result) == 1
    assert str(result[0].id) == sid


def test_get_trajectory_paths_load_matching_trajectories(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo = InMemoryRepository(tmp_path)
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)

    traj_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_bytes(b"traj")

    model = Model(project=tmp_path, repository=repo)
    result = model.get_trajectory_paths()

    assert result == [traj_path]


def test_get_trajectory_paths_falls_back_to_artifact_store(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo = InMemoryRepository(tmp_path)
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)

    storage_root = tmp_path / "remote"
    storage_path = storage_root / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_bytes(b"traj")

    model = Model(
        project=tmp_path / "cache",
        repository=repo,
        artifact_store=FsspecArtifactStore(storage_root),
    )
    result = model.get_trajectory_paths()

    assert len(result) == 1
    assert result[0] == storage_path


def test_get_trajectories_load_matching_trajectories(
    tmp_path, point_receptor, monkeypatch
):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo = InMemoryRepository(tmp_path)
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)

    traj_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_bytes(b"traj")

    sentinel = object()
    monkeypatch.setattr("stilt.model.Trajectories.from_parquet", lambda p: sentinel)

    model = Model(project=tmp_path, repository=repo)
    result = model.get_trajectories()

    assert result == [sentinel]


def test_get_footprints_loads_matching_footprints(
    tmp_path, point_receptor, monkeypatch
):
    from stilt.footprint import Footprint

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_complete(sid, "slv")
    foot_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_slv_foot.nc"
    foot_path.parent.mkdir(parents=True, exist_ok=True)
    foot_path.write_text("stub")

    monkeypatch.setattr(Footprint, "from_netcdf", staticmethod(lambda p: p))

    model = Model(project=tmp_path, repository=repo)
    result = model.get_footprints("slv")

    assert len(result) == 1
    assert result[0].name == f"{sid}_slv_foot.nc"


def test_get_footprint_paths_load_matching_footprints(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo = InMemoryRepository(tmp_path)
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_complete(sid, "slv")

    foot_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_slv_foot.nc"
    foot_path.parent.mkdir(parents=True, exist_ok=True)
    foot_path.write_text("stub")

    model = Model(project=tmp_path, repository=repo)
    result = model.get_footprint_paths("slv")

    assert result == [foot_path]


def test_get_footprint_paths_skips_complete_empty(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo = InMemoryRepository(tmp_path)
    repo.register_many([(sid, point_receptor)], footprint_names=["slv"])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_empty(sid, "slv")

    model = Model(project=tmp_path, repository=repo)
    assert model.get_footprint_paths("slv") == []


def test_get_footprints_empty_when_no_match(tmp_path):
    repo = InMemoryRepository(tmp_path)
    model = Model(project=tmp_path, repository=repo)
    assert model.get_footprints("slv") == []


def test_get_footprints_skips_missing_files(tmp_path, point_receptor, monkeypatch):
    from stilt.footprint import Footprint

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_complete(sid, "slv")

    def _missing(_path):
        raise FileNotFoundError

    monkeypatch.setattr(Footprint, "from_netcdf", staticmethod(_missing))

    model = Model(project=tmp_path, repository=repo)
    assert model.get_footprints("slv") == []


def test_get_footprints_falls_back_to_artifact_store(
    tmp_path, point_receptor, monkeypatch
):
    from stilt.footprint import Footprint

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)], footprint_names=["slv"])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_complete(sid, "slv")

    storage_root = tmp_path / "remote"
    storage_key = storage_root / "simulations" / "by-id" / sid / f"{sid}_slv_foot.nc"
    storage_key.parent.mkdir(parents=True, exist_ok=True)
    storage_key.write_text("stub")

    monkeypatch.setattr(Footprint, "from_netcdf", staticmethod(lambda p: p))

    model = Model(
        project=tmp_path / "cache",
        repository=repo,
        artifact_store=FsspecArtifactStore(storage_root),
    )
    result = model.get_footprints("slv")

    assert len(result) == 1
    assert result[0] == storage_key


def test_get_footprints_filters_by_mets(tmp_path, point_receptor, monkeypatch):
    from stilt.footprint import Footprint

    repo = InMemoryRepository(tmp_path)
    sid_hrrr = str(SimID.from_parts("hrrr", point_receptor))
    sid_nam = str(SimID.from_parts("nam", point_receptor))
    repo.register_many([(sid_hrrr, point_receptor), (sid_nam, point_receptor)])
    for sid in (sid_hrrr, sid_nam):
        repo.mark_trajectory_complete(sid)
        repo.mark_footprint_complete(sid, "slv")
        foot_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_slv_foot.nc"
        foot_path.parent.mkdir(parents=True, exist_ok=True)
        foot_path.write_text("stub")

    monkeypatch.setattr(Footprint, "from_netcdf", staticmethod(lambda p: p))

    cfg = _config(tmp_path, point_receptor)
    cfg.to_yaml(tmp_path / "config.yaml")
    model = Model(project=tmp_path, repository=repo)
    result = model.get_footprints("slv", mets="hrrr")

    assert all("hrrr" in str(p) for p in result)


def test_plot_accessor_is_cached(tmp_path):
    model = Model(project=tmp_path)
    assert model.plot is model.plot


def test_plot_availability_returns_axes(tmp_path, point_receptor):
    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])

    model = Model(project=tmp_path, repository=repo)
    ax = model.plot.availability()

    assert ax.get_title() == "Simulation Availability"


# ---------------------------------------------------------------------------
# model.run() — dispatch and filtering
# ---------------------------------------------------------------------------


class _CapturingExecutor:
    """Fake executor that records start() calls without actually running workers."""

    def __init__(self):
        from stilt.executors import LocalHandle

        self._handle = LocalHandle()
        self.start_calls: list[dict] = []

    def start(
        self,
        project,
        n_workers=1,
        follow=False,
        output_dir=None,
        compute_root=None,
    ):
        self.start_calls.append(
            {
                "project": project,
                "n_workers": n_workers,
                "follow": follow,
                "output_dir": output_dir,
                "compute_root": compute_root,
            }
        )
        return self._handle

    @property
    def was_started(self) -> bool:
        return len(self.start_calls) > 0


def _run_model(tmp_path, point_receptor, executor, skip_existing=True, wait=True):
    """Helper: build a minimal Model and call run()."""
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    return model, model.run(executor=executor, skip_existing=skip_existing, wait=wait)


def test_model_run_dispatches_to_executor(tmp_path, point_receptor):
    """executor.start() is called when there is work to do."""
    exc = _CapturingExecutor()
    _run_model(tmp_path, point_receptor, exc)
    assert exc.was_started
    assert exc.start_calls[0]["compute_root"] == str(
        (tmp_path / "simulations" / "by-id").resolve()
    )


def test_model_run_registers_sims_before_start(tmp_path, point_receptor):
    """repository.register_many is called before executor.start()."""
    from stilt.simulation import SimID

    exc = _CapturingExecutor()
    model, _ = _run_model(tmp_path, point_receptor, exc)

    sid = str(SimID.from_parts("hrrr", point_receptor))
    assert sid in list(model.repository.all_sim_ids())


def test_model_run_forwards_separate_output_dir_and_compute_root(
    tmp_path, point_receptor
):
    """Workers should compute in scratch and publish into the durable output root."""
    project_dir = tmp_path / "project"
    output_dir = tmp_path / "output"
    compute_root = tmp_path / "scratch"

    model = Model(
        project=project_dir,
        output_dir=output_dir,
        compute_root=compute_root,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=False)

    assert exc.was_started
    assert exc.start_calls[0]["project"] == str(output_dir.resolve())
    assert exc.start_calls[0]["output_dir"] is None
    assert exc.start_calls[0]["compute_root"] == str(compute_root.resolve())


def test_model_run_cloud_output_bootstraps_workers_from_output_uri(
    tmp_path, point_receptor
):
    """Remote workers should reconstruct from the durable output URI, not a local project path."""
    project_dir = tmp_path / "project"
    repo = InMemoryRepository(tmp_path / "repo")
    artifact_store = FsspecArtifactStore(tmp_path / "artifacts")

    model = Model(
        project=project_dir,
        output_dir="s3://bucket/project",
        config=_config(tmp_path),
        receptors=[point_receptor],
        repository=repo,
        artifact_store=artifact_store,
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=False)

    assert exc.was_started
    assert exc.start_calls[0]["project"] == "s3://bucket/project"
    assert exc.start_calls[0]["output_dir"] is None


def test_model_run_skip_existing_omits_completed_trajectories(tmp_path, point_receptor):
    """With skip_existing=True, executor is NOT started when all trajs are complete."""
    from stilt.simulation import SimID

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[point_receptor],
        repository=repo,
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=True)

    assert not exc.was_started


def test_model_run_no_skip_resets_and_dispatches(tmp_path, point_receptor):
    """skip_existing=False resets completed sim to pending and starts executor."""
    from stilt.simulation import SimID

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[point_receptor],
        repository=repo,
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=False)

    assert exc.was_started


def test_model_run_returns_completed_handle_when_nothing_to_do(
    tmp_path, point_receptor
):
    """All simulations already complete → early LocalHandle return."""
    from stilt.executors import LocalHandle
    from stilt.simulation import SimID

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[point_receptor],
        repository=repo,
    )
    handle = model.run(executor=_CapturingExecutor(), skip_existing=True)

    assert isinstance(handle, LocalHandle)
    assert handle.wait() is None


def test_model_run_with_custom_artifact_store(tmp_path, point_receptor):
    """artifact_store= kwarg is accepted and bound."""
    artifact_store = FsspecArtifactStore(tmp_path)
    exc = _CapturingExecutor()
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
        artifact_store=artifact_store,
    )
    model.run(executor=exc, skip_existing=False)
    assert exc.was_started


def test_model_run_foot_configs_skip_completed(tmp_path, point_receptor):
    """With footprint configs, executor is NOT started when all footprints are done."""
    from stilt.simulation import SimID

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_complete(sid, "slv")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
        repository=repo,
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=True)

    assert not exc.was_started


def test_model_run_foot_configs_skip_complete_empty(tmp_path, point_receptor):
    """With skip_existing=True, complete-empty footprints are treated as done."""
    from stilt.simulation import SimID

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_empty(sid, "slv")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
        repository=repo,
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=True)

    assert not exc.was_started


def test_model_run_foot_configs_retries_failed_footprint(tmp_path, point_receptor):
    """With skip_existing=True, failed footprint states are re-queued for retry."""
    from stilt.simulation import SimID

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_failed(sid, "slv", "transient")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
        repository=repo,
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=True)

    assert exc.was_started


def test_model_run_no_skip_clears_completed_footprints_for_rerun(
    tmp_path, point_receptor
):
    """Force rerun clears footprint completion so workers regenerate outputs."""
    from stilt.simulation import SimID

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)], footprint_names=["slv"])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_complete(sid, "slv")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
        repository=repo,
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=False)

    run_args = model._build_run_args(sid)
    assert exc.was_started
    assert run_args is not None
    assert run_args.foot_configs is not None
    assert "slv" in run_args.foot_configs
    assert not repo.footprint_completed(sid, "slv")


def test_model_run_skip_existing_does_not_requeue_running_sim(tmp_path, point_receptor):
    """A repeated coordinator run must not duplicate an already-running sim."""
    from stilt.simulation import SimID

    repo = InMemoryRepository(tmp_path)
    sid = str(SimID.from_parts("hrrr", point_receptor))
    repo.register_many([(sid, point_receptor)], footprint_names=["slv"])
    repo.claim_pending(1)

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
        repository=repo,
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=True)

    assert repo.traj_status(sid) == "running"
    assert not exc.was_started


def test_model_submit_registers_sims(tmp_path, point_receptor):
    """Model.submit() registers all met×receptor pairs as pending."""
    from stilt.simulation import SimID

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    sim_ids = model.submit()

    sid = str(SimID.from_parts("hrrr", point_receptor))
    assert sid in sim_ids
    assert model.repository.has(sid)


def test_model_submit_with_batch_id(tmp_path, point_receptor):
    """Model.submit(batch_id=...) labels the batch for progress tracking."""

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    model.submit(batch_id="test_batch")

    completed, total = model.repository.batch_progress("test_batch")
    assert total == 1
    assert completed == 0


def test_model_submit_registers_footprint_targets(tmp_path, point_receptor):
    """Submitted sims are not fully complete until required footprints exist."""
    from stilt.simulation import SimID

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
    )
    sid = str(SimID.from_parts("hrrr", point_receptor))
    model.submit()
    model.repository.mark_trajectory_complete(sid)

    assert sid not in model.repository.completed_simulations()

    model.repository.mark_footprint_complete(sid, "slv")
    assert sid in model.repository.completed_simulations()


def test_model_submit_registers_error_footprint_targets(tmp_path, point_receptor):
    """Error-footprint outputs are part of the requested manifest targets."""
    from stilt.simulation import SimID

    cfg = _config(tmp_path, include_footprint=True)
    cfg.footprints["slv"] = cfg.footprints["slv"].model_copy(update={"error": True})
    model = Model(
        project=tmp_path,
        config=cfg,
        receptors=[point_receptor],
    )
    sid = str(SimID.from_parts("hrrr", point_receptor))

    model.submit()
    model.repository.mark_trajectory_complete(sid)
    model.repository.mark_footprint_complete(sid, "slv")

    assert sid not in model.repository.completed_simulations()

    model.repository.mark_footprint_empty(sid, "slv_error")
    assert sid in model.repository.completed_simulations()
