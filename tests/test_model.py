"""Tests for stilt.model with index/registration boundaries."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from stilt.config import FootprintConfig, Grid, MetConfig, ModelConfig, RuntimeSettings
from stilt.errors import ConfigValidationError
from stilt.execution import (
    SlurmExecutor,
)
from stilt.execution.tasks import plan_simulation_task
from stilt.completion import StatusCounts
from stilt.model import Model as _Model
from stilt.receptors import PointReceptor
from stilt.simulation import SimID
from stilt.storage import LocalStore, ProjectFiles, Storage, Store


def _write_trajectory(output_dir, sim_id, *, error=False):
    """Write a stub trajectory output to disk (by-key completion world)."""
    files = ProjectFiles(output_dir).simulation(sim_id)
    files.directory.mkdir(parents=True, exist_ok=True)
    (files.error_trajectory_path if error else files.trajectory_path).write_bytes(b"x")


def _write_footprint(output_dir, sim_id, name, *, empty=False):
    """Write a stub footprint output (or empty marker) to disk."""
    files = ProjectFiles(output_dir).simulation(sim_id)
    files.directory.mkdir(parents=True, exist_ok=True)
    if empty:
        files.write_empty_footprint_marker(name)
    else:
        files.footprint_path(name).write_bytes(b"x")


def _storage(
    project_dir: Path,
    output_dir: Path,
    store: Store,
    *,
    is_cloud_project: bool = False,
) -> Storage:
    return Storage(
        project_dir=project_dir,
        output_dir=output_dir,
        store=store,
        is_cloud_project=is_cloud_project,
    )


def Model(*args, index=None, storage=None, **kwargs):
    """Test helper that binds fake index/storage after real Model construction."""
    model = _Model(*args, **kwargs)
    if storage is not None:
        model.storage = storage
    if index is not None:
        model._index = index
    return model


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


def test_receptors_require_explicit_or_output_source_when_not_provided(
    tmp_path, point_receptor
):

    model = Model(project=tmp_path, receptors=None)

    with pytest.raises(FileNotFoundError, match="No receptors available"):
        len(model.receptors)


def test_simulation_collection_select_filters_by_time_and_location(tmp_path):
    rec_a = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_b = PointReceptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )

    sid_b = str(SimID.from_parts("hrrr", rec_b))

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_a, rec_b],
    )
    model.register_pending()
    got = model.simulations.select(
        time_range=(pd.Timestamp("2023-01-01 12:30"), pd.Timestamp("2023-01-01 13:30")),
        location_ids={SimID(sid_b).location},
    )

    assert len(got) == 1
    assert str(got[0].id) == sid_b


def test_simulation_collection_ids_filters_by_time_and_location(tmp_path):
    rec_a = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_b = PointReceptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )

    sid_a = str(SimID.from_parts("hrrr", rec_a))
    sid_b = str(SimID.from_parts("hrrr", rec_b))

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_a, rec_b],
    )
    model.manifest.register([(sid_a, rec_a), (sid_b, rec_b)])
    got = model.simulations.ids(
        time_range=(pd.Timestamp("2023-01-01 12:30"), pd.Timestamp("2023-01-01 13:30")),
        location_ids={SimID(sid_b).location},
    )

    assert got == [sid_b]


def test_simulation_collection_ids_rejects_unknown_met(tmp_path, point_receptor):

    model = Model(project=tmp_path, config=_config(tmp_path))

    with pytest.raises(ConfigValidationError, match="Unknown met name"):
        model.simulations.ids(mets="bogus")


def test_trajectory_queries_reject_unknown_met(tmp_path, point_receptor):

    model = Model(project=tmp_path, config=_config(tmp_path))

    with pytest.raises(ConfigValidationError, match="Unknown met name"):
        model.trajectories.paths(mets="bogus")


def test_model_accepts_empty_receptor_list(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    assert len(model.receptors) == 0
    assert list(model.receptors) == []
    assert list(model.register_pending()) == []


def test_register_pending_refreshes_model_inputs_after_explicit_subset(
    tmp_path, point_receptor
):
    model = Model(project=tmp_path, config=_config(tmp_path), receptors=[])

    # Materialize the original empty receptor cache before registration.
    assert len(model.receptors) == 0

    registration = model.register_pending(receptors=[point_receptor])

    [sim_id] = registration
    assert model.receptors[SimID(sim_id).receptor] == point_receptor
    assert str(model.simulations[sim_id].id) == sim_id


def test_simulations_mapping_len_and_contains(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))

    model = Model(project=tmp_path, config=_config(tmp_path))
    model.manifest.register([(sid, point_receptor)])
    sims = model.simulations

    assert sid in sims
    assert len(sims) == 1


def test_simulations_mapping_getitem(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    from stilt.simulation import Simulation

    sim = model.simulations[sid]
    assert isinstance(sim, Simulation)


def test_simulations_mapping_uses_store_for_reload(
    tmp_path, point_receptor, monkeypatch
):
    sid = str(SimID.from_parts("hrrr", point_receptor))

    output_root = tmp_path / "artifacts"
    traj_path = output_root / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_bytes(b"stub")

    class _FakeTraj:
        is_error = False

    model = Model(
        project=tmp_path / "cache",
        config=_config(tmp_path),
        receptors=[point_receptor],
        storage=_storage(
            tmp_path / "cache",
            tmp_path / "cache",
            LocalStore(output_root),
        ),
    )

    monkeypatch.setattr(
        "stilt.simulation.Trajectories.from_parquet", lambda p: _FakeTraj()
    )
    assert model.simulations[sid].status == "complete"
    assert model.simulations[sid].trajectories is not None


def test_simulations_mapping_iter(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    model.manifest.register([(sid, point_receptor)])
    all_ids = list(model.simulations)
    assert sid in all_ids


def test_simulations_mapping_ids_filters_by_time_and_location(tmp_path):
    rec_a = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_b = PointReceptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )

    sid_a = str(SimID.from_parts("hrrr", rec_a))
    sid_b = str(SimID.from_parts("hrrr", rec_b))

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_a, rec_b],
    )
    model.manifest.register([(sid_a, rec_a), (sid_b, rec_b)])
    got = model.simulations.ids(
        time_range=(pd.Timestamp("2023-01-01 12:30"), pd.Timestamp("2023-01-01 13:30")),
        location_ids={SimID(sid_b).location},
    )

    assert got == [sid_b]


def test_simulations_mapping_values(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    model.manifest.register([(sid, point_receptor)])
    vals = list(model.simulations.values())
    assert len(vals) == 1


def test_simulations_mapping_select_filters_by_footprint(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    _write_trajectory(tmp_path, sid)

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    model.manifest.register([(sid, point_receptor)])

    assert model.simulations.select(footprint="slv") == []

    _write_footprint(tmp_path, sid, "slv")
    result = model.simulations.select(footprint="slv")

    assert len(result) == 1
    assert str(result[0].id) == sid


def test_simulation_collection_incomplete_returns_ids_missing_outputs(tmp_path):
    from stilt.storage import ProjectFiles

    rec_a = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_b = PointReceptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )
    sid_a = str(SimID.from_parts("hrrr", rec_a))
    sid_b = str(SimID.from_parts("hrrr", rec_b))

    files_a = ProjectFiles(tmp_path).simulation(sid_a)
    files_a.directory.mkdir(parents=True, exist_ok=True)
    files_a.trajectory_path.write_bytes(b"traj")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[rec_a, rec_b],
    )

    assert model.simulations.incomplete() == [sid_b]


def test_simulation_collection_incomplete_uses_configured_footprints(tmp_path):
    from stilt.storage import ProjectFiles

    rec_complete = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_missing = PointReceptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )
    sid_complete = str(SimID.from_parts("hrrr", rec_complete))
    sid_missing = str(SimID.from_parts("hrrr", rec_missing))

    # Complete sim has the trajectory AND the configured footprint on disk;
    # the missing sim has only the trajectory, so its footprint is still pending.
    files_complete = ProjectFiles(tmp_path).simulation(sid_complete)
    files_complete.directory.mkdir(parents=True, exist_ok=True)
    files_complete.trajectory_path.write_bytes(b"traj")
    files_complete.footprint_path("slv").write_bytes(b"foot")

    files_missing = ProjectFiles(tmp_path).simulation(sid_missing)
    files_missing.directory.mkdir(parents=True, exist_ok=True)
    files_missing.trajectory_path.write_bytes(b"traj")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_complete, rec_missing],
    )

    assert model.simulations.incomplete() == [sid_missing]


def test_trajectory_collection_missing_returns_non_complete_ids(tmp_path):
    rec_complete = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_missing = PointReceptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )
    sid_complete = str(SimID.from_parts("hrrr", rec_complete))
    sid_missing = str(SimID.from_parts("hrrr", rec_missing))
    _write_trajectory(tmp_path, sid_complete)

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_complete, rec_missing],
    )
    model.manifest.register([(sid_complete, rec_complete)])

    assert model.trajectories.missing() == [sid_missing]


def test_named_footprint_collection_missing_excludes_complete_empty(tmp_path):
    rec_complete = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_empty = PointReceptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )
    rec_missing = PointReceptor(
        time=dt.datetime(2023, 1, 1, 14),
        longitude=-111.95,
        latitude=40.77,
        altitude=5.0,
    )
    sid_complete = str(SimID.from_parts("hrrr", rec_complete))
    sid_empty = str(SimID.from_parts("hrrr", rec_empty))
    sid_missing = str(SimID.from_parts("hrrr", rec_missing))
    _write_trajectory(tmp_path, sid_complete)
    _write_trajectory(tmp_path, sid_empty)
    _write_footprint(tmp_path, sid_complete, "slv")
    _write_footprint(tmp_path, sid_empty, "slv", empty=True)

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[rec_complete, rec_empty, rec_missing],
    )

    assert model.footprints["slv"].missing() == [sid_missing]


def test_simulations_mapping_items(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    model.manifest.register([(sid, point_receptor)])
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


def test_model_loads_config_and_receptors_via_store(tmp_path):
    output_root = tmp_path / "remote"
    output_root.mkdir()
    cfg = _config(tmp_path)
    cfg.to_yaml(output_root / "config.yaml")
    (output_root / "receptors.csv").write_text(
        "time,latitude,longitude,altitude\n2023-01-01 12:00:00,40.77,-111.85,5.0\n"
    )

    model = Model(
        project=tmp_path / "cache",
        storage=_storage(
            tmp_path / "cache",
            tmp_path / "cache",
            LocalStore(output_root),
        ),
    )

    assert model.config.mets["hrrr"].directory == tmp_path / "met"
    assert len(model.receptors) == 1


def test_model_uses_runtime_defaults_for_compute_root_and_local_store(tmp_path):
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
    assert isinstance(model.storage.store, LocalStore)


def test_model_uses_runtime_db_url_for_cloud_output(tmp_path, monkeypatch):
    captured: list[str] = []
    runtime = RuntimeSettings(db_url="postgresql://runtime-db/pystilt", max_rows=25)

    monkeypatch.setattr(
        "stilt.service.postgres.PostgresQueue",
        lambda db_url: captured.append(db_url) or object(),
    )

    model = Model(
        project=tmp_path / "project",
        output_dir="s3://bucket/project",
        config=_config(tmp_path),
        runtime=runtime,
        storage=_storage(
            tmp_path / "project",
            tmp_path / "artifacts-local",
            LocalStore(tmp_path / "artifacts"),
        ),
    )

    assert model.queue is not None
    assert captured == ["postgresql://runtime-db/pystilt"]


def test_model_uses_runtime_db_url_for_local_output(tmp_path, monkeypatch):
    captured: list[str] = []
    runtime = RuntimeSettings(db_url="postgresql://runtime-db/pystilt", max_rows=12)

    monkeypatch.setattr(
        "stilt.service.postgres.PostgresQueue",
        lambda db_url: captured.append(db_url) or object(),
    )

    model = Model(
        project=tmp_path / "project",
        config=_config(tmp_path),
        runtime=runtime,
    )

    assert model.queue is not None
    assert captured == ["postgresql://runtime-db/pystilt"]


def test_register_pending_bootstraps_config_and_receptors_to_storage(
    tmp_path, point_receptor
):
    output_root = tmp_path / "artifacts"
    artifact_store = LocalStore(output_root)

    model = Model(
        project="s3://bucket/project",
        config=_config(tmp_path),
        receptors=[point_receptor],
        storage=_storage(
            tmp_path / "cache",
            tmp_path / "cache",
            artifact_store,
            is_cloud_project=True,
        ),
    )

    model.register_pending()

    assert artifact_store.exists("config.yaml")
    assert artifact_store.exists("receptors.csv")

    clone = Model(
        project="s3://bucket/project",
        storage=_storage(
            tmp_path / "cache",
            tmp_path / "cache",
            artifact_store,
            is_cloud_project=True,
        ),
    )
    assert len(clone.receptors) == 1
    assert clone.config.mets["hrrr"].directory == tmp_path / "met"


def test_register_pending_bootstraps_inputs_to_separate_output_root(
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

    model.register_pending()

    assert (output_root / "config.yaml").exists()
    assert (output_root / "receptors.csv").exists()

    clone = Model(project=output_root)
    assert len(clone.receptors) == 1
    assert clone.config.mets["hrrr"].directory == tmp_path / "met"


def test_register_pending_bootstraps_inputs_to_project_root(tmp_path, point_receptor):
    project_dir = tmp_path / "project"

    model = Model(
        project=project_dir,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )

    model.register_pending()

    assert (project_dir / "config.yaml").exists()
    assert (project_dir / "receptors.csv").exists()

    clone = Model(project=project_dir)
    assert len(clone.receptors) == 1
    assert clone.config.mets["hrrr"].directory == tmp_path / "met"


def test_cloud_run_passes_project_uri_to_executor(tmp_path, point_receptor):
    artifact_store = LocalStore(tmp_path / "artifacts")

    model = Model(
        project="s3://bucket/project",
        config=_config(tmp_path),
        receptors=[point_receptor],
        storage=_storage(
            tmp_path / "cache",
            tmp_path / "cache",
            artifact_store,
            is_cloud_project=True,
        ),
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


def test_receptors_support_lookup_by_sim_id(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )

    sim_id = str(SimID.from_parts("hrrr", point_receptor))

    assert model.receptors[SimID(sim_id).receptor] == point_receptor


def test_register_pending_tracks_scene_counts(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )

    sim_ids = model.register_pending(scene_id="scene-a")

    assert len(sim_ids) == 1
    assert model.status(scene_id="scene-a") == StatusCounts(
        total=1,
        completed=0,
        running=0,
        pending=1,
        failed=0,
    )
    assert model.scene_counts() == {
        "scene-a": StatusCounts(
            total=1,
            completed=0,
            running=0,
            pending=1,
            failed=0,
        )
    }


def test_met_config_subgrid_requires_bounds(tmp_path, point_receptor):
    """subgrid_enable=True without subgrid_bounds raises a ValidationError."""
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


def test_simulation_collection_select_empty_when_no_sims(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path))
    result = model.simulations.select()
    assert result == []


def test_simulation_collection_select_filters_by_footprint(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    _write_trajectory(tmp_path, sid)

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    model.manifest.register([(sid, point_receptor)])
    # No footprint registered → filtered out
    result = model.simulations.select(footprint="slv")
    assert result == []

    # Register footprint
    _write_footprint(tmp_path, sid, "slv")
    result = model.simulations.select(footprint="slv")
    assert len(result) == 1
    assert str(result[0].id) == sid


def test_trajectory_accessor_paths_load_matching_trajectories(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    _write_trajectory(tmp_path, sid)

    traj_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_bytes(b"traj")

    model = Model(project=tmp_path)
    model.manifest.register([(sid, point_receptor)])
    result = model.trajectories.paths()

    assert result == [traj_path]


def test_trajectory_accessor_paths_load_matching_error_trajectories(
    tmp_path, point_receptor
):
    sid = str(SimID.from_parts("hrrr", point_receptor))

    error_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_error.parquet"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_bytes(b"error-traj")

    model = Model(project=tmp_path)
    model.manifest.register([(sid, point_receptor)])

    assert model.trajectories.paths(error=True) == [error_path]


def test_trajectory_accessor_paths_fall_back_to_store(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))

    storage_root = tmp_path / "remote"
    storage_path = storage_root / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_bytes(b"traj")

    model = Model(
        project=tmp_path / "cache",
        storage=_storage(
            tmp_path / "cache",
            tmp_path / "cache",
            LocalStore(storage_root),
        ),
    )
    model.manifest.register([(sid, point_receptor)])
    result = model.trajectories.paths()

    assert len(result) == 1
    assert result[0] == storage_path


def test_trajectory_accessor_load_matching_trajectories(
    tmp_path, point_receptor, monkeypatch
):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    _write_trajectory(tmp_path, sid)

    traj_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_bytes(b"traj")

    sentinel = object()
    monkeypatch.setattr(
        "stilt.collections.Trajectories.from_parquet", lambda p: sentinel
    )

    model = Model(project=tmp_path)
    model.manifest.register([(sid, point_receptor)])
    result = model.trajectories.load()

    assert result == [sentinel]


def test_trajectory_accessor_paths_and_load(tmp_path, point_receptor, monkeypatch):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    _write_trajectory(tmp_path, sid)

    traj_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_traj.parquet"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_bytes(b"traj")

    monkeypatch.setattr("stilt.collections.Trajectories.from_parquet", lambda p: p)

    model = Model(project=tmp_path)
    model.manifest.register([(sid, point_receptor)])
    assert model.trajectories.paths() == [traj_path]
    assert model.trajectories.load() == [traj_path]


def test_named_footprint_accessor_loads_matching_footprints_via_path(
    tmp_path, point_receptor, monkeypatch
):
    from stilt.footprint import Footprint

    sid = str(SimID.from_parts("hrrr", point_receptor))
    _write_trajectory(tmp_path, sid)
    _write_footprint(tmp_path, sid, "slv")
    foot_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_slv_foot.nc"
    foot_path.parent.mkdir(parents=True, exist_ok=True)
    foot_path.write_text("stub")

    monkeypatch.setattr(Footprint, "from_netcdf", staticmethod(lambda p: p))

    model = Model(project=tmp_path)
    model.manifest.register([(sid, point_receptor)])
    result = model.footprints["slv"].load()

    assert len(result) == 1
    assert result[0].name == f"{sid}_slv_foot.nc"


def test_named_footprint_accessor_paths_load_matching_footprints(
    tmp_path, point_receptor
):
    sid = str(SimID.from_parts("hrrr", point_receptor))
    _write_trajectory(tmp_path, sid)
    _write_footprint(tmp_path, sid, "slv")

    foot_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_slv_foot.nc"
    foot_path.parent.mkdir(parents=True, exist_ok=True)
    foot_path.write_text("stub")

    model = Model(project=tmp_path)
    model.manifest.register([(sid, point_receptor)])
    result = model.footprints["slv"].paths()

    assert result == [foot_path]


def test_named_footprint_accessor_paths_skip_complete_empty(tmp_path, point_receptor):

    model = Model(project=tmp_path)
    assert model.footprints["slv"].paths() == []


def test_footprint_accessor_names_and_paths(tmp_path):
    rec_complete = PointReceptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    rec_empty = PointReceptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.90,
        latitude=40.77,
        altitude=5.0,
    )
    sid_complete = str(SimID.from_parts("hrrr", rec_complete))
    sid_empty = str(SimID.from_parts("hrrr", rec_empty))
    _write_trajectory(tmp_path, sid_complete)
    _write_trajectory(tmp_path, sid_empty)
    _write_footprint(tmp_path, sid_complete, "slv")
    _write_footprint(tmp_path, sid_empty, "slv", empty=True)

    foot_path = (
        tmp_path
        / "simulations"
        / "by-id"
        / sid_complete
        / f"{sid_complete}_slv_foot.nc"
    )

    model = Model(project=tmp_path, config=_config(tmp_path))
    model.manifest.register([(sid_complete, rec_complete), (sid_empty, rec_empty)])
    assert list(model.footprints) == ["slv"]
    assert model.footprints["slv"].paths() == [foot_path]


def test_named_footprint_accessor_loads_matching_footprints(
    tmp_path, point_receptor, monkeypatch
):
    from stilt.footprint import Footprint

    sid = str(SimID.from_parts("hrrr", point_receptor))
    _write_trajectory(tmp_path, sid)
    _write_footprint(tmp_path, sid, "slv")

    foot_path = tmp_path / "simulations" / "by-id" / sid / f"{sid}_slv_foot.nc"
    foot_path.parent.mkdir(parents=True, exist_ok=True)
    foot_path.write_text("stub")

    monkeypatch.setattr(Footprint, "from_netcdf", staticmethod(lambda p: p))

    model = Model(project=tmp_path)
    model.manifest.register([(sid, point_receptor)])
    assert model.footprints["slv"].load() == [foot_path]


def test_named_footprint_accessor_load_empty_when_no_match(tmp_path):
    model = Model(project=tmp_path)
    assert model.footprints["slv"].load() == []


def test_named_footprint_accessor_load_skips_missing_files(
    tmp_path, point_receptor, monkeypatch
):
    from stilt.footprint import Footprint

    def _missing(_path):
        raise FileNotFoundError

    monkeypatch.setattr(Footprint, "from_netcdf", staticmethod(_missing))

    model = Model(project=tmp_path)
    assert model.footprints["slv"].load() == []


def test_named_footprint_accessor_load_falls_back_to_store(
    tmp_path, point_receptor, monkeypatch
):
    from stilt.footprint import Footprint

    sid = str(SimID.from_parts("hrrr", point_receptor))

    storage_root = tmp_path / "remote"
    storage_key = storage_root / "simulations" / "by-id" / sid / f"{sid}_slv_foot.nc"
    storage_key.parent.mkdir(parents=True, exist_ok=True)
    storage_key.write_text("stub")

    monkeypatch.setattr(Footprint, "from_netcdf", staticmethod(lambda p: p))

    model = Model(
        project=tmp_path / "cache",
        storage=_storage(
            tmp_path / "cache",
            tmp_path / "cache",
            LocalStore(storage_root),
        ),
    )
    model.manifest.register([(sid, point_receptor)])
    result = model.footprints["slv"].load()

    assert len(result) == 1
    assert result[0] == storage_key


def test_named_footprint_accessor_load_filters_by_mets(
    tmp_path, point_receptor, monkeypatch
):
    from stilt.footprint import Footprint

    sid_hrrr = str(SimID.from_parts("hrrr", point_receptor))
    sid_nam = str(SimID.from_parts("nam", point_receptor))
    for sid in (sid_hrrr, sid_nam):
        _write_trajectory(tmp_path, sid)
        _write_footprint(tmp_path, sid, "slv")

    monkeypatch.setattr(Footprint, "from_netcdf", staticmethod(lambda p: p))

    cfg = _config(tmp_path, point_receptor)
    cfg.to_yaml(tmp_path / "config.yaml")
    model = Model(project=tmp_path)
    model.manifest.register([(sid_hrrr, point_receptor), (sid_nam, point_receptor)])
    result = model.footprints["slv"].load(mets="hrrr")

    assert result
    assert all("hrrr" in str(p) for p in result)


def test_plot_accessor_is_cached(tmp_path):
    model = Model(project=tmp_path)
    assert model.plot is model.plot


def test_trajectory_accessor_is_cached(tmp_path):
    model = Model(project=tmp_path)
    assert model.trajectories is model.trajectories


def test_footprint_accessor_is_cached(tmp_path):
    model = Model(project=tmp_path, config=_config(tmp_path))
    assert model.footprints is model.footprints


def test_plot_availability_returns_axes(tmp_path, point_receptor):
    sid = str(SimID.from_parts("hrrr", point_receptor))

    model = Model(project=tmp_path)
    model.manifest.register([(sid, point_receptor)])
    ax = model.plot.availability()

    assert ax.get_title() == "Simulation Availability"


# ---------------------------------------------------------------------------
# model.run() — dispatch and filtering
# ---------------------------------------------------------------------------


class _CapturingExecutor:
    """Fake executor that records start() calls without actually running workers."""

    dispatch = "pull"

    def __init__(self):
        from stilt.execution import LocalHandle

        self._handle = LocalHandle()
        self.start_calls: list[dict] = []

    def start(self, pending: list[str], **kwargs):
        self.start_calls.append(kwargs)
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


def test_model_run_resolves_executor_from_config(tmp_path, point_receptor, monkeypatch):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()
    monkeypatch.setattr("stilt.model.get_executor", lambda execution: exc)

    handle = model.run(skip_existing=False, wait=False)

    assert handle is exc._handle
    assert exc.was_started


def test_model_run_registers_sims_before_start(tmp_path, point_receptor):
    """register_pending records the sim in the manifest before executor.start()."""
    from stilt.simulation import SimID

    exc = _CapturingExecutor()
    model, _ = _run_model(tmp_path, point_receptor, exc)

    sid = str(SimID.from_parts("hrrr", point_receptor))
    assert sid in model.manifest.sim_ids()


def test_model_run_propagates_resolved_skip_to_executor_kwargs(
    tmp_path, point_receptor
):
    exc = _CapturingExecutor()
    _run_model(tmp_path, point_receptor, exc, skip_existing=False)

    assert exc.start_calls[0]["skip_existing"] is False


def test_model_run_forwards_separate_output_dir_and_compute_root(
    tmp_path, point_receptor
):
    """Workers should compute in scratch and publish into the output root."""
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
    assert model.layout.project_root == str(project_dir.resolve())
    assert model.layout.output_root == str(output_dir.resolve())
    assert exc.start_calls[0]["project"] == str(output_dir.resolve())
    assert exc.start_calls[0]["output_dir"] == str(output_dir.resolve())
    assert exc.start_calls[0]["compute_root"] == str(compute_root.resolve())


def test_model_run_cloud_output_bootstraps_workers_from_output_uri(
    tmp_path, point_receptor
):
    """Remote workers should reconstruct from the output URI, not a local project path."""
    project_dir = tmp_path / "project"
    artifact_store = LocalStore(tmp_path / "artifacts")

    model = Model(
        project=project_dir,
        output_dir="s3://bucket/project",
        config=_config(tmp_path),
        receptors=[point_receptor],
        storage=_storage(
            project_dir,
            tmp_path / "artifacts-local",
            artifact_store,
        ),
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=False)

    assert exc.was_started
    assert model.layout.project_root == str(project_dir.resolve())
    assert model.layout.output_root == "s3://bucket/project"
    assert exc.start_calls[0]["project"] == "s3://bucket/project"


def test_model_run_custom_executor_ignores_execution_worker_config(
    tmp_path, point_receptor
):
    config = _config(tmp_path).model_copy(
        update={"execution": {"backend": "local", "n_workers": 99}}
    )
    exc = _CapturingExecutor()
    model = Model(
        project=tmp_path,
        config=config,
        receptors=[point_receptor],
    )

    model.run(executor=exc, skip_existing=False)

    assert exc.start_calls[0].get("n_workers") is None


def test_model_run_rejects_slurm_when_output_root_is_cloud(tmp_path, point_receptor):
    config = _config(tmp_path).model_copy(
        update={"execution": {"backend": "slurm", "n_workers": 1}}
    )
    model = Model(
        project=tmp_path,
        output_dir="s3://bucket/project",
        config=config,
        receptors=[point_receptor],
    )

    with pytest.raises(
        ConfigValidationError, match="requires both project and output roots"
    ):
        model.run(executor=SlurmExecutor(n_workers=1), skip_existing=False)


def test_model_run_skip_existing_omits_completed_trajectories(tmp_path, point_receptor):
    """With skip_existing=True, executor is NOT started when the trajectory exists."""
    from stilt.simulation import SimID
    from stilt.storage import ProjectFiles

    sid = str(SimID.from_parts("hrrr", point_receptor))
    files = ProjectFiles(tmp_path).simulation(sid)
    files.directory.mkdir(parents=True, exist_ok=True)
    files.trajectory_path.write_bytes(b"traj")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=True, rebuild=False)

    assert not exc.was_started


def test_model_run_skip_existing_auto_rebuilds_stale_outputs(tmp_path, point_receptor):
    from stilt.simulation import SimID
    from stilt.storage import ProjectFiles

    sid = str(SimID.from_parts("hrrr", point_receptor))
    files = ProjectFiles(tmp_path).simulation(sid)
    files.directory.mkdir(parents=True, exist_ok=True)
    files.trajectory_path.write_bytes(b"traj")
    files.footprint_path("slv").write_bytes(b"foot")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True)

    assert not exc.was_started


def test_model_run_skip_existing_is_disk_based_regardless_of_rebuild(
    tmp_path, point_receptor
):
    """
    Completion is decided by the outputs on disk, not by a pre-run rebuild.

    Even with ``rebuild=False`` a sim whose outputs already exist is skipped,
    because by-key completion always reads the store.
    """
    from stilt.simulation import SimID
    from stilt.storage import ProjectFiles

    sid = str(SimID.from_parts("hrrr", point_receptor))
    files = ProjectFiles(tmp_path).simulation(sid)
    files.directory.mkdir(parents=True, exist_ok=True)
    files.trajectory_path.write_bytes(b"traj")
    files.footprint_path("slv").write_bytes(b"foot")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()

    model.run(executor=exc, skip_existing=True, rebuild=False)

    assert not exc.was_started


def test_model_run_no_skip_resets_and_dispatches(tmp_path, point_receptor):
    """skip_existing=False resets completed sim to pending and starts executor."""
    from stilt.simulation import SimID

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=False)

    assert exc.was_started


def test_model_run_returns_completed_handle_when_nothing_to_do(
    tmp_path, point_receptor
):
    """All simulations already complete → early LocalHandle return."""
    from stilt.execution import LocalHandle
    from stilt.simulation import SimID

    sid = str(SimID.from_parts("hrrr", point_receptor))
    _write_trajectory(tmp_path, sid)  # trajectory present → nothing to do

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=False),
        receptors=[point_receptor],
    )
    handle = model.run(executor=_CapturingExecutor(), skip_existing=True)

    assert isinstance(handle, LocalHandle)
    assert handle.wait() is None


def test_model_run_with_custom_storage(tmp_path, point_receptor):
    """Tests can bind custom storage after construction for output lookup paths."""
    artifact_store = LocalStore(tmp_path)
    exc = _CapturingExecutor()
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
        storage=_storage(tmp_path, tmp_path, artifact_store),
    )
    model.run(executor=exc, skip_existing=False)
    assert exc.was_started


def test_model_run_foot_configs_skip_completed(tmp_path, point_receptor):
    """With footprint configs, executor is NOT started when all footprints are done."""
    from stilt.simulation import SimID
    from stilt.storage import ProjectFiles

    sid = str(SimID.from_parts("hrrr", point_receptor))
    files = ProjectFiles(tmp_path).simulation(sid)
    files.directory.mkdir(parents=True, exist_ok=True)
    files.trajectory_path.write_bytes(b"traj")
    files.footprint_path("slv").write_bytes(b"foot")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=True, rebuild=False)

    assert not exc.was_started


def test_model_run_foot_configs_skip_complete_empty(tmp_path, point_receptor):
    """With skip_existing=True, an empty-footprint marker is treated as done."""
    from stilt.simulation import SimID
    from stilt.storage import ProjectFiles

    sid = str(SimID.from_parts("hrrr", point_receptor))
    files = ProjectFiles(tmp_path).simulation(sid)
    files.directory.mkdir(parents=True, exist_ok=True)
    files.trajectory_path.write_bytes(b"traj")
    files.write_empty_footprint_marker("slv")

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=True, rebuild=False)

    assert not exc.was_started


def test_model_run_foot_configs_retries_failed_footprint(tmp_path, point_receptor):
    """With skip_existing=True, failed footprint states are re-queued for retry."""
    from stilt.simulation import SimID

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=True, rebuild=False)

    assert exc.was_started


def test_model_run_no_skip_clears_completed_footprints_for_rerun(
    tmp_path, point_receptor
):
    """Force rerun clears footprint completion so workers regenerate outputs."""
    from stilt.simulation import SimID

    sid = str(SimID.from_parts("hrrr", point_receptor))

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
    )
    exc = _CapturingExecutor()
    model.run(executor=exc, skip_existing=False)

    run_args = plan_simulation_task(model, sid)
    assert exc.was_started
    assert run_args is not None
    assert run_args.foot_configs is not None
    assert "slv" in run_args.foot_configs


def test_register_pending_registers_sims(tmp_path, point_receptor):
    """Model.register_pending() registers all met×receptor pairs as pending."""
    from stilt.simulation import SimID

    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )
    sim_ids = model.register_pending()

    sid = str(SimID.from_parts("hrrr", point_receptor))
    assert sid in sim_ids
    assert model.manifest.has(sid)


def test_model_status_reflects_queue_counts(tmp_path, point_receptor):
    model = Model(
        project=tmp_path,
        config=_config(tmp_path),
        receptors=[point_receptor],
    )

    model.register_pending()

    assert model.status() == StatusCounts(
        total=1,
        completed=0,
        running=0,
        pending=1,
        failed=0,
    )


def test_plan_simulation_task_push_uses_stored_inputs_not_repository(
    tmp_path, point_receptor
):
    """Push workers get receptor/skip state from stored inputs, not a registry."""
    config = _config(tmp_path, include_footprint=False).model_copy(
        update={"execution": {"backend": "slurm", "n_workers": 1}}
    )
    sid = str(SimID.from_parts("hrrr", point_receptor))

    model = Model(
        project=tmp_path,
        config=config,
        receptors=[point_receptor],
    )

    run_args = plan_simulation_task(model, sid)

    assert run_args is not None
    assert run_args.receptor.id == point_receptor.id


def test_register_pending_registers_footprint_targets(tmp_path, point_receptor):
    """A sim is not complete until its required footprint exists on disk."""
    from stilt.simulation import SimID

    model = Model(
        project=tmp_path,
        config=_config(tmp_path, include_footprint=True),
        receptors=[point_receptor],
    )
    sid = str(SimID.from_parts("hrrr", point_receptor))
    model.register_pending()
    _write_trajectory(tmp_path, sid)

    assert model.status().completed == 0

    _write_footprint(tmp_path, sid, "slv")
    assert model.status().completed == 1
