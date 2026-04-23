"""Tests for execution tasks, batch execution, and entrypoints."""

import datetime as dt
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from stilt.config import FootprintConfig, Grid, MetConfig, ModelConfig, STILTParams
from stilt.errors import SimulationError
from stilt.execution import (
    SimulationResult,
    SimulationTask,
    execute_batch,
    execute_task,
    pull_simulations,
    push_simulations,
)
from stilt.index import OutputSummary
from stilt.index.sqlite import SqliteIndex
from stilt.meteorology import MetStream
from stilt.model import Model as _Model
from stilt.receptor import Receptor
from stilt.simulation import SimID
from stilt.storage import (
    FsspecStore,
    ProjectFiles,
    SimulationFiles,
    Storage,
)


def Model(*args, index=None, storage=None, **kwargs):
    """Test helper that binds fake index/storage after real Model construction."""
    model = _Model(*args, **kwargs)
    if storage is not None:
        model.storage = storage
    if index is not None:
        model._index = index
    return model


@pytest.fixture
def receptor() -> Receptor:
    return Receptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


@pytest.fixture
def sim_id(receptor: Receptor) -> SimID:
    return SimID.from_parts("hrrr", receptor)


@pytest.fixture
def met(tmp_path) -> MetStream:
    return MetStream(
        "hrrr",
        directory=tmp_path / "met",
        file_format="%Y%m%d_%H",
        file_tres="1h",
    )


@pytest.fixture
def params() -> STILTParams:
    return STILTParams(n_hours=-24, numpar=10)


@pytest.fixture
def storage(tmp_path) -> Storage:
    return Storage(
        project_dir=tmp_path,
        output_dir=tmp_path,
        store=FsspecStore(tmp_path),
    )


@pytest.fixture
def task(tmp_path, sim_id, receptor, met, params, storage) -> SimulationTask:
    return SimulationTask(
        compute_root=tmp_path,
        sim_id=sim_id,
        meteorology=met,
        receptor=receptor,
        params=params,
        storage=storage,
    )


def _make_model_config(tmp_path) -> ModelConfig:
    return ModelConfig(
        mets={
            "hrrr": MetConfig(
                directory=tmp_path / "met",
                file_format="%Y%m%d_%H",
                file_tres="1h",
            )
        }
    )


def _fake_simulation(
    sim_id: SimID, traj_path: Path, *, succeed: bool = True
) -> MagicMock:
    sim = MagicMock()
    sim.id = sim_id
    sim.directory = traj_path.parent
    sim.files = SimulationFiles(traj_path.parent, str(sim_id))
    sim.trajectories_path = traj_path
    sim.log_path = traj_path.parent / "stilt.log"
    sim.error_trajectories_path = traj_path.parent / "error.parquet"
    sim.footprint_path.side_effect = sim.files.footprint_path
    sim.resolve_output.side_effect = lambda path: path if Path(path).exists() else None
    if succeed:
        sim.run_trajectories.return_value = None
    else:
        sim.run_trajectories.side_effect = SimulationError("HYSPLIT failed")
    return sim


def test_run_args_requires_storage(tmp_path, sim_id, receptor, met, params):
    with pytest.raises(ValidationError):
        SimulationTask(
            compute_root=tmp_path,
            sim_id=sim_id,
            meteorology=met,
            receptor=receptor,
            params=params,
        )


def test_execute_task_returns_complete_result(tmp_path, task, monkeypatch):
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()
    log_path = tmp_path / "stilt.log"
    log_path.write_text("ok")
    error_path = tmp_path / "sim_error.parquet"
    error_path.touch()

    fake_sim = _fake_simulation(task.sim_id, traj_path, succeed=True)
    fake_sim.log_path = log_path
    fake_sim.error_trajectories_path = error_path
    monkeypatch.setattr("stilt.execution.execute.Simulation", lambda **kw: fake_sim)

    result = execute_task(task)

    assert result.status == "complete"
    assert result.traj_present is True
    assert result.wrote_traj is True
    assert result.log_path == log_path
    assert result.error_traj_path == error_path


def test_record_result_records_successful_trajectory(tmp_path, sim_id):
    state = SqliteIndex(tmp_path)
    receptor = Receptor(
        time="2023-01-01 12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    state.register([(str(sim_id), receptor)])
    result = SimulationResult(
        sim_id=sim_id,
        status="complete",
        traj_present=True,
        traj_path=tmp_path / "traj.parquet",
        error_traj_path=tmp_path / "error.parquet",
        log_path=tmp_path / "stilt.log",
        wrote_traj=True,
        started_at=dt.datetime.now(dt.timezone.utc),
    )

    state.record(result)

    assert state.summaries([str(sim_id)]).get(
        str(sim_id), OutputSummary()
    ) == OutputSummary(
        traj_present=True,
        error_traj_present=True,
        log_present=True,
        footprints={},
    )
    assert state.counts().completed == 1


def test_execute_task_returns_failed_result_for_simulation_error(
    tmp_path, task, monkeypatch
):
    traj_path = tmp_path / "sim_traj.parquet"
    fake_sim = _fake_simulation(task.sim_id, traj_path, succeed=False)
    monkeypatch.setattr("stilt.execution.execute.Simulation", lambda **kw: fake_sim)

    result = execute_task(task)

    assert result.status == "failed"
    assert result.traj_present is False
    assert result.error == "HYSPLIT failed"


def test_execute_task_logs_simulation_failures_to_stderr(
    tmp_path, task, monkeypatch, caplog
):
    traj_path = tmp_path / "sim_traj.parquet"
    fake_sim = _fake_simulation(task.sim_id, traj_path, succeed=False)
    monkeypatch.setattr("stilt.execution.execute.Simulation", lambda **kw: fake_sim)

    with caplog.at_level(logging.ERROR):
        result = execute_task(task)

    assert result.status == "failed"
    assert str(task.sim_id) in caplog.text
    assert "trajectory" in caplog.text
    assert "HYSPLIT failed" in caplog.text


def test_record_result_records_failed_attempt(tmp_path, sim_id):
    state = SqliteIndex(tmp_path)
    receptor = Receptor(
        time="2023-01-01 12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    state.register([(str(sim_id), receptor)])
    result = SimulationResult(
        sim_id=sim_id,
        status="failed",
        traj_present=False,
        error="boom",
        started_at=dt.datetime.now(dt.timezone.utc),
    )

    state.record(result)

    with state._connect() as conn:
        row = conn.execute(
            "SELECT trajectory_status, error FROM simulations WHERE sim_id=?",
            (str(sim_id),),
        ).fetchone()
    assert row["trajectory_status"] == "failed"
    assert row["error"] == "boom"


def test_record_result_keeps_interrupted_attempt_pending(tmp_path, sim_id):
    state = SqliteIndex(tmp_path)
    receptor = Receptor(
        time="2023-01-01 12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    state.register([(str(sim_id), receptor)])
    result = SimulationResult(
        sim_id=sim_id,
        status="interrupted",
        traj_present=False,
        error="Worker preempted by SIGTERM",
        started_at=dt.datetime.now(dt.timezone.utc),
    )

    state.record(result)

    with state._connect() as conn:
        row = conn.execute(
            "SELECT trajectory_status, error FROM simulations WHERE sim_id=?",
            (str(sim_id),),
        ).fetchone()
    assert row["trajectory_status"] == "pending"
    assert row["error"] == "Worker preempted by SIGTERM"
    assert state.pending_trajectories() == [str(sim_id)]


def test_execute_task_footprint_complete_empty_result(tmp_path, task, monkeypatch):
    foot_config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)
    )
    foot_task = task.model_copy(update={"foot_configs": {"slv": foot_config}})
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()

    fake_sim = _fake_simulation(task.sim_id, traj_path)
    fake_sim.generate_footprint.return_value = None
    monkeypatch.setattr("stilt.execution.execute.Simulation", lambda **kw: fake_sim)

    result = execute_task(foot_task)

    assert result.status == "complete-empty"
    assert result.foot_paths == []
    assert result.empty_footprints == ["slv"]
    assert result.footprint_statuses == {"slv": "complete-empty"}


def test_execute_task_returns_failed_result_for_footprint_simulation_error(
    tmp_path, task, monkeypatch
):
    foot_config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)
    )
    foot_task = task.model_copy(update={"foot_configs": {"slv": foot_config}})
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()

    fake_sim = _fake_simulation(task.sim_id, traj_path)
    fake_sim.generate_footprint.side_effect = SimulationError("Footprint failed")
    monkeypatch.setattr("stilt.execution.execute.Simulation", lambda **kw: fake_sim)

    result = execute_task(foot_task)

    assert result.status == "failed"
    assert result.error == "Footprint failed"
    assert result.footprint_statuses == {"slv": "failed"}


def test_execute_task_skips_existing_footprint_outputs(tmp_path, task, monkeypatch):
    foot_config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)
    )
    foot_task = task.model_copy(
        update={"foot_configs": {"slv": foot_config}, "skip_existing": True}
    )
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()

    fake_sim = _fake_simulation(task.sim_id, traj_path)
    fake_sim.footprint_path("slv").touch()
    fake_sim.generate_footprint.side_effect = AssertionError("should not rerun")
    monkeypatch.setattr("stilt.execution.execute.Simulation", lambda **kw: fake_sim)

    result = execute_task(foot_task)

    assert result.status == "complete"
    assert result.footprint_statuses == {"slv": "complete"}


def test_execute_task_skips_existing_empty_footprint_markers(
    tmp_path, task, monkeypatch
):
    foot_config = FootprintConfig(
        grid=Grid(xmin=-114.0, xmax=-111.0, ymin=39.0, ymax=42.0, xres=0.1, yres=0.1)
    )
    foot_task = task.model_copy(
        update={"foot_configs": {"slv": foot_config}, "skip_existing": True}
    )
    traj_path = tmp_path / "sim_traj.parquet"
    traj_path.touch()

    fake_sim = _fake_simulation(task.sim_id, traj_path)
    fake_sim.files.empty_footprint_path("slv").touch()
    fake_sim.generate_footprint.side_effect = AssertionError("should not rerun")
    monkeypatch.setattr("stilt.execution.execute.Simulation", lambda **kw: fake_sim)

    result = execute_task(foot_task)

    assert result.status == "complete-empty"
    assert result.footprint_statuses == {"slv": "complete-empty"}


def test_execute_batch_interrupt_returns_interrupted_result(task, monkeypatch):
    def _raise_interrupt(_task):
        raise KeyboardInterrupt

    monkeypatch.setattr("stilt.execution.execute.execute_task", _raise_interrupt)

    results = execute_batch([task], n_cores=1)

    assert len(results) == 1
    assert results[0].status == "interrupted"
    assert not (task.compute_root / str(task.sim_id)).exists()


def test_execute_batch_returns_completed_results_without_side_effect_logs(
    task, monkeypatch
):
    started_at = dt.datetime.now(dt.timezone.utc)

    monkeypatch.setattr(
        "stilt.execution.execute.execute_task",
        lambda _task: SimulationResult(
            sim_id=task.sim_id,
            status="complete",
            traj_present=True,
            started_at=started_at,
        ),
    )

    results = execute_batch([task], n_cores=1)

    assert len(results) == 1
    assert results[0].status == "complete"
    assert not (task.compute_root / str(task.sim_id)).exists()


def test_execute_batch_continues_after_uncaught_simulation_error(
    tmp_path, task, monkeypatch, caplog
):
    other_receptor = Receptor(
        time=dt.datetime(2023, 1, 1, 13),
        longitude=-111.86,
        latitude=40.78,
        altitude=10.0,
    )
    second_task = task.model_copy(
        update={
            "sim_id": SimID.from_parts("hrrr", other_receptor),
            "receptor": other_receptor,
        }
    )
    started_at = dt.datetime.now(dt.timezone.utc)

    def _fake_execute_task(current_task):
        if current_task.sim_id == task.sim_id:
            raise SimulationError("uncaught boom")
        return SimulationResult(
            sim_id=current_task.sim_id,
            status="complete",
            traj_present=True,
            started_at=started_at,
            finished_at=started_at,
        )

    monkeypatch.setattr("stilt.execution.execute.execute_task", _fake_execute_task)

    with caplog.at_level(logging.ERROR):
        results = execute_batch([task, second_task], n_cores=1)

    assert [result.status for result in results] == ["failed", "complete"]
    assert results[0].error == "uncaught boom"
    assert results[0].log_path is not None
    assert results[0].log_path.exists()
    assert "uncaught boom" in results[0].log_path.read_text()
    assert str(task.sim_id) in caplog.text


def test_execute_batch_pool_guard_converts_uncaught_exceptions(
    task, monkeypatch, caplog
):
    class _FakePool:
        def __init__(self, _n_cores, initializer=None):
            del initializer

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, batch):
            return [func(item) for item in batch]

    monkeypatch.setattr("stilt.execution.execute.multiprocessing.Pool", _FakePool)
    monkeypatch.setattr(
        "stilt.execution.execute.execute_task",
        lambda _task: (_ for _ in ()).throw(RuntimeError("pool boom")),
    )

    with caplog.at_level(logging.ERROR):
        [result] = execute_batch([task], n_cores=2)

    assert result.status == "error"
    assert result.error == "pool boom"
    assert result.log_path is not None
    assert result.log_path.exists()
    assert "pool boom" in result.log_path.read_text()
    assert str(task.sim_id) in caplog.text


def test_push_simulations_persists_results(tmp_path, receptor, monkeypatch):
    state = SqliteIndex(tmp_path)
    model = Model(
        project=tmp_path,
        config=_make_model_config(tmp_path),
        receptors=[receptor],
        index=state,
    )
    [sim_id] = model.register_pending()
    started_at = dt.datetime.now(dt.timezone.utc)

    def _fake_execute_batch(batch, n_cores=1):
        del batch, n_cores
        traj_file = (
            ProjectFiles(Path(model.layout.output_dir))
            .simulation(sim_id)
            .trajectory_path
        )
        traj_file.parent.mkdir(parents=True, exist_ok=True)
        traj_file.write_bytes(b"traj")
        result = SimulationResult(
            sim_id=SimID(sim_id),
            status="complete",
            traj_present=True,
            started_at=started_at,
            finished_at=started_at,
        )
        return [result]

    monkeypatch.setattr(
        "stilt.execution.entrypoints.execute_batch",
        _fake_execute_batch,
    )

    push_simulations(model, [sim_id])

    state.rebuild()

    assert state.counts().completed == 1


class _FakeUnitOfWork:
    def __init__(self, sim_id: str, state: SqliteIndex):
        self.sim_id = sim_id
        self._state = state
        self._released = False

    def release(self):
        self._released = True

    def record(self, result: SimulationResult) -> None:
        self._state.record(result)

    @property
    def released(self):
        return self._released


class _FakePullState:
    def __init__(self, state: SqliteIndex, sim_ids: list[str]):
        self._state = state
        self._sim_ids = list(sim_ids)
        self.released: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._state, name)

    @contextmanager
    def claim_one(self) -> Iterator[_FakeUnitOfWork | None]:
        if not self._sim_ids:
            yield None
            return
        sim_id = self._sim_ids.pop(0)
        uow = _FakeUnitOfWork(sim_id=sim_id, state=self._state)
        yield uow
        if uow.released:
            self.released.append(sim_id)


def test_pull_simulations_drains_queue(tmp_path, receptor, monkeypatch):
    state = SqliteIndex(tmp_path)
    model = Model(
        project=tmp_path,
        config=_make_model_config(tmp_path),
        receptors=[receptor],
        index=state,
    )
    [sim_id] = model.register_pending()
    fake_state = _FakePullState(state, [sim_id])
    model = Model(
        project=tmp_path,
        config=_make_model_config(tmp_path),
        receptors=[receptor],
        index=fake_state,
    )
    started_at = dt.datetime.now(dt.timezone.utc)

    monkeypatch.setattr(
        "stilt.execution.entrypoints.execute_task",
        lambda task: SimulationResult(
            sim_id=task.sim_id,
            status="complete",
            traj_present=True,
            started_at=started_at,
        ),
    )

    pull_simulations(model, follow=False)

    assert state.counts().completed == 1
    assert fake_state.released == []


def test_pull_simulations_requires_postgres_state(tmp_path):
    model = SimpleNamespace(index=SqliteIndex(tmp_path))

    with pytest.raises(ValueError, match="claim-capable index backend"):
        pull_simulations(model, follow=False)
