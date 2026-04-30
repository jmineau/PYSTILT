"""Tests for the SQLite index backend."""

import datetime as dt
import uuid

import pytest

from stilt.execution import SimulationResult
from stilt.index import IndexCounts, OutputSummary
from stilt.index import sqlite as sqlite_mod
from stilt.index.sqlite import SqliteIndex
from stilt.index.updates import index_update_from_summary
from stilt.receptors import PointReceptor, Receptor
from stilt.simulation import SimID
from stilt.storage import (
    ProjectFiles,
    footprint_filename,
    trajectory_filename,
)


def _record_result(
    state: SqliteIndex,
    sim_id: str,
    *,
    status: str = "complete",
    traj_present: bool = True,
    footprint_statuses: dict[str, str] | None = None,
    error: str | None = None,
    finished_at: dt.datetime | None = None,
) -> None:
    footprints = dict(state.summaries([sim_id]).get(sim_id, OutputSummary()).footprints)
    if footprint_statuses is not None:
        footprints.update(footprint_statuses)
    state.record(
        SimulationResult(
            sim_id=SimID(sim_id),
            status=status,
            traj_present=traj_present,
            footprint_statuses=footprints,
            error=error,
            started_at=finished_at,
            finished_at=finished_at,
        )
    )


def _state_call(state: SqliteIndex, method: str, *args, **kwargs):
    if method == "output_summary":
        [sim_id] = args
        return state.summaries([sim_id]).get(sim_id, OutputSummary())
    if method == "output_summaries":
        return state.summaries(*args, **kwargs)
    if method == "get_receptor":
        [sim_id] = args
        return state.receptors_for([sim_id])[sim_id]
    if method == "get_receptors":
        return state.receptors_for(*args, **kwargs)
    return getattr(state, method)(*args, **kwargs)


def _force_traj_status(state: SqliteIndex, sim_id: str, status: str) -> None:
    with state._connect() as conn:
        conn.execute(
            "UPDATE simulations SET trajectory_status=? WHERE sim_id=?",
            (status, sim_id),
        )


def _mark_trajectory_complete(state: SqliteIndex, sim_id: str) -> None:
    _record_result(state, sim_id)


def _mark_trajectory_failed(
    state: SqliteIndex,
    sim_id: str,
    error: str = "",
) -> None:
    _record_result(
        state,
        sim_id,
        status="failed",
        traj_present=False,
        error=error or None,
    )


def _mark_footprint_complete(
    state: SqliteIndex,
    sim_id: str,
    name: str,
) -> None:
    _record_result(state, sim_id, footprint_statuses={name: "complete"})


def _mark_footprint_empty(
    state: SqliteIndex,
    sim_id: str,
    name: str,
) -> None:
    _record_result(state, sim_id, footprint_statuses={name: "complete-empty"})


def _mark_footprint_failed(
    state: SqliteIndex,
    sim_id: str,
    name: str,
    error: str = "",
) -> None:
    _record_result(
        state,
        sim_id,
        status="error",
        footprint_statuses={name: "failed"},
        error=error or None,
    )


def _record_outputs(
    state: SqliteIndex,
    sim_id: str,
    summary: OutputSummary,
) -> None:
    with state._connect() as conn:
        state._store_update_conn(conn, sim_id, index_update_from_summary(summary))


def _make_receptor(
    lon: float = -111.85,
    lat: float = 40.77,
    altitude: float = 5.0,
    time: str = "202301011200",
) -> Receptor:
    return PointReceptor(time=time, longitude=lon, latitude=lat, altitude=altitude)


def _sid(receptor: Receptor, met: str = "hrrr") -> str:
    return f"{met}_{receptor.id}"


def _memory_state(tmp_path) -> SqliteIndex:
    return SqliteIndex(
        tmp_path,
        db_path=f"file:{uuid.uuid4().hex}?mode=memory&cache=shared",
        uri=True,
    )


def test_supports_wal_on_local_filesystem(monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite_mod, "_mount_type", lambda _path: "ext4")

    assert sqlite_mod._supports_wal(tmp_path / "index.db") is True


def test_disables_wal_on_nfs_filesystem(monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite_mod, "_mount_type", lambda _path: "nfs")

    assert sqlite_mod._supports_wal(tmp_path / "index.db") is False
    with pytest.warns(UserWarning, match="SQLite WAL disabled"):
        SqliteIndex(tmp_path)


@pytest.mark.parametrize(
    "state_factory",
    [SqliteIndex, _memory_state],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_register_and_query_simulations(tmp_path, state_factory):
    state = state_factory(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    _state_call(state, "register", sim_id, receptor)
    _state_call(state, "register", [(sim_id, receptor)])

    assert _state_call(state, "has", sim_id)
    assert _state_call(state, "count") == 1
    assert _state_call(state, "sim_ids") == [sim_id]
    assert _state_call(state, "output_summary", sim_id) == OutputSummary()
    assert _state_call(state, "pending_trajectories") == [sim_id]


@pytest.mark.parametrize(
    "state_factory",
    [SqliteIndex, _memory_state],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_trajectory_and_footprint_status_flow(tmp_path, state_factory):
    state = state_factory(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    _state_call(state, "register", [(sim_id, receptor)], footprint_names=["slv"])
    _mark_trajectory_complete(state, sim_id)

    assert _state_call(state, "output_summary", sim_id) == OutputSummary(
        traj_present=True,
    )
    assert state.counts().completed == 0

    _mark_footprint_empty(state, sim_id, "slv")

    assert _state_call(state, "output_summary", sim_id) == OutputSummary(
        traj_present=True,
        footprints={"slv": "complete-empty"},
    )
    assert state.counts().completed == 1


def test_failed_footprint_does_not_count_as_complete(tmp_path):
    state = SqliteIndex(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    _state_call(state, "register", [(sim_id, receptor)], footprint_names=["slv"])
    _mark_trajectory_complete(state, sim_id)
    _mark_footprint_failed(state, sim_id, "slv", "boom")

    assert _state_call(state, "output_summary", sim_id) == OutputSummary(
        traj_present=True,
        footprints={"slv": "failed"},
    )
    assert state.counts().completed == 0


@pytest.mark.parametrize(
    "state_factory",
    [SqliteIndex, _memory_state],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_get_receptor_roundtrip(tmp_path, state_factory):
    state = state_factory(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    _state_call(state, "register", sim_id, receptor)
    loaded = _state_call(state, "get_receptor", sim_id)

    assert isinstance(loaded, PointReceptor)
    assert loaded.longitude == pytest.approx(receptor.longitude)
    assert loaded.latitude == pytest.approx(receptor.latitude)
    assert loaded.altitude == pytest.approx(receptor.altitude)


def test_reset_runtime_state_resets_running_to_pending(tmp_path):
    state = SqliteIndex(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    _state_call(state, "register", sim_id, receptor)
    _force_traj_status(state, sim_id, "running")

    state.rebuild()

    assert _state_call(state, "pending_trajectories") == [sim_id]


@pytest.mark.parametrize(
    "state_factory",
    [SqliteIndex, _memory_state],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_rebuild_clears_stale_complete_artifacts(tmp_path, state_factory):
    state = state_factory(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    _state_call(state, "register", sim_id, receptor)
    _mark_trajectory_complete(state, sim_id)
    ProjectFiles(tmp_path).by_id_dir.mkdir(parents=True, exist_ok=True)

    state.rebuild()

    assert _state_call(state, "pending_trajectories") == [sim_id]
    assert _state_call(state, "output_summary", sim_id) == OutputSummary()


def test_counts_partition_completed_pending_running_and_failed(tmp_path):
    state = SqliteIndex(tmp_path)
    complete_receptor = _make_receptor(time="202301011200")
    pending_receptor = _make_receptor(time="202301011300")
    running_receptor = _make_receptor(time="202301011400")
    failed_receptor = _make_receptor(time="202301011500")

    complete_sim_id = _sid(complete_receptor)
    pending_sim_id = _sid(pending_receptor)
    running_sim_id = _sid(running_receptor)
    failed_sim_id = _sid(failed_receptor)

    _state_call(
        state,
        "register",
        [
            (complete_sim_id, complete_receptor),
            (pending_sim_id, pending_receptor),
            (running_sim_id, running_receptor),
            (failed_sim_id, failed_receptor),
        ],
        footprint_names=["slv"],
    )
    _mark_trajectory_complete(state, complete_sim_id)
    _mark_footprint_complete(state, complete_sim_id, "slv")
    _mark_trajectory_complete(state, pending_sim_id)
    _force_traj_status(state, running_sim_id, "running")
    _mark_trajectory_failed(state, failed_sim_id, "boom")

    assert state.counts().completed == 1
    assert _state_call(state, "pending_trajectories") == [pending_sim_id]
    assert state.counts() == IndexCounts(
        total=4,
        completed=1,
        running=1,
        pending=1,
        failed=1,
    )


def test_scene_counts_partition_registered_simulations(tmp_path):
    state = SqliteIndex(tmp_path)
    rec_a = _make_receptor(time="202301011200")
    rec_b = _make_receptor(time="202301011300")
    rec_c = _make_receptor(time="202301011400")

    sim_a = _sid(rec_a)
    sim_b = _sid(rec_b)
    sim_c = _sid(rec_c)

    state.register([(sim_a, rec_a)], scene_id="scene-a")
    state.register([(sim_b, rec_b)], scene_id="scene-a")
    state.register([(sim_c, rec_c)], scene_id="scene-b")

    assert state.counts(scene_id="scene-a") == IndexCounts(
        total=2,
        completed=0,
        running=0,
        pending=2,
        failed=0,
    )
    assert state.scene_counts() == {
        "scene-a": IndexCounts(
            total=2,
            completed=0,
            running=0,
            pending=2,
            failed=0,
        ),
        "scene-b": IndexCounts(
            total=1,
            completed=0,
            running=0,
            pending=1,
            failed=0,
        ),
    }


def test_record_artifacts_roundtrip(tmp_path):
    state = SqliteIndex(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    _state_call(state, "register", sim_id, receptor)
    summary = OutputSummary(
        traj_present=True,
        error_traj_present=True,
        log_present=True,
        footprints={"slv": "complete", "fine": "complete-empty"},
    )

    _record_outputs(state, sim_id, summary)

    assert _state_call(state, "output_summary", sim_id) == summary


def test_rebuild_rebuilds_from_output_artifacts(tmp_path):
    state = SqliteIndex(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    state.register([(sim_id, receptor)], footprint_names=["slv"])

    sim_dir = ProjectFiles(tmp_path).by_id_dir / sim_id
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir / trajectory_filename(sim_id)).write_bytes(b"traj")
    (sim_dir / footprint_filename(sim_id, "slv")).write_bytes(b"foot")

    state.rebuild()

    assert state.summaries([sim_id]).get(sim_id, OutputSummary()) == OutputSummary(
        traj_present=True,
        footprints={"slv": "complete"},
    )
    assert state.counts().completed == 1


def test_rebuild_marks_logged_failures_failed_until_reset(tmp_path):
    state = SqliteIndex(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    state.register([(sim_id, receptor)])

    sim_dir = ProjectFiles(tmp_path).by_id_dir / sim_id
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir / "stilt.log").write_text("failure")

    state.rebuild()

    with state._connect() as conn:
        row = conn.execute(
            "SELECT trajectory_status FROM simulations WHERE sim_id=?",
            (sim_id,),
        ).fetchone()
    assert row["trajectory_status"] == "failed"
    assert sim_id not in _state_call(state, "pending_trajectories")

    _state_call(state, "reset_to_pending", [sim_id])

    assert _state_call(state, "pending_trajectories") == [sim_id]


def test_failed_results_prevent_retry_until_reset(tmp_path):
    state = SqliteIndex(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    _state_call(state, "register", sim_id, receptor)
    _mark_trajectory_failed(state, sim_id, "transient")

    with state._connect() as conn:
        row = conn.execute(
            "SELECT trajectory_status, error FROM simulations WHERE sim_id=?",
            (sim_id,),
        ).fetchone()
    assert sim_id not in _state_call(state, "pending_trajectories")
    assert row["trajectory_status"] == "failed"
    assert row["error"] == "transient"

    _state_call(state, "reset_to_pending", [sim_id])

    assert _state_call(state, "pending_trajectories") == [sim_id]
    with state._connect() as conn:
        row = conn.execute(
            "SELECT trajectory_status, error FROM simulations WHERE sim_id=?",
            (sim_id,),
        ).fetchone()
    assert row["trajectory_status"] == "pending"
    assert row["error"] is None


def test_reset_to_pending_clear_outputs_removes_completion_state(tmp_path):
    state = SqliteIndex(tmp_path)
    receptor = _make_receptor()
    sim_id = _sid(receptor)

    _state_call(state, "register", [(sim_id, receptor)], footprint_names=["slv"])
    _mark_trajectory_complete(state, sim_id)
    _mark_footprint_complete(state, sim_id, "slv")

    assert state.counts().completed == 1

    _state_call(state, "reset_to_pending", [sim_id], clear_outputs=True)

    assert _state_call(state, "output_summary", sim_id) == OutputSummary()
    assert _state_call(state, "pending_trajectories") == [sim_id]
    assert state.counts().completed == 0


def test_max_rows_prunes_oldest_terminal_rows_on_register(tmp_path):
    state = SqliteIndex(tmp_path, max_rows=2)
    old_receptor = _make_receptor(time="202301011200")
    active_receptor = _make_receptor(time="202301011300")
    new_receptor = _make_receptor(time="202301011400")
    old_sim_id = _sid(old_receptor)
    active_sim_id = _sid(active_receptor)
    new_sim_id = _sid(new_receptor)

    _state_call(
        state,
        "register",
        [(old_sim_id, old_receptor), (active_sim_id, active_receptor)],
    )
    _mark_trajectory_complete(state, old_sim_id)

    _state_call(state, "register", new_sim_id, new_receptor)

    assert _state_call(state, "count") == 2
    assert not _state_call(state, "has", old_sim_id)
    assert _state_call(state, "has", active_sim_id)
    assert _state_call(state, "has", new_sim_id)


def test_max_rows_soft_cap_keeps_active_rows_when_no_terminal_candidates(tmp_path):
    state = SqliteIndex(tmp_path, max_rows=1)
    receptor_a = _make_receptor(time="202301011200")
    receptor_b = _make_receptor(time="202301011300")
    sim_id_a = _sid(receptor_a)
    sim_id_b = _sid(receptor_b)

    _state_call(state, "register", [(sim_id_a, receptor_a), (sim_id_b, receptor_b)])

    assert _state_call(state, "count") == 2
    assert _state_call(state, "pending_trajectories") == [sim_id_a, sim_id_b]


def test_max_rows_prunes_oldest_terminal_rows_after_rebuild(tmp_path):
    state = SqliteIndex(tmp_path, max_rows=1)
    old_receptor = _make_receptor(time="202301011200")
    new_receptor = _make_receptor(time="202301011300")
    old_sim_id = _sid(old_receptor)
    new_sim_id = _sid(new_receptor)

    _state_call(
        state, "register", [(old_sim_id, old_receptor), (new_sim_id, new_receptor)]
    )

    old_dir = ProjectFiles(tmp_path).by_id_dir / old_sim_id
    new_dir = ProjectFiles(tmp_path).by_id_dir / new_sim_id
    old_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)
    (old_dir / trajectory_filename(old_sim_id)).write_bytes(b"traj")
    (new_dir / trajectory_filename(new_sim_id)).write_bytes(b"traj")

    state.rebuild()

    assert _state_call(state, "count") == 1
    assert not _state_call(state, "has", old_sim_id)
    assert _state_call(state, "has", new_sim_id)
