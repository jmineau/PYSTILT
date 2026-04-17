"""Tests for stilt.repositories SQLite file and shared-memory modes."""

import datetime as dt

import pytest

from stilt.receptor import Receptor
from stilt.repositories import (
    ArtifactSummary,
    SimulationAttempt,
    SimulationClaim,
    SQLiteRepository,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_receptor(
    lon=-111.85, lat=40.77, altitude=5.0, time="202301011200"
) -> Receptor:
    return Receptor(time=time, longitude=lon, latitude=lat, altitude=altitude)


def _sid(receptor: Receptor, met: str = "hrrr") -> str:
    time_str = receptor.time.strftime("%Y%m%d%H%M")
    return f"{met}_{time_str}_{receptor.location_id}"


def _memory_repo(tmp_path) -> SQLiteRepository:
    """Build a SQLite repository using shared in-memory storage."""
    return SQLiteRepository.in_memory(tmp_path)


# ---------------------------------------------------------------------------
# SQLiteRepository
# ---------------------------------------------------------------------------


class TestSQLiteRepository:
    def test_init_creates_db(self, tmp_path):
        SQLiteRepository(tmp_path)
        assert (tmp_path / "simulations" / "state.sqlite").exists()

    def test_register_single_receptor(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        assert repo.has(sid)

    def test_register_many_is_idempotent(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register_many([(sid, r)])
        repo.register_many([(sid, r)])  # second call - no-op
        assert repo.count() == 1

    def test_count_and_all_sim_ids(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        repo.register_many([(_sid(r1), r1), (_sid(r2), r2)])
        assert repo.count() == 2
        ids = repo.all_sim_ids()
        assert _sid(r1) in ids
        assert _sid(r2) in ids

    def test_has_returns_false_for_unknown(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        assert not repo.has("nonexistent_sim_id")

    def test_traj_status_starts_as_pending(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        assert repo.traj_status(sid) == "pending"

    def test_traj_status_none_for_unknown(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        assert repo.traj_status("no_such_id") is None

    def test_pending_trajectories(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        assert sid in repo.pending_trajectories()

    def test_sync_is_no_op(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.sync()  # should not raise or change state
        assert repo.traj_status(sid) == "pending"

    # -- mark_*() round-trip tests -------------------------------------------

    def test_mark_trajectory_complete(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_trajectory_complete(sid)
        assert repo.traj_status(sid) == "complete"
        assert sid in repo.completed_trajectories()

    def test_mark_trajectory_failed(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_trajectory_failed(sid, "met missing")
        assert repo.traj_status(sid) == "failed"
        assert sid not in repo.completed_trajectories()

    def test_mark_footprint_complete(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_footprint_complete(sid, "slv")
        assert repo.footprint_completed(sid, "slv")
        assert not repo.footprint_completed(sid, "other")
        assert repo.footprint_status(sid, "slv") == "complete"

    def test_mark_footprint_empty(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_footprint_empty(sid, "slv")
        assert repo.footprint_completed(sid, "slv")
        assert repo.footprint_status(sid, "slv") == "complete-empty"

    def test_mark_footprint_failed(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_footprint_failed(sid, "slv", "oops")
        assert not repo.footprint_completed(sid, "slv")
        assert repo.footprint_status(sid, "slv") == "failed"

    def test_mark_trajectory_complete_clears_error(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_trajectory_failed(sid, "transient error")
        repo.mark_trajectory_complete(sid)
        assert repo.traj_status(sid) == "complete"

    # -- other methods -------------------------------------------------------

    def test_get_receptor_roundtrip_point(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        loaded = repo.get_receptor(sid)
        assert loaded.kind == "point"
        assert loaded.longitude == pytest.approx(r.longitude)
        assert loaded.latitude == pytest.approx(r.latitude)
        assert loaded.altitude == pytest.approx(r.altitude)

    def test_get_receptor_raises_for_unknown(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        with pytest.raises(KeyError, match="sim_id not found"):
            repo.get_receptor("nonexistent")

    def test_to_dataframe(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        df = repo.to_dataframe()
        assert sid in df.index
        assert "traj_status" in df.columns

    def test_to_dataframe_empty(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        df = repo.to_dataframe()
        assert df.empty


# ---------------------------------------------------------------------------
# SQLiteRepository shared-memory mode
# ---------------------------------------------------------------------------


class TestSQLiteMemoryRepository:
    def test_register_and_has(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        assert repo.has(sid)

    def test_register_many_is_idempotent(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register_many([(sid, r)])
        repo.register_many([(sid, r)])
        assert repo.count() == 1

    def test_count_and_all_sim_ids(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        repo.register_many([(_sid(r1), r1), (_sid(r2), r2)])
        assert repo.count() == 2

    def test_has_returns_false_for_unknown(self, tmp_path):
        repo = _memory_repo(tmp_path)
        assert not repo.has("no_such")

    def test_traj_status_starts_as_pending(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        assert repo.traj_status(sid) == "pending"

    def test_traj_status_none_for_unknown(self, tmp_path):
        repo = _memory_repo(tmp_path)
        assert repo.traj_status("no_such") is None

    def test_pending_trajectories(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        assert sid in repo.pending_trajectories()

    def test_sync_is_no_op(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.sync()  # should not raise or change state
        assert repo.traj_status(sid) == "pending"

    # -- mark_*() round-trip tests -------------------------------------------

    def test_mark_trajectory_complete(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_trajectory_complete(sid)
        assert repo.traj_status(sid) == "complete"
        assert sid in repo.completed_trajectories()

    def test_mark_trajectory_failed(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_trajectory_failed(sid, "bad met")
        assert repo.traj_status(sid) == "failed"

    def test_mark_footprint_complete(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_footprint_complete(sid, "slv")
        assert repo.footprint_completed(sid, "slv")
        assert repo.footprint_status(sid, "slv") == "complete"

    def test_mark_footprint_empty(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_footprint_empty(sid, "slv")
        assert repo.footprint_completed(sid, "slv")
        assert repo.footprint_status(sid, "slv") == "complete-empty"

    def test_mark_footprint_failed(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_footprint_failed(sid, "slv", "oops")
        assert not repo.footprint_completed(sid, "slv")
        assert repo.footprint_status(sid, "slv") == "failed"

    def test_get_receptor_roundtrip(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        loaded = repo.get_receptor(sid)
        assert loaded.kind == "point"
        assert loaded.longitude == pytest.approx(r.longitude)

    def test_get_receptor_raises_for_unknown(self, tmp_path):
        repo = _memory_repo(tmp_path)
        with pytest.raises(KeyError):
            repo.get_receptor("no_such")

    def test_to_dataframe(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        df = repo.to_dataframe()
        assert sid in df.index

    def test_to_dataframe_empty(self, tmp_path):
        repo = _memory_repo(tmp_path)
        df = repo.to_dataframe()
        assert df.empty
        assert "traj_status" in df.columns


# ---------------------------------------------------------------------------
# claim_pending / release_claim — SQLite
# ---------------------------------------------------------------------------


class TestSQLiteClaimPending:
    def test_claim_pending_claims_returns_claim_objects(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)

        claims = repo.claim_pending_claims(1, worker_id="worker-a", lease_ttl=60.0)

        assert len(claims) == 1
        assert claims[0].sim_id == sid
        assert claims[0].worker_id == "worker-a"
        assert repo.claim_is_current(sid, claims[0].claim_token)

    def test_claim_pending_returns_pending_ids(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        claimed = repo.claim_pending(1)
        assert claimed == [sid]
        assert repo.traj_status(sid) == "running"
        claims = repo.list_claims()
        assert len(claims) == 1
        assert claims[0].sim_id == sid

    def test_claim_pending_respects_limit(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        r3 = _make_receptor(time="202301011400")
        repo.register_many([(_sid(r1), r1), (_sid(r2), r2), (_sid(r3), r3)])
        claimed = repo.claim_pending(2)
        assert len(claimed) == 2

    def test_claim_pending_skips_running(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        repo.register_many([(_sid(r1), r1), (_sid(r2), r2)])
        first = repo.claim_pending(1)
        second = repo.claim_pending(1)
        assert len(first) == 1
        assert len(second) == 1
        assert first[0] != second[0]

    def test_claim_pending_empty_when_drained(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        repo.register(_sid(r), r)
        repo.claim_pending(1)
        assert repo.claim_pending(1) == []

    def test_release_claim_returns_to_pending(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.claim_pending(1)
        assert repo.traj_status(sid) == "running"
        repo.release_claim([sid])
        assert repo.traj_status(sid) == "pending"
        assert repo.list_claims() == []

    def test_release_claim_empty_list_is_noop(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        repo.release_claim([])  # should not raise

    def test_release_claim_ignores_non_running(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.release_claim([sid])
        assert repo.traj_status(sid) == "pending"

    def test_reset_to_pending_does_not_demote_running(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.claim_pending(1)

        repo.reset_to_pending([sid])

        assert repo.traj_status(sid) == "running"

    def test_reclaim_expired_claims_returns_pending(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)

        claims = repo.claim_pending_claims(1, lease_ttl=-1.0)
        assert len(claims) == 1

        reclaimed = repo.reclaim_expired_claims()

        assert reclaimed == [sid]
        assert repo.traj_status(sid) == "pending"
        assert repo.list_claims() == []


# ---------------------------------------------------------------------------
# claim_pending / release_claim — SQLite shared-memory
# ---------------------------------------------------------------------------


class TestSQLiteMemoryClaimPending:
    def test_claim_pending_claims_returns_claim_objects(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)

        claims = repo.claim_pending_claims(1, worker_id="worker-a", lease_ttl=60.0)

        assert len(claims) == 1
        assert claims[0].sim_id == sid
        assert claims[0].worker_id == "worker-a"
        assert repo.claim_is_current(sid, claims[0].claim_token)

    def test_claim_pending_returns_pending_ids(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        claimed = repo.claim_pending(1)
        assert claimed == [sid]
        assert repo.traj_status(sid) == "running"
        claims = repo.list_claims()
        assert len(claims) == 1
        assert claims[0].sim_id == sid

    def test_claim_pending_respects_limit(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        r3 = _make_receptor(time="202301011400")
        repo.register_many([(_sid(r1), r1), (_sid(r2), r2), (_sid(r3), r3)])
        claimed = repo.claim_pending(2)
        assert len(claimed) == 2

    def test_claim_pending_empty_when_drained(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        repo.register(_sid(r), r)
        repo.claim_pending(1)
        assert repo.claim_pending(1) == []

    def test_release_claim_returns_to_pending(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.claim_pending(1)
        assert repo.traj_status(sid) == "running"
        repo.release_claim([sid])
        assert repo.traj_status(sid) == "pending"
        assert repo.list_claims() == []

    def test_release_claim_ignores_non_running(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.release_claim([sid])
        assert repo.traj_status(sid) == "pending"

    def test_reset_to_pending_does_not_demote_running(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.claim_pending(1)

        repo.reset_to_pending([sid])

        assert repo.traj_status(sid) == "running"

    def test_reclaim_expired_claims_returns_pending(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)

        claims = repo.claim_pending_claims(1, lease_ttl=-1.0)
        assert len(claims) == 1

        reclaimed = repo.reclaim_expired_claims()

        assert reclaimed == [sid]
        assert repo.traj_status(sid) == "pending"
        assert repo.list_claims() == []


# ---------------------------------------------------------------------------
# Stale claim recovery via rebuild — SQLite
# ---------------------------------------------------------------------------


class TestSQLiteRebuildStaleRecovery:
    def test_reset_runtime_state_resets_stale_running_to_pending(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.claim_pending(1)

        repo.reset_runtime_state()

        assert repo.traj_status(sid) == "pending"
        assert repo.list_claims() == []

    def test_rebuild_resets_stale_running_to_pending(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.claim_pending(1)
        assert repo.traj_status(sid) == "running"

        from stilt.artifacts import simulation_index_path

        simulation_index_path(tmp_path).mkdir(parents=True, exist_ok=True)

        repo.rebuild()
        assert repo.traj_status(sid) == "pending"

    @pytest.mark.parametrize(
        ("repo_factory", "repo_mode"),
        [
            (SQLiteRepository, "sqlite-file"),
            (_memory_repo, "sqlite-memory"),
        ],
        ids=["sqlite-file", "sqlite-memory"],
    )
    def test_rebuild_clears_stale_completed_artifacts(
        self, tmp_path, repo_factory, repo_mode
    ):
        del repo_mode
        repo = repo_factory(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register(sid, r)
        repo.mark_trajectory_complete(sid)

        repo.rebuild()

        assert repo.traj_status(sid) == "pending"
        assert repo.artifact_summary(sid) == ArtifactSummary()


# ---------------------------------------------------------------------------
# Batch tracking — SQLite
# ---------------------------------------------------------------------------


class TestSQLiteBatchTracking:
    def test_register_many_with_batch_id(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        repo.register_many([(_sid(r1), r1), (_sid(r2), r2)], batch_id="jan")
        assert repo.batch_progress("jan") == (0, 2)

    def test_batch_progress_after_completion(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        sid1, sid2 = _sid(r1), _sid(r2)
        repo.register_many([(sid1, r1), (sid2, r2)], batch_id="jan")
        repo.mark_trajectory_complete(sid1)
        assert repo.batch_progress("jan") == (1, 2)

    def test_batch_progress_requires_footprint_targets(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register_many([(sid, r)], batch_id="jan", footprint_names=["slv"])
        repo.mark_trajectory_complete(sid)
        assert repo.batch_progress("jan") == (0, 1)

        repo.mark_footprint_complete(sid, "slv")
        assert repo.batch_progress("jan") == (1, 1)

    def test_reused_batch_id_preserves_union_total(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        repo.register_many([(_sid(r1), r1), (_sid(r2), r2)], batch_id="jan")
        repo.register_many([(_sid(r1), r1)], batch_id="jan")

        assert repo.batch_progress("jan") == (0, 2)

    def test_batch_progress_unknown_batch(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        assert repo.batch_progress("nonexistent") == (0, 0)

    def test_all_batches(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        repo.register_many([(_sid(r1), r1)], batch_id="a")
        repo.register_many([(_sid(r2), r2)], batch_id="b")
        batches = repo.all_batches()
        assert len(batches) == 2
        names = [b[0] for b in batches]
        assert "a" in names
        assert "b" in names

    def test_register_many_without_batch_has_no_batch(self, tmp_path):
        repo = SQLiteRepository(tmp_path)
        r = _make_receptor()
        repo.register_many([(_sid(r), r)])
        assert repo.all_batches() == []


# ---------------------------------------------------------------------------
# Batch tracking — SQLite shared-memory
# ---------------------------------------------------------------------------


class TestSQLiteMemoryBatchTracking:
    def test_register_many_with_batch_id(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        repo.register_many([(_sid(r1), r1), (_sid(r2), r2)], batch_id="jan")
        assert repo.batch_progress("jan") == (0, 2)

    def test_batch_progress_after_completion(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        sid1, sid2 = _sid(r1), _sid(r2)
        repo.register_many([(sid1, r1), (sid2, r2)], batch_id="jan")
        repo.mark_trajectory_complete(sid1)
        assert repo.batch_progress("jan") == (1, 2)

    def test_batch_progress_requires_footprint_targets(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        sid = _sid(r)
        repo.register_many([(sid, r)], batch_id="jan", footprint_names=["slv"])
        repo.mark_trajectory_complete(sid)
        assert repo.batch_progress("jan") == (0, 1)

        repo.mark_footprint_complete(sid, "slv")
        assert repo.batch_progress("jan") == (1, 1)

    def test_reused_batch_id_preserves_union_total(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        repo.register_many([(_sid(r1), r1), (_sid(r2), r2)], batch_id="jan")
        repo.register_many([(_sid(r1), r1)], batch_id="jan")

        assert repo.batch_progress("jan") == (0, 2)

    def test_batch_progress_unknown_batch(self, tmp_path):
        repo = _memory_repo(tmp_path)
        assert repo.batch_progress("nonexistent") == (0, 0)

    def test_all_batches(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r1 = _make_receptor(time="202301011200")
        r2 = _make_receptor(time="202301011300")
        repo.register_many([(_sid(r1), r1)], batch_id="a")
        repo.register_many([(_sid(r2), r2)], batch_id="b")
        batches = repo.all_batches()
        assert len(batches) == 2

    def test_register_many_without_batch_has_no_batch(self, tmp_path):
        repo = _memory_repo(tmp_path)
        r = _make_receptor()
        repo.register_many([(_sid(r), r)])
        assert repo.all_batches() == []


@pytest.mark.parametrize(
    ("repo_factory", "repo_mode"),
    [
        (SQLiteRepository, "sqlite-file"),
        (_memory_repo, "sqlite-memory"),
    ],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_completed_simulations_require_all_footprint_targets(
    tmp_path, repo_factory, repo_mode
):
    del repo_mode
    repo = repo_factory(tmp_path)
    r = _make_receptor()
    sid = _sid(r)
    repo.register_many([(sid, r)], footprint_names=["coarse", "fine"])
    repo.mark_trajectory_complete(sid)

    assert sid not in repo.completed_simulations()


@pytest.mark.parametrize(
    ("repo_factory", "repo_mode"),
    [
        (SQLiteRepository, "sqlite-file"),
        (_memory_repo, "sqlite-memory"),
    ],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_artifact_summary_defaults_empty(tmp_path, repo_factory, repo_mode):
    del repo_mode
    repo = repo_factory(tmp_path)
    r = _make_receptor()
    sid = _sid(r)
    repo.register(sid, r)

    summary = repo.artifact_summary(sid)

    assert summary == ArtifactSummary()


@pytest.mark.parametrize(
    ("repo_factory", "repo_mode"),
    [
        (SQLiteRepository, "sqlite-file"),
        (_memory_repo, "sqlite-memory"),
    ],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_record_artifacts_roundtrip(tmp_path, repo_factory, repo_mode):
    del repo_mode
    repo = repo_factory(tmp_path)
    r = _make_receptor()
    sid = _sid(r)
    repo.register(sid, r)

    repo.record_artifacts(
        sid,
        ArtifactSummary(
            traj_present=True,
            error_traj_present=True,
            log_present=True,
            footprints={"slv": "complete", "fine": "complete-empty"},
        ),
    )

    assert repo.artifact_summary(sid) == ArtifactSummary(
        traj_present=True,
        error_traj_present=True,
        log_present=True,
        footprints={"slv": "complete", "fine": "complete-empty"},
    )


@pytest.mark.parametrize(
    ("repo_factory", "repo_mode"),
    [
        (SQLiteRepository, "sqlite-file"),
        (_memory_repo, "sqlite-memory"),
    ],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_claim_roundtrip(tmp_path, repo_factory, repo_mode):
    del repo_mode
    repo = repo_factory(tmp_path)
    r = _make_receptor()
    sid = _sid(r)
    repo.register(sid, r)
    now = dt.datetime(2026, 4, 13, 12, tzinfo=dt.timezone.utc)
    claim = SimulationClaim(
        sim_id=sid,
        claim_token="claim-1",
        worker_id="worker-a",
        claimed_at=now,
        heartbeat_at=now,
        expires_at=now + dt.timedelta(minutes=30),
    )

    repo.upsert_claim(claim)

    assert repo.list_claims() == [claim]

    repo.delete_claim(sid, "claim-1")
    assert repo.list_claims() == []


@pytest.mark.parametrize(
    ("repo_factory", "repo_mode"),
    [
        (SQLiteRepository, "sqlite-file"),
        (_memory_repo, "sqlite-memory"),
    ],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_attempt_roundtrip(tmp_path, repo_factory, repo_mode):
    del repo_mode
    repo = repo_factory(tmp_path)
    r = _make_receptor()
    sid = _sid(r)
    repo.register(sid, r)
    attempt = SimulationAttempt(
        attempt_id="attempt-1",
        sim_id=sid,
        claim_token="claim-1",
        started_at=dt.datetime(2026, 4, 13, 12, tzinfo=dt.timezone.utc),
        finished_at=dt.datetime(2026, 4, 13, 12, 5, tzinfo=dt.timezone.utc),
        outcome="failed",
        terminal=False,
        error="transient",
    )

    repo.record_attempt(attempt)

    assert repo.list_attempts(sid) == [attempt]


@pytest.mark.parametrize(
    ("repo_factory", "repo_mode"),
    [
        (SQLiteRepository, "sqlite-file"),
        (_memory_repo, "sqlite-memory"),
    ],
    ids=["sqlite-file", "sqlite-memory"],
)
def test_clear_footprints_removes_terminal_success(tmp_path, repo_factory, repo_mode):
    del repo_mode
    repo = repo_factory(tmp_path)
    r = _make_receptor()
    sid = _sid(r)
    repo.register_many([(sid, r)], footprint_names=["slv"])
    repo.mark_trajectory_complete(sid)
    repo.mark_footprint_complete(sid, "slv")

    assert sid in repo.completed_simulations()

    repo.clear_footprints([sid], names=["slv"])

    assert not repo.footprint_completed(sid, "slv")
    assert repo.footprint_status(sid, "slv") is None
    assert sid not in repo.completed_simulations()
