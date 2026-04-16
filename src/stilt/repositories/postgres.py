"""PostgreSQL repository backend."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import pandas as pd

from stilt.receptor import Receptor
from stilt.simulation import SimID

from .protocol import (
    ArtifactSummary,
    SimulationAttempt,
    SimulationClaim,
    _normalize_footprint_names,
    simulation_complete,
    simulation_pending,
    trajectory_status_from_state,
)

POSTGRES_PENDING_SIMULATIONS_SQL = """
SELECT COUNT(*)
FROM simulations AS s
LEFT JOIN artifact_index AS ai ON ai.sim_id = s.sim_id
LEFT JOIN claims AS c
  ON c.sim_id = s.sim_id
 AND c.expires_at >= NOW()
LEFT JOIN LATERAL (
  SELECT terminal
  FROM attempts AS a
  WHERE a.sim_id = s.sim_id
  ORDER BY a.started_at DESC, a.attempt_id DESC
  LIMIT 1
) AS latest ON TRUE
WHERE c.sim_id IS NULL
  AND COALESCE(latest.terminal, FALSE) = FALSE
  AND NOT (
    COALESCE(ai.traj_present, FALSE) = TRUE
    AND NOT EXISTS (
      SELECT 1
      FROM jsonb_array_elements_text(s.footprint_targets) AS target(name)
      WHERE COALESCE(ai.footprints ->> target.name, '')
        NOT IN ('complete', 'complete-empty')
    )
  )
"""


class _TransactionalClaimRepository:
    """Repository view bound to one open PostgreSQL claim transaction."""

    def __init__(
        self,
        owner: PostgreSQLRepository,
        conn: Any,
        claim: SimulationClaim,
    ) -> None:
        self._owner = owner
        self._conn = conn
        self._claim = claim

    def mark_trajectory_complete(self, sim_id: str) -> None:
        """Mark one trajectory complete inside the open claim transaction."""
        self._conn.execute(
            "UPDATE simulations SET trajectory_status = 'complete', error = NULL"
            " WHERE sim_id = %s",
            (sim_id,),
        )
        summary = self._owner._artifact_summary_from_conn(self._conn, sim_id)
        self._owner._store_artifact_summary(
            self._conn,
            sim_id,
            ArtifactSummary(
                traj_present=True,
                error_traj_present=summary.error_traj_present,
                log_present=summary.log_present,
                footprints=summary.footprints,
            ),
        )

    def mark_trajectory_failed(self, sim_id: str, error: str = "") -> None:
        """Mark one trajectory failed inside the open claim transaction."""
        self._conn.execute(
            "UPDATE simulations SET trajectory_status = 'failed', error = %s"
            " WHERE sim_id = %s",
            (error or None, sim_id),
        )
        summary = self._owner._artifact_summary_from_conn(self._conn, sim_id)
        self._owner._store_artifact_summary(
            self._conn,
            sim_id,
            ArtifactSummary(
                traj_present=False,
                error_traj_present=summary.error_traj_present,
                log_present=summary.log_present,
                footprints=summary.footprints,
            ),
        )

    def mark_footprint_complete(self, sim_id: str, name: str) -> None:
        """Mark one named footprint complete inside the open claim transaction."""
        key = name or "default"
        self._conn.execute(
            "UPDATE simulations"
            " SET footprint_status = footprint_status || jsonb_build_object(%s, 'complete')"
            " WHERE sim_id = %s",
            (key, sim_id),
        )
        summary = self._owner._artifact_summary_from_conn(self._conn, sim_id)
        footprints = dict(summary.footprints)
        footprints[name] = "complete"
        self._owner._store_artifact_summary(
            self._conn,
            sim_id,
            ArtifactSummary(
                traj_present=summary.traj_present,
                error_traj_present=summary.error_traj_present,
                log_present=summary.log_present,
                footprints=footprints,
            ),
        )

    def mark_footprint_empty(self, sim_id: str, name: str) -> None:
        """Mark one named footprint complete-empty inside the transaction."""
        key = name or "default"
        self._conn.execute(
            "UPDATE simulations"
            " SET footprint_status = footprint_status || jsonb_build_object(%s, 'complete-empty')"
            " WHERE sim_id = %s",
            (key, sim_id),
        )
        summary = self._owner._artifact_summary_from_conn(self._conn, sim_id)
        footprints = dict(summary.footprints)
        footprints[name] = "complete-empty"
        self._owner._store_artifact_summary(
            self._conn,
            sim_id,
            ArtifactSummary(
                traj_present=summary.traj_present,
                error_traj_present=summary.error_traj_present,
                log_present=summary.log_present,
                footprints=footprints,
            ),
        )

    def mark_footprint_failed(self, sim_id: str, name: str, error: str = "") -> None:
        """Mark one named footprint failed inside the open claim transaction."""
        key = name or "default"
        self._conn.execute(
            "UPDATE simulations"
            " SET footprint_status = footprint_status || jsonb_build_object(%s, 'failed')"
            " WHERE sim_id = %s",
            (key, sim_id),
        )
        summary = self._owner._artifact_summary_from_conn(self._conn, sim_id)
        footprints = dict(summary.footprints)
        footprints[name] = "failed"
        self._owner._store_artifact_summary(
            self._conn,
            sim_id,
            ArtifactSummary(
                traj_present=summary.traj_present,
                error_traj_present=summary.error_traj_present,
                log_present=summary.log_present,
                footprints=footprints,
            ),
        )

    def artifact_summary(self, sim_id: SimID | str) -> ArtifactSummary:
        """Return the artifact summary for one simulation within the transaction."""
        return self._owner._artifact_summary_from_conn(self._conn, str(sim_id))

    def record_artifacts(self, sim_id: str, summary: ArtifactSummary) -> None:
        """Persist one artifact summary within the open transaction."""
        self._owner._store_artifact_summary(self._conn, sim_id, summary)

    def record_attempt(self, attempt: SimulationAttempt) -> None:
        """Insert one attempt row within the open claim transaction."""
        self._conn.execute(
            "INSERT INTO attempts "
            "(attempt_id, sim_id, claim_token, started_at, finished_at, outcome, terminal, error) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                attempt.attempt_id,
                attempt.sim_id,
                attempt.claim_token,
                attempt.started_at,
                attempt.finished_at,
                attempt.outcome,
                attempt.terminal,
                attempt.error,
            ),
        )

    def heartbeat_claim(
        self,
        sim_id: str,
        claim_token: str,
        lease_ttl: float = 1800.0,
    ) -> bool:
        """Refresh the active lease when the claim still belongs to this unit."""
        if sim_id != self._claim.sim_id or claim_token != self._claim.claim_token:
            return False
        return self._owner.heartbeat_claim(sim_id, claim_token, lease_ttl=lease_ttl)

    def claim_is_current(self, sim_id: str, claim_token: str) -> bool:
        """Return whether this unit still owns the provided claim token."""
        return sim_id == self._claim.sim_id and claim_token == self._claim.claim_token


class PostgreSQLClaimUnitOfWork:
    """One claimed PostgreSQL simulation with a bound transactional repository."""

    def __init__(
        self,
        owner: PostgreSQLRepository,
        conn: Any,
        claim: SimulationClaim,
    ) -> None:
        self.claim = claim
        self.repository = _TransactionalClaimRepository(owner, conn, claim)
        self._released = False

    def release(self) -> None:
        """Mark this unit of work for rollback instead of commit."""
        self._released = True

    @property
    def released(self) -> bool:
        """Return whether the caller requested rollback on exit."""
        return self._released


class PostgreSQLRepository:
    """Simulation repository backed by PostgreSQL."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS simulations (
            sim_id             TEXT        NOT NULL PRIMARY KEY,
            receptor           JSONB       NOT NULL,
            trajectory_status  TEXT,
            error              TEXT,
            footprint_status   JSONB       NOT NULL DEFAULT '{}'::jsonb,
            footprint_targets  JSONB       NOT NULL DEFAULT '[]'::jsonb
        );
        CREATE TABLE IF NOT EXISTS batches (
            batch_id    TEXT    NOT NULL PRIMARY KEY,
            total       INTEGER NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS batch_simulations (
            batch_id    TEXT    NOT NULL,
            sim_id      TEXT    NOT NULL,
            PRIMARY KEY (batch_id, sim_id)
        );
        CREATE TABLE IF NOT EXISTS artifact_index (
            sim_id               TEXT        NOT NULL PRIMARY KEY,
            traj_present         BOOLEAN     NOT NULL DEFAULT FALSE,
            error_traj_present   BOOLEAN     NOT NULL DEFAULT FALSE,
            log_present          BOOLEAN     NOT NULL DEFAULT FALSE,
            footprints           JSONB       NOT NULL DEFAULT '{}'::jsonb,
            updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS claims (
            sim_id               TEXT        NOT NULL PRIMARY KEY,
            claim_token          TEXT        NOT NULL UNIQUE,
            claim_mode           TEXT        NOT NULL DEFAULT 'lease',
            worker_id            TEXT        NOT NULL,
            claimed_at           TIMESTAMPTZ NOT NULL,
            heartbeat_at         TIMESTAMPTZ NOT NULL,
            expires_at           TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id           TEXT        NOT NULL PRIMARY KEY,
            sim_id               TEXT        NOT NULL,
            claim_token          TEXT,
            started_at           TIMESTAMPTZ NOT NULL,
            finished_at          TIMESTAMPTZ,
            outcome              TEXT        NOT NULL,
            terminal             BOOLEAN     NOT NULL DEFAULT FALSE,
            error                TEXT
        );
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._ensure_schema()

    def _connect(self) -> Any:
        """Open one psycopg connection configured for dict-style rows."""
        try:
            import psycopg
            import psycopg.rows
        except ImportError as exc:
            raise ImportError(
                "PostgreSQLRepository requires psycopg. "
                "Install it with: pip install 'pystilt[cloud]'"
            ) from exc
        return psycopg.connect(  # type: ignore[call-arg]
            self._db_url,
            row_factory=cast(Any, psycopg.rows.dict_row),
        )

    def _ensure_schema(self) -> None:
        """Create the base PostgreSQL schema and additive columns."""
        with self._connect() as conn:
            conn.execute(self._SCHEMA)
            conn.execute(
                "ALTER TABLE simulations ADD COLUMN IF NOT EXISTS "
                "footprint_targets JSONB NOT NULL DEFAULT '[]'::jsonb"
            )
            conn.execute(
                "ALTER TABLE claims ADD COLUMN IF NOT EXISTS "
                "claim_mode TEXT NOT NULL DEFAULT 'lease'"
            )

    def _upsert_claim_with_mode(
        self,
        claim: SimulationClaim,
        *,
        mode: str,
    ) -> None:
        """Insert or replace one claim row using the provided claim mode."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO claims "
                "(sim_id, claim_token, claim_mode, worker_id, claimed_at, heartbeat_at, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (sim_id) DO UPDATE SET "
                "claim_token = EXCLUDED.claim_token, "
                "claim_mode = EXCLUDED.claim_mode, "
                "worker_id = EXCLUDED.worker_id, "
                "claimed_at = EXCLUDED.claimed_at, "
                "heartbeat_at = EXCLUDED.heartbeat_at, "
                "expires_at = EXCLUDED.expires_at",
                (
                    claim.sim_id,
                    claim.claim_token,
                    mode,
                    claim.worker_id,
                    claim.claimed_at,
                    claim.heartbeat_at,
                    claim.expires_at,
                ),
            )

    def _artifact_summary_from_conn(
        self,
        conn: Any,
        sim_id: str,
    ) -> ArtifactSummary:
        """Load the artifact summary for one simulation from an open connection."""
        row = conn.execute(
            "SELECT traj_present, error_traj_present, log_present, footprints "
            "FROM artifact_index WHERE sim_id = %s",
            (sim_id,),
        ).fetchone()
        if row is None:
            return ArtifactSummary()
        return ArtifactSummary(
            traj_present=bool(row["traj_present"]),
            error_traj_present=bool(row["error_traj_present"]),
            log_present=bool(row["log_present"]),
            footprints=row["footprints"] or {},
        )

    def _store_artifact_summary(
        self,
        conn: Any,
        sim_id: str,
        summary: ArtifactSummary,
    ) -> None:
        """Upsert one artifact summary inside an existing transaction."""
        conn.execute(
            "INSERT INTO artifact_index "
            "(sim_id, traj_present, error_traj_present, log_present, footprints, updated_at) "
            "VALUES (%s, %s, %s, %s, %s::jsonb, NOW()) "
            "ON CONFLICT (sim_id) DO UPDATE SET "
            "traj_present = EXCLUDED.traj_present, "
            "error_traj_present = EXCLUDED.error_traj_present, "
            "log_present = EXCLUDED.log_present, "
            "footprints = EXCLUDED.footprints, "
            "updated_at = EXCLUDED.updated_at",
            (
                sim_id,
                summary.traj_present,
                summary.error_traj_present,
                summary.log_present,
                json.dumps(summary.footprints),
            ),
        )

    def _latest_attempts_from_conn(
        self,
        conn: Any,
    ) -> dict[str, SimulationAttempt]:
        """Return the latest recorded attempt for each simulation."""
        rows = conn.execute(
            "SELECT attempt_id, sim_id, claim_token, started_at, finished_at, "
            "outcome, terminal, error "
            "FROM attempts ORDER BY started_at, attempt_id"
        ).fetchall()
        latest: dict[str, SimulationAttempt] = {}
        for row in rows:
            latest[row["sim_id"]] = SimulationAttempt(
                attempt_id=row["attempt_id"],
                sim_id=row["sim_id"],
                claim_token=row["claim_token"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                outcome=row["outcome"],
                terminal=bool(row["terminal"]),
                error=row["error"],
            )
        return latest

    def _state_maps(
        self,
        conn: Any,
    ) -> tuple[
        dict[str, list[str]],
        dict[str, ArtifactSummary],
        set[str],
        dict[str, SimulationAttempt],
    ]:
        """Collect requested outputs, artifact state, claims, and attempts."""
        sim_rows = conn.execute(
            "SELECT sim_id, footprint_targets FROM simulations"
        ).fetchall()
        requested = {row["sim_id"]: row["footprint_targets"] or [] for row in sim_rows}
        summary_rows = conn.execute(
            "SELECT sim_id, traj_present, error_traj_present, log_present, footprints "
            "FROM artifact_index"
        ).fetchall()
        summaries = {
            row["sim_id"]: ArtifactSummary(
                traj_present=bool(row["traj_present"]),
                error_traj_present=bool(row["error_traj_present"]),
                log_present=bool(row["log_present"]),
                footprints=row["footprints"] or {},
            )
            for row in summary_rows
        }
        active_claims = {
            row["sim_id"]
            for row in conn.execute(
                "SELECT sim_id FROM claims WHERE expires_at >= NOW()"
            ).fetchall()
        }
        latest_attempts = self._latest_attempts_from_conn(conn)
        return requested, summaries, active_claims, latest_attempts

    def register_many(
        self,
        pairs: list[tuple[str, Receptor]],
        batch_id: str | None = None,
        footprint_names: list[str] | None = None,
    ) -> None:
        """Register many simulations, optionally grouped under one batch."""
        targets = json.dumps(_normalize_footprint_names(footprint_names))
        with self._connect() as conn:
            for sim_id, receptor in pairs:
                conn.execute(
                    "INSERT INTO simulations (sim_id, receptor, footprint_targets)"
                    " VALUES (%s, %s, %s::jsonb)"
                    " ON CONFLICT (sim_id) DO NOTHING",
                    (sim_id, json.dumps(receptor.to_dict()), targets),
                )
                conn.execute(
                    "UPDATE simulations SET footprint_targets = %s::jsonb"
                    " WHERE sim_id = %s",
                    (targets, sim_id),
                )
                conn.execute(
                    "INSERT INTO artifact_index (sim_id)"
                    " VALUES (%s)"
                    " ON CONFLICT (sim_id) DO NOTHING",
                    (sim_id,),
                )
            if batch_id is not None:
                conn.execute(
                    "INSERT INTO batches (batch_id, total)"
                    " VALUES (%s, 0)"
                    " ON CONFLICT (batch_id)"
                    " DO UPDATE SET total = EXCLUDED.total",
                    (batch_id,),
                )
                for sim_id, _ in pairs:
                    conn.execute(
                        "INSERT INTO batch_simulations (batch_id, sim_id)"
                        " VALUES (%s, %s)"
                        " ON CONFLICT DO NOTHING",
                        (batch_id, sim_id),
                    )
                total_row = conn.execute(
                    "SELECT COUNT(*) AS n FROM batch_simulations WHERE batch_id = %s",
                    (batch_id,),
                ).fetchone()
                conn.execute(
                    "UPDATE batches SET total = %s WHERE batch_id = %s",
                    (total_row["n"] if total_row else 0, batch_id),
                )

    def all_sim_ids(self) -> list[str]:
        """List every simulation id stored in PostgreSQL."""
        with self._connect() as conn:
            rows = conn.execute("SELECT sim_id FROM simulations").fetchall()
        return [r["sim_id"] for r in rows]

    def completed_trajectories(self) -> list[str]:
        """List simulations with completed trajectory artifacts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sim_id FROM artifact_index WHERE traj_present = TRUE"
            ).fetchall()
        return [r["sim_id"] for r in rows]

    def completed_simulations(self) -> list[str]:
        """List simulations whose trajectories and requested footprints are done."""
        with self._connect() as conn:
            requested, summaries, _, _ = self._state_maps(conn)
        return [
            sim_id
            for sim_id, targets in requested.items()
            if simulation_complete(
                summaries.get(sim_id, ArtifactSummary()),
                targets,
            )
        ]

    def has(self, sim_id: SimID | str) -> bool:
        """Return whether one simulation id exists in the repository."""
        sid = str(sim_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM simulations WHERE sim_id = %s", (sid,)
            ).fetchone()
        return row is not None

    def count(self) -> int:
        """Return the number of registered simulations."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM simulations").fetchone()
        return row["n"] if row else 0

    def pending_trajectories(self) -> list[str]:
        """List simulations that are still pending trajectory work."""
        with self._connect() as conn:
            requested, summaries, active_claims, latest_attempts = self._state_maps(
                conn
            )
        return [
            sim_id
            for sim_id, targets in requested.items()
            if simulation_pending(
                summaries.get(sim_id, ArtifactSummary()),
                targets,
                active_claim=sim_id in active_claims,
                latest_attempt=latest_attempts.get(sim_id),
            )
        ]

    def failed_sim_ids(self) -> list[str]:
        """List simulations whose legacy trajectory status is failed."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sim_id FROM simulations WHERE trajectory_status = 'failed'"
            ).fetchall()
        return [r["sim_id"] for r in rows]

    def footprint_status(self, sim_id: SimID | str, name: str = "") -> str | None:
        """Return the stored status for one named footprint."""
        return self.artifact_summary(sim_id).footprints.get(name)

    def footprint_completed(self, sim_id: SimID | str, name: str = "") -> bool:
        """Return whether one named footprint has a complete terminal state."""
        return self.footprint_status(sim_id, name) in {"complete", "complete-empty"}

    def get_receptor(self, sim_id: SimID | str) -> Receptor:
        """Reconstruct one receptor definition from PostgreSQL JSON."""
        sid = str(sim_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT receptor FROM simulations WHERE sim_id = %s", (sid,)
            ).fetchone()
        if row is None:
            raise KeyError(sim_id)
        return Receptor.from_dict(json.loads(row["receptor"]))

    def rebuild(self) -> None:
        """Reset transient running state after an interrupted service process."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE simulations SET trajectory_status = NULL"
                " WHERE trajectory_status = 'running'"
            )
            conn.execute("DELETE FROM claims")

    def mark_trajectory_complete(self, sim_id: str) -> None:
        """Mark one trajectory run complete and persist its artifact state."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE simulations SET trajectory_status = 'complete', error = NULL"
                " WHERE sim_id = %s",
                (sim_id,),
            )
            conn.execute("DELETE FROM claims WHERE sim_id = %s", (sim_id,))
            summary = self._artifact_summary_from_conn(conn, sim_id)
            self._store_artifact_summary(
                conn,
                sim_id,
                ArtifactSummary(
                    traj_present=True,
                    error_traj_present=summary.error_traj_present,
                    log_present=summary.log_present,
                    footprints=summary.footprints,
                ),
            )

    def mark_trajectory_failed(self, sim_id: str, error: str = "") -> None:
        """Mark one trajectory run failed and store its error string."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE simulations SET trajectory_status = 'failed', error = %s"
                " WHERE sim_id = %s",
                (error or None, sim_id),
            )
            conn.execute("DELETE FROM claims WHERE sim_id = %s", (sim_id,))
            summary = self._artifact_summary_from_conn(conn, sim_id)
            self._store_artifact_summary(
                conn,
                sim_id,
                ArtifactSummary(
                    traj_present=False,
                    error_traj_present=summary.error_traj_present,
                    log_present=summary.log_present,
                    footprints=summary.footprints,
                ),
            )

    def mark_footprint_complete(self, sim_id: str, name: str) -> None:
        """Mark one named footprint complete for a simulation."""
        key = name or "default"
        with self._connect() as conn:
            conn.execute(
                "UPDATE simulations"
                " SET footprint_status = footprint_status || jsonb_build_object(%s, 'complete')"
                " WHERE sim_id = %s",
                (key, sim_id),
            )
            summary = self._artifact_summary_from_conn(conn, sim_id)
            footprints = dict(summary.footprints)
            footprints[name] = "complete"
            self._store_artifact_summary(
                conn,
                sim_id,
                ArtifactSummary(
                    traj_present=summary.traj_present,
                    error_traj_present=summary.error_traj_present,
                    log_present=summary.log_present,
                    footprints=footprints,
                ),
            )

    def mark_footprint_empty(self, sim_id: str, name: str) -> None:
        """Mark one named footprint complete-empty for a simulation."""
        key = name or "default"
        with self._connect() as conn:
            conn.execute(
                "UPDATE simulations"
                " SET footprint_status = footprint_status || jsonb_build_object(%s, 'complete-empty')"
                " WHERE sim_id = %s",
                (key, sim_id),
            )
            summary = self._artifact_summary_from_conn(conn, sim_id)
            footprints = dict(summary.footprints)
            footprints[name] = "complete-empty"
            self._store_artifact_summary(
                conn,
                sim_id,
                ArtifactSummary(
                    traj_present=summary.traj_present,
                    error_traj_present=summary.error_traj_present,
                    log_present=summary.log_present,
                    footprints=footprints,
                ),
            )

    def mark_footprint_failed(self, sim_id: str, name: str, error: str = "") -> None:
        """Mark one named footprint failed for a simulation."""
        key = name or "default"
        with self._connect() as conn:
            conn.execute(
                "UPDATE simulations"
                " SET footprint_status = footprint_status || jsonb_build_object(%s, 'failed')"
                " WHERE sim_id = %s",
                (key, sim_id),
            )
            summary = self._artifact_summary_from_conn(conn, sim_id)
            footprints = dict(summary.footprints)
            footprints[name] = "failed"
            self._store_artifact_summary(
                conn,
                sim_id,
                ArtifactSummary(
                    traj_present=summary.traj_present,
                    error_traj_present=summary.error_traj_present,
                    log_present=summary.log_present,
                    footprints=footprints,
                ),
            )

    def claim_pending_claims(
        self,
        n: int = 1,
        worker_id: str = "legacy",
        lease_ttl: float = 1800.0,
    ) -> list[SimulationClaim]:
        """Lease up to ``n`` pending simulations and return their claims."""
        with self._connect() as conn:
            rows = conn.execute(
                "UPDATE simulations SET trajectory_status = 'running'"
                " WHERE sim_id IN ("
                "   SELECT s.sim_id "
                "   FROM simulations AS s "
                "   LEFT JOIN artifact_index AS ai ON ai.sim_id = s.sim_id "
                "   LEFT JOIN claims AS c "
                "     ON c.sim_id = s.sim_id AND c.claim_mode = 'lease' AND c.expires_at >= NOW() "
                "   LEFT JOIN LATERAL ("
                "     SELECT terminal "
                "     FROM attempts AS a "
                "     WHERE a.sim_id = s.sim_id "
                "     ORDER BY a.started_at DESC, a.attempt_id DESC "
                "     LIMIT 1"
                "   ) AS latest ON TRUE "
                "   WHERE c.sim_id IS NULL "
                "     AND COALESCE(latest.terminal, FALSE) = FALSE "
                "     AND NOT ("
                "       COALESCE(ai.traj_present, FALSE) = TRUE "
                "       AND NOT EXISTS ("
                "         SELECT 1 "
                "         FROM jsonb_array_elements_text(s.footprint_targets) AS target(name) "
                "         WHERE COALESCE(ai.footprints ->> target.name, '') "
                "           NOT IN ('complete', 'complete-empty')"
                "       )"
                "     ) "
                "   ORDER BY s.sim_id "
                "   LIMIT %s "
                "   FOR UPDATE OF s SKIP LOCKED"
                " ) RETURNING sim_id",
                (n,),
            ).fetchall()
            claims: list[SimulationClaim] = []
            if rows:
                now = dt.datetime.now(dt.timezone.utc)
                expires_at = now + dt.timedelta(seconds=lease_ttl)
                claims = [
                    SimulationClaim(
                        sim_id=row["sim_id"],
                        claim_token=uuid.uuid4().hex,
                        worker_id=worker_id,
                        claimed_at=now,
                        heartbeat_at=now,
                        expires_at=expires_at,
                    )
                    for row in rows
                ]
                conn.executemany(
                    "INSERT INTO claims "
                    "(sim_id, claim_token, claim_mode, worker_id, claimed_at, heartbeat_at, expires_at) "
                    "VALUES (%s, %s, 'lease', %s, %s, %s, %s) "
                    "ON CONFLICT (sim_id) DO UPDATE SET "
                    "claim_token = EXCLUDED.claim_token, "
                    "claim_mode = EXCLUDED.claim_mode, "
                    "worker_id = EXCLUDED.worker_id, "
                    "claimed_at = EXCLUDED.claimed_at, "
                    "heartbeat_at = EXCLUDED.heartbeat_at, "
                    "expires_at = EXCLUDED.expires_at",
                    [
                        (
                            claim.sim_id,
                            claim.claim_token,
                            claim.worker_id,
                            claim.claimed_at,
                            claim.heartbeat_at,
                            claim.expires_at,
                        )
                        for claim in claims
                    ],
                )
        return claims

    @contextmanager
    def begin_claim_uow(
        self,
        *,
        worker_id: str = "legacy",
        lease_ttl: float = 1800.0,
    ) -> Iterator[PostgreSQLClaimUnitOfWork | None]:
        """Yield one transactional claim unit of work or ``None`` if none are pending."""
        conn = self._connect()
        claim: SimulationClaim | None = None
        uow: PostgreSQLClaimUnitOfWork | None = None
        try:
            row = conn.execute(
                "UPDATE simulations SET trajectory_status = 'running' "
                "WHERE sim_id IN ("
                "   SELECT s.sim_id "
                "   FROM simulations AS s "
                "   LEFT JOIN artifact_index AS ai ON ai.sim_id = s.sim_id "
                "   LEFT JOIN claims AS c "
                "     ON c.sim_id = s.sim_id AND c.claim_mode = 'lease' AND c.expires_at >= NOW() "
                "   LEFT JOIN LATERAL ("
                "     SELECT terminal "
                "     FROM attempts AS a "
                "     WHERE a.sim_id = s.sim_id "
                "     ORDER BY a.started_at DESC, a.attempt_id DESC "
                "     LIMIT 1"
                "   ) AS latest ON TRUE "
                "   WHERE c.sim_id IS NULL "
                "     AND COALESCE(latest.terminal, FALSE) = FALSE "
                "     AND NOT ("
                "       COALESCE(ai.traj_present, FALSE) = TRUE "
                "       AND NOT EXISTS ("
                "         SELECT 1 "
                "         FROM jsonb_array_elements_text(s.footprint_targets) AS target(name) "
                "         WHERE COALESCE(ai.footprints ->> target.name, '') "
                "           NOT IN ('complete', 'complete-empty')"
                "       )"
                "     ) "
                "   ORDER BY s.sim_id "
                "   LIMIT 1 "
                "   FOR UPDATE OF s SKIP LOCKED"
                ") RETURNING sim_id",
            ).fetchone()
            if row is None:
                conn.rollback()
                yield None
                return

            now = dt.datetime.now(dt.timezone.utc)
            claim = SimulationClaim(
                sim_id=row["sim_id"],
                claim_token=uuid.uuid4().hex,
                worker_id=worker_id,
                claimed_at=now,
                heartbeat_at=now,
                expires_at=now + dt.timedelta(seconds=lease_ttl),
            )
            self._upsert_claim_with_mode(claim, mode="uow")
            uow = PostgreSQLClaimUnitOfWork(self, conn, claim)
            yield uow
            if uow.released:
                conn.rollback()
            else:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if claim is not None:
                self.delete_claim(claim.sim_id, claim.claim_token)
            conn.close()

    def claim_pending(
        self,
        n: int = 1,
        worker_id: str = "legacy",
        lease_ttl: float = 1800.0,
    ) -> list[str]:
        """Lease pending simulations and return only their ids."""
        return [
            claim.sim_id
            for claim in self.claim_pending_claims(
                n=n,
                worker_id=worker_id,
                lease_ttl=lease_ttl,
            )
        ]

    def release_claim(self, sim_ids: list[str]) -> None:
        """Release claims and running status for the provided simulation ids."""
        with self._connect() as conn:
            conn.executemany(
                "UPDATE simulations SET trajectory_status = NULL WHERE sim_id = %s",
                [(sid,) for sid in sim_ids],
            )
            conn.executemany(
                "DELETE FROM claims WHERE sim_id = %s",
                [(sid,) for sid in sim_ids],
            )

    def release_claims(self, claims: list[SimulationClaim]) -> None:
        """Release only the exact claim tokens provided."""
        if not claims:
            return
        with self._connect() as conn:
            for claim in claims:
                deleted = conn.execute(
                    "DELETE FROM claims WHERE sim_id = %s AND claim_token = %s RETURNING sim_id",
                    (claim.sim_id, claim.claim_token),
                ).fetchone()
                if deleted is None:
                    continue
                conn.execute(
                    "UPDATE simulations SET trajectory_status = NULL"
                    " WHERE sim_id = %s AND trajectory_status = 'running'",
                    (claim.sim_id,),
                )

    def heartbeat_claim(
        self,
        sim_id: str,
        claim_token: str,
        lease_ttl: float = 1800.0,
    ) -> bool:
        """Refresh one active claim lease if the token still matches."""
        now = dt.datetime.now(dt.timezone.utc)
        expires_at = now + dt.timedelta(seconds=lease_ttl)
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE claims SET heartbeat_at = %s, expires_at = %s "
                "WHERE sim_id = %s AND claim_token = %s RETURNING sim_id",
                (now, expires_at, sim_id, claim_token),
            ).fetchone()
        return row is not None

    def claim_is_current(self, sim_id: str, claim_token: str) -> bool:
        """Return whether one claim token is still current for a simulation."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM claims WHERE sim_id = %s AND claim_token = %s",
                (sim_id, claim_token),
            ).fetchone()
        return row is not None

    def reclaim_expired_claims(self) -> list[str]:
        """Return expired claimed simulations back to the pending state."""
        now = dt.datetime.now(dt.timezone.utc)
        with self._connect() as conn:
            rows = conn.execute(
                "DELETE FROM claims WHERE expires_at < %s RETURNING sim_id",
                (now,),
            ).fetchall()
            sim_ids = [row["sim_id"] for row in rows]
            if sim_ids:
                conn.executemany(
                    "UPDATE simulations SET trajectory_status = NULL "
                    "WHERE sim_id = %s AND trajectory_status = 'running'",
                    [(sid,) for sid in sim_ids],
                )
        return sim_ids

    def reset_to_pending(self, sim_ids: list[str]) -> None:
        """Clear terminal state so simulations can be retried."""
        with self._connect() as conn:
            conn.executemany(
                "UPDATE simulations SET trajectory_status = NULL "
                "WHERE sim_id = %s AND trajectory_status != 'running'",
                [(sid,) for sid in sim_ids],
            )
            conn.executemany(
                "DELETE FROM claims WHERE sim_id = %s AND expires_at < NOW()",
                [(sid,) for sid in sim_ids],
            )
            conn.executemany(
                "DELETE FROM attempts WHERE sim_id = %s AND terminal = TRUE",
                [(sid,) for sid in sim_ids],
            )

    def clear_footprints(
        self, sim_ids: list[str], names: list[str] | None = None
    ) -> None:
        """Clear stored footprint state for selected simulations and names."""
        with self._connect() as conn:
            if names:
                for name in names:
                    conn.executemany(
                        "UPDATE simulations "
                        "SET footprint_status = footprint_status - %s "
                        "WHERE sim_id = %s",
                        [(name, sid) for sid in sim_ids],
                    )
                for sim_id in sim_ids:
                    summary = self._artifact_summary_from_conn(conn, sim_id)
                    footprints = dict(summary.footprints)
                    for name in names:
                        footprints.pop(name, None)
                    self._store_artifact_summary(
                        conn,
                        sim_id,
                        ArtifactSummary(
                            traj_present=summary.traj_present,
                            error_traj_present=summary.error_traj_present,
                            log_present=summary.log_present,
                            footprints=footprints,
                        ),
                    )
                return
            conn.executemany(
                "UPDATE simulations SET footprint_status = '{}'::jsonb WHERE sim_id = %s",
                [(sid,) for sid in sim_ids],
            )
            for sim_id in sim_ids:
                summary = self._artifact_summary_from_conn(conn, sim_id)
                self._store_artifact_summary(
                    conn,
                    sim_id,
                    ArtifactSummary(
                        traj_present=summary.traj_present,
                        error_traj_present=summary.error_traj_present,
                        log_present=summary.log_present,
                        footprints={},
                    ),
                )

    def batch_progress(self, batch_id: str) -> tuple[int, int]:
        """Return completed and total counts for one batch."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT total FROM batches WHERE batch_id = %s",
                (batch_id,),
            ).fetchone()
        if row is None:
            return (0, 0)
        completed_set = set(self.completed_simulations())
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sim_id FROM batch_simulations WHERE batch_id = %s",
                (batch_id,),
            ).fetchall()
        completed = sum(1 for batch_row in rows if batch_row["sim_id"] in completed_set)
        return (completed, row["total"])

    def all_batches(self) -> list[tuple[str, int, int]]:
        """List all batches with their completed and total counts."""
        completed_set = set(self.completed_simulations())
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT batch_id, total, created_at FROM batches ORDER BY created_at DESC"
            ).fetchall()
            batch_members = conn.execute(
                "SELECT batch_id, sim_id FROM batch_simulations"
            ).fetchall()
        members_by_batch: dict[str, set[str]] = {}
        for row in batch_members:
            members_by_batch.setdefault(row["batch_id"], set()).add(row["sim_id"])
        return [
            (
                row["batch_id"],
                sum(
                    1
                    for sim_id in members_by_batch.get(row["batch_id"], set())
                    if sim_id in completed_set
                ),
                row["total"],
            )
            for row in rows
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Return the simulations table as a pandas DataFrame."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sim_id, trajectory_status, error FROM simulations"
            ).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=pd.Index(["trajectory_status", "error"]),
                index=pd.Index([], name="sim_id"),
            )
        df = pd.DataFrame(rows)
        return df.set_index("sim_id")

    def traj_status(self, sim_id: SimID | str) -> str | None:
        """Return the derived trajectory status for one simulation."""
        sid = str(sim_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT trajectory_status FROM simulations WHERE sim_id = %s", (sid,)
            ).fetchone()
            if row is None:
                return None
            legacy_status = row["trajectory_status"]
            summary = self._artifact_summary_from_conn(conn, sid)
            active_claim = (
                conn.execute(
                    "SELECT 1 FROM claims WHERE sim_id = %s AND expires_at >= NOW()",
                    (sid,),
                ).fetchone()
                is not None
            )
            latest_row = conn.execute(
                "SELECT attempt_id, sim_id, claim_token, started_at, finished_at, "
                "outcome, terminal, error "
                "FROM attempts WHERE sim_id = %s "
                "ORDER BY started_at DESC, attempt_id DESC LIMIT 1",
                (sid,),
            ).fetchone()
        latest_attempt = (
            SimulationAttempt(
                attempt_id=latest_row["attempt_id"],
                sim_id=latest_row["sim_id"],
                claim_token=latest_row["claim_token"],
                started_at=latest_row["started_at"],
                finished_at=latest_row["finished_at"],
                outcome=latest_row["outcome"],
                terminal=bool(latest_row["terminal"]),
                error=latest_row["error"],
            )
            if latest_row is not None
            else None
        )
        if (
            latest_attempt is None
            and not summary.traj_present
            and legacy_status == "failed"
        ):
            return "failed"
        return trajectory_status_from_state(
            summary,
            active_claim=active_claim,
            latest_attempt=latest_attempt,
        )

    def sync(self) -> None:
        """No-op — state is updated via ``mark_*()``."""

    def artifact_summary(self, sim_id: SimID | str) -> ArtifactSummary:
        """Return the artifact summary recorded for one simulation."""
        sid = str(sim_id)
        with self._connect() as conn:
            return self._artifact_summary_from_conn(conn, sid)

    def record_artifacts(self, sim_id: str, summary: ArtifactSummary) -> None:
        """Persist one artifact summary for a simulation."""
        with self._connect() as conn:
            self._store_artifact_summary(conn, sim_id, summary)

    def list_claims(self) -> list[SimulationClaim]:
        """List all currently recorded claims."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sim_id, claim_token, worker_id, claimed_at, heartbeat_at, expires_at "
                "FROM claims ORDER BY claimed_at"
            ).fetchall()
        return [
            SimulationClaim(
                sim_id=row["sim_id"],
                claim_token=row["claim_token"],
                worker_id=row["worker_id"],
                claimed_at=row["claimed_at"],
                heartbeat_at=row["heartbeat_at"],
                expires_at=row["expires_at"],
            )
            for row in rows
        ]

    def upsert_claim(self, claim: SimulationClaim) -> None:
        """Insert or replace one lease claim row."""
        self._upsert_claim_with_mode(claim, mode="lease")

    def delete_claim(self, sim_id: str, claim_token: str | None = None) -> None:
        """Delete one claim, optionally requiring a matching token."""
        with self._connect() as conn:
            if claim_token is None:
                conn.execute("DELETE FROM claims WHERE sim_id = %s", (sim_id,))
            else:
                conn.execute(
                    "DELETE FROM claims WHERE sim_id = %s AND claim_token = %s",
                    (sim_id, claim_token),
                )

    def list_attempts(
        self, sim_id: SimID | str | None = None
    ) -> list[SimulationAttempt]:
        """List recorded attempts, optionally restricted to one simulation."""
        with self._connect() as conn:
            if sim_id is None:
                rows = conn.execute(
                    "SELECT attempt_id, sim_id, claim_token, started_at, finished_at, "
                    "outcome, terminal, error FROM attempts ORDER BY started_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT attempt_id, sim_id, claim_token, started_at, finished_at, "
                    "outcome, terminal, error FROM attempts "
                    "WHERE sim_id = %s ORDER BY started_at",
                    (str(sim_id),),
                ).fetchall()
        return [
            SimulationAttempt(
                attempt_id=row["attempt_id"],
                sim_id=row["sim_id"],
                claim_token=row["claim_token"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                outcome=row["outcome"],
                terminal=bool(row["terminal"]),
                error=row["error"],
            )
            for row in rows
        ]

    def record_attempt(self, attempt: SimulationAttempt) -> None:
        """Insert or update one recorded simulation attempt."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO attempts "
                "(attempt_id, sim_id, claim_token, started_at, finished_at, outcome, terminal, error) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (attempt_id) DO UPDATE SET "
                "sim_id = EXCLUDED.sim_id, "
                "claim_token = EXCLUDED.claim_token, "
                "started_at = EXCLUDED.started_at, "
                "finished_at = EXCLUDED.finished_at, "
                "outcome = EXCLUDED.outcome, "
                "terminal = EXCLUDED.terminal, "
                "error = EXCLUDED.error",
                (
                    attempt.attempt_id,
                    attempt.sim_id,
                    attempt.claim_token,
                    attempt.started_at,
                    attempt.finished_at,
                    attempt.outcome,
                    attempt.terminal,
                    attempt.error,
                ),
            )
