"""SQLite repository backend."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from stilt.artifacts import simulation_index_path, simulation_state_db_path
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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS simulations (
    sim_id       TEXT PRIMARY KEY,
    time         TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'point',
    altitude_ref TEXT NOT NULL DEFAULT 'agl',
    traj_status  TEXT NOT NULL DEFAULT 'pending',
    error        TEXT,
    footprint_targets TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS receptor_points (
    sim_id     TEXT NOT NULL REFERENCES simulations(sim_id),
    point_idx  INTEGER NOT NULL,
    lati       REAL NOT NULL,
    long       REAL NOT NULL,
    zagl       REAL NOT NULL,
    PRIMARY KEY (sim_id, point_idx)
);

CREATE TABLE IF NOT EXISTS footprints (
    sim_id  TEXT NOT NULL REFERENCES simulations(sim_id),
    name    TEXT NOT NULL,
    status  TEXT NOT NULL DEFAULT 'complete',
    PRIMARY KEY (sim_id, name)
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    total      INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS batch_simulations (
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    sim_id     TEXT NOT NULL REFERENCES simulations(sim_id),
    PRIMARY KEY (batch_id, sim_id)
);

CREATE TABLE IF NOT EXISTS artifact_index (
    sim_id               TEXT PRIMARY KEY REFERENCES simulations(sim_id),
    traj_present         INTEGER NOT NULL DEFAULT 0,
    error_traj_present   INTEGER NOT NULL DEFAULT 0,
    log_present          INTEGER NOT NULL DEFAULT 0,
    footprints           TEXT NOT NULL DEFAULT '{}',
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS claims (
    sim_id               TEXT PRIMARY KEY REFERENCES simulations(sim_id),
    claim_token          TEXT NOT NULL UNIQUE,
    worker_id            TEXT NOT NULL,
    claimed_at           TEXT NOT NULL,
    heartbeat_at         TEXT NOT NULL,
    expires_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id           TEXT PRIMARY KEY,
    sim_id               TEXT NOT NULL REFERENCES simulations(sim_id),
    claim_token          TEXT,
    started_at           TEXT NOT NULL,
    finished_at          TEXT,
    outcome              TEXT NOT NULL,
    terminal             INTEGER NOT NULL DEFAULT 0,
    error                TEXT
);
"""


class _ManagedConnection(sqlite3.Connection):
    """SQLite connection whose context manager also closes the handle."""

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        suppress = super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return suppress


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_dt(value: str) -> dt.datetime:
    """Parse one ISO-8601 timestamp from repository state."""
    return dt.datetime.fromisoformat(value)


class SQLiteRepository:
    """SQLite-backed repository for a STILT project directory."""

    _project_dir: Path

    def __init__(
        self,
        project_dir: Path,
        *,
        db_path: str | Path | None = None,
        uri: bool = False,
    ):
        self._project_dir = project_dir
        self._db_path: str | Path = (
            db_path if db_path is not None else simulation_state_db_path(project_dir)
        )
        self._uri = uri
        self._keepalive: sqlite3.Connection | None = None
        if self._is_memory_db:
            self._keepalive = self._connect()
        self._init_db()

    @classmethod
    def in_memory(cls, project_dir: Path | None = None) -> SQLiteRepository:
        """Create a SQLite repository backed by a shared in-memory database."""
        root = (project_dir or Path(tempfile.mkdtemp(prefix="pystilt_repo_"))).resolve()
        name = uuid.uuid4().hex
        return cls(
            root,
            db_path=f"file:pystilt-{name}?mode=memory&cache=shared",
            uri=True,
        )

    @property
    def _is_memory_db(self) -> bool:
        """Return whether this repository uses a shared in-memory SQLite URI."""
        return (
            self._uri
            and isinstance(self._db_path, str)
            and "mode=memory" in self._db_path
        )

    def _init_db(self) -> None:
        """Create the SQLite schema and apply additive migrations."""
        if not self._is_memory_db:
            assert isinstance(self._db_path, Path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            if not self._is_memory_db:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
        self._migrate_db()

    def _migrate_db(self) -> None:
        """Apply additive schema migrations needed for the current alpha."""
        with self._connect() as conn:
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(simulations)").fetchall()
            }
            if "footprint_targets" not in columns:
                conn.execute(
                    "ALTER TABLE simulations ADD COLUMN footprint_targets "
                    "TEXT NOT NULL DEFAULT '[]'"
                )
            if "altitude_ref" not in columns:
                conn.execute(
                    "ALTER TABLE simulations ADD COLUMN altitude_ref "
                    "TEXT NOT NULL DEFAULT 'agl'"
                )

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        """Open one SQLite connection configured for row-style access."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=timeout,
            uri=self._uri,
            factory=_ManagedConnection,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def close(self) -> None:
        """Close the keepalive connection for shared in-memory databases."""
        if self._keepalive is not None:
            self._keepalive.close()
            self._keepalive = None

    def __del__(self) -> None:
        self.close()

    def _artifact_summary_from_conn(
        self,
        conn: sqlite3.Connection,
        sim_id: str,
    ) -> ArtifactSummary:
        """Load the artifact summary for one simulation from an open connection."""
        row = conn.execute(
            "SELECT traj_present, error_traj_present, log_present, footprints "
            "FROM artifact_index WHERE sim_id=?",
            (sim_id,),
        ).fetchone()
        if row is None:
            return ArtifactSummary()
        return ArtifactSummary(
            traj_present=bool(row["traj_present"]),
            error_traj_present=bool(row["error_traj_present"]),
            log_present=bool(row["log_present"]),
            footprints=json.loads(row["footprints"] or "{}"),
        )

    def _store_artifact_summary(
        self,
        conn: sqlite3.Connection,
        sim_id: str,
        summary: ArtifactSummary,
    ) -> None:
        """Upsert one artifact summary inside an existing transaction."""
        conn.execute(
            "INSERT INTO artifact_index "
            "(sim_id, traj_present, error_traj_present, log_present, footprints, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(sim_id) DO UPDATE SET "
            "traj_present=excluded.traj_present, "
            "error_traj_present=excluded.error_traj_present, "
            "log_present=excluded.log_present, "
            "footprints=excluded.footprints, "
            "updated_at=excluded.updated_at",
            (
                sim_id,
                int(summary.traj_present),
                int(summary.error_traj_present),
                int(summary.log_present),
                json.dumps(summary.footprints),
                _now_iso(),
            ),
        )

    def _latest_attempts_from_conn(
        self,
        conn: sqlite3.Connection,
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
                started_at=_parse_dt(row["started_at"]),
                finished_at=_parse_dt(row["finished_at"])
                if row["finished_at"]
                else None,
                outcome=row["outcome"],
                terminal=bool(row["terminal"]),
                error=row["error"],
            )
        return latest

    def _state_maps(
        self,
        conn: sqlite3.Connection,
    ) -> tuple[
        dict[str, list[str]],
        dict[str, ArtifactSummary],
        set[str],
        dict[str, SimulationAttempt],
    ]:
        """Collect requested outputs, artifact state, claims, and latest attempts."""
        sim_rows = conn.execute(
            "SELECT sim_id, footprint_targets FROM simulations"
        ).fetchall()
        requested = {
            row["sim_id"]: json.loads(row["footprint_targets"] or "[]")
            for row in sim_rows
        }
        summary_rows = conn.execute(
            "SELECT sim_id, traj_present, error_traj_present, log_present, footprints "
            "FROM artifact_index"
        ).fetchall()
        summaries = {
            row["sim_id"]: ArtifactSummary(
                traj_present=bool(row["traj_present"]),
                error_traj_present=bool(row["error_traj_present"]),
                log_present=bool(row["log_present"]),
                footprints=json.loads(row["footprints"] or "{}"),
            )
            for row in summary_rows
        }
        now = _now_iso()
        active_claims = {
            row["sim_id"]
            for row in conn.execute(
                "SELECT sim_id FROM claims WHERE expires_at >= ?",
                (now,),
            ).fetchall()
        }
        latest_attempts = self._latest_attempts_from_conn(conn)
        return requested, summaries, active_claims, latest_attempts

    def register(self, sim_id: str, receptor: Receptor) -> None:
        """Register one simulation and its receptor in the repository."""
        self.register_many([(sim_id, receptor)])

    def register_many(
        self,
        pairs: list[tuple[str, Receptor]],
        batch_id: str | None = None,
        footprint_names: list[str] | None = None,
    ) -> None:
        """Register many simulations, optionally grouped under one batch."""
        targets_json = json.dumps(_normalize_footprint_names(footprint_names))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for sim_id, receptor in pairs:
                conn.execute(
                    "INSERT OR IGNORE INTO simulations (sim_id, time, kind, altitude_ref) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        sim_id,
                        receptor.time.isoformat(),
                        receptor.kind,
                        receptor.altitude_ref,
                    ),
                )
                conn.execute(
                    "UPDATE simulations SET altitude_ref=? WHERE sim_id=?",
                    (receptor.altitude_ref, sim_id),
                )
                conn.execute(
                    "UPDATE simulations SET footprint_targets=? WHERE sim_id=?",
                    (targets_json, sim_id),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO artifact_index (sim_id, updated_at)"
                    " VALUES (?, ?)",
                    (sim_id, _now_iso()),
                )
                for i, (lat, lon, hgt) in enumerate(
                    zip(
                        receptor.latitudes,
                        receptor.longitudes,
                        receptor.altitudes,
                        strict=False,
                    )
                ):
                    conn.execute(
                        "INSERT OR IGNORE INTO receptor_points "
                        "(sim_id, point_idx, lati, long, zagl) VALUES (?, ?, ?, ?, ?)",
                        (sim_id, i, float(lat), float(lon), float(hgt)),
                    )
            if batch_id is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO batches (batch_id, total) VALUES (?, 0)",
                    (batch_id,),
                )
                for sim_id, _ in pairs:
                    conn.execute(
                        "INSERT OR IGNORE INTO batch_simulations (batch_id, sim_id) "
                        "VALUES (?, ?)",
                        (batch_id, sim_id),
                    )
                total = conn.execute(
                    "SELECT COUNT(*) FROM batch_simulations WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()[0]
                conn.execute(
                    "UPDATE batches SET total = ? WHERE batch_id = ?",
                    (total, batch_id),
                )

    def sync(self) -> None:
        """No-op — state is updated directly via ``mark_*()``."""

    def mark_trajectory_complete(self, sim_id: str) -> None:
        """Mark one trajectory run complete and persist its artifact state."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE simulations SET traj_status='complete', error=NULL WHERE sim_id=?",
                (sim_id,),
            )
            conn.execute("DELETE FROM claims WHERE sim_id=?", (sim_id,))
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
                "UPDATE simulations SET traj_status='failed', error=? WHERE sim_id=?",
                (error, sim_id),
            )
            conn.execute("DELETE FROM claims WHERE sim_id=?", (sim_id,))
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
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO footprints (sim_id, name, status) VALUES (?, ?, 'complete')",
                (sim_id, name),
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
        """Mark one named footprint complete with an empty result."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO footprints (sim_id, name, status) VALUES (?, ?, 'complete-empty')",
                (sim_id, name),
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
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO footprints (sim_id, name, status) VALUES (?, ?, 'failed')",
                (sim_id, name),
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

    def has(self, sim_id: SimID | str) -> bool:
        """Return whether the repository knows about one simulation id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM simulations WHERE sim_id=?", (str(sim_id),)
            ).fetchone()
        return row is not None

    def count(self) -> int:
        """Return the number of registered simulations."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]

    def all_sim_ids(self) -> list[str]:
        """List every simulation id stored in the repository."""
        with self._connect() as conn:
            return [
                r[0] for r in conn.execute("SELECT sim_id FROM simulations").fetchall()
            ]

    def completed_trajectories(self) -> list[str]:
        """List simulations with completed trajectory artifacts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sim_id FROM artifact_index WHERE traj_present = 1"
            ).fetchall()
        return [row["sim_id"] for row in rows]

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

    def pending_trajectories(self) -> list[str]:
        """List simulations that still need trajectory work claimed."""
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

    def traj_status(self, sim_id: SimID | str) -> str | None:
        """Return the derived trajectory status for one simulation."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT traj_status FROM simulations WHERE sim_id=?", (str(sim_id),)
            ).fetchone()
            if row is None:
                return None
            legacy_status = row["traj_status"]
            summary = self._artifact_summary_from_conn(conn, str(sim_id))
            active_claim = (
                conn.execute(
                    "SELECT 1 FROM claims WHERE sim_id=? AND expires_at >= ?",
                    (str(sim_id), _now_iso()),
                ).fetchone()
                is not None
            )
            latest_row = conn.execute(
                "SELECT attempt_id, sim_id, claim_token, started_at, finished_at, "
                "outcome, terminal, error "
                "FROM attempts WHERE sim_id=? "
                "ORDER BY started_at DESC, attempt_id DESC LIMIT 1",
                (str(sim_id),),
            ).fetchone()
        latest_attempt = (
            SimulationAttempt(
                attempt_id=latest_row["attempt_id"],
                sim_id=latest_row["sim_id"],
                claim_token=latest_row["claim_token"],
                started_at=_parse_dt(latest_row["started_at"]),
                finished_at=_parse_dt(latest_row["finished_at"])
                if latest_row["finished_at"]
                else None,
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

    def footprint_status(self, sim_id: SimID | str, name: str) -> str | None:
        """Return the stored status for one named footprint."""
        return self.artifact_summary(sim_id).footprints.get(name)

    def footprint_completed(self, sim_id: SimID | str, name: str) -> bool:
        """Return whether one named footprint has a terminal complete state."""
        return self.footprint_status(sim_id, name) in {"complete", "complete-empty"}

    def get_receptor(self, sim_id: SimID | str) -> Receptor:
        """Reconstruct one receptor definition from repository rows."""
        sim_id = str(sim_id)
        with self._connect() as conn:
            sim_row = conn.execute(
                "SELECT time, altitude_ref FROM simulations WHERE sim_id=?",
                (sim_id,),
            ).fetchone()
            points = conn.execute(
                "SELECT lati, long, zagl FROM receptor_points "
                "WHERE sim_id=? ORDER BY point_idx",
                (sim_id,),
            ).fetchall()
        if sim_row is None:
            raise KeyError(f"sim_id not found in repository: {sim_id!r}")
        return Receptor(
            time=sim_row["time"],
            latitude=[p["lati"] for p in points],
            longitude=[p["long"] for p in points],
            altitude=[p["zagl"] for p in points],
            altitude_ref=sim_row["altitude_ref"],
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the simulations table as a pandas DataFrame."""
        with self._connect() as conn:
            df = pd.read_sql("SELECT * FROM simulations", conn)
        return df.set_index("sim_id")

    def rebuild(self) -> None:
        """Rebuild repository state from on-disk simulation artifacts."""
        by_id = simulation_index_path(self._project_dir)
        if not by_id.exists():
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE simulations SET traj_status = 'pending'"
                " WHERE traj_status = 'running'"
            )
            conn.execute("DELETE FROM claims")
        with self._connect() as conn:
            existing = {
                row[0] for row in conn.execute("SELECT sim_id FROM simulations")
            }
        for sim_dir in sorted(by_id.iterdir()):
            if not sim_dir.is_dir():
                continue
            sim_id = sim_dir.name
            traj_file = sim_dir / f"{sim_id}_traj.parquet"

            if traj_file.exists():
                if sim_id not in existing:
                    try:
                        meta = pq.ParquetFile(traj_file).schema_arrow.metadata
                        receptor = Receptor.from_dict(
                            json.loads(meta[b"stilt:receptor"])
                        )
                        self.register_many([(sim_id, receptor)])
                    except Exception:
                        try:
                            sid = SimID(sim_id)
                            with self._connect() as conn:
                                conn.execute(
                                    "INSERT OR IGNORE INTO simulations (sim_id, time)"
                                    " VALUES (?, ?)",
                                    (sim_id, sid.time.isoformat()),
                                )
                        except ValueError:
                            continue

                with self._connect() as conn:
                    conn.execute(
                        "UPDATE simulations SET traj_status = 'complete'"
                        " WHERE sim_id = ? AND traj_status != 'complete'",
                        (sim_id,),
                    )
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

            for foot_file in sim_dir.glob(f"{sim_id}*_foot.nc"):
                stem = foot_file.stem
                if not stem.endswith("_foot"):
                    continue
                without_foot = stem[: -len("_foot")]
                if without_foot == sim_id:
                    name = ""
                elif without_foot.startswith(f"{sim_id}_"):
                    name = without_foot[len(sim_id) + 1 :]
                else:
                    continue
                with self._connect() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO footprints (sim_id, name, status)"
                        " VALUES (?, ?, 'complete')",
                        (sim_id, name),
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

    def claim_pending_claims(
        self,
        n: int = 1,
        worker_id: str = "legacy",
        lease_ttl: float = 1800.0,
    ) -> list[SimulationClaim]:
        """Lease up to ``n`` pending simulations and return their claims."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            requested, summaries, active_claims, latest_attempts = self._state_maps(
                conn
            )
            sim_ids = [
                sim_id
                for sim_id, targets in requested.items()
                if simulation_pending(
                    summaries.get(sim_id, ArtifactSummary()),
                    targets,
                    active_claim=sim_id in active_claims,
                    latest_attempt=latest_attempts.get(sim_id),
                )
            ][:n]
            claims: list[SimulationClaim] = []
            if sim_ids:
                now = dt.datetime.now(dt.timezone.utc)
                expires_at = now + dt.timedelta(seconds=lease_ttl)
                claims = [
                    SimulationClaim(
                        sim_id=sim_id,
                        claim_token=uuid.uuid4().hex,
                        worker_id=worker_id,
                        claimed_at=now,
                        heartbeat_at=now,
                        expires_at=expires_at,
                    )
                    for sim_id in sim_ids
                ]
                placeholders = ",".join("?" for _ in sim_ids)
                conn.execute(
                    f"UPDATE simulations SET traj_status = 'running'"
                    f" WHERE sim_id IN ({placeholders})",
                    sim_ids,
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO claims "
                    "(sim_id, claim_token, worker_id, claimed_at, heartbeat_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            claim.sim_id,
                            claim.claim_token,
                            claim.worker_id,
                            claim.claimed_at.isoformat(),
                            claim.heartbeat_at.isoformat(),
                            claim.expires_at.isoformat(),
                        )
                        for claim in claims
                    ],
                )
            conn.execute("COMMIT")
        return claims

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
        """Release trajectory claims for the given simulation ids."""
        if not sim_ids:
            return
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in sim_ids)
            conn.execute(
                f"UPDATE simulations SET traj_status = 'pending'"
                f" WHERE sim_id IN ({placeholders}) AND traj_status = 'running'",
                sim_ids,
            )
            conn.execute(
                f"DELETE FROM claims WHERE sim_id IN ({placeholders})",
                sim_ids,
            )

    def release_claims(self, claims: list[SimulationClaim]) -> None:
        """Release only the exact claim tokens provided."""
        if not claims:
            return
        with self._connect() as conn:
            for claim in claims:
                current = conn.execute(
                    "SELECT 1 FROM claims WHERE sim_id=? AND claim_token=?",
                    (claim.sim_id, claim.claim_token),
                ).fetchone()
                if current is None:
                    continue
                conn.execute(
                    "DELETE FROM claims WHERE sim_id=? AND claim_token=?",
                    (claim.sim_id, claim.claim_token),
                )
                conn.execute(
                    "UPDATE simulations SET traj_status='pending'"
                    " WHERE sim_id=? AND traj_status='running'",
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
                "UPDATE claims SET heartbeat_at=?, expires_at=? "
                "WHERE sim_id=? AND claim_token=? RETURNING sim_id",
                (now.isoformat(), expires_at.isoformat(), sim_id, claim_token),
            ).fetchone()
        return row is not None

    def claim_is_current(self, sim_id: str, claim_token: str) -> bool:
        """Return whether one claim token is still current for a simulation."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM claims WHERE sim_id=? AND claim_token=?",
                (sim_id, claim_token),
            ).fetchone()
        return row is not None

    def reclaim_expired_claims(self) -> list[str]:
        """Return expired claimed simulations back to the pending state."""
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sim_id FROM claims WHERE expires_at < ?",
                (now,),
            ).fetchall()
            sim_ids = [row["sim_id"] for row in rows]
            if not sim_ids:
                return []
            placeholders = ",".join("?" for _ in sim_ids)
            conn.execute(
                f"DELETE FROM claims WHERE sim_id IN ({placeholders})",
                sim_ids,
            )
            conn.execute(
                f"UPDATE simulations SET traj_status='pending'"
                f" WHERE sim_id IN ({placeholders}) AND traj_status='running'",
                sim_ids,
            )
        return sim_ids

    def reset_to_pending(self, sim_ids: list[str]) -> None:
        """Clear terminal state so simulations can be retried from scratch."""
        if not sim_ids:
            return
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in sim_ids)
            conn.execute(
                f"UPDATE simulations SET traj_status = 'pending', error = NULL"
                f" WHERE sim_id IN ({placeholders}) AND traj_status != 'running'",
                sim_ids,
            )
            conn.execute(
                f"DELETE FROM claims WHERE sim_id IN ({placeholders}) "
                "AND expires_at < ?",
                [*sim_ids, _now_iso()],
            )
            conn.execute(
                f"DELETE FROM attempts WHERE sim_id IN ({placeholders}) AND terminal = 1",
                sim_ids,
            )

    def clear_footprints(
        self, sim_ids: list[str], names: list[str] | None = None
    ) -> None:
        """Clear stored footprint state for selected simulations and names."""
        if not sim_ids:
            return
        with self._connect() as conn:
            sim_placeholders = ",".join("?" for _ in sim_ids)
            if names:
                name_placeholders = ",".join("?" for _ in names)
                conn.execute(
                    "DELETE FROM footprints "
                    f"WHERE sim_id IN ({sim_placeholders}) "
                    f"AND name IN ({name_placeholders})",
                    [*sim_ids, *names],
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
            else:
                conn.execute(
                    f"DELETE FROM footprints WHERE sim_id IN ({sim_placeholders})",
                    sim_ids,
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
                "SELECT total FROM batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        if row is None:
            return (0, 0)
        completed_set = set(self.completed_simulations())
        with self._connect() as conn:
            batch_rows = conn.execute(
                "SELECT sim_id FROM batch_simulations WHERE batch_id = ?",
                (batch_id,),
            ).fetchall()
        completed = sum(
            1 for batch_row in batch_rows if batch_row["sim_id"] in completed_set
        )
        return (completed, row["total"])

    def all_batches(self) -> list[tuple[str, int, int]]:
        """List all batches with their completed and total counts."""
        completed_set = set(self.completed_simulations())
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT batch_id, total, created_at FROM batches ORDER BY created_at"
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

    def artifact_summary(self, sim_id: SimID | str) -> ArtifactSummary:
        """Return the artifact summary recorded for one simulation."""
        with self._connect() as conn:
            return self._artifact_summary_from_conn(conn, str(sim_id))

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
                claimed_at=_parse_dt(row["claimed_at"]),
                heartbeat_at=_parse_dt(row["heartbeat_at"]),
                expires_at=_parse_dt(row["expires_at"]),
            )
            for row in rows
        ]

    def upsert_claim(self, claim: SimulationClaim) -> None:
        """Insert or replace one claim row."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO claims "
                "(sim_id, claim_token, worker_id, claimed_at, heartbeat_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(sim_id) DO UPDATE SET "
                "claim_token=excluded.claim_token, "
                "worker_id=excluded.worker_id, "
                "claimed_at=excluded.claimed_at, "
                "heartbeat_at=excluded.heartbeat_at, "
                "expires_at=excluded.expires_at",
                (
                    claim.sim_id,
                    claim.claim_token,
                    claim.worker_id,
                    claim.claimed_at.isoformat(),
                    claim.heartbeat_at.isoformat(),
                    claim.expires_at.isoformat(),
                ),
            )

    def delete_claim(self, sim_id: str, claim_token: str | None = None) -> None:
        """Delete one claim, optionally requiring a matching token."""
        with self._connect() as conn:
            if claim_token is None:
                conn.execute("DELETE FROM claims WHERE sim_id=?", (sim_id,))
            else:
                conn.execute(
                    "DELETE FROM claims WHERE sim_id=? AND claim_token=?",
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
                    "WHERE sim_id=? ORDER BY started_at",
                    (str(sim_id),),
                ).fetchall()
        return [
            SimulationAttempt(
                attempt_id=row["attempt_id"],
                sim_id=row["sim_id"],
                claim_token=row["claim_token"],
                started_at=_parse_dt(row["started_at"]),
                finished_at=(
                    _parse_dt(row["finished_at"])
                    if row["finished_at"] is not None
                    else None
                ),
                outcome=row["outcome"],
                terminal=bool(row["terminal"]),
                error=row["error"],
            )
            for row in rows
        ]

    def record_attempt(self, attempt: SimulationAttempt) -> None:
        """Insert or replace one recorded simulation attempt."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO attempts "
                "(attempt_id, sim_id, claim_token, started_at, finished_at, outcome, terminal, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attempt.attempt_id,
                    attempt.sim_id,
                    attempt.claim_token,
                    attempt.started_at.isoformat(),
                    attempt.finished_at.isoformat()
                    if attempt.finished_at is not None
                    else None,
                    attempt.outcome,
                    int(attempt.terminal),
                    attempt.error,
                ),
            )
