"""Shared SQL output-index base used by SQLite and PostgreSQL backends."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from stilt.receptor import Receptor
from stilt.simulation import SimID

from .protocol import (
    IndexCounts,
    OutputSummary,
    SimulationIndex,
    _normalize_registration_pairs,
)
from .sql import (
    apply_rebuild_records,
    count_rows,
    count_where,
    existing_sim_ids,
    group_counts,
    index_counts,
    ordered_sim_ids,
    output_summary_mapping,
    receptor_mapping,
    registration_rows,
    reset_for_rebuild,
    total_rows,
    update_values,
    validate_required_columns,
)
from .updates import IndexUpdate, index_update_from_result, index_update_from_summary

if TYPE_CHECKING:
    from stilt.execution import SimulationResult


class _SqlIndex(SimulationIndex, abc.ABC):
    """
    Dialect-parameterised SQL simulation index.

    Concrete backends fill in a handful of dialect constants, a connection
    factory, a ``WHERE sim_id IN (...)`` dispatcher, and a column-introspection
    helper.  All CRUD logic lives here.
    """

    # Dialect constants populated by subclasses.
    _ph: ClassVar[str]  # parameter placeholder, "?" or "%s"
    _now_sql: ClassVar[str]  # current-timestamp expression
    _true_sql: ClassVar[str]  # boolean-true literal
    _false_sql: ClassVar[str]  # boolean-false literal
    _empty_footprints_sql: ClassVar[str]  # empty-dict JSON literal
    _jsonb_cast: ClassVar[str]  # cast suffix for JSON params ("" or "::jsonb")
    _bool_encode: ClassVar[Callable[[bool], Any]]  # int for SQLite, bool for Postgres
    _backend_name: ClassVar[str]  # used in schema error messages
    _completed_where: ClassVar[str]
    _pending_where: ClassVar[str]
    _prune_eligible_where: ClassVar[str]

    _max_rows: int | None

    # ------------------------------------------------------------------
    # Dialect-specific primitives — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _connect(self, *args: Any, **kwargs: Any) -> Any:
        """Return a context-managed DB connection with dict-style row access."""

    @abc.abstractmethod
    def _table_columns(self, conn: Any, table: str) -> set[str]:
        """Return column names present on *table* in the connected schema."""

    @abc.abstractmethod
    def _execute_match_ids(
        self,
        conn: Any,
        *,
        prefix: str,
        sim_ids: list[str],
        suffix: str = "",
        prefix_params: tuple = (),
        fetch: bool = False,
    ) -> list[Any]:
        """
        Run ``{prefix} WHERE sim_id <IN-OR-ANY> {suffix}`` across *sim_ids*.

        SQLite chunks the id list into multiple ``IN (?, ?, ...)`` statements;
        PostgreSQL issues one ``sim_id = ANY(%s)`` query.  When ``fetch`` is
        true, returns accumulated ``fetchall()`` rows in input order.
        """

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _validate_schema(self) -> None:
        with self._connect() as conn:
            validate_required_columns(
                self._table_columns(conn, "simulations"),
                backend_name=self._backend_name,
            )

    # ------------------------------------------------------------------
    # SimulationIndex surface
    # ------------------------------------------------------------------

    def record(self, result: SimulationResult) -> None:
        with self._connect() as conn:
            self._record_result_conn(conn, result)

    def register(
        self,
        pairs: str | tuple[str, Receptor] | list[tuple[str, Receptor]],
        receptor: Receptor | None = None,
        footprint_names: list[str] | None = None,
        scene_id: str | None = None,
    ) -> None:
        normalized = _normalize_registration_pairs(pairs, receptor)
        with self._connect() as conn:
            self._register_many_conn(
                conn,
                normalized,
                footprint_names=footprint_names,
                scene_id=scene_id,
            )
            self._prune_to_max_rows_conn(conn)

    def sim_ids(self) -> list[str]:
        with self._connect() as conn:
            return ordered_sim_ids(conn)

    def has(self, sim_id: SimID | str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT 1 FROM simulations WHERE sim_id = {self._ph}",
                (str(sim_id),),
            ).fetchone()
        return row is not None

    def count(self) -> int:
        with self._connect() as conn:
            return total_rows(conn)

    def counts(self, scene_id: str | None = None) -> IndexCounts:
        if scene_id is not None:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT scene_id, trajectory_status, traj_present, error_traj_present, "
                    "log_present, footprints, footprint_targets "
                    f"FROM simulations WHERE scene_id = {self._ph}",
                    (scene_id,),
                ).fetchall()
            return count_rows(rows)
        with self._connect() as conn:
            total = int(total_rows(conn))
            completed = count_where(conn, self._completed_where)
            pending = count_where(conn, self._pending_where)
            running = count_where(conn, "s.trajectory_status = 'running'")
        return index_counts(
            total=total,
            completed=completed,
            pending=pending,
            running=running,
        )

    def scene_counts(self) -> dict[str, IndexCounts]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT scene_id, trajectory_status, traj_present, error_traj_present, "
                "log_present, footprints, footprint_targets "
                "FROM simulations WHERE scene_id IS NOT NULL ORDER BY scene_id, sim_id"
            ).fetchall()
        return group_counts(rows)

    def receptors_for(self, sim_ids: list[str]) -> dict[str, Receptor]:
        if not sim_ids:
            return {}
        with self._connect() as conn:
            rows = self._execute_match_ids(
                conn,
                prefix="SELECT sim_id, receptor FROM simulations",
                sim_ids=sim_ids,
                fetch=True,
            )
        return receptor_mapping(rows)

    def reset_to_pending(
        self,
        sim_ids: list[str],
        *,
        clear_outputs: bool = False,
    ) -> None:
        if not sim_ids:
            return
        with self._connect() as conn:
            self._execute_match_ids(
                conn,
                prefix=(
                    "UPDATE simulations SET trajectory_status = 'pending', "
                    f"error = NULL, updated_at = {self._now_sql}"
                ),
                sim_ids=sim_ids,
                suffix="AND trajectory_status != 'running'",
            )
            if clear_outputs:
                self._execute_match_ids(
                    conn,
                    prefix=(
                        f"UPDATE simulations SET traj_present = {self._false_sql}, "
                        f"error_traj_present = {self._false_sql}, "
                        f"log_present = {self._false_sql}, "
                        f"footprints = {self._empty_footprints_sql}, "
                        f"updated_at = {self._now_sql}"
                    ),
                    sim_ids=sim_ids,
                )

    def pending_trajectories(self) -> list[str]:
        with self._connect() as conn:
            return ordered_sim_ids(conn, self._pending_where)

    def summaries(
        self,
        sim_ids: list[str] | None = None,
    ) -> dict[str, OutputSummary]:
        if sim_ids == []:
            return {}
        with self._connect() as conn:
            if sim_ids is None:
                rows = conn.execute(
                    "SELECT sim_id, traj_present, error_traj_present, log_present, footprints "
                    "FROM simulations"
                ).fetchall()
            else:
                rows = self._execute_match_ids(
                    conn,
                    prefix=(
                        "SELECT sim_id, traj_present, error_traj_present, "
                        "log_present, footprints FROM simulations"
                    ),
                    sim_ids=sim_ids,
                    fetch=True,
                )
        return output_summary_mapping(rows)

    # ------------------------------------------------------------------
    # Shared connection-scoped helpers (used by claim contexts as well)
    # ------------------------------------------------------------------

    def _store_update_conn(
        self,
        conn: Any,
        sim_id: str,
        update: IndexUpdate,
    ) -> None:
        conn.execute(
            "UPDATE simulations SET "
            f"trajectory_status = {self._ph}, error = {self._ph}, "
            f"traj_present = {self._ph}, error_traj_present = {self._ph}, "
            f"log_present = {self._ph}, footprints = {self._ph}{self._jsonb_cast}, "
            f"updated_at = {self._now_sql} "
            f"WHERE sim_id = {self._ph}",
            (*update_values(update, encode_bool=self._bool_encode), sim_id),
        )

    def _record_result_conn(self, conn: Any, result: SimulationResult) -> None:
        self._store_update_conn(
            conn,
            str(result.sim_id),
            index_update_from_result(result),
        )
        self._prune_to_max_rows_conn(conn)

    def _register_many_conn(
        self,
        conn: Any,
        pairs: list[tuple[str, Receptor]],
        footprint_names: list[str] | None = None,
        scene_id: str | None = None,
    ) -> None:
        if not pairs:
            return
        rows = registration_rows(
            pairs,
            footprint_names=footprint_names,
            scene_id=scene_id,
        )
        cast = self._jsonb_cast
        conn.executemany(
            "INSERT INTO simulations (sim_id, scene_id, receptor, footprint_targets) "
            f"VALUES ({self._ph}, {self._ph}, {self._ph}{cast}, {self._ph}{cast}) "
            "ON CONFLICT(sim_id) DO UPDATE SET "
            "scene_id = COALESCE(excluded.scene_id, simulations.scene_id), "
            "receptor = excluded.receptor, "
            "footprint_targets = excluded.footprint_targets, "
            f"updated_at = {self._now_sql}",
            rows,
        )

    def _prune_to_max_rows_conn(self, conn: Any) -> list[str]:
        if self._max_rows is None:
            return []
        total = total_rows(conn)
        overflow = total - self._max_rows
        if overflow <= 0:
            return []
        rows = conn.execute(
            f"SELECT s.sim_id FROM simulations AS s WHERE {self._prune_eligible_where} "
            f"ORDER BY s.created_at, s.sim_id LIMIT {self._ph}",
            (overflow,),
        ).fetchall()
        sim_ids = [row["sim_id"] for row in rows]
        if not sim_ids:
            return []
        self._execute_match_ids(
            conn,
            prefix="DELETE FROM simulations",
            sim_ids=sim_ids,
        )
        return sim_ids

    # ------------------------------------------------------------------
    # Rebuild (scan-from-disk)
    # ------------------------------------------------------------------

    def _rebuild_apply(self, conn: Any, records: list[Any]) -> None:
        reset_for_rebuild(
            conn,
            now_sql=self._now_sql,
            false_sql=self._false_sql,
            empty_footprints_sql=self._empty_footprints_sql,
        )
        existing = existing_sim_ids(conn)
        apply_rebuild_records(
            records,
            existing_ids=existing,
            register_missing=lambda sim_id, receptor: self._register_many_conn(
                conn, [(sim_id, receptor)]
            ),
            store_update=lambda sim_id, summary: self._store_update_conn(
                conn,
                sim_id,
                index_update_from_summary(summary),
            ),
        )
        self._prune_to_max_rows_conn(conn)


__all__ = ["_SqlIndex"]
