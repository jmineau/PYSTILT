"""Additional PostgreSQL index unit tests that do not require a live database."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from stilt.index.postgres import PostgresClaim, PostgresIndex


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        if isinstance(self._rows, list):
            return self._rows[0] if self._rows else None
        return self._rows

    def fetchall(self):
        if self._rows is None:
            return []
        if isinstance(self._rows, list):
            return self._rows
        return [self._rows]


class _ClaimConn:
    def __init__(self, row=None):
        self.row = row
        self.queries: list[tuple[str, tuple | None]] = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, query, params=None):
        self.queries.append((" ".join(query.split()), params))
        return _FakeResult(self.row)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _context_connect(conn):
    @contextmanager
    def _connect() -> Iterator[_ClaimConn]:
        yield conn

    return _connect


def test_postgres_index_table_columns_returns_column_names():
    state = object.__new__(PostgresIndex)
    conn = type(
        "_Conn",
        (),
        {
            "execute": lambda self, query, params: _FakeResult(
                [{"column_name": "sim_id"}, {"column_name": "updated_at"}]
            )
        },
    )()

    assert state._table_columns(conn, "simulations") == {"sim_id", "updated_at"}


def test_postgres_index_execute_match_ids_supports_fetch_and_suffix():
    captured: list[tuple[str, tuple]] = []

    class _Conn:
        def execute(self, query, params):
            captured.append((query, params))
            return _FakeResult([{"sim_id": "a"}])

    state = object.__new__(PostgresIndex)

    rows = state._execute_match_ids(
        _Conn(),
        prefix="SELECT sim_id FROM simulations",
        sim_ids=["a", "b"],
        suffix="ORDER BY sim_id",
        prefix_params=("x",),
        fetch=True,
    )

    assert rows == [{"sim_id": "a"}]
    assert captured == [
        (
            "SELECT sim_id FROM simulations WHERE sim_id = ANY(%s) ORDER BY sim_id",
            ("x", ["a", "b"]),
        )
    ]


def test_postgres_claim_records_result_through_index():
    calls = []
    claim = PostgresClaim(
        sim_id="sim-1",
        _index=type(
            "_Index",
            (),
            {
                "_record_result_conn": lambda self, conn, result: calls.append(
                    (conn, result)
                )
            },
        )(),
        _conn="conn",
    )

    claim.record("result")

    assert calls == [("conn", "result")]
    assert claim.released is False
    claim.release()
    assert claim.released is True


def test_postgres_claim_one_rolls_back_when_queue_is_empty(monkeypatch):
    conn = _ClaimConn(row=None)
    state = object.__new__(PostgresIndex)
    state._pending_where = "trajectory_status = 'pending'"
    monkeypatch.setattr(state, "_connect", _context_connect(conn))

    with state.claim_one() as claim:
        assert claim is None

    assert conn.rollbacks == 1
    assert conn.commits == 0


def test_postgres_claim_one_commits_when_claim_is_consumed(monkeypatch):
    conn = _ClaimConn(row={"sim_id": "sim-1"})
    state = object.__new__(PostgresIndex)
    state._pending_where = "trajectory_status = 'pending'"
    monkeypatch.setattr(state, "_connect", _context_connect(conn))

    with state.claim_one() as claim:
        assert claim is not None
        assert claim.sim_id == "sim-1"

    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_postgres_claim_one_rolls_back_released_claim(monkeypatch):
    conn = _ClaimConn(row={"sim_id": "sim-1"})
    state = object.__new__(PostgresIndex)
    state._pending_where = "trajectory_status = 'pending'"
    monkeypatch.setattr(state, "_connect", _context_connect(conn))

    with state.claim_one() as claim:
        assert claim is not None
        claim.release()

    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_postgres_claim_one_rolls_back_when_body_raises(monkeypatch):
    conn = _ClaimConn(row={"sim_id": "sim-1"})
    state = object.__new__(PostgresIndex)
    state._pending_where = "trajectory_status = 'pending'"
    monkeypatch.setattr(state, "_connect", _context_connect(conn))

    with pytest.raises(RuntimeError, match="boom"), state.claim_one():
        raise RuntimeError("boom")

    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_postgres_index_ensure_schema_executes_schema_then_validates(monkeypatch):
    executed: list[str] = []
    conn = type("_Conn", (), {"execute": lambda self, sql: executed.append(sql)})()
    state = object.__new__(PostgresIndex)
    validated: list[bool] = []
    monkeypatch.setattr(state, "_connect", _context_connect(conn))
    monkeypatch.setattr(state, "_validate_schema", lambda: validated.append(True))

    state._ensure_schema()

    assert executed == [state._SCHEMA]
    assert validated == [True]


def test_postgres_index_rebuild_without_output_root_resets_running_rows(monkeypatch):
    executed: list[str] = []
    conn = type(
        "_Conn",
        (),
        {"execute": lambda self, sql: executed.append(" ".join(sql.split()))},
    )()
    state = object.__new__(PostgresIndex)
    state._output_root = None
    monkeypatch.setattr(state, "_connect", _context_connect(conn))

    state.rebuild()

    assert executed == [
        "UPDATE simulations SET trajectory_status = 'pending', updated_at = NOW() WHERE trajectory_status = 'running'"
    ]


def test_postgres_index_rebuild_with_output_root_scans_and_applies(monkeypatch):
    conn = object()
    state = object.__new__(PostgresIndex)
    state._output_root = "s3://bucket/project"
    calls: list[tuple[object, list[str]]] = []
    monkeypatch.setattr(state, "_connect", _context_connect(conn))
    monkeypatch.setattr(
        "stilt.index.postgres.scan_output_simulations",
        lambda root: [f"record:{root}"],
    )
    monkeypatch.setattr(
        state,
        "_rebuild_apply",
        lambda current, records: calls.append((current, records)),
    )

    state.rebuild()

    assert calls == [(conn, ["record:s3://bucket/project"])]


def test_postgres_index_db_url_property_returns_original_value():
    state = object.__new__(PostgresIndex)
    state._db_url = "postgresql://example"

    assert state.db_url == "postgresql://example"
