"""Unit tests for PostgreSQL index pruning helpers."""

from stilt.index.postgres import PostgresIndex


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


class _FakeConn:
    def __init__(self, *, total: int, candidates: list[str]):
        self.total = total
        self.candidates = list(candidates)
        self.deleted: list[str] = []

    def execute(self, query, params=None):
        normalized = " ".join(query.split())
        if normalized == "SELECT COUNT(*) AS n FROM simulations":
            return _FakeResult({"n": self.total})
        if normalized.startswith("SELECT s.sim_id FROM simulations AS s WHERE"):
            limit = params[0]
            return _FakeResult(
                [{"sim_id": sim_id} for sim_id in self.candidates[:limit]]
            )
        if normalized == "DELETE FROM simulations WHERE sim_id = ANY(%s)":
            self.deleted.extend(params[0])
            self.total -= len(params[0])
            return _FakeResult([])
        raise AssertionError(f"Unexpected query: {normalized}")


def _postgres_state(max_rows: int | None) -> PostgresIndex:
    state = object.__new__(PostgresIndex)
    state._max_rows = max_rows
    return state


def test_postgres_state_prunes_oldest_terminal_rows_to_soft_cap():
    state = _postgres_state(2)
    conn = _FakeConn(total=4, candidates=["sim-old", "sim-mid", "sim-new"])

    pruned = state._prune_to_max_rows_conn(conn)

    assert pruned == ["sim-old", "sim-mid"]
    assert conn.deleted == ["sim-old", "sim-mid"]


def test_postgres_state_soft_cap_keeps_active_rows_when_no_terminal_candidates():
    state = _postgres_state(2)
    conn = _FakeConn(total=5, candidates=[])

    pruned = state._prune_to_max_rows_conn(conn)

    assert pruned == []
    assert conn.deleted == []
