"""Tests for shared SQL index predicate rendering."""

from stilt.index.sql import SqlPredicateDialect, build_index_predicates


def test_build_index_predicates_renders_shared_completion_semantics():
    predicates = build_index_predicates(
        SqlPredicateDialect(
            true_sql="TRUE",
            false_sql="FALSE",
            target_rows_sql="target_table AS target",
            footprint_status_sql="target.status",
        )
    )

    assert "COALESCE(s.traj_present, FALSE) = TRUE" in predicates.completed
    assert "FROM target_table AS target" in predicates.completed
    assert "target.status" in predicates.completed
    assert "'complete-empty'" in predicates.completed
    assert predicates.completed in predicates.pending
    assert predicates.completed in predicates.prune_eligible
