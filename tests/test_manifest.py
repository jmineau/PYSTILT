"""Tests for the simulation registry manifest (.stilt/manifest.parquet)."""

import datetime as dt

from stilt.manifest import Manifest
from stilt.receptors import PointReceptor
from stilt.simulation import SimID
from stilt.storage import LocalStore


def _rec(lon=-111.85):
    return PointReceptor(
        time=dt.datetime(2023, 1, 1, 12), longitude=lon, latitude=40.77, altitude=5.0
    )


def _sid(rec, met="hrrr"):
    return str(SimID.from_parts(met, rec))


def test_empty_manifest(tmp_path):
    m = Manifest(LocalStore(tmp_path))
    assert m.sim_ids() == []
    assert m.count() == 0
    assert m.has("anything") is False
    assert m.sim_ids_by_scene() == {}


def test_register_and_read(tmp_path):
    m = Manifest(LocalStore(tmp_path))
    rec = _rec()
    sid = _sid(rec)
    m.register([(sid, rec)], footprint_names=["slv"], scene_id="tower-1")

    assert m.sim_ids() == [sid]
    assert m.has(sid) is True
    assert m.count() == 1
    assert m.receptors_for([sid]) == {sid: rec}
    assert m.sim_ids_by_scene() == {"tower-1": [sid]}
    assert (tmp_path / ".stilt" / "manifest.parquet").exists()


def test_register_upsert_no_duplicates(tmp_path):
    m = Manifest(LocalStore(tmp_path))
    rec = _rec()
    sid = _sid(rec)
    m.register([(sid, rec)], scene_id="a")
    m.register([(sid, rec)], scene_id="b")  # re-register, last wins

    assert m.sim_ids() == [sid]
    assert m.count() == 1
    assert m.sim_ids_by_scene() == {"b": [sid]}


def test_register_accumulates_across_calls(tmp_path):
    m = Manifest(LocalStore(tmp_path))
    rec_a, rec_b = _rec(-111.85), _rec(-111.90)
    sid_a, sid_b = _sid(rec_a), _sid(rec_b)
    m.register([(sid_a, rec_a)], scene_id="s1")
    m.register([(sid_b, rec_b)], scene_id="s2")

    assert sorted(m.sim_ids()) == sorted([sid_a, sid_b])
    assert m.sim_ids_by_scene() == {"s1": [sid_a], "s2": [sid_b]}


def test_scene_none_excluded_from_scene_grouping(tmp_path):
    m = Manifest(LocalStore(tmp_path))
    rec = _rec()
    sid = _sid(rec)
    m.register([(sid, rec)])  # no scene

    assert m.has(sid) is True
    assert m.sim_ids_by_scene() == {}
