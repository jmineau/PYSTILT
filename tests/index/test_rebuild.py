"""Unit tests for output-index rebuild helpers."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from stilt.index.rebuild import (
    _footprint_name_from_filename,
    _receptor_from_parquet,
    scan_output_simulations,
)
from stilt.receptor import Receptor


def test_footprint_name_from_filename_handles_default_named_and_invalid_cases():
    sim_id = "hrrr_202301011200_-111.85_40.77_5"

    assert (
        _footprint_name_from_filename(sim_id, f"{sim_id}_foot.nc", suffix="_foot.nc")
        == ""
    )
    assert (
        _footprint_name_from_filename(
            sim_id, f"{sim_id}_fine_foot.nc", suffix="_foot.nc"
        )
        == "fine"
    )
    assert (
        _footprint_name_from_filename(sim_id, "other_foot.nc", suffix="_foot.nc")
        is None
    )
    assert (
        _footprint_name_from_filename(
            sim_id, f"{sim_id}_fine_foot.empty", suffix="_foot.nc"
        )
        is None
    )


def test_receptor_from_parquet_returns_none_for_missing_metadata(monkeypatch, tmp_path):
    class _ArrowSchema:
        metadata = None

    class _ParquetFile:
        schema_arrow = _ArrowSchema()

    monkeypatch.setattr(
        "stilt.index.rebuild.pq.ParquetFile",
        lambda path: _ParquetFile(),
    )

    assert _receptor_from_parquet(tmp_path / "traj.parquet") is None
    assert _receptor_from_parquet(None) is None


def test_receptor_from_parquet_returns_none_on_read_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "stilt.index.rebuild.pq.ParquetFile",
        lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert _receptor_from_parquet(tmp_path / "traj.parquet") is None


def test_receptor_from_parquet_loads_receptor_from_metadata(monkeypatch, tmp_path):
    receptor = Receptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )

    class _ArrowSchema:
        metadata = {b"stilt:receptor": json.dumps(receptor.to_dict()).encode("utf-8")}

    class _ParquetFile:
        schema_arrow = _ArrowSchema()

    monkeypatch.setattr(
        "stilt.index.rebuild.pq.ParquetFile",
        lambda path: _ParquetFile(),
    )

    loaded = _receptor_from_parquet(tmp_path / "traj.parquet")

    assert loaded == receptor


def test_scan_output_simulations_summarizes_outputs_and_empty_markers(
    monkeypatch, tmp_path
):
    receptor = Receptor(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    keys = [
        "simulations/by-id/sim-a/sim-a_traj.parquet",
        "simulations/by-id/sim-a/sim-a_error.parquet",
        "simulations/by-id/sim-a/stilt.log",
        "simulations/by-id/sim-a/sim-a_foot.nc",
        "simulations/by-id/sim-a/sim-a_fine_foot.empty",
        "simulations/by-id/sim-a/sim-a_fine_foot.nc",
        "simulations/by-id/sim-b/sim-b_error.parquet",
        "simulations/by-id/sim-b/sim-b_coarse_foot.empty",
        "simulations/by-id/sim-c/README.txt",
        "simulations/particles/sim-a_traj.parquet",
    ]

    class _Store:
        def list_prefix(self, prefix: str):
            assert prefix == "simulations/by-id"
            return keys

        def local_path(self, key: str) -> Path:
            return tmp_path / Path(key).name

    local_calls: list[str] = []
    monkeypatch.setattr("stilt.index.rebuild.make_store", lambda root: _Store())

    def _fake_receptor_from_parquet(path: Path) -> Receptor | None:
        local_calls.append(path.name)
        return receptor if path.name == "sim-a_traj.parquet" else None

    monkeypatch.setattr(
        "stilt.index.rebuild._receptor_from_parquet",
        _fake_receptor_from_parquet,
    )

    records = scan_output_simulations(tmp_path)

    assert [record.sim_id for record in records] == ["sim-a", "sim-b"]
    assert records[0].summary.traj_present is True
    assert records[0].summary.error_traj_present is True
    assert records[0].summary.log_present is True
    assert records[0].summary.footprints == {"": "complete", "fine": "complete"}
    assert records[0].receptor == receptor
    assert records[1].summary.traj_present is False
    assert records[1].summary.error_traj_present is True
    assert records[1].summary.footprints == {"coarse": "complete-empty"}
    assert records[1].receptor is None
    assert local_calls == ["sim-a_traj.parquet", "sim-b_error.parquet"]
