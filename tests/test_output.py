"""Tests for stilt.output: the output directory of runs and footprints."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from stilt.config import Variant
from stilt.footprint import jacobian, open_footprints
from stilt.footprint.config import FootprintConfig
from stilt.footprint.targets import Mesh
from stilt.identity import footprint_hash, footprint_settings, settings_hash
from stilt.meteorology import MetConfig
from stilt.output import Output
from stilt.particles import particles_metadata, write_particles
from stilt.receptors import PointReceptor
from stilt.spatial import Grid

from .fixtures.factories import make_receptor, make_variant
from .fixtures.footprints import as_footprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.1, yres=0.1)
MET = MetConfig(directory="/data/hrrr", file_format="%Y%m%d_%H", file_tres="6h")


def _variant(
    name: str = "hrrr", footprint: FootprintConfig | None = None, **overrides
) -> Variant:
    """A resolved variant for the tests; *overrides* change the transport fields."""
    return make_variant(name, met_config=MET, footprint=footprint, **overrides)


VARIANT = _variant()
#: VARIANT with footprints on GRID.
FEET = _variant(footprint=FootprintConfig(grid=GRID))


def _receptor(hour: int = 12, day: int = 15) -> PointReceptor:
    return make_receptor(dt.datetime(2024, 7, day, hour))


def _trajectories(receptor: PointReceptor, n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(int(receptor.time.timestamp()) % 1000)
    steps = np.arange(-1, -11, -1, dtype=float)
    indx = np.repeat(np.arange(1, n + 1, dtype=float), len(steps))
    time = np.tile(steps, n)
    data = pd.DataFrame(
        {
            "time": time,
            "particle": indx,
            "lon": -111.85 + rng.normal(0, 0.1, len(time)),
            "lat": 40.77 + rng.normal(0, 0.1, len(time)),
            "zagl": rng.uniform(0, 500, len(time)),
            "foot": rng.uniform(0, 0.1, len(time)),
        }
    )
    data["datetime"] = pd.Timestamp(receptor.time) + pd.to_timedelta(
        data["time"], unit="min"
    )
    return data


def _footprint(
    receptor: PointReceptor, hours=(-2, -1, 0), seed: int = 0, name: str = "hrrr"
) -> xr.DataArray:
    """A footprint on GRID with a few non-zero cells per layer."""
    x_axis, y_axis = GRID.axes
    rng = np.random.default_rng(seed)
    values = np.zeros((len(hours), len(y_axis), len(x_axis)))
    for t in range(len(hours)):
        for _ in range(4):
            values[t, rng.integers(len(y_axis)), rng.integers(len(x_axis))] = (
                rng.uniform(0.01, 1.0)
            )
    times = [pd.Timestamp(receptor.time) + pd.Timedelta(hours=h) for h in hours]
    data = xr.DataArray(
        values,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": y_axis, "lon": x_axis},
    )
    return as_footprint(data, receptor, FootprintConfig(grid=GRID), name)


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------


def _write(out: Output, variant: Variant = VARIANT, receptor=None) -> Path:
    """Write one receptor's particles for *variant*, making its folder."""
    receptor = receptor if receptor is not None else _receptor()
    return out.write_particles(variant, receptor, _trajectories(receptor), [])


def test_footprint_files_record_the_geometry_hash(tmp_path):
    """A footprint read back from its folder knows the geometry its grid was derived for."""
    from dataclasses import replace

    from stilt.footprint import read_footprint

    out = Output(tmp_path / "output")
    spec = {"kind": "windows", "coords": [(-111.85, 40.77)], "size": 0.2}
    variant = replace(
        VARIANT,
        footprint=FootprintConfig(grid=GRID, geometry=spec),
        geometry_hash="deadbeef00",
    )
    receptor = _receptor()
    path = out.write_footprint(variant, _footprint(receptor))
    out.write_empty_footprint(variant, _receptor(hour=13), "outside_domain")
    assert read_footprint(path).stilt.geometry_hash == "deadbeef00"
    folder = out.folder("footprints", variant)
    record = yaml.safe_load((folder / "_settings.yaml").read_text())
    assert record["settings"]["geometry_hash"] == "deadbeef00"
    # The same footprint config without the geometry hash is another folder.
    assert out.folder("footprints", replace(variant, geometry_hash=None)) is None


def test_run_folder_is_name_and_hash_with_settings_file(tmp_path):
    out = Output(tmp_path / "output")
    assert out.folder("particles", VARIANT) is None
    assert out.path("particles", VARIANT, _receptor().id) is None
    _write(out)
    digest = VARIANT.particles_hash
    folder = out.folder("particles", VARIANT)
    assert folder == tmp_path / "output" / "particles" / f"settings=hrrr-{digest[:6]}"
    record = yaml.safe_load((folder / "_settings.yaml").read_text())
    assert record["name"] == "hrrr"
    assert record["hash"] == digest
    assert record["settings"]["numpar"] == 100
    assert (
        "exe_dir" not in record["settings"]
        and "directory" not in record["settings"]["met"]
    )
    assert out.hashes("particles") == {f"hrrr-{digest[:6]}": digest}
    assert out.hashes("footprints") == {}


def test_same_settings_under_another_name_share_the_folder(tmp_path):
    out = Output(tmp_path / "output")
    _write(out)
    renamed = _variant("hrrr-main")
    assert out.folder("particles", renamed) == out.folder("particles", VARIANT)
    _write(out, renamed, _receptor(hour=13))
    assert len(out.hashes("particles")) == 1


def test_changed_settings_make_a_new_folder_beside_the_old(tmp_path):
    out = Output(tmp_path / "output")
    _write(out)
    changed = _variant(ziscale=0.8)
    _write(out, changed)
    first, second = out.folder("particles", VARIANT), out.folder("particles", changed)
    assert second != first
    assert second is not None and second.name.startswith("settings=hrrr-")
    assert set(out.hashes("particles")) == {
        first.name.removeprefix("settings="),
        second.name.removeprefix("settings="),
    }
    assert out.folder("particles", _variant(numpar=7)) is None


def test_kind_is_particles_or_footprints(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        Output(tmp_path).present("logs", VARIANT)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------


def test_particles_round_trip_in_date_folders(tmp_path):
    from stilt.particles import read_particles

    out = Output(tmp_path / "output")
    receptor = _receptor()
    traj = _trajectories(receptor)
    path = out.write_particles(VARIANT, receptor, traj, [])
    folder = out.folder("particles", VARIANT)
    assert path == folder / "date=2024-07-15" / f"{receptor.id}.parquet"
    assert path == out.path("particles", VARIANT, receptor.id)
    assert out.present("particles", VARIANT) == {receptor.id}

    back = read_particles(path)
    meta = particles_metadata(path)
    assert meta.receptor == receptor
    assert meta.settings == VARIANT.run_settings
    pd.testing.assert_frame_equal(
        back[traj.columns], traj, check_dtype=True, check_like=True
    )


def test_particles_store_time_and_index_as_int32(tmp_path):
    import pyarrow.parquet as pq

    path = _write(Output(tmp_path / "output"))
    schema = pq.read_schema(path)
    assert str(schema.field("time").type) == "int32"
    assert str(schema.field("particle").type) == "int32"
    assert "datetime" not in schema.names
    assert str(schema.field("lon").type) == "double"


def test_particle_files_name_their_receptor_in_a_column(tmp_path):
    """A scan of the particles tree tells receptors apart; one file reads without it (#105)."""
    from collections import Counter

    import pyarrow.dataset as pads

    from stilt.particles import read_particles

    out = Output(tmp_path / "output")
    receptors = [_receptor(hour=6), _receptor(hour=18, day=16)]
    for receptor in receptors:
        out.write_particles(VARIANT, receptor, _trajectories(receptor, n=10), [])

    scan = pads.dataset(out.folder("particles", VARIANT), format="parquet").to_table()
    counts = Counter(scan.column("receptor").to_pylist())
    assert counts == {str(r.id): 100 for r in receptors}

    back = read_particles(out.path("particles", VARIANT, receptors[0].id))
    assert "receptor" not in back.columns


def test_particle_file_without_a_receptor_column_still_reads(tmp_path):
    """Files written before the receptor column read as before."""
    import pyarrow.parquet as pq

    from stilt.particles import read_particles

    receptor = _receptor()
    traj = _trajectories(receptor)
    path = write_particles(
        tmp_path / "p.parquet", traj, receptor, VARIANT.run_settings, []
    )
    table = pq.ParquetFile(path).read()
    old = table.drop_columns(["receptor"]).replace_schema_metadata(
        table.schema.metadata
    )
    pq.write_table(old, path)
    back = read_particles(path)
    pd.testing.assert_frame_equal(back[traj.columns], traj, check_like=True)


def test_particles_reject_fractional_time(tmp_path):
    receptor = _receptor()
    traj = _trajectories(receptor)
    traj.loc[0, "time"] = -1.5
    with pytest.raises(ValueError, match="whole numbers"):
        Output(tmp_path / "output").write_particles(VARIANT, receptor, traj, [])


def test_particles_table_is_every_file_in_date_order(tmp_path):
    out = Output(tmp_path / "output")
    for day, hour in [(16, 0), (15, 18), (15, 6)]:
        _write(out, receptor=_receptor(hour=hour, day=day))
    table = out.table("particles", VARIANT)
    ids = list(dict.fromkeys(table.column("receptor").to_pylist()))
    assert [i[:10] for i in ids] == ["2024071506", "2024071518", "2024071600"]


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------


def test_footprint_folder_is_variant_name_and_combined_hash(tmp_path):
    out = Output(tmp_path / "output")
    config = FootprintConfig(grid=GRID, smooth_factor=1.0)
    variant = _variant(footprint=config)
    receptor = _receptor()
    out.write_footprint(variant, _footprint(receptor))
    digest = footprint_hash(VARIANT.particles_hash, footprint_settings(config, None))
    assert digest == settings_hash(
        {
            "particles": VARIANT.particles_hash,
            "footprint": {**config.model_dump(mode="json"), "geometry_hash": None},
        }
    )
    folder = out.folder("footprints", variant)
    assert folder == tmp_path / "output" / "footprints" / f"settings=hrrr-{digest[:6]}"
    particles = out.folder("particles", variant)
    assert particles is not None  # made with the footprint folder
    record = yaml.safe_load((folder / "_settings.yaml").read_text())
    assert record["particles"] == particles.name.removeprefix("settings=")
    assert out.hashes("footprints") == {f"hrrr-{digest[:6]}": digest}

    smooth = _variant(
        "hrrr-smooth", footprint=config.model_copy(update={"smooth_factor": 0.5})
    )
    out.write_footprint(smooth, _footprint(receptor))
    other = out.folder("footprints", smooth)
    assert other != folder and other.name.startswith("settings=hrrr-smooth-")
    assert len(out.hashes("footprints")) == 2
    # The same settings under another variant name reuse the folder.
    assert out.folder("footprints", _variant("renamed", footprint=config)) == folder
    # The same footprint settings on other particles are another folder.
    assert out.folder("footprints", _variant(footprint=config, numpar=200)) is None


def test_footprint_round_trip_is_exact_in_float32(tmp_path):
    from stilt.footprint import read_footprint
    from stilt.footprint.io import _empty_reason

    out = Output(tmp_path / "output")
    receptor = _receptor()
    foot = _footprint(receptor, hours=(-3, -2, -1, 0), seed=1)
    path = out.write_footprint(FEET, foot)
    folder = out.folder("footprints", FEET)
    assert path == folder / "date=2024-07-15" / f"{receptor.id}.parquet"
    assert folder.parent == tmp_path / "output" / "footprints"

    back = read_footprint(path)
    assert back is not None
    assert back.stilt.receptor == receptor
    assert back.stilt.name == "hrrr"
    assert list(back.dims) == ["time", "lat", "lon"]
    assert back.shape == foot.shape
    np.testing.assert_array_equal(back["time"].values, foot["time"].values)
    np.testing.assert_array_equal(
        back.values, foot.values.astype(np.float32).astype(np.float64)
    )
    assert _empty_reason(path) is None


def test_footprint_all_zero_layer_keeps_its_shape(tmp_path):
    from stilt.footprint import read_footprint

    foot = _footprint(_receptor(), hours=(-2, -1, 0))
    foot[1] = 0.0
    back = read_footprint(Output(tmp_path / "output").write_footprint(FEET, foot))
    assert back.shape == foot.shape
    assert float(back[1].sum()) == 0.0


def test_empty_footprint_is_a_file_with_no_rows_and_a_reason(tmp_path):
    from stilt.footprint import read_footprint
    from stilt.footprint.io import _empty_reason

    out = Output(tmp_path / "output")
    receptor = _receptor()
    path = out.write_empty_footprint(FEET, receptor, "outside_domain")
    assert path.exists()
    assert read_footprint(path) is None
    assert _empty_reason(path) == "outside_domain"
    assert out.present("footprints", FEET) == {receptor.id}


def test_footprint_coordinates_off_by_rounding_still_match(tmp_path):
    """A stored footprint's axis can differ from the grid's by 1e-14, as real files do."""
    from stilt.footprint import read_footprint

    foot = _footprint(_receptor(), seed=5)
    nudged = foot.assign_coords(
        lat=foot["lat"].values + 5e-15, lon=foot["lon"].values - 5e-15
    )
    back = read_footprint(Output(tmp_path / "output").write_footprint(FEET, nudged))
    np.testing.assert_array_equal(back.values, foot.values.astype(np.float32))


def test_footprint_on_another_grid_is_rejected(tmp_path):
    foot = _footprint(_receptor())
    shifted = foot.assign_coords(lon=foot["lon"].values + 0.03)
    with pytest.raises(ValueError, match="not cells of the grid"):
        Output(tmp_path / "output").write_footprint(FEET, shifted)


def test_a_variant_without_a_grid_has_no_footprint_folder(tmp_path):
    out = Output(tmp_path / "output")
    assert out.folder("footprints", VARIANT) is None
    assert out.present("footprints", VARIANT) == frozenset()
    with pytest.raises(ValueError, match="no grid"):
        out.write_empty_footprint(VARIANT, _receptor(), "outside_domain")


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


def test_complete_is_particles_and_a_footprint_when_the_variant_has_a_grid(tmp_path):
    out = Output(tmp_path / "output")
    a, b, c = _receptor(6), _receptor(12), _receptor(18)
    for receptor in (a, b):
        _write(out, FEET, receptor)
    out.write_footprint(FEET, _footprint(a))
    out.write_empty_footprint(FEET, c, "outside_domain")  # a footprint, no particles
    ids = [a.id, b.id, c.id]
    assert out.complete(FEET, ids) == {a.id}
    assert out.complete(VARIANT, ids) == {a.id, b.id}  # particles alone


def test_completed_is_the_one_rule():
    from stilt.output import completed

    assert completed(frozenset({"a", "b"}), None) == {"a", "b"}
    assert completed(frozenset({"a", "b"}), frozenset({"b", "c"})) == {"b"}


# ---------------------------------------------------------------------------
# Failure records
# ---------------------------------------------------------------------------


def test_failure_records_live_in_the_logs_of_the_folder_that_failed(tmp_path):
    out = Output(tmp_path / "output")
    rid = _receptor().id
    assert out.failure("particles", VARIANT, rid) is None
    out.record_failure("particles", VARIANT, rid, {"step": "particles", "reason": "X"})
    name = out.folder("particles", VARIANT).name
    assert (tmp_path / "output" / "logs" / name / "date=2024-07-15").is_dir()
    assert out.failure("particles", VARIANT, rid)["reason"] == "X"
    assert out.failures("particles", VARIANT) == {
        rid: {"step": "particles", "reason": "X"}
    }
    assert out.failures("particles", VARIANT, [_receptor(day=20).id]) == {}
    assert out.failure("footprints", FEET, rid) is None
    out.clear_failure("particles", VARIANT, rid)
    assert out.failure("particles", VARIANT, rid) is None


def test_logs_and_kept_workdirs_sit_beside_the_particles(tmp_path):
    out = Output(tmp_path / "output")
    rid = _receptor().id
    assert out.log_path(VARIANT, rid) is None and out.kept_workdir(VARIANT, rid) is None
    log = out.write_log(VARIANT, rid, "hycs_std ran\n")
    name = out.folder("particles", VARIANT).name
    assert log == tmp_path / "output" / "logs" / name / "date=2024-07-15" / f"{rid}.log"
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "CONTROL").write_text("x")
    kept = out.keep_workdir(VARIANT, rid, workdir)
    assert kept == out.kept_workdir(VARIANT, rid)
    assert (kept / "CONTROL").read_text() == "x"


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


@pytest.fixture
def written_footprints(tmp_path):
    out = Output(tmp_path / "output")
    feet_by_id = {}
    for k, hour in enumerate([6, 12, 18]):
        receptor = _receptor(hour=hour)
        foot = _footprint(receptor, hours=(-3, -2, -1, 0), seed=10 + k)
        out.write_footprint(FEET, foot)
        feet_by_id[str(receptor.id)] = foot
    empty = _receptor(hour=23)
    out.write_empty_footprint(FEET, empty, "outside_domain")
    return out, feet_by_id, str(empty.id)


def _bins():
    edges = pd.date_range("2024-07-15 00:00", "2024-07-16 00:00", freq="6h")
    return pd.IntervalIndex.from_breaks(edges, closed="left")


@pytest.mark.parametrize(
    "target",
    [
        Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.25, yres=0.25),
        Mesh.from_windows([(-111.85, 40.75), (-111.65, 40.9)], (0.3, 0.3)),
    ],
    ids=["coarser-grid", "mesh"],
)
def test_jacobian_matches_footprint_aggregate(written_footprints, target):
    out, feet_by_id, empty_id = written_footprints
    bins = _bins()
    table = out.table("footprints", FEET)
    H = jacobian(table, FEET.footprint, target, bins, [*feet_by_id, empty_id])

    assert list(H.receptors) == list(feet_by_id)
    assert H.empty == [empty_id]
    assert H.missing == []
    assert H.data.shape == (3, len(bins) * len(target.index))
    assert H.columns.names == ["time", "cell"]

    frame = H.to_frame()
    for rid, foot in feet_by_id.items():
        expected = foot.stilt.aggregate(target, bins)  # (cells × bins)
        got = frame.loc[rid].to_numpy().reshape(len(bins), len(target.index)).T
        np.testing.assert_allclose(got, expected.to_numpy(), rtol=1e-6, atol=1e-12)


def test_table_reads_the_date_folder_as_a_date32_column(written_footprints):
    import pyarrow as pa

    out, feet_by_id, empty_id = written_footprints
    table = out.table("footprints", FEET)
    assert table.schema.field("date").type == pa.date32()
    assert set(table.column("date").to_pylist()) == {dt.date(2024, 7, 15)}
    assert set(table.column("receptor").to_pylist()) == set(feet_by_id)
    assert out.table("footprints", FEET, []).num_rows == 0
    assert out.table("footprints", FEET, []).schema.field("date").type == pa.date32()
    one = next(iter(feet_by_id))
    assert set(out.table("footprints", FEET, [one]).column("receptor").to_pylist()) == {
        one
    }


def test_jacobian_rows_are_the_requested_receptors(written_footprints):
    """Cells of receptors in the table that were not asked for are left out."""
    out, feet_by_id, empty_id = written_footprints
    ids = list(feet_by_id)
    target = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.25, yres=0.25)
    table = out.table("footprints", FEET)
    H = jacobian(table, FEET.footprint, target, _bins(), [ids[1], empty_id])
    alone = jacobian(
        out.table("footprints", FEET, [ids[1]]),
        FEET.footprint,
        target,
        _bins(),
        [ids[1]],
    )
    assert list(H.receptors) == [ids[1]]
    assert H.empty == [empty_id]
    assert H.missing == []
    np.testing.assert_array_equal(H.data.toarray(), alone.data.toarray())


@pytest.mark.parametrize("closed", ["right", "both", "neither"])
def test_jacobian_rejects_bins_not_closed_on_the_left(written_footprints, closed):
    """A right-closed bin would put footprint hours in the wrong bin without a word."""
    out, _, _ = written_footprints
    left = _bins()
    bins = pd.IntervalIndex.from_arrays(left.left, left.right, closed=closed)
    with pytest.raises(ValueError, match="closed on the left"):
        jacobian(out.table("footprints", FEET), FEET.footprint, GRID, bins, [])


def test_jacobian_with_no_footprints_is_empty(tmp_path):
    table = Output(tmp_path / "output").table("footprints", FEET)
    H = jacobian(table, FEET.footprint, GRID, _bins(), [])
    assert H.data.shape == (0, len(_bins()) * len(GRID.index))
    assert list(H.receptors) == []


def test_jacobian_to_xarray_holds_the_matrix(written_footprints):
    out, feet_by_id, empty_id = written_footprints
    target = Mesh.from_windows([(-111.85, 40.75), (-111.65, 40.9)], (0.3, 0.3))
    bins = _bins()
    H = jacobian(
        out.table("footprints", FEET),
        FEET.footprint,
        target,
        bins,
        [*feet_by_id, empty_id],
        missing=["not-run"],
    )

    dense = H.to_xarray(dense=True)
    assert dense.dims == ("receptor", "time", "cell")
    assert list(dense.receptor.values) == list(H.receptors)
    assert list(dense.cell.values) == list(target.index)
    np.testing.assert_array_equal(dense.time.values, bins.left.values)
    np.testing.assert_array_equal(
        dense.values.reshape(len(H.receptors), -1), H.to_frame().to_numpy()
    )
    assert dense.attrs["empty"] == [empty_id]
    assert dense.attrs["missing"] == ["not-run"]

    pytest.importorskip("sparse")
    lazy = H.to_xarray()
    assert lazy.data.fill_value == 0
    np.testing.assert_array_equal(lazy.data.todense(), dense.values)


def test_jacobian_to_xarray_keeps_grid_cells_as_tuples(written_footprints):
    out, feet_by_id, _ = written_footprints
    target = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.25, yres=0.25)
    H = jacobian(
        out.table("footprints", FEET), FEET.footprint, target, _bins(), list(feet_by_id)
    )
    cells = H.to_xarray(dense=True).cell.values
    assert list(cells) == list(target.index)


# ---------------------------------------------------------------------------
# Footprints as one dataset
# ---------------------------------------------------------------------------


def test_open_footprints_stacks_receptors_on_the_hour(tmp_path):
    out = Output(tmp_path / "output")
    early, late, empty = _receptor(hour=6), _receptor(hour=18), _receptor(hour=23)
    hours = {early.id: [-3, -1], late.id: [-2, -1, 0]}
    feet = {
        r.id: _footprint(r, hours=hours[r.id], seed=k)
        for k, r in enumerate((early, late))
    }
    for foot in feet.values():
        out.write_footprint(FEET, foot)
    out.write_empty_footprint(FEET, empty, "outside_domain")
    paths = [out.path("footprints", FEET, r.id) for r in (late, empty, early)]

    ds = open_footprints(paths)

    assert list(ds.receptor.values) == [late.id, early.id]
    assert ds.attrs["empty"] == [empty.id]
    assert ds.attrs["stilt_name"] == "hrrr"
    assert ds.foot.dims == ("receptor", "hour", "lat", "lon")
    assert list(ds.hour.values) == [-3, -2, -1, 0]
    assert ds.foot.dtype == np.float32
    assert ds.foot.chunks is not None
    for rid, foot in feet.items():
        one = ds.sel(receptor=rid, hour=hours[rid])
        np.testing.assert_array_equal(one.time.values, foot.time.values)
        np.testing.assert_allclose(one.foot.values, foot.values, rtol=1e-6)
    # Hours a receptor has no layer for are zero, and still have a time.
    assert float(ds.foot.sel(receptor=early.id, hour=-2).sum()) == 0
    assert ds.time.sel(receptor=early.id, hour=0).values == np.datetime64(early.time)


def test_open_footprints_reads_in_blocks_by_date_folder(tmp_path):
    out = Output(tmp_path / "output")
    receptors = [_receptor(6, day=15), _receptor(12, day=15), _receptor(6, day=16)]
    for k, r in enumerate(receptors):
        out.write_footprint(FEET, _footprint(r, seed=k))
    ds = open_footprints(out.path("footprints", FEET, r.id) for r in receptors)
    assert ds.foot.chunks[0] == (2, 1)
    assert float(ds.foot.sum()) > 0


def test_open_footprints_refuses_mixed_settings(tmp_path):
    out = Output(tmp_path / "output")
    other = _variant("fine", FootprintConfig(grid=GRID, smooth_factor=0.5))
    r = _receptor()
    out.write_footprint(FEET, _footprint(r))
    out.write_footprint(other, _footprint(r, name="fine"))
    paths = [out.path("footprints", v, r.id) for v in (FEET, other)]
    with pytest.raises(ValueError, match="different settings"):
        open_footprints(paths)
    with pytest.raises(ValueError, match="No footprint files"):
        open_footprints([])


# ---------------------------------------------------------------------------
# Two workers writing the same file at once
# ---------------------------------------------------------------------------


def _interleave(monkeypatch, competitor):
    """Make *competitor* run just before the first rename of the code under test."""
    import stilt.output as output_module

    real_replace = output_module.os.replace
    state = {"raced": False}

    def replace(src, dst):
        if not state["raced"]:
            state["raced"] = True
            competitor()
        real_replace(src, dst)

    monkeypatch.setattr(output_module.os, "replace", replace)


def test_two_workers_starting_a_run_at_once_both_succeed(tmp_path, monkeypatch):
    """Both wrote `_settings.tmp`; the second rename found it already moved."""
    path = tmp_path / "output"
    rid = _receptor().id
    _interleave(monkeypatch, lambda: Output(path).write_log(VARIANT, rid, ""))

    Output(path).write_log(VARIANT, rid, "")

    folder = Output(path).folder("particles", VARIANT)
    record = yaml.safe_load((folder / "_settings.yaml").read_text())
    assert record["hash"] == VARIANT.particles_hash
    assert [p.name for p in folder.iterdir()] == ["_settings.yaml"]


def test_two_workers_writing_one_receptor_at_once_both_succeed(tmp_path, monkeypatch):
    """The other worker's cleanup removed the shared temporary file."""
    from stilt.particles import read_particles

    receptor = _receptor()
    traj = _trajectories(receptor)
    out = Output(tmp_path / "output")
    other = Output(tmp_path / "output")
    out.write_log(VARIANT, receptor.id, "")  # both find the folder
    other.folder("particles", VARIANT)
    _interleave(monkeypatch, lambda: other.write_particles(VARIANT, receptor, traj, []))

    path = out.write_particles(VARIANT, receptor, traj, [])

    assert len(read_particles(path)) == len(traj)
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no stray files


# ---------------------------------------------------------------------------
# Which receptors have results
# ---------------------------------------------------------------------------


def test_present_lists_only_the_date_folders_asked_for(tmp_path, monkeypatch):
    """A few simulations of a large project must not list every date folder."""
    import stilt.output as output_module

    out = Output(tmp_path / "output")
    day15, day16 = _receptor(12, day=15), _receptor(12, day=16)
    for receptor in (day15, day16):
        _write(out, receptor=receptor)
    never_run = _receptor(12, day=20)
    out.folder("particles", VARIANT)  # the folder is known; only receptors are listed

    listed: list[str] = []
    real_scandir = output_module.os.scandir

    def scandir(path):
        listed.append(Path(path).name)
        return real_scandir(path)

    monkeypatch.setattr(output_module.os, "scandir", scandir)
    found = out.present("particles", VARIANT, [day15.id, never_run.id])

    assert found == {day15.id}
    assert sorted(listed) == ["date=2024-07-15", "date=2024-07-20"]
    assert out.present("particles", VARIANT) == {day15.id, day16.id}


# ---------------------------------------------------------------------------
# What each file records
# ---------------------------------------------------------------------------


def test_each_result_file_names_its_settings_and_the_version_that_wrote_it(tmp_path):
    """A file copied out of the output directory still says where it came from."""
    import pyarrow.parquet as pq

    import stilt

    receptor = _receptor()
    out = Output(tmp_path / "output")
    particles = out.write_particles(FEET, receptor, _trajectories(receptor), [])
    footprint = out.write_footprint(FEET, _footprint(receptor))
    empty = out.write_empty_footprint(FEET, _receptor(13), "outside_domain")

    meta = pq.read_schema(particles).metadata
    assert meta[b"stilt:hash"].decode() == FEET.particles_hash
    assert meta[b"stilt:pystilt"].decode() == stilt.__version__
    for path in (footprint, empty):
        meta = pq.read_schema(path).metadata
        assert meta[b"stilt:hash"].decode() == FEET.footprint_hash
        assert meta[b"stilt:pystilt"].decode() == stilt.__version__


def test_a_stored_footprint_file_opens_on_its_own(tmp_path):
    """The file records its grid, so read_footprint needs nothing else (#107)."""
    import shutil

    from stilt.footprint import read_footprint

    receptor = _receptor()
    path = Output(tmp_path / "output").write_footprint(
        FEET, _footprint(receptor, seed=3)
    )

    copied = tmp_path / "elsewhere.parquet"  # away from the folder's _settings.yaml
    shutil.copy(path, copied)
    alone = read_footprint(copied)
    assert alone is not None
    xr.testing.assert_identical(alone, read_footprint(path))
    assert alone.stilt.grid == GRID
    assert alone.stilt.receptor == receptor


def test_a_footprint_file_without_its_settings_is_refused(tmp_path):
    """Each file is read alone, so a file must record its settings."""
    import pyarrow.parquet as pq

    from stilt.footprint import read_footprint

    path = Output(tmp_path / "output").write_footprint(
        FEET, _footprint(_receptor(), seed=3)
    )
    table = pq.ParquetFile(path).read()
    meta = {k: v for k, v in table.schema.metadata.items() if k != b"stilt:footprint"}
    pq.write_table(table.replace_schema_metadata(meta), path)

    with pytest.raises(ValueError, match="does not record its footprint settings"):
        read_footprint(path)


def test_a_footprint_folder_stored_with_projection_is_found_by_crs(tmp_path):
    """Folders written when the grid said ``projection`` are found by a ``crs`` config."""
    variant = _variant(footprint=FootprintConfig(grid=GRID))
    out = Output(tmp_path / "output")
    out.write_empty_footprint(variant, _receptor(), "outside_domain")
    folder = out.folder("footprints", variant)
    record_path = folder / "_settings.yaml"
    record = yaml.safe_load(record_path.read_text())
    grid = record["settings"]["grid"]
    grid["projection"] = grid.pop("crs")
    record_path.write_text(yaml.safe_dump(record))

    assert Output(tmp_path / "output").folder("footprints", variant) == folder


def test_a_lookup_reads_each_folder_once(tmp_path, monkeypatch):
    """A lookup that misses lists the tree again but reads only the folders it has not seen."""
    out = Output(tmp_path / "output")
    _write(out)
    _write(out, _variant(ziscale=0.8))

    import stilt.output

    reads: list[str] = []
    real = stilt.output.read_run_settings

    def counting(stored):
        reads.append(stored["numpar"])
        return real(stored)

    monkeypatch.setattr(stilt.output, "read_run_settings", counting)
    fresh = Output(tmp_path / "output")
    for _ in range(3):  # a variant that has not run misses every time
        assert fresh.folder("particles", _variant(numpar=7)) is None
    assert len(reads) == 2  # each existing folder once

    # A folder another worker creates is found on the next lookup, which
    # reads that folder alone.
    _write(Output(tmp_path / "output"), _variant(numpar=7))
    before = len(reads)
    assert fresh.folder("particles", _variant(numpar=7)) is not None
    assert len(reads) == before + 1


def test_a_footprint_folder_reads_its_particles_folder_once(tmp_path, monkeypatch):
    out = Output(tmp_path / "output")
    out.write_empty_footprint(FEET, _receptor(), "outside_domain")

    import stilt.output

    reads: list[int] = []
    real = stilt.output.read_run_settings
    monkeypatch.setattr(
        stilt.output, "read_run_settings", lambda s: reads.append(1) or real(s)
    )
    fresh = Output(tmp_path / "output")
    assert fresh.folder("footprints", FEET) == out.folder("footprints", FEET)
    assert fresh.folder("particles", FEET) == out.folder("particles", FEET)
    assert len(reads) == 1


def test_an_ensembles_realizations_are_partitions_of_one_folder(tmp_path):
    from dataclasses import replace

    out = Output(tmp_path / "output")
    ensemble = replace(_variant("err", krand=4), realizations=2)
    receptor = _receptor()
    paths = [
        out.write_particles(ensemble, receptor, _trajectories(receptor), [], k)
        for k in (0, 1)
    ]
    folder = out.folder("particles", ensemble)
    assert [p.parent.parent for p in paths] == [
        folder / "realization=0",
        folder / "realization=1",
    ]
    assert particles_metadata(paths[1]).realization == 1
    assert out.present("particles", ensemble, realization=1) == {receptor.id}
    assert out.present("particles", ensemble, [receptor.id], 0) == {receptor.id}
    out.record_failure("particles", ensemble, receptor.id, {"reason": "X"}, 1)
    assert out.failure("particles", ensemble, receptor.id, 1) == {"reason": "X"}
    assert out.failure("particles", ensemble, receptor.id, 0) is None
    with pytest.raises(ValueError, match="realizations"):
        out.path("particles", ensemble, receptor.id)  # an ensemble needs its number
    with pytest.raises(ValueError, match="realizations"):
        out.path("particles", VARIANT, receptor.id, 0)
    table = out.table("particles", ensemble, realization=0)
    assert set(table.column("receptor").to_pylist()) == {receptor.id}


def test_a_stored_footprint_says_what_made_it(tmp_path):
    from stilt.footprint import read_footprint

    out = Output(tmp_path / "output")
    receptor = _receptor()
    path = out.write_footprint(FEET, _footprint(receptor))

    foot = read_footprint(path)

    assert foot is not None
    assert foot.attrs["stilt_particles_hash"] == FEET.particles_hash
    assert foot.attrs["stilt_version"]
    nc = foot.stilt.to_netcdf(tmp_path / "foot.nc")
    back = read_footprint(nc)
    assert back is not None
    assert back.attrs["stilt_particles_hash"] == FEET.particles_hash
