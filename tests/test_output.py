"""Tests for runtraj.output: the output directory of runs and footprints."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from runtraj.identity import settings_hash, stage_hash
from runtraj.output import Output
from runtraj.spatial import Grid
from scipy import sparse

from stilt.footprint import Jacobian, jacobian, open_footprints
from stilt.footprint.config import FootprintConfig, footprint_settings
from stilt.footprint.stage import store_empty_footprint, store_footprint
from stilt.footprint.targets import Mesh

from .conftest import STAND_IN_VERSION
from .fixtures.output import (
    FEET,
    GRID,
    VARIANT,
    as_written_before_the_rename,
    fake_footprint,
    fake_particles,
    output_variant,
    release_at,
    write_one,
)

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------


def test_footprint_files_record_the_geometry_hash(tmp_path):
    """A footprint read back from its folder knows the geometry its grid was derived for."""
    from stilt.footprint import read_footprint

    out = Output(tmp_path / "output")
    spec = {"kind": "windows", "coords": [(-111.85, 40.77)], "size": 0.2}
    config = FootprintConfig(grid=GRID, geometry=spec)
    variant = output_variant(
        footprint=config.model_copy(update={"geometry_hash": "deadbeef00"})
    )
    release = release_at()
    path = store_footprint(out, variant, fake_footprint(release))
    store_empty_footprint(out, variant, release_at(hour=13))
    assert read_footprint(path).stilt.geometry_hash == "deadbeef00"
    folder = out.folder("footprint", variant)
    record = yaml.safe_load((folder / "_settings.yaml").read_text())
    assert record["settings"]["geometry_hash"] == "deadbeef00"
    # The same footprint config without the geometry hash is another folder.
    assert out.folder("footprint", output_variant(footprint=config)) is None


def test_run_folder_is_name_and_hash_with_settings_file(tmp_path):
    out = Output(tmp_path / "output")
    assert out.folder("particles", VARIANT) is None
    assert out.path("particles", VARIANT, release_at().id) is None
    write_one(out)
    digest = VARIANT.particles_hash
    folder = out.folder("particles", VARIANT)
    assert folder == tmp_path / "output" / "particles" / f"settings=hrrr-{digest[:6]}"
    record = yaml.safe_load((folder / "_settings.yaml").read_text())
    assert record["name"] == "hrrr"
    assert record["hash"] == digest
    assert record["settings"]["numpar"] == 100
    # The version hycs_std prints is recorded beside the settings, not in them.
    assert record["model_version"] == STAND_IN_VERSION
    assert "version" not in record["settings"]["model"]
    assert (
        "directory" not in record["settings"]
        and "directory" not in record["settings"]["met"]
    )
    assert out.hashes("particles") == {f"hrrr-{digest[:6]}": digest}
    assert out.hashes("footprint") == {}


def test_kind_is_particles_or_footprints(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        Output(tmp_path).present("logs", VARIANT)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Files written before a receptor was renamed a release
# ---------------------------------------------------------------------------

OLD = "written before receptors were renamed releases"


def test_a_footprint_file_from_before_the_rename_is_refused_with_its_name(tmp_path):
    from stilt.footprint import read_footprint

    out = Output(tmp_path / "output")
    path = as_written_before_the_rename(
        store_footprint(out, FEET, fake_footprint(release_at(), hours=(-1, 0)))
    )
    with pytest.raises(ValueError, match=OLD) as error:
        read_footprint(path)
    assert str(path) in str(error.value)
    with pytest.raises(ValueError, match=OLD):
        open_footprints([path])


def test_a_scan_of_footprints_from_before_the_rename_is_refused(tmp_path):
    out = Output(tmp_path / "output")
    for hour in (6, 18):
        store_footprint(
            out, FEET, fake_footprint(release_at(hour=hour), hours=(-1, 0), seed=hour)
        )
    old = as_written_before_the_rename(
        out.path("footprint", FEET, release_at(hour=6).id)
    )
    with pytest.raises(ValueError, match=OLD) as error:
        out.table("footprint", FEET)
    assert str(old) in str(error.value)


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------


def test_footprint_folder_is_variant_name_and_combined_hash(tmp_path):
    out = Output(tmp_path / "output")
    config = FootprintConfig(grid=GRID, smooth_factor=1.0)
    variant = output_variant(footprint=config)
    release = release_at()
    store_footprint(out, variant, fake_footprint(release))
    digest = stage_hash("footprint", footprint_settings(config), VARIANT.particles_hash)
    assert digest == settings_hash(
        {
            "particles": VARIANT.particles_hash,
            "footprint": {**config.model_dump(mode="json"), "geometry_hash": None},
        }
    )
    folder = out.folder("footprint", variant)
    assert folder == tmp_path / "output" / "footprint" / f"settings=hrrr-{digest[:6]}"
    particles = out.folder("particles", variant)
    assert particles is not None  # made with the footprint folder
    record = yaml.safe_load((folder / "_settings.yaml").read_text())
    assert list(record) == [
        "name",
        "hash",
        "stage",
        "class",
        "reads",
        "versions",
        "settings",
    ]
    assert (record["name"], record["hash"], record["stage"]) == (
        "hrrr",
        digest,
        "footprint",
    )
    assert record["class"] == "stilt.footprint.FootprintStage"
    # What the footprints were made from: the particles folder, and its hash.
    assert record["reads"] == {
        "particles": {
            "folder": particles.name.removeprefix("settings="),
            "hash": VARIANT.particles_hash,
        }
    }
    assert list(record["versions"]) == ["runtraj", "pystilt"]
    assert record["settings"] == footprint_settings(config)
    assert out.hashes("footprint") == {f"hrrr-{digest[:6]}": digest}

    smooth = output_variant(
        "hrrr-smooth", footprint=config.model_copy(update={"smooth_factor": 0.5})
    )
    store_footprint(out, smooth, fake_footprint(release))
    other = out.folder("footprint", smooth)
    assert other != folder and other.name.startswith("settings=hrrr-smooth-")
    assert len(out.hashes("footprint")) == 2
    # The same settings under another variant name reuse the folder.
    assert (
        out.folder("footprint", output_variant("renamed", footprint=config)) == folder
    )
    # The same footprint settings on other particles are another folder.
    assert out.folder("footprint", output_variant(footprint=config, numpar=200)) is None


def test_footprint_round_trip_is_exact_in_float32(tmp_path):
    from stilt.footprint import read_footprint

    out = Output(tmp_path / "output")
    release = release_at()
    foot = fake_footprint(release, hours=(-3, -2, -1, 0), seed=1)
    path = store_footprint(out, FEET, foot)
    folder = out.folder("footprint", FEET)
    assert path == folder / "date=2024-07-15" / f"{release.id}.parquet"
    assert folder.parent == tmp_path / "output" / "footprint"

    back = read_footprint(path)
    assert back is not None
    assert back.stilt.release == release
    assert back.stilt.name == "hrrr"
    assert list(back.dims) == ["time", "lat", "lon"]
    assert back.shape == foot.shape
    np.testing.assert_array_equal(back["time"].values, foot["time"].values)
    np.testing.assert_array_equal(
        back.values, foot.values.astype(np.float32).astype(np.float64)
    )
    assert _metadata(path)[b"stilt:empty"] == b"false"


def test_footprint_all_zero_layer_keeps_its_shape(tmp_path):
    from stilt.footprint import read_footprint

    foot = fake_footprint(release_at(), hours=(-2, -1, 0))
    foot[1] = 0.0
    back = read_footprint(store_footprint(Output(tmp_path / "output"), FEET, foot))
    assert back.shape == foot.shape
    assert float(back[1].sum()) == 0.0


def _metadata(path) -> dict[bytes, bytes]:
    """Return the metadata of a result file."""
    import pyarrow.parquet as pq

    return pq.read_schema(path).metadata


def test_empty_footprint_is_a_file_with_no_rows_marked_empty(tmp_path):
    from stilt.footprint import read_footprint

    out = Output(tmp_path / "output")
    release = release_at()
    path = store_empty_footprint(out, FEET, release)
    assert path.exists()
    assert read_footprint(path) is None
    assert _metadata(path)[b"stilt:empty"] == b"true"
    assert out.present("footprint", FEET) == {release.id}


def test_footprint_coordinates_off_by_rounding_still_match(tmp_path):
    """A stored footprint's axis can differ from the grid's by 1e-14, as real files do."""
    from stilt.footprint import read_footprint

    foot = fake_footprint(release_at(), seed=5)
    nudged = foot.assign_coords(
        lat=foot["lat"].values + 5e-15, lon=foot["lon"].values - 5e-15
    )
    back = read_footprint(store_footprint(Output(tmp_path / "output"), FEET, nudged))
    np.testing.assert_array_equal(back.values, foot.values.astype(np.float32))


def test_footprint_on_another_grid_is_rejected(tmp_path):
    foot = fake_footprint(release_at())
    shifted = foot.assign_coords(lon=foot["lon"].values + 0.03)
    with pytest.raises(ValueError, match="not cells of the grid"):
        store_footprint(Output(tmp_path / "output"), FEET, shifted)


def test_a_variant_without_a_grid_has_no_footprint_folder(tmp_path):
    out = Output(tmp_path / "output")
    assert out.folder("footprint", VARIANT) is None
    assert out.present("footprint", VARIANT) == frozenset()
    with pytest.raises(ValueError, match="Variant 'hrrr' has no footprint stage"):
        store_empty_footprint(out, VARIANT, release_at())


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


def test_complete_is_particles_and_a_footprint_when_the_variant_has_a_grid(tmp_path):
    out = Output(tmp_path / "output")
    a, b, c = release_at(6), release_at(12), release_at(18)
    for release in (a, b):
        write_one(out, FEET, release)
    store_footprint(out, FEET, fake_footprint(a))
    store_empty_footprint(out, FEET, c)  # a footprint, no particles
    ids = [a.id, b.id, c.id]
    assert out.complete(FEET, ids) == {a.id}
    assert out.complete(VARIANT, ids) == {a.id, b.id}  # particles alone


# ---------------------------------------------------------------------------
# Failure records
# ---------------------------------------------------------------------------


def test_failure_records_live_in_the_logs_of_the_folder_that_failed(tmp_path):
    out = Output(tmp_path / "output")
    rid = release_at().id
    assert out.failure("particles", VARIANT, rid) is None
    out.record_failure("particles", VARIANT, rid, {"stage": "particles", "reason": "X"})
    name = out.folder("particles", VARIANT).name
    assert (tmp_path / "output" / "logs" / name / "date=2024-07-15").is_dir()
    assert out.failure("particles", VARIANT, rid)["reason"] == "X"
    assert out.failures("particles", VARIANT) == {
        rid: {"stage": "particles", "reason": "X"}
    }
    assert out.failures("particles", VARIANT, [release_at(day=20).id]) == {}
    assert out.failure("footprint", FEET, rid) is None
    out.clear_failure("particles", VARIANT, rid)
    assert out.failure("particles", VARIANT, rid) is None


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


@pytest.fixture
def written_footprints(tmp_path):
    out = Output(tmp_path / "output")
    feet_by_id = {}
    for k, hour in enumerate([6, 12, 18]):
        release = release_at(hour=hour)
        foot = fake_footprint(release, hours=(-3, -2, -1, 0), seed=10 + k)
        store_footprint(out, FEET, foot)
        feet_by_id[str(release.id)] = foot
    empty = release_at(hour=23)
    store_empty_footprint(out, FEET, empty)
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
    table = out.table("footprint", FEET)
    H = jacobian(
        table, FEET.stages.get("footprint"), target, bins, [*feet_by_id, empty_id]
    )

    assert list(H.releases) == list(feet_by_id)
    assert H.empty == [empty_id]
    assert H.missing == []
    assert H.data.shape == (3, len(bins) * len(target.index))
    # The target's own levels: lon, lat for a grid; cell for a mesh.
    assert H.columns.names == ["time", *target.index.names]

    frame = H.to_frame()
    for rid, foot in feet_by_id.items():
        expected = foot.stilt.aggregate(target, bins)  # (cells × bins)
        got = frame.loc[rid].to_numpy().reshape(len(bins), len(target.index)).T
        np.testing.assert_allclose(got, expected.to_numpy(), rtol=1e-6, atol=1e-12)


#: FEET with 15-minute footprint layers.
QUARTERS = output_variant(footprint=FootprintConfig(grid=GRID, tres="15min"))


def test_jacobian_of_fifteen_minute_layers_matches_footprint_aggregate(tmp_path):
    """Each stored layer goes to the bin its 15-minute start falls in, not the hour's."""
    out = Output(tmp_path / "output")
    release = release_at(hour=12)
    foot = fake_footprint(release, hours=(-5, -4, -3, -2, -1), tres="15min")
    store_footprint(out, QUARTERS, foot)
    edges = pd.date_range("2024-07-15 10:30", "2024-07-15 12:00", freq="30min")
    bins = pd.IntervalIndex.from_breaks(edges, closed="left")
    target = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.25, yres=0.25)

    H = jacobian(
        out.table("footprint", QUARTERS),
        QUARTERS.stages.get("footprint"),
        target,
        bins,
        [str(release.id)],
    )

    expected = foot.stilt.aggregate(target, bins)
    # The layer at 10:45 is in the first bin, so not every bin is empty.
    assert (expected.to_numpy().sum(axis=0) > 0).all()
    got = H.to_frame().iloc[0].to_numpy().reshape(len(bins), len(target.index)).T
    np.testing.assert_allclose(got, expected.to_numpy(), rtol=1e-6, atol=1e-12)


def test_jacobian_sparse_row_matches_dense(written_footprints, monkeypatch):
    """A row too long to sum densely is summed sparsely, to the same matrix."""
    import stilt.footprint.aggregation as aggregation

    out, feet_by_id, empty_id = written_footprints
    target = Mesh.from_windows([(-111.85, 40.75), (-111.65, 40.9)], (0.3, 0.3))
    table = out.table("footprint", FEET)
    dense = jacobian(
        table, FEET.stages.get("footprint"), target, _bins(), [*feet_by_id, empty_id]
    )
    monkeypatch.setattr(aggregation, "_DENSE_ROW_MAX", 0)
    sparse = jacobian(
        table, FEET.stages.get("footprint"), target, _bins(), [*feet_by_id, empty_id]
    )

    assert list(sparse.releases) == list(dense.releases)
    assert sparse.empty == dense.empty == [empty_id]
    np.testing.assert_allclose(
        sparse.data.toarray(), dense.data.toarray(), rtol=1e-12, atol=0
    )


def test_jacobian_takes_a_plain_release_column(written_footprints):
    """The release column may be strings rather than dictionary-encoded."""
    import pyarrow as pa

    out, feet_by_id, _ = written_footprints
    table = out.table("footprint", FEET)
    plain = table.set_column(
        table.schema.get_field_index("release"),
        "release",
        table.column("release").cast(pa.string()),
    )
    ids = list(feet_by_id)
    H = jacobian(table, FEET.stages.get("footprint"), GRID, _bins(), ids)
    H_plain = jacobian(plain, FEET.stages.get("footprint"), GRID, _bins(), ids)
    assert list(H_plain.releases) == ids
    np.testing.assert_array_equal(H_plain.data.toarray(), H.data.toarray())


def test_read_returns_one_releases_stored_columns(written_footprints):
    out, feet_by_id, empty_id = written_footprints
    one = next(iter(feet_by_id))
    table = out.read("footprint", FEET, one)
    assert table.column_names == ["release", "hour", "y", "x", "foot"]
    assert set(table.column("release").to_pylist()) == {one}
    many = out.table("footprint", FEET, [one])
    assert table.num_rows == many.num_rows == np.count_nonzero(feet_by_id[one].values)
    cells = out.read("footprint", FEET, one, columns=["hour", "foot"])
    assert cells.column_names == ["hour", "foot"]
    assert out.read("footprint", FEET, empty_id).num_rows == 0
    with pytest.raises(FileNotFoundError):
        out.read("footprint", FEET, "202407150000_-111.9_40.8_5")


def test_table_reads_the_date_folder_as_a_date32_column(written_footprints):
    import pyarrow as pa

    out, feet_by_id, empty_id = written_footprints
    table = out.table("footprint", FEET)
    assert table.schema.field("date").type == pa.date32()
    assert set(table.column("date").to_pylist()) == {dt.date(2024, 7, 15)}
    assert set(table.column("release").to_pylist()) == set(feet_by_id)
    assert out.table("footprint", FEET, []).num_rows == 0
    assert out.table("footprint", FEET, []).schema.field("date").type == pa.date32()
    one = next(iter(feet_by_id))
    assert set(out.table("footprint", FEET, [one]).column("release").to_pylist()) == {
        one
    }


def test_jacobian_rows_are_the_requested_releases(written_footprints):
    """Cells of releases in the table that were not asked for are left out."""
    out, feet_by_id, empty_id = written_footprints
    ids = list(feet_by_id)
    target = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.25, yres=0.25)
    table = out.table("footprint", FEET)
    H = jacobian(
        table, FEET.stages.get("footprint"), target, _bins(), [ids[1], empty_id]
    )
    alone = jacobian(
        out.table("footprint", FEET, [ids[1]]),
        FEET.stages.get("footprint"),
        target,
        _bins(),
        [ids[1]],
    )
    assert list(H.releases) == [ids[1]]
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
        jacobian(
            out.table("footprint", FEET), FEET.stages.get("footprint"), GRID, bins, []
        )


def test_jacobian_with_no_footprints_is_empty(tmp_path):
    table = Output(tmp_path / "output").table("footprint", FEET)
    H = jacobian(table, FEET.stages.get("footprint"), GRID, _bins(), [])
    assert H.data.shape == (0, len(_bins()) * len(GRID.index))
    assert list(H.releases) == []


def test_jacobian_to_xarray_holds_the_matrix(written_footprints):
    out, feet_by_id, empty_id = written_footprints
    target = Mesh.from_windows([(-111.85, 40.75), (-111.65, 40.9)], (0.3, 0.3))
    bins = _bins()
    H = jacobian(
        out.table("footprint", FEET),
        FEET.stages.get("footprint"),
        target,
        bins,
        [*feet_by_id, empty_id],
        missing=["not-run"],
    )

    dense = H.to_xarray(dense=True)
    assert dense.dims == ("release", "time", "cell")
    assert list(dense.release.values) == list(H.releases)
    assert list(dense.cell.values) == list(target.index)
    np.testing.assert_array_equal(dense.time.values, bins.left.values)
    np.testing.assert_array_equal(
        dense.values.reshape(len(H.releases), -1), H.to_frame().to_numpy()
    )
    assert dense.attrs["empty"] == [empty_id]
    assert dense.attrs["missing"] == ["not-run"]

    pytest.importorskip("sparse")
    lazy = H.to_xarray()
    assert lazy.data.fill_value == 0
    np.testing.assert_array_equal(lazy.data.todense(), dense.values)


def test_a_grid_jacobian_names_its_cell_levels(written_footprints):
    out, feet_by_id, _ = written_footprints
    target = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.25, yres=0.25)
    bins = _bins()
    H = jacobian(
        out.table("footprint", FEET),
        FEET.stages.get("footprint"),
        target,
        bins,
        list(feet_by_id),
    )

    assert H.columns.names == ["time", "lon", "lat"]
    assert list(H.columns[: len(target.index)].droplevel("time")) == list(target.index)
    dense = H.to_xarray(dense=True)
    assert list(dense.cell.values) == list(target.index)
    grid = dense.unstack("cell")
    assert grid.dims == ("release", "time", "lon", "lat")
    np.testing.assert_allclose(float(grid.sum()), H.data.sum())

    sparse_frame = H.to_frame(sparse=True)
    assert isinstance(sparse_frame.dtypes.iloc[0], pd.SparseDtype)
    np.testing.assert_array_equal(
        sparse_frame.sparse.to_dense().to_numpy(), H.to_frame().to_numpy()
    )


def test_a_sparse_jacobian_frame_keeps_the_stored_cells_and_fills_zero():
    # An empty column, a stored zero and a full column; a cell not stored is 0.
    data = sparse.csr_matrix(np.array([[0.0, 2.0, 0.0, 1.0], [0.0, 0.0, 0.0, 3.0]]))
    data.data[0] = 0.0
    columns = pd.MultiIndex.from_product([[0], [1, 2, 3, 4]], names=["time", "cell"])
    H = Jacobian(data, pd.Index(["a", "b"]), columns, empty=[], missing=[])

    frame = H.to_frame(sparse=True)

    assert list(frame.columns) == list(columns)
    assert list(frame.index) == ["a", "b"]
    arrays = [frame.iloc[:, j].array for j in range(frame.shape[1])]
    assert [a.fill_value for a in arrays] == [0.0] * 4
    assert [a.sp_index.to_int_index().indices.tolist() for a in arrays] == [
        [],
        [0],
        [],
        [0, 1],
    ]
    assert [a.sp_values.tolist() for a in arrays] == [[], [0.0], [], [1.0, 3.0]]
    np.testing.assert_array_equal(frame.sparse.to_dense().to_numpy(), H.data.toarray())


# ---------------------------------------------------------------------------
# Footprints as one dataset
# ---------------------------------------------------------------------------


def test_open_footprints_stacks_releases_on_the_hour(tmp_path):
    out = Output(tmp_path / "output")
    early, late, empty = release_at(hour=6), release_at(hour=18), release_at(hour=23)
    hours = {early.id: [-3, -1], late.id: [-2, -1, 0]}
    feet = {
        r.id: fake_footprint(r, hours=hours[r.id], seed=k)
        for k, r in enumerate((early, late))
    }
    for foot in feet.values():
        store_footprint(out, FEET, foot)
    store_empty_footprint(out, FEET, empty)
    paths = [out.path("footprint", FEET, r.id) for r in (late, empty, early)]

    ds = open_footprints(paths)

    assert list(ds.release.values) == [late.id, early.id]
    assert ds.attrs["empty"] == [empty.id]
    assert ds.attrs["stilt_name"] == "hrrr"
    assert ds.foot.dims == ("release", "hour", "lat", "lon")
    assert list(ds.hour.values) == [-3, -2, -1, 0]
    assert ds.foot.dtype == np.float32
    assert ds.foot.chunks is not None
    for rid, foot in feet.items():
        one = ds.sel(release=rid, hour=hours[rid])
        np.testing.assert_array_equal(one.time.values, foot.time.values)
        np.testing.assert_allclose(one.foot.values, foot.values, rtol=1e-6)
    # Hours a release has no layer for are zero, and still have a time.
    assert float(ds.foot.sel(release=early.id, hour=-2).sum()) == 0
    assert ds.time.sel(release=early.id, hour=0).values == np.datetime64(early.time)


def test_open_footprints_times_fifteen_minute_layers(tmp_path):
    out = Output(tmp_path / "output")
    release = release_at(hour=12)
    foot = fake_footprint(release, hours=(-3, -1), tres="15min")
    store_footprint(out, QUARTERS, foot)

    ds = open_footprints([out.path("footprint", QUARTERS, release.id)])

    assert ds.foot.dims == ("release", "minute", "lat", "lon")
    assert list(ds.minute.values) == [-45, -30, -15]
    np.testing.assert_array_equal(
        ds.time.sel(release=release.id).values,
        pd.DatetimeIndex(["2024-07-15 11:15", "2024-07-15 11:30", "2024-07-15 11:45"]),
    )
    one = ds.sel(release=release.id, minute=[-45, -15])
    np.testing.assert_allclose(one.foot.values, foot.values, rtol=1e-6)


#: FEET with 1-day and 6-hour footprint layers.
DAYS = output_variant(footprint=FootprintConfig(grid=GRID, tres="1D"))
SIX_HOURS = output_variant(footprint=FootprintConfig(grid=GRID, tres="6h"))


@pytest.mark.parametrize(
    ("variant", "tres", "dim", "values", "first"),
    [
        (DAYS, "1D", "day", [-2, -1], "2024-07-13 12:00"),
        (SIX_HOURS, "6h", "hour", [-12, -6], "2024-07-15 00:00"),
    ],
)
def test_open_footprints_counts_layers_in_the_unit_of_tres(
    tmp_path, variant, tres, dim, values, first
):
    out = Output(tmp_path / "output")
    release = release_at(hour=12)
    store_footprint(out, variant, fake_footprint(release, hours=(-2, -1), tres=tres))

    ds = open_footprints([out.path("footprint", variant, release.id)])

    assert ds.foot.dims == ("release", dim, "lat", "lon")
    assert list(ds[dim].values) == values
    assert ds.time.sel(release=release.id).values[0] == np.datetime64(first)


def test_open_footprints_reads_in_blocks_by_date_folder(tmp_path):
    out = Output(tmp_path / "output")
    releases = [
        release_at(6, day=15),
        release_at(12, day=15),
        release_at(6, day=16),
    ]
    for k, r in enumerate(releases):
        store_footprint(out, FEET, fake_footprint(r, seed=k))
    ds = open_footprints(out.path("footprint", FEET, r.id) for r in releases)
    assert ds.foot.chunks[0] == (2, 1)
    assert float(ds.foot.sum()) > 0


def test_open_footprints_refuses_mixed_settings(tmp_path):
    out = Output(tmp_path / "output")
    other = output_variant("fine", FootprintConfig(grid=GRID, smooth_factor=0.5))
    r = release_at()
    store_footprint(out, FEET, fake_footprint(r))
    store_footprint(out, other, fake_footprint(r, name="fine"))
    paths = [out.path("footprint", v, r.id) for v in (FEET, other)]
    with pytest.raises(ValueError, match="different settings"):
        open_footprints(paths)
    with pytest.raises(ValueError, match="No footprint files"):
        open_footprints([])


# ---------------------------------------------------------------------------
# Two workers writing the same file at once
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Which releases have results
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# What each file records
# ---------------------------------------------------------------------------


def test_each_result_file_names_its_settings_and_the_version_that_wrote_it(tmp_path):
    """A file copied out of the output directory still says where it came from."""
    import pyarrow.parquet as pq
    import runtraj

    import stilt

    release = release_at()
    out = Output(tmp_path / "output")
    particles = out.write_particles(FEET, release, fake_particles(release), [])
    footprint = store_footprint(out, FEET, fake_footprint(release))
    empty = store_empty_footprint(out, FEET, release_at(13))

    meta = pq.read_schema(particles).metadata
    assert meta[b"runtraj:hash"].decode() == FEET.particles_hash
    assert meta[b"runtraj:version"].decode() == runtraj.__version__
    for path in (footprint, empty):
        meta = pq.read_schema(path).metadata
        assert meta[b"runtraj:hash"].decode() == FEET.stage_hashes.get("footprint")
        assert meta[b"runtraj:version"].decode() == runtraj.__version__
        assert meta[b"stilt:pystilt"].decode() == stilt.__version__


def test_a_stored_footprint_file_opens_on_its_own(tmp_path):
    """The file records its grid, so read_footprint needs nothing else (#107)."""
    import shutil

    from stilt.footprint import read_footprint

    release = release_at()
    path = store_footprint(
        Output(tmp_path / "output"), FEET, fake_footprint(release, seed=3)
    )

    copied = tmp_path / "elsewhere.parquet"  # away from the folder's _settings.yaml
    shutil.copy(path, copied)
    alone = read_footprint(copied)
    assert alone is not None
    xr.testing.assert_identical(alone, read_footprint(path))
    assert alone.stilt.grid == GRID
    assert alone.stilt.release == release


def test_a_footprint_file_without_its_settings_is_refused(tmp_path):
    """Each file is read alone, so a file must record its settings."""
    import pyarrow.parquet as pq

    from stilt.footprint import read_footprint

    path = store_footprint(
        Output(tmp_path / "output"), FEET, fake_footprint(release_at(), seed=3)
    )
    table = pq.ParquetFile(path).read()
    meta = {k: v for k, v in table.schema.metadata.items() if k != b"stilt:footprint"}
    pq.write_table(table.replace_schema_metadata(meta), path)

    with pytest.raises(ValueError, match="does not record its footprint settings"):
        read_footprint(path)


def test_a_footprint_folder_stored_with_projection_is_found_by_crs(tmp_path):
    """Folders written when the grid said ``projection`` are found by a ``crs`` config."""
    variant = output_variant(footprint=FootprintConfig(grid=GRID))
    out = Output(tmp_path / "output")
    store_empty_footprint(out, variant, release_at())
    folder = out.folder("footprint", variant)
    record_path = folder / "_settings.yaml"
    record = yaml.safe_load(record_path.read_text())
    grid = record["settings"]["grid"]
    grid["projection"] = grid.pop("crs")
    record_path.write_text(yaml.safe_dump(record))

    assert Output(tmp_path / "output").folder("footprint", variant) == folder


def test_a_footprint_folder_reads_its_particles_folder_once(tmp_path, monkeypatch):
    out = Output(tmp_path / "output")
    store_empty_footprint(out, FEET, release_at())

    import runtraj.output

    reads: list[int] = []
    real = runtraj.output.read_run_settings
    monkeypatch.setattr(
        runtraj.output, "read_run_settings", lambda s: reads.append(1) or real(s)
    )
    fresh = Output(tmp_path / "output")
    assert fresh.folder("footprint", FEET) == out.folder("footprint", FEET)
    assert fresh.folder("particles", FEET) == out.folder("particles", FEET)
    assert len(reads) == 1


def test_a_stored_footprint_says_what_made_it(tmp_path):
    from stilt.footprint import read_footprint

    out = Output(tmp_path / "output")
    release = release_at()
    path = store_footprint(out, FEET, fake_footprint(release))

    foot = read_footprint(path)

    assert foot is not None
    assert foot.attrs["stilt_particles_hash"] == FEET.particles_hash
    assert foot.attrs["stilt_version"]
    nc = foot.stilt.to_netcdf(tmp_path / "foot.nc")
    back = read_footprint(nc)
    assert back is not None
    assert back.attrs["stilt_particles_hash"] == FEET.particles_hash
