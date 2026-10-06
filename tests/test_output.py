"""Tests for stilt.output: the output directory of runs and footprints."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from stilt.footprint import jacobian
from stilt.footprint.config import FootprintConfig
from stilt.footprint.targets import Mesh
from stilt.identity import footprint_hash, footprint_settings, settings_hash
from stilt.meteorology import MetConfig
from stilt.output import Output
from stilt.particles import particles_metadata, write_particles
from stilt.receptors import PointReceptor
from stilt.spatial import Grid
from stilt.transport import ModelInfo
from stilt.transport.hysplit import HysplitConfig
from stilt.variants import Variant

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
    return Variant(
        name=name,
        group=name,
        met="hrrr",
        met_config=MET,
        transport=HysplitConfig(**{"n_hours": -24, "numpar": 100, **overrides}),
        model=ModelInfo(version="v5.1.0"),
        footprint=footprint,
    )


VARIANT = _variant()
#: VARIANT with footprints on GRID.
FEET = _variant(footprint=FootprintConfig(grid=GRID))


def _receptor(hour: int = 12, day: int = 15) -> PointReceptor:
    return PointReceptor(
        time=dt.datetime(2024, 7, day, hour),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


def _trajectories(receptor: PointReceptor, n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(int(receptor.time.timestamp()) % 1000)
    steps = np.arange(-1, -11, -1, dtype=float)
    indx = np.repeat(np.arange(1, n + 1, dtype=float), len(steps))
    time = np.tile(steps, n)
    data = pd.DataFrame(
        {
            "time": time,
            "indx": indx,
            "long": -111.85 + rng.normal(0, 0.1, len(time)),
            "lati": 40.77 + rng.normal(0, 0.1, len(time)),
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
# Identity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_footprint_files_record_the_geometry_hash(tmp_path):
    """A footprint read back from its folder knows the geometry its grid was derived for."""
    from dataclasses import replace

    out = Output(tmp_path / "output")
    spec = {"kind": "windows", "coords": [(-111.85, 40.77)], "size": 0.2}
    variant = replace(
        VARIANT,
        footprint=FootprintConfig(grid=GRID, geometry=spec),
        geometry_hash="deadbeef00",
    )
    feet = out.footprints(variant)
    assert feet.geometry_hash == "deadbeef00"
    assert out.find_footprints(variant) == feet
    receptor = _receptor()
    feet.write(_footprint(receptor))
    feet.write_empty(_receptor(hour=13), "outside_domain")
    assert feet.read(str(receptor.id)).stilt.geometry_hash == "deadbeef00"
    record = yaml.safe_load((feet.path / "_settings.yaml").read_text())
    assert record["settings"]["geometry_hash"] == "deadbeef00"
    # The same footprint config without the geometry hash is another folder.
    assert out.footprints(replace(variant, geometry_hash=None)) != feet


def test_run_folder_is_name_and_hash_with_settings_file(tmp_path):
    out = Output(tmp_path / "output")
    run = out.particles(VARIANT)
    digest = VARIANT.particles_hash
    assert run.key == f"hrrr-{digest[:6]}"
    assert run.path == tmp_path / "output" / "particles" / f"settings=hrrr-{digest[:6]}"
    assert run.logs_dir == tmp_path / "output" / "logs" / f"settings=hrrr-{digest[:6]}"
    record = yaml.safe_load((run.path / "_settings.yaml").read_text())
    assert record["name"] == "hrrr"
    assert record["hash"] == digest
    assert record["settings"]["numpar"] == 100
    assert (
        "exe_dir" not in record["settings"]
        and "directory" not in record["settings"]["met"]
    )
    assert run.settings == VARIANT.run_settings  # maxpar is stored as numpar
    assert [r.path for r in out.particle_sets()] == [run.path]


def test_same_settings_under_another_name_share_the_folder(tmp_path):
    out = Output(tmp_path / "output")
    first = out.particles(VARIANT)
    second = out.particles(_variant("hrrr-main"))
    assert second.path == first.path
    assert second.name == "hrrr"


def test_changed_settings_make_a_new_folder_beside_the_old(tmp_path):
    out = Output(tmp_path / "output")
    first = out.particles(VARIANT)
    second = out.particles(_variant(ziscale=0.8))
    assert second.path != first.path
    assert second.key.startswith("hrrr-")
    assert {r.path for r in out.particle_sets()} == {first.path, second.path}
    assert out.find_particles(VARIANT).path == first.path
    assert out.find_particles(_variant(numpar=7)) is None


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------


def test_particles_round_trip_in_date_folders(tmp_path):
    run = Output(tmp_path / "output").particles(VARIANT)
    receptor = _receptor()
    traj = _trajectories(receptor)
    path = run.write(receptor, traj, [])
    assert path == run.path / "date=2024-07-15" / f"{receptor.id}.parquet"
    assert run.has(str(receptor.id))
    assert list(run.files()) == [str(receptor.id)]

    back = run.read(str(receptor.id))
    meta = particles_metadata(path)
    assert meta.receptor == receptor
    assert meta.settings == run.settings
    pd.testing.assert_frame_equal(
        back[traj.columns], traj, check_dtype=True, check_like=True
    )


def test_particles_store_time_and_index_as_int32(tmp_path):
    import pyarrow.parquet as pq

    run = Output(tmp_path / "output").particles(VARIANT)
    receptor = _receptor()
    traj = _trajectories(receptor)
    path = run.write(receptor, traj, [])
    schema = pq.read_schema(path)
    assert str(schema.field("time").type) == "int32"
    assert str(schema.field("indx").type) == "int32"
    assert "datetime" not in schema.names
    assert str(schema.field("long").type) == "double"


def test_particle_files_name_their_receptor_in_a_column(tmp_path):
    """A scan of the particles tree tells receptors apart; one file reads without it (#105)."""
    import pyarrow.dataset as pads

    run = Output(tmp_path / "output").particles(VARIANT)
    receptors = [_receptor(hour=6), _receptor(hour=18, day=16)]
    for receptor in receptors:
        run.write(receptor, _trajectories(receptor, n=10), [])

    from collections import Counter

    scan = pads.dataset(run.path, format="parquet").to_table()
    counts = Counter(scan.column("receptor").to_pylist())
    assert counts == {str(r.id): 100 for r in receptors}

    back = run.read(str(receptors[0].id))
    assert "receptor" not in back.columns


def test_particle_file_without_a_receptor_column_still_reads(tmp_path):
    """Files written before the receptor column read as before."""
    run = Output(tmp_path / "output").particles(VARIANT)
    receptor = _receptor()
    traj = _trajectories(receptor)
    import pyarrow.parquet as pq

    path = write_particles(run.file(str(receptor.id)), traj, receptor, run.settings, [])
    table = pq.ParquetFile(path).read()
    old = table.drop_columns(["receptor"]).replace_schema_metadata(
        table.schema.metadata
    )
    pq.write_table(old, path)
    back = run.read(str(receptor.id))
    pd.testing.assert_frame_equal(back[traj.columns], traj, check_like=True)


def test_particles_reject_fractional_time(tmp_path):
    run = Output(tmp_path / "output").particles(VARIANT)
    receptor = _receptor()
    traj = _trajectories(receptor)
    traj.loc[0, "time"] = -1.5
    with pytest.raises(ValueError, match="whole numbers"):
        run.write(receptor, traj, [])


def test_receptors_listed_in_date_order(tmp_path):
    run = Output(tmp_path / "output").particles(VARIANT)
    for day, hour in [(16, 0), (15, 18), (15, 6)]:
        run.write(
            _receptor(hour=hour, day=day),
            _trajectories(_receptor(hour=hour, day=day)),
            [],
        )
    ids = list(run.files())
    assert ids == sorted(ids)
    assert [i[:10] for i in ids] == ["2024071506", "2024071518", "2024071600"]


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------


def test_footprint_folder_is_variant_name_and_combined_hash(tmp_path):
    out = Output(tmp_path / "output")
    run = out.particles(VARIANT)
    config = FootprintConfig(grid=GRID, smooth_factor=1.0)
    feet = out.footprints(_variant(footprint=config))
    digest = footprint_hash(run.hash, footprint_settings(config, None))
    assert digest == settings_hash(
        {
            "particles": VARIANT.particles_hash,
            "footprint": {**config.model_dump(mode="json"), "geometry_hash": None},
        }
    )
    assert feet.key == f"hrrr-{digest[:6]}"
    assert feet.path == out.path / "footprints" / f"settings=hrrr-{digest[:6]}"
    assert feet.config == config
    assert feet.grid == GRID
    assert feet.particles == run
    record = yaml.safe_load((feet.path / "_settings.yaml").read_text())
    assert record["particles"] == run.key

    smooth = config.model_copy(update={"smooth_factor": 0.5})
    other = out.footprints(_variant("hrrr-smooth", footprint=smooth))
    assert other.path != feet.path
    assert other.key.startswith("hrrr-smooth-")
    assert set(out.footprint_sets()) == {feet, other}
    assert {f.particles_key for f in out.footprint_sets()} == {run.key}
    # The same settings under another variant name reuse the folder.
    assert out.footprints(_variant("renamed", footprint=config)) == feet

    # The same footprint settings on other particles are another folder.
    other_run = out.footprints(_variant(footprint=config, numpar=200))
    assert other_run.path != feet.path


def test_footprint_round_trip_is_exact_in_float32(tmp_path):
    run = Output(tmp_path / "output").particles(VARIANT)
    feet = run.output.footprints(FEET)
    receptor = _receptor()
    foot = _footprint(receptor, hours=(-3, -2, -1, 0), seed=1)
    path = feet.write(foot)
    assert path == feet.path / "date=2024-07-15" / f"{receptor.id}.parquet"
    assert feet.path.parent == tmp_path / "output" / "footprints"

    back = feet.read(str(receptor.id))
    assert back is not None
    assert back.stilt.receptor == receptor
    assert back.stilt.name == "hrrr"
    assert list(back.dims) == ["time", "lat", "lon"]
    assert back.shape == foot.shape
    np.testing.assert_array_equal(back["time"].values, foot["time"].values)
    np.testing.assert_array_equal(
        back.values, foot.values.astype(np.float32).astype(np.float64)
    )
    assert feet.empty_reason(str(receptor.id)) is None


def test_footprint_all_zero_layer_keeps_its_shape(tmp_path):
    run = Output(tmp_path / "output").particles(VARIANT)
    feet = run.output.footprints(FEET)
    foot = _footprint(_receptor(), hours=(-2, -1, 0))
    foot[1] = 0.0
    feet.write(foot)
    back = feet.read(str(foot.stilt.receptor.id))
    assert back.shape == foot.shape
    assert float(back[1].sum()) == 0.0


def test_empty_footprint_is_a_file_with_no_rows_and_a_reason(tmp_path):
    run = Output(tmp_path / "output").particles(VARIANT)
    feet = run.output.footprints(FEET)
    receptor = _receptor()
    feet.write_empty(receptor, "outside_domain")
    assert feet.has(str(receptor.id))
    assert feet.read(str(receptor.id)) is None
    assert feet.empty_reason(str(receptor.id)) == "outside_domain"
    assert list(feet.files()) == [str(receptor.id)]


def test_footprint_coordinates_off_by_rounding_still_match(tmp_path):
    """A stored footprint's axis can differ from the grid's by 1e-14, as real files do."""
    run = Output(tmp_path / "output").particles(VARIANT)
    feet = run.output.footprints(FEET)
    foot = _footprint(_receptor(), seed=5)
    nudged = foot.assign_coords(
        lat=foot["lat"].values + 5e-15, lon=foot["lon"].values - 5e-15
    )
    feet.write(nudged)
    back = feet.read(str(foot.stilt.receptor.id))
    np.testing.assert_array_equal(back.values, foot.values.astype(np.float32))


def test_footprint_on_another_grid_is_rejected(tmp_path):
    run = Output(tmp_path / "output").particles(VARIANT)
    feet = run.output.footprints(FEET)
    foot = _footprint(_receptor())
    shifted = foot.assign_coords(lon=foot["lon"].values + 0.03)
    with pytest.raises(ValueError, match="not cells of the grid"):
        feet.write(shifted)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


@pytest.fixture
def written_footprints(tmp_path):
    run = Output(tmp_path / "output").particles(VARIANT)
    feet = run.output.footprints(FEET)
    feet_by_id = {}
    for k, hour in enumerate([6, 12, 18]):
        receptor = _receptor(hour=hour)
        foot = _footprint(receptor, hours=(-3, -2, -1, 0), seed=10 + k)
        feet.write(foot)
        feet_by_id[str(receptor.id)] = foot
    empty = _receptor(hour=23)
    feet.write_empty(empty, "outside_domain")
    return feet, feet_by_id, str(empty.id)


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
    feet, feet_by_id, empty_id = written_footprints
    bins = _bins()
    H = jacobian(feet.table(), feet.config, target, bins, [*feet_by_id, empty_id])

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

    feet, feet_by_id, empty_id = written_footprints
    table = feet.table()
    assert table.schema.field("date").type == pa.date32()
    assert set(table.column("date").to_pylist()) == {dt.date(2024, 7, 15)}
    assert set(table.column("receptor").to_pylist()) == set(feet_by_id)
    assert feet.table({}).num_rows == 0
    assert feet.table({}).schema.field("date").type == pa.date32()


def test_jacobian_rows_are_the_requested_receptors(written_footprints):
    """Cells of receptors in the table that were not asked for are left out."""
    feet, feet_by_id, empty_id = written_footprints
    ids = list(feet_by_id)
    target = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.25, yres=0.25)
    H = jacobian(feet.table(), feet.config, target, _bins(), [ids[1], empty_id])
    alone = jacobian(
        feet.table(feet.files([ids[1]])), feet.config, target, _bins(), [ids[1]]
    )
    assert list(H.receptors) == [ids[1]]
    assert H.empty == [empty_id]
    assert H.missing == []
    np.testing.assert_array_equal(H.data.toarray(), alone.data.toarray())


@pytest.mark.parametrize("closed", ["right", "both", "neither"])
def test_jacobian_rejects_bins_not_closed_on_the_left(written_footprints, closed):
    """A right-closed bin would put footprint hours in the wrong bin without a word."""
    feet, _, _ = written_footprints
    left = _bins()
    bins = pd.IntervalIndex.from_arrays(left.left, left.right, closed=closed)
    with pytest.raises(ValueError, match="closed on the left"):
        jacobian(feet.table(), feet.config, GRID, bins, [])


def test_jacobian_with_no_footprints_is_empty(tmp_path):
    run = Output(tmp_path / "output").particles(VARIANT)
    feet = run.output.footprints(FEET)
    H = jacobian(feet.table(), feet.config, GRID, _bins(), [])
    assert H.data.shape == (0, len(_bins()) * len(GRID.index))
    assert list(H.receptors) == []


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
    _interleave(monkeypatch, lambda: Output(path).particles(VARIANT))

    run = Output(path).particles(VARIANT)

    record = yaml.safe_load((run.path / "_settings.yaml").read_text())
    assert record["hash"] == VARIANT.particles_hash
    assert [p.name for p in run.path.iterdir()] == ["_settings.yaml"]


def test_two_workers_writing_one_receptor_at_once_both_succeed(tmp_path, monkeypatch):
    """The other worker's cleanup removed the shared temporary file."""
    receptor = _receptor()
    traj = _trajectories(receptor)
    run = Output(tmp_path / "output").particles(VARIANT)
    other = Output(tmp_path / "output").particles(VARIANT)
    _interleave(monkeypatch, lambda: other.write(receptor, traj, []))

    path = run.write(receptor, traj, [])

    assert len(run.read(str(receptor.id))) == len(traj)
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no stray files


# ---------------------------------------------------------------------------
# Which receptors have results
# ---------------------------------------------------------------------------


def test_receptors_among_lists_only_the_date_folders_asked_for(tmp_path, monkeypatch):
    """A few simulations of a large project must not list every date folder."""
    import stilt.output as output_module

    run = Output(tmp_path / "output").particles(VARIANT)
    day15, day16 = _receptor(12, day=15), _receptor(12, day=16)
    for receptor in (day15, day16):
        run.write(receptor, _trajectories(receptor), [])
    never_run = _receptor(12, day=20)

    listed: list[str] = []
    real_scandir = output_module.os.scandir

    def scandir(path):
        listed.append(Path(path).name)
        return real_scandir(path)

    monkeypatch.setattr(output_module.os, "scandir", scandir)
    found = list(run.files(among=[day15.id, never_run.id]))

    assert found == [day15.id]
    assert sorted(listed) == ["date=2024-07-15", "date=2024-07-20"]
    assert list(run.files()) == [day15.id, day16.id]


# ---------------------------------------------------------------------------
# Provenance in each file
# ---------------------------------------------------------------------------


def test_each_result_file_names_its_settings_and_the_version_that_wrote_it(tmp_path):
    """A file copied out of the output directory still says where it came from."""
    import pyarrow.parquet as pq

    import stilt

    receptor = _receptor()
    run = Output(tmp_path / "output").particles(VARIANT)
    particles = run.write(receptor, _trajectories(receptor), [])
    feet = run.output.footprints(FEET)
    footprint = feet.write(_footprint(receptor))
    empty = feet.write_empty(_receptor(13), "outside_domain")

    meta = pq.read_schema(particles).metadata
    assert meta[b"stilt:hash"].decode() == run.hash == VARIANT.particles_hash
    assert meta[b"stilt:pystilt"].decode() == stilt.__version__
    for path in (footprint, empty):
        meta = pq.read_schema(path).metadata
        assert meta[b"stilt:hash"].decode() == feet.hash
        assert meta[b"stilt:pystilt"].decode() == stilt.__version__


def test_a_stored_footprint_file_opens_on_its_own(tmp_path):
    """The file records its grid, so read_footprint needs nothing else (#107)."""
    import shutil

    from stilt.footprint import read_footprint

    run = Output(tmp_path / "output").particles(VARIANT)
    feet = run.output.footprints(FEET)
    receptor = _receptor()
    path = feet.write(_footprint(receptor, seed=3))

    copied = tmp_path / "elsewhere.parquet"  # away from the folder's _settings.yaml
    shutil.copy(path, copied)
    alone = read_footprint(copied)
    assert alone is not None
    xr.testing.assert_identical(alone, feet.read(str(receptor.id)))
    assert alone.stilt.grid == GRID
    assert alone.stilt.receptor == receptor


def test_a_footprint_file_without_its_settings_is_refused(tmp_path):
    """The folder reads each file alone, so a file must record its settings."""
    import pyarrow.parquet as pq

    run = Output(tmp_path / "output").particles(VARIANT)
    feet = run.output.footprints(FEET)
    receptor = _receptor()
    path = feet.write(_footprint(receptor, seed=3))
    table = pq.ParquetFile(path).read()
    meta = {k: v for k, v in table.schema.metadata.items() if k != b"stilt:footprint"}
    pq.write_table(table.replace_schema_metadata(meta), path)

    with pytest.raises(ValueError, match="does not record its footprint settings"):
        feet.read(str(receptor.id))


def test_a_footprint_folder_stored_with_projection_is_found_by_crs(tmp_path):
    """Folders written when the grid said ``projection`` are found by a ``crs`` config."""
    config = FootprintConfig(grid=GRID)
    feet = Output(tmp_path / "output").footprints(_variant(footprint=config))
    record_path = feet.path / "_settings.yaml"
    record = yaml.safe_load(record_path.read_text())
    grid = record["settings"]["grid"]
    grid["projection"] = grid.pop("crs")
    record_path.write_text(yaml.safe_dump(record))

    again = Output(tmp_path / "output").footprints(_variant(footprint=config))
    assert again.key == feet.key


def test_a_lookup_reads_each_folder_once(tmp_path, monkeypatch):
    """A lookup that misses lists the tree again but reads only the folders it has not seen."""
    out = Output(tmp_path / "output")
    out.particles(VARIANT)
    out.particles(_variant(ziscale=0.8))

    import stilt.output

    reads: list[str] = []
    real = stilt.output.read_run_settings

    def counting(stored):
        reads.append(stored["numpar"])
        return real(stored)

    monkeypatch.setattr(stilt.output, "read_run_settings", counting)
    fresh = Output(tmp_path / "output")
    for _ in range(3):  # a variant that has not run misses every time
        assert fresh.find_particles(_variant(numpar=7)) is None
    assert len(reads) == 2  # each existing folder once

    # A folder another worker creates is found on the next lookup, which
    # reads that folder alone.
    Output(tmp_path / "output").particles(_variant(numpar=7))
    before = len(reads)
    assert fresh.find_particles(_variant(numpar=7)) is not None
    assert len(reads) == before + 1


def test_a_footprint_folder_finds_its_particles_folder_in_the_cache(tmp_path):
    out = Output(tmp_path / "output")
    run = out.particles(VARIANT)
    feet = run.output.footprints(FEET)
    fresh = Output(tmp_path / "output")
    found = fresh.footprint_sets()[0]
    assert found == feet
    assert found.particles is fresh.particle_sets()[0]
