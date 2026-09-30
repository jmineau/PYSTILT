"""Tests for stilt.output: the output directory of runs and footprints."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from stilt.config import (
    FootprintConfig,
    Grid,
    MetConfig,
    STILTParams,
    TransportSettings,
)
from stilt.config.transport import settings_hash
from stilt.footprint import Footprint
from stilt.geometry import Mesh
from stilt.output import Footprints, Output
from stilt.receptors import PointReceptor
from stilt.trajectory import Trajectories

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.1, yres=0.1)
MET = MetConfig(directory="/data/hrrr", file_format="%Y%m%d_%H", file_tres="6h")


def _settings(**overrides) -> TransportSettings:
    """Transport settings for the tests; *overrides* change the transport fields."""
    params = STILTParams(**{"n_hours": -24, "numpar": 100, **overrides})
    return TransportSettings.build(params, MET)


SETTINGS = _settings()


def _receptor(hour: int = 12, day: int = 15) -> PointReceptor:
    return PointReceptor(
        time=dt.datetime(2024, 7, day, hour),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )


def _trajectories(receptor: PointReceptor, n: int = 50) -> Trajectories:
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
    return Trajectories(
        receptor=receptor,
        params=STILTParams(n_hours=-24, numpar=n),
        met_files=[],
        data=data,
    )


def _footprint(
    receptor: PointReceptor, hours=(-2, -1, 0), seed: int = 0, name: str = "hrrr"
) -> Footprint:
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
    return Footprint(
        receptor=receptor, config=FootprintConfig(grid=GRID), data=data, name=name
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_run_folder_is_name_and_hash_with_settings_file(tmp_path):
    out = Output(tmp_path / "output")
    run = out.run("hrrr", SETTINGS)
    digest = SETTINGS.hash
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
    assert run.settings.identity() == SETTINGS.identity()  # maxpar is stored as numpar
    assert [r.path for r in out.runs()] == [run.path]


def test_same_settings_under_another_name_share_the_folder(tmp_path):
    out = Output(tmp_path / "output")
    first = out.run("hrrr", SETTINGS)
    second = out.run("hrrr-main", _settings())
    assert second.path == first.path
    assert second.name == "hrrr"


def test_changed_settings_make_a_new_folder_beside_the_old(tmp_path):
    out = Output(tmp_path / "output")
    first = out.run("hrrr", SETTINGS)
    second = out.run("hrrr", _settings(ziscale=0.8))
    assert second.path != first.path
    assert second.key.startswith("hrrr-")
    assert {r.path for r in out.runs()} == {first.path, second.path}
    assert out.find_run(SETTINGS).path == first.path
    assert out.find_run(_settings(numpar=7)) is None


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------


def test_particles_round_trip_in_date_folders(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    receptor = _receptor()
    traj = _trajectories(receptor)
    path = run.write_particles(traj)
    assert path == run.particles_dir / "date=2024-07-15" / f"{receptor.id}.parquet"
    assert run.has_particles(str(receptor.id))
    assert run.receptors() == [str(receptor.id)]

    back = run.read_particles(str(receptor.id))
    assert back.receptor == receptor
    assert back.params == traj.params
    pd.testing.assert_frame_equal(
        back.data[traj.data.columns], traj.data, check_dtype=True, check_like=True
    )


def test_particles_store_time_and_index_as_int32(tmp_path):
    import pyarrow.parquet as pq

    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    traj = _trajectories(_receptor())
    path = run.write_particles(traj)
    schema = pq.read_schema(path)
    assert str(schema.field("time").type) == "int32"
    assert str(schema.field("indx").type) == "int32"
    assert "datetime" not in schema.names
    assert str(schema.field("long").type) == "double"


def test_particles_reject_fractional_time(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    traj = _trajectories(_receptor())
    traj.data.loc[0, "time"] = -1.5
    with pytest.raises(ValueError, match="whole numbers"):
        run.write_particles(traj)


def test_receptors_listed_in_date_order(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    for day, hour in [(16, 0), (15, 18), (15, 6)]:
        run.write_particles(_trajectories(_receptor(hour=hour, day=day)))
    ids = run.receptors()
    assert ids == sorted(ids)
    assert [i[:10] for i in ids] == ["2024071506", "2024071518", "2024071600"]


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------


def test_footprint_folder_is_variant_name_and_combined_hash(tmp_path):
    out = Output(tmp_path / "output")
    run = out.run("hrrr", SETTINGS)
    config = FootprintConfig(grid=GRID, smooth_factor=1.0)
    feet = run.footprints(config)
    digest = Footprints.hash_for(run.hash, config)
    assert digest == settings_hash(
        {"particles": SETTINGS.hash, "footprint": config.model_dump(mode="json")}
    )
    assert feet.key == f"hrrr-{digest[:6]}"
    assert feet.path == out.footprints_dir / f"settings=hrrr-{digest[:6]}"
    assert feet.config == config
    assert feet.grid == GRID
    assert feet.run == run
    record = yaml.safe_load((feet.path / "_settings.yaml").read_text())
    assert record["particles"] == run.key

    other = run.footprints(
        config.model_copy(update={"smooth_factor": 0.5}), name="hrrr-smooth"
    )
    assert other.path != feet.path
    assert other.key.startswith("hrrr-smooth-")
    assert set(run.footprint_sets()) == {feet, other}
    assert set(out.footprint_sets()) == {feet, other}
    assert run.footprints(config) == feet

    # The same footprint settings on other particles are another folder.
    other_run = out.run("hrrr", _settings(numpar=200))
    assert other_run.footprints(config).path != feet.path
    assert other_run.footprint_sets() != run.footprint_sets()


def test_footprint_round_trip_is_exact_in_float32(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    feet = run.footprints(FootprintConfig(grid=GRID))
    receptor = _receptor()
    foot = _footprint(receptor, hours=(-3, -2, -1, 0), seed=1)
    path = feet.write(foot)
    assert path == feet.path / "date=2024-07-15" / f"{receptor.id}.parquet"
    assert feet.path.parent == Output(tmp_path / "output").footprints_dir

    back = feet.read(str(receptor.id))
    assert back is not None
    assert back.receptor == receptor
    assert back.name == "hrrr"
    assert list(back.data.dims) == ["time", "lat", "lon"]
    assert back.data.shape == foot.data.shape
    np.testing.assert_array_equal(back.data["time"].values, foot.data["time"].values)
    np.testing.assert_array_equal(
        back.data.values, foot.data.values.astype(np.float32).astype(np.float64)
    )
    assert feet.empty_reason(str(receptor.id)) is None


def test_footprint_all_zero_layer_keeps_its_shape(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    feet = run.footprints(FootprintConfig(grid=GRID))
    foot = _footprint(_receptor(), hours=(-2, -1, 0))
    foot.data[1] = 0.0
    feet.write(foot)
    back = feet.read(str(foot.receptor.id))
    assert back.data.shape == foot.data.shape
    assert float(back.data[1].sum()) == 0.0


def test_empty_footprint_is_a_file_with_no_rows_and_a_reason(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    feet = run.footprints(FootprintConfig(grid=GRID))
    receptor = _receptor()
    feet.write_empty(receptor, "outside_domain")
    assert feet.has(str(receptor.id))
    assert feet.read(str(receptor.id)) is None
    assert feet.empty_reason(str(receptor.id)) == "outside_domain"
    assert feet.receptors() == [str(receptor.id)]


def test_footprint_coordinates_off_by_rounding_still_match(tmp_path):
    """A stored footprint's axis can differ from the grid's by 1e-14, as real files do."""
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    feet = run.footprints(FootprintConfig(grid=GRID))
    foot = _footprint(_receptor(), seed=5)
    nudged = foot.data.assign_coords(
        lat=foot.data["lat"].values + 5e-15, lon=foot.data["lon"].values - 5e-15
    )
    feet.write(Footprint(receptor=foot.receptor, config=foot.config, data=nudged))
    back = feet.read(str(foot.receptor.id))
    np.testing.assert_array_equal(back.data.values, foot.data.values.astype(np.float32))


def test_footprint_on_another_grid_is_rejected(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    feet = run.footprints(FootprintConfig(grid=GRID))
    foot = _footprint(_receptor())
    shifted = foot.data.assign_coords(lon=foot.data["lon"].values + 0.03)
    with pytest.raises(ValueError, match="not cells of the folder's grid"):
        feet.write(Footprint(receptor=foot.receptor, config=foot.config, data=shifted))


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


@pytest.fixture
def written_footprints(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    feet = run.footprints(FootprintConfig(grid=GRID))
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
    H = feet.jacobian(target, bins)

    assert list(H.receptors) == list(feet_by_id)
    assert H.empty == [empty_id]
    assert H.missing == []
    assert H.data.shape == (3, len(bins) * len(target.index))
    assert H.columns.names == ["time", "cell"]

    frame = H.to_frame()
    for rid, foot in feet_by_id.items():
        expected = foot.aggregate(target, bins)  # (cells × bins)
        got = frame.loc[rid].to_numpy().reshape(len(bins), len(target.index)).T
        np.testing.assert_allclose(got, expected.to_numpy(), rtol=1e-6, atol=1e-12)


def test_table_reads_the_date_folder_as_a_date32_column(written_footprints):
    import pyarrow as pa

    feet, feet_by_id, empty_id = written_footprints
    table = feet.table()
    assert table.schema.field("date").type == pa.date32()
    assert set(table.column("date").to_pylist()) == {dt.date(2024, 7, 15)}
    assert set(table.column("receptor").to_pylist()) == set(feet_by_id)
    assert feet.table([]).num_rows == 0
    assert feet.table([]).schema.field("date").type == pa.date32()


def test_jacobian_selection_and_missing(written_footprints):
    feet, feet_by_id, empty_id = written_footprints
    ids = list(feet_by_id)
    target = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.25, yres=0.25)
    H = feet.jacobian(
        target, _bins(), receptors=[ids[1], "202407151300_-111.85_40.77_5", empty_id]
    )
    assert list(H.receptors) == [ids[1]]
    assert H.empty == [empty_id]
    assert H.missing == ["202407151300_-111.85_40.77_5"]


@pytest.mark.parametrize("closed", ["right", "both", "neither"])
def test_jacobian_rejects_bins_not_closed_on_the_left(written_footprints, closed):
    """A right-closed bin would put footprint hours in the wrong bin without a word."""
    feet, _, _ = written_footprints
    left = _bins()
    bins = pd.IntervalIndex.from_arrays(left.left, left.right, closed=closed)
    with pytest.raises(ValueError, match="closed on the left"):
        feet.jacobian(GRID, bins)


def test_jacobian_with_no_footprints_is_empty(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    feet = run.footprints(FootprintConfig(grid=GRID))
    H = feet.jacobian(GRID, _bins())
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
    _interleave(monkeypatch, lambda: Output(path).run("hrrr", SETTINGS))

    run = Output(path).run("hrrr", SETTINGS)

    record = yaml.safe_load((run.particles_dir / "_settings.yaml").read_text())
    assert record["hash"] == SETTINGS.hash
    assert [p.name for p in run.particles_dir.iterdir()] == ["_settings.yaml"]


def test_two_workers_writing_one_receptor_at_once_both_succeed(tmp_path, monkeypatch):
    """The other worker's cleanup removed the shared temporary file."""
    receptor = _receptor()
    traj = _trajectories(receptor)
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    other = Output(tmp_path / "output").run("hrrr", SETTINGS)
    _interleave(monkeypatch, lambda: other.write_particles(traj))

    path = run.write_particles(traj)

    assert len(run.read_particles(str(receptor.id)).data) == len(traj.data)
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no stray files


# ---------------------------------------------------------------------------
# Which receptors have results
# ---------------------------------------------------------------------------


def test_receptors_among_lists_only_the_date_folders_asked_for(tmp_path, monkeypatch):
    """A few simulations of a large project must not list every date folder."""
    import stilt.output as output_module

    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    day15, day16 = _receptor(12, day=15), _receptor(12, day=16)
    for receptor in (day15, day16):
        run.write_particles(_trajectories(receptor))
    never_run = _receptor(12, day=20)

    listed: list[str] = []
    real_scandir = output_module.os.scandir

    def scandir(path):
        listed.append(Path(path).name)
        return real_scandir(path)

    monkeypatch.setattr(output_module.os, "scandir", scandir)
    found = run.receptors(among=[day15.id, never_run.id])

    assert found == [day15.id]
    assert sorted(listed) == ["date=2024-07-15", "date=2024-07-20"]
    assert run.receptors() == [day15.id, day16.id]
