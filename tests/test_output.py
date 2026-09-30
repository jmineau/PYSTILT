"""Tests for stilt.output: the output directory of runs and footprints."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from stilt.config import FootprintConfig, Grid, STILTParams
from stilt.footprint import Footprint
from stilt.geometry import Mesh
from stilt.output import Output, footprint_label, settings_hash
from stilt.receptors import PointReceptor
from stilt.trajectory import Trajectories

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID = Grid(xmin=-112.0, xmax=-111.5, ymin=40.5, ymax=41.0, xres=0.1, yres=0.1)
SETTINGS = {"n_hours": -24, "numpar": 100, "met": {"source": "hrrr", "file_tres": "6h"}}


def _receptor(hour: int = 12, day: int = 15) -> PointReceptor:
    return PointReceptor(dt.datetime(2024, 7, day, hour), -111.85, 40.77, 5.0)


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


def test_settings_hash_ignores_key_order_and_number_spelling():
    a = {"numpar": 1000, "ziscale": 1.0, "grid": {"xres": 0.01, "yres": 0.01}}
    b = {"grid": {"yres": 0.01, "xres": 0.01}, "ziscale": 1, "numpar": 1000.0}
    assert settings_hash(a) == settings_hash(b)


def test_settings_hash_changes_with_any_value():
    base = {"numpar": 1000, "ziscale": 1.0}
    assert settings_hash(base) != settings_hash({**base, "ziscale": 0.8})
    assert settings_hash(base) != settings_hash({**base, "extra": None})


def test_footprint_label_from_grid():
    assert footprint_label(GRID) == "0.1deg"
    assert footprint_label(GRID.model_copy(update={"yres": 0.05})) == "0.1x0.05deg"
    utm = Grid(
        xmin=-112,
        xmax=-111.5,
        ymin=40.5,
        ymax=41,
        xres=1000,
        yres=1000,
        projection="EPSG:32612",
    )
    assert footprint_label(utm) == "1000m"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_run_folder_is_name_and_hash_with_settings_file(tmp_path):
    out = Output(tmp_path / "output")
    run = out.run("hrrr", SETTINGS)
    digest = settings_hash(SETTINGS)
    assert run.path == tmp_path / "output" / f"hrrr-{digest[:6]}"
    record = yaml.safe_load((run.path / "settings.yaml").read_text())
    assert record["name"] == "hrrr"
    assert record["hash"] == digest
    assert record["settings"]["numpar"] == 100
    assert out.runs() == [run] or [r.path for r in out.runs()] == [run.path]


def test_same_settings_under_another_name_share_the_folder(tmp_path):
    out = Output(tmp_path / "output")
    first = out.run("hrrr", SETTINGS)
    second = out.run("hrrr-main", dict(SETTINGS))
    assert second.path == first.path
    assert second.name == "hrrr"


def test_changed_settings_make_a_new_folder_beside_the_old(tmp_path):
    out = Output(tmp_path / "output")
    first = out.run("hrrr", SETTINGS)
    second = out.run("hrrr", {**SETTINGS, "ziscale": 0.8})
    assert second.path != first.path
    assert second.path.name.startswith("hrrr-")
    assert {r.path for r in out.runs()} == {first.path, second.path}
    assert out.find_run(SETTINGS).path == first.path
    assert out.find_run({"other": 1}) is None


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


def test_footprint_folder_is_label_and_hash(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    config = FootprintConfig(grid=GRID, smooth_factor=1.0)
    feet = run.footprints(config)
    digest = settings_hash(config.model_dump(mode="json"))
    assert feet.path == run.footprints_dir / f"0.1deg-{digest[:6]}"
    assert feet.config == config
    assert feet.grid == GRID

    other = run.footprints(config.model_copy(update={"smooth_factor": 0.5}))
    assert other.path != feet.path
    assert other.path.name.startswith("0.1deg-")
    assert {f.path for f in run.footprint_sets()} == {feet.path, other.path}
    assert run.footprints(config).path == feet.path


def test_footprint_round_trip_is_exact_in_float32(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    feet = run.footprints(FootprintConfig(grid=GRID))
    receptor = _receptor()
    foot = _footprint(receptor, hours=(-3, -2, -1, 0), seed=1)
    path = feet.write(foot)
    assert path == feet.path / "date=2024-07-15" / f"{receptor.id}.parquet"

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


def test_jacobian_with_no_footprints_is_empty(tmp_path):
    run = Output(tmp_path / "output").run("hrrr", SETTINGS)
    feet = run.footprints(FootprintConfig(grid=GRID))
    H = feet.jacobian(GRID, _bins())
    assert H.data.shape == (0, len(_bins()) * len(GRID.index))
    assert list(H.receptors) == []


# ---------------------------------------------------------------------------
# Settings of a run, and converting a project
# ---------------------------------------------------------------------------


def _project_model(tmp_path):
    """A model with three variants: hrrr, a footprint-only derivative, and an error run without a grid."""
    from stilt.config import MetConfig, ModelConfig
    from stilt.model import Model

    met = MetConfig(directory=tmp_path / "met", file_format="%Y%m%d_%H", file_tres="6h")
    coarse = {"grid": {"xres": 0.25, "yres": 0.25}}
    config = ModelConfig(
        mets={"hrrr": met},
        grid=GRID,
        numpar=100,
        variants={
            "hrrr": {},
            "hrrr-coarse": {"from": "hrrr", **coarse},
            "hrrr-err": {
                "siguverr": 2.0,
                "tluverr": 100.0,
                "zcoruverr": 200.0,
                "horcoruverr": 10.0,
                "grid": None,
            },
        },
    )
    receptors = [_receptor(hour=6), _receptor(hour=12)]
    return Model(project=tmp_path / "project", receptors=receptors, config=config)


def test_transport_settings_ignore_footprint_and_unrecorded_fields(tmp_path):
    from stilt.output import transport_settings

    model = _project_model(tmp_path)
    met = model.config.mets["hrrr"]
    base = transport_settings(model.variants["hrrr"], met)
    derived = transport_settings(model.variants["hrrr-coarse"], met)
    error = transport_settings(model.variants["hrrr-err"], met)
    assert base == derived
    assert base != error
    assert "grid" not in base and "smooth_factor" not in base
    assert "timeout" not in base and "exe_dir" not in base
    assert "directory" not in base["met"]
    assert base["met"]["file_tres"] == "6h"

    moved = met.model_copy(update={"directory": tmp_path / "elsewhere"})
    assert settings_hash(
        transport_settings(model.variants["hrrr"], moved)
    ) == settings_hash(base)


def test_convert_project_shares_runs_and_keeps_empty_footprints(tmp_path):
    from stilt.output import convert_project

    model = _project_model(tmp_path)
    r6, r12 = model.receptors[0], model.receptors[1]
    for receptor in (r6, r12):
        for variant in ("hrrr", "hrrr-err"):
            sim = model.simulation((receptor.id, variant))
            sim.directory.mkdir(parents=True, exist_ok=True)
            _trajectories(receptor).to_parquet(sim.trajectories_path)
    # hrrr: one real footprint and one empty marker; hrrr-coarse: one footprint.
    _footprint(r6, seed=3).to_netcdf(model.simulation((r6.id, "hrrr")).footprint_path)
    model.simulation((r12.id, "hrrr")).write_empty_footprint_marker("outside_domain")
    coarse = model.simulation((r6.id, "hrrr-coarse"))
    coarse_grid = coarse.footprint_config.grid
    x_axis, y_axis = coarse_grid.axes
    data = xr.DataArray(
        np.ones((1, len(y_axis), len(x_axis))),
        dims=["time", "lat", "lon"],
        coords={"time": [pd.Timestamp(r6.time)], "lat": y_axis, "lon": x_axis},
    )
    Footprint(
        receptor=r6, config=coarse.footprint_config, data=data, name="hrrr-coarse"
    ).to_netcdf(coarse.footprint_path)

    out = Output(tmp_path / "output")
    report = convert_project(model, out)
    # 6 simulations: hrrr ×2 and hrrr-err ×2 complete, hrrr-coarse: r6 complete, r12 incomplete.
    assert report.particles == 4
    assert report.footprints == 3  # r6 hrrr, r12 hrrr (empty), r6 hrrr-coarse
    assert report.incomplete == 1

    runs = {run.name: run for run in out.runs()}
    assert set(runs) == {"hrrr", "hrrr-err"}
    assert runs["hrrr"].receptors() == [str(r6.id), str(r12.id)]
    assert runs["hrrr-err"].receptors() == [str(r6.id), str(r12.id)]
    assert runs["hrrr-err"].footprint_sets() == []

    sets = {f.path.name.split("-")[0]: f for f in runs["hrrr"].footprint_sets()}
    assert set(sets) == {"0.1deg", "0.25deg"}
    fine, coarse_set = sets["0.1deg"], sets["0.25deg"]
    assert fine.read(str(r6.id)) is not None
    assert fine.read(str(r12.id)) is None
    assert fine.empty_reason(str(r12.id)) == "outside_domain"
    assert coarse_set.receptors() == [str(r6.id)]
    np.testing.assert_array_equal(coarse_set.read(str(r6.id)).data.values, 1.0)

    again = convert_project(model, out)
    assert again.particles == 0 and again.particles_skipped == 4
    assert again.footprints == 0 and again.footprints_skipped == 3
