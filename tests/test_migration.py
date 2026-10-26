"""
Tests for reading STILT-R and X-STILT ``by-id`` output (``stilt.migration``).

The sample project in ``fixtures/stilt_r_project`` was written by STILT-R's
own code (``write_control``, ``saveRDS`` of the list ``simulation_step``
saves, and ``calc_footprint`` with ``smooth_factor`` 0.5) from made-up
particles, with the ``params.RDS`` rslurm keeps beside ``out/``:
``fixtures/r_helpers/write_by_id.r`` says how, and writes it again with
``Rscript write_by_id.r <STILT_R_DIR> <folder>``.
"""

from __future__ import annotations

import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from stilt import (
    ColumnReceptor,
    Grid,
    MultiPointReceptor,
    PointReceptor,
    Project,
    calc_footprint,
)
from stilt.migration import (
    _axis,
    _error_settings,
    _particles,
    _receptor,
    _rslurm_params,
    _smooth_factor,
    import_stilt_r,
    read_stilt_r,
)
from stilt.particles import particles_metadata

PROJECT = Path(__file__).parent / "fixtures" / "stilt_r_project"
SAMPLE = PROJECT / "out" / "by-id"
PARAMS = PROJECT / "_rslurm_stilt_r_project" / "params.RDS"
POINT = "201507050000_-111.85_40.77_5"
COLUMN = "201507051800_-111.9_40.7_X"

rdata = pytest.importorskip("rdata")


def _read(by_id: Path = SAMPLE, **kwargs) -> dict[str, tuple]:
    """Return every simulation of a by-id folder, by its STILT-R simulation id."""
    sims = list(read_stilt_r(by_id, **kwargs))
    names = sorted(p.name for p in by_id.iterdir() if p.is_dir())
    return dict(zip(names, sims, strict=True))


def _raw_foot(name: str) -> xr.DataArray:
    """Return a sample footprint as STILT-R wrote it, read with xarray alone."""
    with xr.open_dataset(SAMPLE / name / f"{name}_foot.nc", decode_times=False) as ds:
        return ds["foot"].load()


def _write_points(control: Path, points: list[str]) -> None:
    """Replace the release points of a CONTROL file with *points* (``"lat lon height"``)."""
    lines = control.read_text().splitlines()
    n = int(lines[1])
    control.write_text(
        "\n".join([lines[0], str(len(points)), *points, *lines[2 + n :]])
    )


def _set_n_hours(control: Path, n_hours: int) -> None:
    """Replace the run length of a CONTROL file."""
    lines = control.read_text().splitlines()
    lines[2 + int(lines[1])] = str(n_hours)
    control.write_text("\n".join(lines))


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    """Return the by-id folder of a copy of the sample project, to change."""
    return (
        Path(shutil.copytree(PROJECT, tmp_path / "stilt_r_project")) / "out" / "by-id"
    )


@pytest.fixture
def bare(tmp_path: Path) -> Path:
    """Return a copy of the sample by-id folder without rslurm's params.RDS."""
    return Path(shutil.copytree(SAMPLE, tmp_path / "project" / "out" / "by-id"))


def test_point_receptor_is_the_one_stilt_r_ran():
    receptor, _, _ = _read()[POINT]
    assert receptor == PointReceptor(
        time=dt.datetime(2015, 7, 5), longitude=-111.85, latitude=40.77, altitude=5
    )
    assert receptor.id == POINT


def test_receptor_coordinates_are_the_numbers_in_control(copy):
    # The landfill receptor: HYSPLIT read these digits from CONTROL.
    _write_points(copy / POINT / "CONTROL", ["40.7454482983855 -112.043938842752 5"])
    receptor, _, _ = _read(copy)[POINT]
    assert (receptor.latitude, receptor.longitude) == (
        40.7454482983855,
        -112.043938842752,
    )
    assert receptor.location_id == "-112.043938842752_40.7454482983855_5"


def test_receptor_coordinates_do_not_come_from_the_rds(copy):
    # The .rds keeps the receptor as run_stilt.r gave it; CONTROL is what ran.
    _write_points(copy / POINT / "CONTROL", ["40.78 -111.86 10"])
    receptor, _, _ = _read(copy)[POINT]
    assert receptor.coords() == [(40.78, -111.86, 10.0)]


def test_slant_points_in_control_are_a_multipoint_receptor(copy):
    # X-STILT's slant columns release from a point per height, each its own place.
    points = ["40.5 -112.0 100", "40.51 -112.01 1000", "40.52 -112.02 2000"]
    _write_points(copy / POINT / "CONTROL", points)
    receptor, _, _ = _read(copy)[POINT]
    assert isinstance(receptor, MultiPointReceptor)
    assert receptor.longitudes == (-112.0, -112.01, -112.02)
    assert receptor.altitudes == (100.0, 1000.0, 2000.0)


def test_a_folder_without_control_raises_naming_it(copy):
    (copy / COLUMN / "CONTROL").unlink()
    with pytest.raises(FileNotFoundError, match=f"1 folders .* first .*{COLUMN}"):
        read_stilt_r(copy)


def test_a_control_that_cannot_be_read_raises(copy):
    (copy / POINT / "CONTROL").write_text("15 07 05 00 00\n1\n")
    with pytest.raises(ValueError, match=f"{POINT}/CONTROL cannot be read"):
        _read(copy)


def test_a_receptor_without_its_time_raises():
    with pytest.raises(ValueError, match="has no run_time"):
        _receptor({"lati": 40.5}, None, Path("x_traj.rds"))


def test_particles_without_a_schema_column_raise():
    receptor, particles, _ = _read()[POINT]
    stilt_r = particles.drop(columns=["time", "zagl"]).rename(
        columns={"particle": "indx", "age": "time", "lon": "long", "lat": "lati"}
    )
    with pytest.raises(ValueError, match="x_traj.rds: .*'zagl'"):
        _particles(stilt_r, receptor, Path("x_traj.rds"))


def test_an_rds_without_particles_raises(copy):
    rdata.write_rds(copy / POINT / f"{POINT}_traj.rds", {"file": "x", "params": {}})
    with pytest.raises(ValueError, match="not a STILT-R trajectory file"):
        _read(copy)


def test_column_receptor_spans_the_heights_and_keeps_seconds():
    receptor, _, _ = _read()[COLUMN]
    assert receptor == ColumnReceptor(
        time=dt.datetime(2015, 7, 5, 18, 0, 30),
        longitude=-111.9,
        latitude=40.7,
        bottom=0,
        top=1000,
    )


def test_particle_columns_get_pystilt_names():
    _, particles, _ = _read()[POINT]
    for stilt_r in ("indx", "long", "lati", "xhgt"):
        assert stilt_r not in particles.columns
    assert {"particle", "age", "lon", "lat", "zagl", "foot"} <= set(particles.columns)
    assert "foot_no_hnf_dilution" in particles.columns
    assert sorted(particles["particle"].unique()) == list(range(1, 11))
    # STILT-R's time is minutes since release, negative backward.
    assert particles["age"].min() == -120 and particles["age"].max() == -1
    assert particles["particle"].dtype == "float64"
    assert particles["age"].dtype == "float64"


def test_time_is_the_receptor_time_plus_age():
    receptor, particles, _ = _read()[COLUMN]
    expected = pd.Timestamp(receptor.time) + pd.to_timedelta(particles["age"], "min")
    pd.testing.assert_series_equal(particles["time"], expected, check_names=False)
    assert particles["time"].max() == pd.Timestamp("2015-07-05 17:59:30")


def test_column_release_height_is_stilt_r_xhgt():
    _, particles, _ = _read()[COLUMN]
    # 10 particles spread over 0 to 1000 m: slab centers 50, 150, ... 950.
    expected = (particles["particle"] - 0.5) * 100
    np.testing.assert_allclose(particles["release_height"], expected)


def test_hourly_footprint_layers_are_stamped_as_stilt_r_stamps_them():
    _, _, foot = _read()[POINT]
    raw = _raw_foot(POINT)
    stamps = pd.to_datetime(raw["time"].to_numpy(), unit="s")
    assert list(stamps) == [
        pd.Timestamp("2015-07-04 22:00"),
        pd.Timestamp("2015-07-04 23:00"),
    ]
    np.testing.assert_array_equal(foot["time"].to_numpy(), stamps.to_numpy())
    assert foot.stilt.config.time_integrate is False


def test_footprint_values_are_the_file_values_on_its_grid():
    _, _, foot = _read()[POINT]
    raw = _raw_foot(POINT).transpose("time", "lat", "lon")
    assert foot.dims == ("time", "lat", "lon")
    np.testing.assert_array_equal(foot.to_numpy(), raw.to_numpy().astype("float64"))
    np.testing.assert_allclose(foot["lon"], raw["lon"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(foot["lat"], raw["lat"], rtol=0, atol=1e-9)
    grid = foot.stilt.grid
    assert (grid.xmin, grid.xmax, grid.ymin, grid.ymax) == (-112.2, -111.6, 40.6, 41.0)
    assert (grid.xres, grid.yres) == (0.02, 0.025)


def test_time_integrated_footprint_is_one_layer_at_the_receptor_time():
    receptor, _, foot = _read()[COLUMN]
    assert foot.sizes["time"] == 1
    assert foot["time"].to_numpy()[0] == np.datetime64(receptor.time)
    assert foot.stilt.config.time_integrate is True
    assert foot.stilt.receptor == receptor


@pytest.mark.parametrize("name", [POINT, COLUMN])
def test_footprint_matches_pystilt_footprint_of_the_particles(name):
    # STILT-R made its footprint from these particles; PYSTILT's matches
    # STILT-R's at rtol=1e-7. This holds only when the columns, ages,
    # receptor time, grid, and layer stamps are all read right.
    receptor, particles, foot = _read()[name]
    config = foot.stilt.config
    assert config.smooth_factor == 0.5
    mine = calc_footprint(
        particles,
        receptor,
        config.grid,
        smooth_factor=config.smooth_factor,
        time_integrate=config.time_integrate,
    )
    assert mine is not None
    xr.testing.assert_allclose(mine, foot, rtol=1e-7, atol=1e-12)


def test_a_folder_without_a_footprint_gives_none(copy):
    (copy / POINT / f"{POINT}_foot.nc").unlink()
    _, particles, foot = _read(copy)[POINT]
    assert foot is None
    assert not particles.empty


def test_other_footprint_files_are_not_read(copy):
    # X-STILT writes footprints at other resolutions beside the main one.
    (copy / COLUMN / f"{COLUMN}_0.1x0.1_foot.nc").write_bytes(b"not netcdf")
    _, _, foot = _read(copy)[COLUMN]
    assert foot is not None


def test_a_footprint_without_particles_is_skipped_with_a_warning(copy):
    (copy / POINT / f"{POINT}_traj.rds").unlink()
    with pytest.warns(UserWarning, match="footprint but no _traj.rds"):
        sims = list(read_stilt_r(copy))
    assert [r.id for r, _, _ in sims] == ["201507051800_-111.9_40.7_X0-1000"]


def test_layers_that_are_not_whole_hours_raise(copy):
    path = copy / POINT / f"{POINT}_foot.nc"
    with xr.open_dataset(path, decode_times=False) as ds:
        shifted = ds.load()
    shifted["time"] = shifted["time"] + 1800
    path.unlink()
    shifted.to_netcdf(path)
    with pytest.raises(ValueError, match="not whole hours"):
        _read(copy)


def test_a_projected_footprint_raises(copy):
    path = copy / POINT / f"{POINT}_foot.nc"
    with xr.open_dataset(path, decode_times=False) as ds:
        projected = ds.load().rename({"lon": "x", "lat": "y"})
    path.unlink()
    projected.to_netcdf(path)
    with pytest.raises(ValueError, match="longitude/latitude grid"):
        _read(copy)


def test_uneven_footprint_cells_raise(copy):
    path = copy / POINT / f"{POINT}_foot.nc"
    with xr.open_dataset(path, decode_times=False) as ds:
        uneven = ds.load()
    lon = uneven["lon"].to_numpy().copy()
    lon[3] += 0.005
    uneven = uneven.assign_coords(lon=lon)
    path.unlink()
    uneven.to_netcdf(path)
    with pytest.raises(ValueError, match="not evenly spaced"):
        _read(copy)


def test_an_axis_of_one_cell_raises():
    with pytest.raises(ValueError, match="two lon cells"):
        _axis(np.array([-112.0]), "lon", Path("x_foot.nc"))


def test_a_folder_that_is_not_by_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="by-id"):
        read_stilt_r(tmp_path)
    with pytest.raises(FileNotFoundError, match="not a folder"):
        read_stilt_r(tmp_path / "missing")


def test_without_rdata_the_error_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdata", None)
    with pytest.raises(ImportError, match=r"pystilt\[stilt-r\]"):
        read_stilt_r(SAMPLE)


def test_params_rds_columns_are_read_past_its_functions():
    # rslurm's table holds the before_trajec and before_footprint functions,
    # which rdata cannot convert; the columns needed are read alone.
    path, values = _rslurm_params(SAMPLE, rdata)
    assert path == PARAMS.resolve()
    assert values == {"smooth_factor": {0.5}, "n_hours": {-2.0}, "numpar": {10.0}}


def test_smooth_factor_comes_from_params_rds():
    for _, _, foot in read_stilt_r(SAMPLE):
        assert foot.stilt.config.smooth_factor == 0.5


def test_smooth_factor_given_without_params_rds(bare):
    receptor, particles, foot = _read(bare, smooth_factor=0.5)[POINT]
    assert foot.stilt.config.smooth_factor == 0.5
    mine = calc_footprint(particles, receptor, foot.stilt.grid, smooth_factor=0.5)
    xr.testing.assert_allclose(mine, foot, rtol=1e-7, atol=1e-12)


def test_a_footprint_without_a_smooth_factor_raises_saying_how(bare):
    with pytest.raises(ValueError, match=r"params.RDS .* smooth_factor=1"):
        _read(bare)


def test_runs_without_footprints_need_no_smooth_factor(bare):
    for foot in bare.glob("*/*_foot.nc"):
        foot.unlink()
    sims = _read(bare)
    assert all(foot is None for _, _, foot in sims.values())


def test_smooth_factor_given_and_in_params_rds_must_agree():
    assert _read(smooth_factor=0.5)[POINT][2].stilt.config.smooth_factor == 0.5
    with pytest.raises(
        ValueError, match="smooth_factor=1 was given, but .* records 0.5"
    ):
        _read(smooth_factor=1)


def test_params_rds_of_another_run_length_raises(copy):
    _set_n_hours(copy / POINT / "CONTROL", -24)
    with pytest.raises(ValueError, match="disagree: it gives n_hours -2, .* give -24"):
        _read(copy)


def test_params_rds_of_another_numpar_raises():
    rslurm = (PARAMS, {"smooth_factor": {1.0}, "n_hours": {-2.0}, "numpar": {10.0}})
    foot = Path("x_foot.nc")
    assert _smooth_factor(foot, -2, {"numpar": [10.0]}, rslurm, None) == 1
    with pytest.raises(ValueError, match="numpar 10, and the run's files give 1000"):
        _smooth_factor(foot, -2, {"numpar": [1000.0]}, rslurm, None)


def test_params_rds_with_several_smooth_factors_raises():
    values = {"smooth_factor": {1.0, 2.0}, "n_hours": {-2.0}, "numpar": {10.0}}
    with pytest.raises(ValueError, match="gives smooth_factor 1, 2; one value"):
        _smooth_factor(Path("x_foot.nc"), -2, {}, (PARAMS, values), None)


def test_params_rds_that_is_not_a_table_raises(copy):
    rdata.write_rds(
        copy.parents[1] / "_rslurm_stilt_r_project" / "params.RDS", {"a": 1}
    )
    with pytest.raises(ValueError, match="not a table of STILT-R runs"):
        read_stilt_r(copy)


# -- reading the transport-error particles ------------------------------------


def test_error_reads_the_perturbed_particles_and_no_footprint(copy):
    shutil.rmtree(copy / COLUMN)
    ((receptor, particles, _),) = list(read_stilt_r(copy))
    ((same, perturbed, foot),) = list(read_stilt_r(copy, error=True))
    assert same == receptor
    assert foot is None
    assert list(perturbed.columns) == list(particles.columns)
    assert not np.array_equal(perturbed["lon"], particles["lon"])


def test_error_on_a_run_without_transport_error_raises(copy):
    shutil.rmtree(copy / POINT)
    with pytest.raises(ValueError, match="no transport-error particles"):
        list(read_stilt_r(copy, error=True))


# -- importing into an output directory ----------------------------------------

VERSION = "0e290a6"


def _by_id(tmp_path: Path, *names: str, rslurm: bool = True) -> Path:
    """Return the by-id folder of a STILT-R project holding the sample's runs *names*."""
    project = tmp_path / "landfill"
    by_id = project / "out" / "by-id"
    for name in names:
        shutil.copytree(SAMPLE / name, by_id / name)
    if rslurm:
        shutil.copytree(PARAMS.parent, project / "_rslurm_landfill")
    return by_id


def _import(by_id: Path, output: Path, capsys, **kwargs) -> tuple[list, dict]:
    """Import *by_id* into *output*; return the receptors and the config it printed."""
    receptors = import_stilt_r(by_id, output, VERSION, **kwargs)
    return receptors, yaml.safe_load(capsys.readouterr().out)


def _project(tmp_path: Path, by_id: Path, capsys, **kwargs) -> Project:
    """Import *by_id* and return a project whose config.yaml is what the import printed."""
    output = tmp_path / "output"
    receptors, config = _import(by_id, output, capsys, **kwargs)
    return Project.init(tmp_path / "py", receptors, output=str(output), **config)


def _settings(output: Path, kind: str, folder: str) -> dict:
    """Return the settings block of a settings folder's _settings.yaml."""
    # The folder name ends in six characters of the hash; "stilt-r-*" would
    # also match "stilt-r-err-...".
    (path,) = (output / kind).glob(f"settings={folder}-??????")
    path = path / "_settings.yaml"
    return yaml.safe_load(path.read_text())["settings"]


def test_imported_runs_are_the_results_of_the_printed_variants(tmp_path, capsys):
    by_id = _by_id(tmp_path, POINT)
    project = _project(tmp_path, by_id, capsys)
    assert list(project.variants) == ["stilt-r", "stilt-r-err"]
    status = project.status()
    assert list(status.state) == ["complete", "complete"]

    ((receptor, particles, foot),) = list(read_stilt_r(by_id))
    sim = project.simulation(POINT, "stilt-r")
    assert sim.receptor == receptor
    for column in ("particle", "age", "lon", "lat", "zagl", "foot"):
        np.testing.assert_array_equal(
            sim.particles[column].to_numpy(), particles[column].to_numpy()
        )
    assert foot is not None
    np.testing.assert_allclose(sim.footprint.to_numpy(), foot.to_numpy(), rtol=1e-6)
    ((_, perturbed, _),) = list(read_stilt_r(by_id, error=True))
    np.testing.assert_array_equal(
        project.simulation(POINT, "stilt-r-err").particles["lon"].to_numpy(),
        perturbed["lon"].to_numpy(),
    )


def test_jacobian_sums_the_imported_footprints(tmp_path, capsys):
    by_id = _by_id(tmp_path, POINT)
    project = _project(tmp_path, by_id, capsys)
    ((_, _, foot),) = list(read_stilt_r(by_id))
    target = Grid(xmin=-112.2, xmax=-111.6, ymin=40.6, ymax=41.0, xres=0.1, yres=0.1)
    bins = pd.IntervalIndex.from_breaks(
        pd.date_range("2015-07-04 22:00", "2015-07-05 00:00", freq="1h"),
        closed="left",
    )
    sims = project.simulations
    H = project.jacobian(sims[sims.variant == "stilt-r"], target, bins)
    assert list(H.receptors) == [POINT]
    expected = foot.stilt.aggregate(target, bins)
    np.testing.assert_allclose(
        H.to_frame().iloc[0].to_numpy().reshape(len(bins), -1).T,
        expected.to_numpy(),
        rtol=1e-6,
    )


def test_the_settings_are_what_the_files_record(tmp_path, capsys):
    output = tmp_path / "output"
    _import(_by_id(tmp_path, POINT), output, capsys)
    run = _settings(output, "particles", "stilt-r")
    # SETUP.CFG
    assert run["numpar"] == 10
    assert run["kmix0"] == 150
    assert run["frvs"] == 0.01
    # CONTROL
    assert run["n_hours"] == -2
    assert run["emisshrs"] == 0.01
    assert run["z_top"] == 25000
    # The particles have STILT-R's near-field correction.
    assert run["hnf_plume"] is True
    # What no file holds is unset: no seed, no error, no mixed-layer factor.
    assert run["seed"] is None
    assert run["siguverr"] is None
    assert run["ziscale"] == 1
    assert run["model"] == {"name": "stilt-r", "version": VERSION, "data_files": None}
    assert run["met"] == {"met_path": "met"}
    error = _settings(output, "particles", "stilt-r-err")
    assert {
        k: error[k] for k in ("siguverr", "tluverr", "zcoruverr", "horcoruverr")
    } == {
        "siguverr": 2.6,
        "tluverr": 260,
        "zcoruverr": 450,
        "horcoruverr": 14,
    }
    assert error["sigzierr"] is None
    assert {k: v for k, v in error.items() if not k.endswith("err")} == {
        k: v for k, v in run.items() if not k.endswith("err")
    }
    foot = _settings(output, "footprints", "stilt-r")
    assert foot["smooth_factor"] == 0.5
    assert foot["time_integrate"] is False
    assert foot["grid"]["yres"] == 0.025
    path = next((output / "particles").rglob(f"{POINT}.parquet"))
    assert [str(p) for p in particles_metadata(path).met_files] == [
        "met/20150704_18-23_hrrr",
        "met/20150705_00-05_hrrr",
    ]


def test_runs_of_another_length_land_in_another_folder(tmp_path, capsys):
    output = tmp_path / "output"
    _import(_by_id(tmp_path / "a", POINT), output, capsys, name="a")
    longer = _by_id(tmp_path / "b", POINT, rslurm=False)
    _set_n_hours(longer / POINT / "CONTROL", -3)
    (longer / POINT / f"{POINT}_foot.nc").unlink()
    _import(longer, output, capsys, name="b")
    folders = sorted(p.name for p in (output / "particles").iterdir())
    assert [f.rsplit("-", 1)[0] for f in folders] == [
        "settings=a",
        "settings=a-err",
        "settings=b",
        "settings=b-err",
    ]
    assert _settings(output, "particles", "b")["n_hours"] == -3


def test_runs_with_other_settings_stop_before_anything_is_written(tmp_path, capsys):
    by_id = _by_id(tmp_path, POINT, COLUMN)
    setup = by_id / COLUMN / "SETUP.CFG"
    setup.write_text(setup.read_text().replace("KMIX0=150,", "KMIX0=250,"))
    with pytest.raises(ValueError, match=r"different HYSPLIT settings: kmix0 is 250"):
        import_stilt_r(by_id, tmp_path / "output", VERSION)
    assert not (tmp_path / "output").exists()


def test_footprints_with_other_settings_stop(tmp_path):
    # The sample's two runs have footprints on different grids.
    with pytest.raises(ValueError, match="different footprint settings"):
        import_stilt_r(_by_id(tmp_path, POINT, COLUMN), tmp_path / "output", VERSION)


def test_ziscale_comes_from_the_zicontrol_file(tmp_path, capsys):
    by_id = _by_id(tmp_path, POINT)
    setup = by_id / POINT / "SETUP.CFG"
    setup.write_text(setup.read_text().replace("ZICONTROLTF=0,", "ZICONTROLTF=1,"))
    with pytest.raises(FileNotFoundError, match="no ZICONTROL"):
        import_stilt_r(by_id, tmp_path / "output", VERSION)
    (by_id / POINT / "ZICONTROL").write_text("3\n0.8\n0.9\n")
    with pytest.raises(ValueError, match="as many factors"):
        import_stilt_r(by_id, tmp_path / "output", VERSION)
    (by_id / POINT / "ZICONTROL").write_text("2\n0.8\n0.9\n")
    _import(by_id, tmp_path / "output", capsys)
    assert _settings(tmp_path / "output", "particles", "stilt-r")["ziscale"] == [
        0.8,
        0.9,
    ]


@pytest.mark.parametrize(
    ("old", "new", "error", "match"),
    [
        ("KMSL=0,", "KMSL=1,", ValueError, "above sea level"),
        ("KRND=6,", "KRND=6,\nNEWKEY=1,", ValueError, "NEWKEY"),
        ("WINDERRTF=1,", "WINDERRTF=0,", ValueError, "particle_error_params"),
        ("KRAND=4,", "KRAND=5,", ValueError, "not valid HYSPLIT settings"),
    ],
)
def test_a_setup_cfg_the_import_cannot_record_stops(tmp_path, old, new, error, match):
    by_id = _by_id(tmp_path, POINT)
    setup = by_id / POINT / "SETUP.CFG"
    setup.write_text(setup.read_text().replace(old, new))
    with pytest.raises(error, match=match):
        import_stilt_r(by_id, tmp_path / "output", VERSION)


def test_a_transport_error_flag_without_the_particles_stops(tmp_path):
    by_id = _by_id(tmp_path, COLUMN)
    setup = by_id / COLUMN / "SETUP.CFG"
    setup.write_text(setup.read_text().replace("WINDERRTF=0,", "WINDERRTF=1,"))
    with pytest.raises(ValueError, match="holds no particle_error"):
        import_stilt_r(by_id, tmp_path / "output", VERSION)


def test_a_run_without_setup_cfg_stops(tmp_path):
    by_id = _by_id(tmp_path, POINT)
    (by_id / POINT / "SETUP.CFG").unlink()
    with pytest.raises(FileNotFoundError, match="has no SETUP.CFG"):
        import_stilt_r(by_id, tmp_path / "output", VERSION)


def test_several_met_directories_need_the_folder_that_holds_them(tmp_path, capsys):
    by_id = _by_id(tmp_path, POINT)
    control = by_id / POINT / "CONTROL"
    text = control.read_text()
    control.write_text(
        text.replace("met/ \n20150704", "/data/hrrr/2015a/ \n20150704").replace(
            "met/ \n20150705", "/data/hrrr/2015b/ \n20150705"
        )
    )
    with pytest.raises(ValueError, match="2 met directories"):
        import_stilt_r(by_id, tmp_path / "output", VERSION)
    with pytest.raises(ValueError, match="is not in met_path /data/nam"):
        import_stilt_r(by_id, tmp_path / "output", VERSION, met_path="/data/nam")
    _, config = _import(by_id, tmp_path / "output", capsys, met_path="/data/hrrr/")
    assert config["mets"] == {"stilt-r": {"met_path": "/data/hrrr"}}


def test_a_run_whose_log_says_no_particle_reached_the_grid_is_empty(tmp_path, capsys):
    by_id = _by_id(tmp_path, POINT)
    # A second run of the same settings, at another place, without a footprint.
    other = "201507050000_-111.85_40.78_5"
    shutil.copytree(by_id / POINT, by_id / other)
    for path in (by_id / other).iterdir():
        path.rename(path.with_name(path.name.replace(POINT, other)))
    (by_id / other / f"{other}_foot.nc").unlink()
    _write_points(by_id / other / "CONTROL", ["40.78 -111.85 5"])
    (by_id / other / "stilt.log").write_text(
        "No non-zero footprint values found within the footprint domain. \n"
    )
    project = _project(tmp_path, by_id, capsys)
    status = project.status()
    assert list(status.state) == ["complete"] * 4
    _import(by_id, tmp_path / "output", capsys)  # kept, not written again
    sim = project.simulation(other, "stilt-r")
    assert sim.has_footprint
    assert sim.footprint is None
    assert sim.log.startswith("No non-zero footprint")


def test_a_run_without_a_footprint_or_a_log_saying_why_is_not_complete(
    tmp_path, capsys
):
    by_id = _by_id(tmp_path, POINT)
    other = "201507050000_-111.85_40.78_5"
    shutil.copytree(by_id / POINT, by_id / other)
    for path in (by_id / other).iterdir():
        path.rename(path.with_name(path.name.replace(POINT, other)))
    (by_id / other / f"{other}_foot.nc").unlink()
    _write_points(by_id / other / "CONTROL", ["40.78 -111.85 5"])
    project = _project(tmp_path, by_id, capsys)
    status = project.status().set_index(["receptor", "variant"])
    assert status.loc[(other, "stilt-r"), "particles"]
    assert status.loc[(other, "stilt-r"), "state"] == "pending"


def test_importing_again_keeps_what_is_there(tmp_path, capsys):
    by_id = _by_id(tmp_path, POINT)
    output = tmp_path / "output"
    _, first = _import(by_id, output, capsys)
    files = sorted(output.rglob("*.parquet"))
    written = [p.stat().st_mtime_ns for p in files]
    _, again = _import(by_id, output, capsys)
    assert again == first
    assert sorted(output.rglob("*.parquet")) == files
    assert [p.stat().st_mtime_ns for p in files] == written


def test_a_printed_variant_missing_a_setting_does_not_load(tmp_path, capsys):
    receptors, config = _import(_by_id(tmp_path, POINT), tmp_path / "output", capsys)
    del config["variants"]["stilt-r"]["kmix0"]
    with pytest.raises(ValueError, match="missing: kmix0"):
        Project.init(tmp_path / "py", receptors, **config)


def test_the_name_must_be_a_variant_name(tmp_path):
    with pytest.raises(ValueError, match="must be a variant name"):
        import_stilt_r(
            _by_id(tmp_path, POINT), tmp_path / "output", VERSION, name="STILT R"
        )


def test_runs_without_transport_error_give_one_variant(tmp_path, capsys):
    project = _project(tmp_path, _by_id(tmp_path, COLUMN), capsys)
    assert list(project.variants) == ["stilt-r"]
    assert list(project.status().state) == ["complete"]


def test_error_settings_need_every_setting_stilt_r_saves(tmp_path):
    with pytest.raises(ValueError, match="particle_error_params has no siguverr"):
        _error_settings({"winderrtf": 1}, tmp_path / "x_traj.rds")
