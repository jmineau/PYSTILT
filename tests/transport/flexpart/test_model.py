"""Tests for FLEXPART's config, met, input files, and model, with a stand-in for its executable."""

import datetime as dt
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import stilt
from stilt.exceptions import MeteorologyError, SimulationError
from stilt.receptors import ColumnReceptor, MultiPointReceptor, PointReceptor
from stilt.transport import get_model
from stilt.transport.flexpart import FlexpartConfig, FlexpartMetConfig, FlexpartModel
from stilt.transport.flexpart.config import flexpart_version
from stilt.transport.flexpart.inputs import command, partoptions, releases
from stilt.transport.flexpart.met import catalog, files_for

pytest.importorskip("eccodes")
pytest.importorskip("cfgrib")

from ...fixtures.grib import write_gfs

SAMPLE = Path(__file__).parent / "data" / "partoutput_20230601180000_init.nc"
RECEPTOR = PointReceptor(
    time="2023-06-01 18:00", longitude=-111.848, latitude=40.766, altitude=25
)
WINDOW = (dt.datetime(2023, 5, 31, 18), dt.datetime(2023, 6, 1, 18))

#: A stand-in for FLEXPART: copies the real sample into output/ and says it worked.
FAKE = """#!/bin/bash
test -f options/COMMAND && test -f options/RELEASES && test -f AVAILABLE || exit 2
cp {sample} output/partoutput_20230601180000_init.nc
echo " CONGRATULATIONS: YOU HAVE SUCCESSFULLY COMPLETED A FLEXPART MODEL RUN!"
"""


@pytest.fixture
def checkout(tmp_path):
    """A FLEXPART checkout with a stand-in executable and the option files it reads."""
    root = tmp_path / "flexpart"
    (root / "src").mkdir(parents=True)
    (root / "options" / "SPECIES").mkdir(parents=True)
    for name in ("IGBP_int1.dat", "sfcdata.t", "sfcdepo.t"):
        (root / "options" / name).write_text("data\n")
    (root / "options" / "SPECIES" / "SPECIES_AIRTRACER").write_text(
        '&SPECIES_PARAMS\n PSPECIES="AIRTRACER",\n PWEIGHTMOLAR=29.0,\n /\n'
    )
    (root / "version").write_text("11.0-test\n")
    exe = root / "src" / "FLEXPART"
    exe.write_text(FAKE.format(sample=SAMPLE))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return root


@pytest.fixture
def gfs(tmp_path):
    """GRIB files every 12 h around the sample run's window."""
    directory = tmp_path / "gfs"
    directory.mkdir()
    for time in (
        "2023-05-31 06:00",
        "2023-05-31 18:00",
        "2023-06-01 06:00",
        "2023-06-01 18:00",
    ):
        write_gfs(directory / f"gfs_{time[:10]}_{time[11:13]}.grb2", time)
    return directory


def config(checkout, **kw):
    return FlexpartConfig(flexpart_dir=checkout, **kw)


# -- config --------------------------------------------------------------------


def test_config_refuses_a_seed(checkout):
    with pytest.raises(ValidationError, match="FLEXPART has no seed"):
        config(checkout, seed=1)


def test_config_needs_lsynctime_to_divide_loutstep(checkout):
    with pytest.raises(ValidationError, match="must divide loutstep"):
        config(checkout, loutstep=3600, lsynctime=700)


def test_config_records_threads_but_not_where_the_checkout_is(checkout):
    settings = config(checkout).settings()
    assert "flexpart_dir" not in settings
    assert settings["threads"] == 1
    assert settings["hnf_plume"] is False


def test_config_without_a_checkout_cannot_run():
    with pytest.raises(ValueError, match="Set flexpart_dir"):
        _ = FlexpartConfig().executable


def test_version_from_a_version_file(checkout):
    assert flexpart_version(checkout) == "11.0-test"


def test_version_without_file_or_git(tmp_path):
    with pytest.raises(FileNotFoundError, match="no 'version' file"):
        flexpart_version(tmp_path)


def test_met_config_takes_a_bare_directory_and_records_the_grid(gfs):
    met = FlexpartMetConfig.model_validate(str(gfs))
    assert met.directory == gfs
    assert met.settings() == {"source": "gfs", "grid": "1x1 deg, 3x3"}


def test_get_model_finds_flexpart():
    assert isinstance(get_model("flexpart"), FlexpartModel)


# -- met files -----------------------------------------------------------------


def test_files_for_a_window_include_one_beyond_each_end(gfs):
    files = files_for(
        catalog(gfs), (dt.datetime(2023, 5, 31, 20), dt.datetime(2023, 6, 1, 5))
    )
    assert list(files) == [dt.datetime(2023, 5, 31, 18), dt.datetime(2023, 6, 1, 6)]


def test_files_for_a_window_the_files_do_not_cover(gfs):
    with pytest.raises(MeteorologyError, match="do not cover"):
        files_for(catalog(gfs), (dt.datetime(2023, 6, 1, 12), dt.datetime(2023, 6, 2)))


def test_files_for_a_window_with_a_missing_file(gfs):
    (gfs / "gfs_2023-06-01_06.grb2").unlink()
    with pytest.raises(MeteorologyError, match="a file is missing"):
        files_for(catalog(gfs), WINDOW)


# -- input files ---------------------------------------------------------------


def test_command_runs_backward_over_the_window(checkout, gfs):
    text = command(config(checkout), FlexpartMetConfig(directory=gfs), WINDOW)
    for line in (
        "LDIRECT=-1,",
        "IBDATE=20230531,",
        "IBTIME=180000,",
        "IEDATE=20230601,",
        "IETIME=180000,",
        "IPOUT=1,",
        "IOUT=0,",
        "IOUTPUTFOREACHRELEASE=1,",
        "NXSHIFT=0,",
    ):
        assert f" {line}\n" in text
    assert text.startswith("&COMMAND\n") and text.endswith(" /\n")


def test_command_times_before_ten_are_integers(checkout, gfs):
    window = (dt.datetime(2023, 6, 1, 6), dt.datetime(2023, 6, 1, 9, 30))
    text = command(
        config(checkout, n_hours=3), FlexpartMetConfig(directory=gfs), window
    )
    assert " LDIRECT=1,\n" in text
    assert " IBTIME=60000,\n" in text and " IETIME=93000,\n" in text


def test_releases_of_a_point(checkout):
    text = releases(RECEPTOR, 1000)
    assert text.count("&RELEASE\n") == 1
    assert " PARTS=1000,\n" in text and " ZKIND=1,\n" in text
    assert " Z1=25.0,\n" in text and " Z2=25.0,\n" in text


def test_releases_of_a_column_stack_one_particle_per_slab():
    column = ColumnReceptor(
        time="2023-06-01 18:00", longitude=-111.8, latitude=40.7, bottom=0, top=100
    )
    text = releases(column, 4)
    assert text.count("&RELEASE\n") == 4
    assert text.count(" PARTS=1,\n") == 4
    for z1, z2 in [(0.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 100.0)]:
        assert f" Z1={z1},\n Z2={z2},\n" in text


def test_releases_of_a_multipoint_split_the_particles():
    receptor = MultiPointReceptor(
        time="2023-06-01 18:00",
        longitudes=[-111.8, -111.9, -112.0],
        latitudes=[40.7, 40.8, 40.9],
        altitudes=[1000, 2000, 3000],
        altitude_ref="msl",
    )
    text = releases(receptor, 10)
    assert [line for line in text.splitlines() if "PARTS" in line] == [
        " PARTS=4,",
        " PARTS=3,",
        " PARTS=3,",
    ]
    assert text.count(" ZKIND=2,\n") == 3


def test_partoptions_turn_on_what_foot_reads():
    text = partoptions()
    for name in ("LONGITUDE", "HEIGHT", "MIXINGHEIGHT", "DENSITY", "PRESSURE"):
        assert f" {name}=.true.,\n" in text
    assert " PV=.false.,\n" in text


# -- the model -----------------------------------------------------------------


def test_run_writes_the_inputs_and_reads_the_particles(checkout, gfs, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    run = FlexpartModel().run(
        RECEPTOR, config(checkout), FlexpartMetConfig(directory=gfs), WINDOW, workdir
    )

    assert len(run.particles) == 5 * 24
    assert "CONGRATULATIONS" in run.log
    assert [p.name for p in run.met_files] == [
        "gfs_2023-05-31_18.grb2",
        "gfs_2023-06-01_06.grb2",
        "gfs_2023-06-01_18.grb2",
    ]
    assert (workdir / "pathnames").read_text() == (
        "./options/\n./output/\n./met/\n./AVAILABLE\n"
    )
    assert sorted(p.name for p in (workdir / "met").iterdir()) == [
        p.name for p in run.met_files
    ]
    assert (workdir / "options" / "SPECIES" / "SPECIES_001").is_file()
    assert (workdir / "options" / "IGBP_int1.dat").is_symlink()


def test_run_trajectories_computes_foot_from_the_grib_files(checkout, gfs, tmp_path):
    particles = stilt.run_trajectories(
        RECEPTOR,
        {"directory": str(gfs)},
        model="flexpart",
        flexpart_dir=str(checkout),
        workdir=tmp_path / "work",
    )

    below = particles["zagl"] < 0.5 * particles["mlht"]
    assert (particles.loc[below, "foot"] > 0).all()
    assert (particles.loc[~below, "foot"] == 0).all()
    assert "release_height" not in particles.columns


def test_run_fails_with_flexpart_s_output(checkout, gfs, tmp_path):
    exe = checkout / "src" / "FLEXPART"
    exe.write_text("#!/bin/bash\necho ' #### FLEXPART MODEL ERROR! bad'\nexit 1\n")
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(SimulationError, match="MODEL ERROR") as caught:
        FlexpartModel().run(
            RECEPTOR,
            config(checkout),
            FlexpartMetConfig(directory=gfs),
            WINDOW,
            workdir,
        )
    assert caught.value.reason == "EXIT_STATUS"
    assert "MODEL ERROR" in caught.value.log


def test_run_without_particle_output(checkout, gfs, tmp_path):
    exe = checkout / "src" / "FLEXPART"
    exe.write_text("#!/bin/bash\necho ' CONGRATULATIONS'\n")
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(SimulationError, match="no partoutput") as caught:
        FlexpartModel().run(
            RECEPTOR,
            config(checkout),
            FlexpartMetConfig(directory=gfs),
            WINDOW,
            workdir,
        )
    assert caught.value.reason == "NO_PARTICLE_DATA"


def test_run_timeout(checkout, gfs, tmp_path):
    exe = checkout / "src" / "FLEXPART"
    exe.write_text("#!/bin/bash\nsleep 30\n")
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(SimulationError, match="timeout") as caught:
        FlexpartModel().run(
            RECEPTOR,
            config(checkout),
            FlexpartMetConfig(directory=gfs),
            WINDOW,
            workdir,
            timeout=1,
        )
    assert caught.value.reason == "TIMEOUT"


def test_run_refuses_missing_met_before_running(checkout, gfs, tmp_path):
    shutil.rmtree(checkout / "src")  # would fail if it ran
    window = (dt.datetime(2023, 6, 2), dt.datetime(2023, 6, 3))
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(MeteorologyError):
        FlexpartModel().run(
            RECEPTOR,
            config(checkout),
            FlexpartMetConfig(directory=gfs),
            window,
            workdir,
        )
    assert not np.any([p.name == "options" for p in workdir.iterdir()])


def test_a_project_runs_flexpart(checkout, gfs, tmp_path):
    from stilt import Project

    project = Project.init(
        tmp_path / "proj",
        receptors=[RECEPTOR],
        model="flexpart",
        flexpart_dir=str(checkout),
        mets={"gfs": {"directory": str(gfs)}},
        variants={"gfs": {}},
    )

    project.run(compute_root=tmp_path / "compute")

    assert project.status()["state"].tolist() == ["complete"]
    # A new Project reads the result back from the output directory.
    particles = Project(tmp_path / "proj").particles()
    assert len(particles) == 5 * 24
    assert "foot" in particles.columns
