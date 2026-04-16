"""Tests for stilt.hysplit low-level helpers."""

from pathlib import Path

import pytest

from stilt.config import STILTParams
from stilt.errors import HYSPLITFailureError, NoParticleOutputError
from stilt.hysplit import ControlFile, HYSPLITRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_control(
    receptor, n_hours=-24, met_files=None, emisshrs=0.01, w_option=0, z_top=25000.0
):
    if met_files is None:
        met_files = [Path("/met/hrrr_2023010100")]
    return ControlFile(
        receptor=receptor,
        emisshrs=emisshrs,
        n_hours=n_hours,
        w_option=w_option,
        z_top=z_top,
        met_files=met_files,
    )


# ---------------------------------------------------------------------------
# ControlFile.write / .read roundtrip - single-point receptor
# ---------------------------------------------------------------------------


def test_control_file_roundtrip_single_point(point_receptor, tmp_path):
    cf = _make_control(point_receptor)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)

    assert loaded.receptor == point_receptor
    assert loaded.n_hours == -24
    assert loaded.w_option == 0
    assert loaded.z_top == 25000.0
    assert len(loaded.met_files) == 1


def test_control_file_roundtrip_emisshrs(point_receptor, tmp_path):
    cf = _make_control(point_receptor, emisshrs=0.5)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)
    assert loaded.emisshrs == pytest.approx(0.5)


def test_control_file_roundtrip_fractional_emisshrs(point_receptor, tmp_path):
    cf = _make_control(point_receptor, emisshrs=0.01)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)
    assert loaded.emisshrs == pytest.approx(0.01)


def test_control_file_roundtrip_n_hours_forward(point_receptor, tmp_path):
    cf = _make_control(point_receptor, n_hours=24)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)
    assert loaded.n_hours == 24


def test_control_file_roundtrip_receptor_time(point_receptor, tmp_path):
    cf = _make_control(point_receptor)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)
    assert loaded.receptor.time == point_receptor.time


def test_control_file_roundtrip_receptor_coords(point_receptor, tmp_path):
    cf = _make_control(point_receptor)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)
    assert loaded.receptor.longitude == pytest.approx(point_receptor.longitude)
    assert loaded.receptor.latitude == pytest.approx(point_receptor.latitude)
    assert loaded.receptor.altitude == pytest.approx(point_receptor.altitude)


def test_control_file_roundtrip_multiple_met_files(point_receptor, tmp_path):
    met = [
        Path("/met/hrrr_2022123100"),
        Path("/met/hrrr_2023010100"),
    ]
    cf = _make_control(point_receptor, met_files=met)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)
    assert len(loaded.met_files) == 2
    assert loaded.met_files[0].name == "hrrr_2022123100"
    assert loaded.met_files[1].name == "hrrr_2023010100"


def test_control_file_roundtrip_column_receptor(column_receptor, tmp_path):
    cf = _make_control(column_receptor)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)
    assert loaded.receptor.kind == "column"
    assert loaded.receptor.bottom == pytest.approx(column_receptor.bottom)
    assert loaded.receptor.top == pytest.approx(column_receptor.top)


def test_control_file_roundtrip_multipoint_receptor(multipoint_receptor, tmp_path):
    cf = _make_control(multipoint_receptor)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)
    assert loaded.receptor.kind == "multipoint"
    assert len(loaded.receptor) == len(multipoint_receptor)


def test_control_file_read_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ControlFile.read(tmp_path / "CONTROL")


# ---------------------------------------------------------------------------
# read_particle_dat
# ---------------------------------------------------------------------------


def _write_particle_dat(path: Path, rows: list[list[float]]) -> None:
    """Write a minimal PARTICLE_STILT.DAT with one dummy header line."""
    lines = ["header"]
    lines.extend(" ".join(str(v) for v in row) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def _make_runner(tmp_path, point_receptor, rm_dat_default=True) -> HYSPLITRunner:
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        rm_dat=rm_dat_default,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    return HYSPLITRunner(
        directory=tmp_path,
        receptor=point_receptor,
        params=params,
        met_files=[tmp_path / "met" / "dummy"],
        exe_dir=tmp_path,
    )


def test_read_particles_parses_expected_columns(tmp_path, point_receptor):
    runner = _make_runner(tmp_path, point_receptor)
    dat = tmp_path / "PARTICLE_STILT.DAT"
    _write_particle_dat(
        dat,
        rows=[
            [-60, 1, -111.9, 40.7, 10.0, 1e-5],
            [-120, 1, -112.0, 40.6, 20.0, 2e-5],
        ],
    )

    df = runner._read_particles(rm_dat=False)
    assert len(df) == 2
    assert list(df.columns) == runner.params.varsiwant
    assert df["indx"].iloc[0] == 1


def test_read_particles_removes_dat_files_when_requested(tmp_path, point_receptor):
    runner = _make_runner(tmp_path, point_receptor)
    dat = tmp_path / "PARTICLE_STILT.DAT"
    dat2 = tmp_path / "PARTICLE.DAT"
    _write_particle_dat(dat, rows=[[-60, 1, -111.9, 40.7, 10.0, 1e-5]])
    dat2.write_text("unused\n")

    _ = runner._read_particles(rm_dat=True)
    assert not dat.exists()
    assert not dat2.exists()


def test_read_particles_raises_domain_error_when_file_missing(tmp_path, point_receptor):
    runner = _make_runner(tmp_path, point_receptor)

    with pytest.raises(NoParticleOutputError, match="PARTICLE_STILT.DAT"):
        runner._read_particles(rm_dat=False)


def test_run_persists_fortran_runtime_output_on_failure(tmp_path, point_receptor):
    runner = _make_runner(tmp_path, point_receptor)
    exe = tmp_path / "hycs_std"
    exe.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'Fortran runtime error: File already opened in another unit'\n"
        "exit 2\n"
    )
    exe.chmod(0o755)

    with pytest.raises(HYSPLITFailureError):
        runner._run(timeout=5)

    assert "Fortran runtime error" in runner.log_path.read_text()


# ---------------------------------------------------------------------------
# HYSPLITRunner._write_setup
# ---------------------------------------------------------------------------


def test_write_setup_creates_cfg(tmp_path, point_receptor):
    runner = _make_runner(tmp_path, point_receptor)
    runner._write_setup(winderrtf=0)
    cfg = tmp_path / "SETUP.CFG"
    assert cfg.exists()
    content = cfg.read_text()
    assert "numpar" in content.lower()
    assert "varsiwant" in content.lower()
    assert "kmsl=0" in content.lower()


def test_write_setup_sets_winderrtf(tmp_path, point_receptor):
    runner = _make_runner(tmp_path, point_receptor)
    runner._write_setup(winderrtf=3)
    cfg = tmp_path / "SETUP.CFG"
    content = cfg.read_text()
    assert "winderrtf" in content.lower()


def test_write_setup_removes_existing_cfg(tmp_path, point_receptor):
    runner = _make_runner(tmp_path, point_receptor)
    cfg = tmp_path / "SETUP.CFG"
    cfg.write_text("old content")
    runner._write_setup(winderrtf=0)
    assert "old content" not in cfg.read_text()


def test_write_setup_derives_kmsl_from_msl_receptor(tmp_path, point_receptor):
    receptor = point_receptor.__class__(
        time=point_receptor.time,
        longitude=point_receptor.longitude,
        latitude=point_receptor.latitude,
        altitude=1500.0,
        altitude_ref="msl",
    )
    runner = _make_runner(tmp_path, receptor)
    runner._write_setup(winderrtf=0)
    content = (tmp_path / "SETUP.CFG").read_text().lower()
    assert "kmsl=1" in content


def test_write_setup_rejects_conflicting_explicit_kmsl(tmp_path, point_receptor):
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        rm_dat=True,
        kmsl=0,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    receptor = point_receptor.__class__(
        time=point_receptor.time,
        longitude=point_receptor.longitude,
        latitude=point_receptor.latitude,
        altitude=1500.0,
        altitude_ref="msl",
    )
    runner = HYSPLITRunner(
        directory=tmp_path,
        receptor=receptor,
        params=params,
        met_files=[tmp_path / "met" / "dummy"],
        exe_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="conflicts with receptor altitude_ref"):
        runner._write_setup(winderrtf=0)


# ---------------------------------------------------------------------------
# HYSPLITRunner._write_winderr / _write_zierr
# ---------------------------------------------------------------------------


def _make_runner_with_xyerr(tmp_path, point_receptor) -> HYSPLITRunner:
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        siguverr=1.0,
        tluverr=60.0,
        zcoruverr=500.0,
        horcoruverr=40.0,
    )
    return HYSPLITRunner(
        directory=tmp_path,
        receptor=point_receptor,
        params=params,
        met_files=[],
        exe_dir=tmp_path,
    )


def _make_runner_with_zierr(tmp_path, point_receptor) -> HYSPLITRunner:
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        sigzierr=0.6,
        tlzierr=60.0,
        horcorzierr=40.0,
    )
    return HYSPLITRunner(
        directory=tmp_path,
        receptor=point_receptor,
        params=params,
        met_files=[],
        exe_dir=tmp_path,
    )


def test_write_winderr_creates_file(tmp_path, point_receptor):
    runner = _make_runner_with_xyerr(tmp_path, point_receptor)
    runner._write_winderr()
    winderr = tmp_path / "WINDERR"
    assert winderr.exists()
    lines = winderr.read_text().strip().splitlines()
    assert len(lines) == 4  # siguverr, tluverr, zcoruverr, horcoruverr


def test_write_winderr_no_op_without_xyerr(tmp_path, point_receptor):
    """No WINDERR file when XY params are None."""
    runner = _make_runner(tmp_path, point_receptor)
    runner._write_winderr()
    assert not (tmp_path / "WINDERR").exists()


def test_write_zierr_creates_file(tmp_path, point_receptor):
    runner = _make_runner_with_zierr(tmp_path, point_receptor)
    runner._write_zierr()
    zierr = tmp_path / "ZIERR"
    assert zierr.exists()
    lines = zierr.read_text().strip().splitlines()
    assert len(lines) == 3  # sigzierr, tlzierr, horcorzierr


def test_write_zierr_no_op_without_zierr(tmp_path, point_receptor):
    """No ZIERR file when ZI params are None."""
    runner = _make_runner(tmp_path, point_receptor)
    runner._write_zierr()
    assert not (tmp_path / "ZIERR").exists()


def test_write_zicontrol_creates_file_from_shared_vector(tmp_path, point_receptor):
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        zicontroltf=1,
        ziscale=[0.8, 0.8, 0.9],
    )
    runner = HYSPLITRunner(
        directory=tmp_path,
        receptor=point_receptor,
        params=params,
        met_files=[],
        exe_dir=tmp_path,
    )

    runner._write_zicontrol()

    zicontrol = tmp_path / "ZICONTROL"
    assert zicontrol.exists()
    assert zicontrol.read_text().strip().splitlines() == ["3", "0.8", "0.8", "0.9"]


def test_write_zicontrol_expands_scalar_to_run_length(tmp_path, point_receptor):
    params = STILTParams(
        n_hours=-4,
        numpar=10,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        zicontroltf=1,
        ziscale=0.8,
    )
    runner = HYSPLITRunner(
        directory=tmp_path,
        receptor=point_receptor,
        params=params,
        met_files=[],
        exe_dir=tmp_path,
    )

    runner._write_zicontrol()

    lines = (tmp_path / "ZICONTROL").read_text().strip().splitlines()
    assert lines == ["4", "0.8", "0.8", "0.8", "0.8"]


def test_write_zicontrol_rejects_multiple_nested_vectors(tmp_path, point_receptor):
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        zicontroltf=1,
        ziscale=[[0.8, 0.8], [0.9, 0.9]],
    )
    runner = HYSPLITRunner(
        directory=tmp_path,
        receptor=point_receptor,
        params=params,
        met_files=[],
        exe_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="Per-simulation ziscale lists"):
        runner._write_zicontrol()


# ---------------------------------------------------------------------------
# HYSPLITRunner.prepare()
# ---------------------------------------------------------------------------


def test_prepare_writes_control_and_setup(tmp_path, point_receptor):
    """prepare() creates CONTROL and SETUP.CFG in the sim directory."""
    exe_dir = tmp_path / "exe"
    exe_dir.mkdir()
    (exe_dir / "hycs_std").write_text("fake binary")

    sim_dir = tmp_path / "sim"
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    runner = HYSPLITRunner(
        directory=sim_dir,
        receptor=point_receptor,
        params=params,
        met_files=[tmp_path / "met" / "dummy"],
        exe_dir=exe_dir,
    )
    runner.prepare()

    assert (sim_dir / "CONTROL").exists()
    assert (sim_dir / "SETUP.CFG").exists()
    assert (sim_dir / "hycs_std").is_symlink()


def test_prepare_writes_zicontrol_when_enabled(tmp_path, point_receptor):
    exe_dir = tmp_path / "exe"
    exe_dir.mkdir()
    (exe_dir / "hycs_std").write_text("fake binary")

    sim_dir = tmp_path / "sim"
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        zicontroltf=1,
        ziscale=[1.0] * 24,
    )
    runner = HYSPLITRunner(
        directory=sim_dir,
        receptor=point_receptor,
        params=params,
        met_files=[tmp_path / "met" / "dummy"],
        exe_dir=exe_dir,
    )

    runner.prepare()

    zicontrol = sim_dir / "ZICONTROL"
    assert zicontrol.exists()
    lines = zicontrol.read_text().strip().splitlines()
    assert lines[0] == "24"
