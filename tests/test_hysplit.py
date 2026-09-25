"""Tests for stilt.hysplit low-level helpers."""

from pathlib import Path

import pytest

from stilt.config import STILTParams
from stilt.errors import (
    HYSPLITFailureError,
    HYSPLITTimeoutError,
    NoParticleOutputError,
)
from stilt.hysplit import HYSPLITDriver
from stilt.hysplit.control import ControlFile
from stilt.receptors import ColumnReceptor, MultiPointReceptor

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
    assert isinstance(loaded.receptor, ColumnReceptor)
    assert loaded.receptor.bottom == pytest.approx(column_receptor.bottom)
    assert loaded.receptor.top == pytest.approx(column_receptor.top)


def test_control_file_roundtrip_multipoint_receptor(multipoint_receptor, tmp_path):
    cf = _make_control(multipoint_receptor)
    path = tmp_path / "CONTROL"
    cf.write(path)
    loaded = ControlFile.read(path)
    assert isinstance(loaded.receptor, MultiPointReceptor)
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


def _make_runner(tmp_path, point_receptor, rm_dat_default=True) -> HYSPLITDriver:
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        hnf_plume=False,
        rm_dat=rm_dat_default,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    return HYSPLITDriver(
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
    runner.log_path.write_text("previous attempt\n")
    exe = tmp_path / "hycs_std"
    exe.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'Fortran runtime error: File already opened in another unit'\n"
        "exit 2\n"
    )
    exe.chmod(0o755)

    with pytest.raises(HYSPLITFailureError):
        runner._run(timeout=5)

    log_text = runner.log_path.read_text()
    assert "previous attempt" in log_text
    assert "Fortran runtime error" in log_text


def test_run_raises_clear_error_when_executable_missing(tmp_path, point_receptor):
    runner = _make_runner(tmp_path, point_receptor)

    with pytest.raises(FileNotFoundError, match="HYSPLIT executable not found"):
        runner._run(timeout=5)


def test_run_times_out_and_keeps_labeled_log_output(tmp_path, point_receptor):
    runner = _make_runner(tmp_path, point_receptor)
    exe = tmp_path / "hycs_std"
    exe.write_text("#!/usr/bin/env bash\necho 'starting main run'\nsleep 30\n")
    exe.chmod(0o755)

    with pytest.raises(HYSPLITTimeoutError, match="timed out"):
        runner._run(timeout=1)

    log_text = runner.log_path.read_text()
    assert "=== main run ===" in log_text
    assert "starting main run" in log_text


def test_execute_ignores_stale_main_particles_after_error_run_timeout(
    tmp_path, point_receptor, monkeypatch
):
    runner = HYSPLITDriver(
        directory=tmp_path,
        receptor=point_receptor,
        params=STILTParams(
            n_hours=-24,
            numpar=10,
            hnf_plume=False,
            rm_dat=False,
            siguverr=1.0,
            tluverr=60.0,
            zcoruverr=500.0,
            horcoruverr=40.0,
            varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        ),
        met_files=[tmp_path / "met" / "dummy"],
        exe_dir=tmp_path,
    )

    def fake_run(timeout: int | None, *, label: str = "main") -> None:
        if label == "main":
            _write_particle_dat(
                runner.particle_stilt_path,
                rows=[[-60, 1, -111.9, 40.7, 10.0, 1e-5]],
            )
            return
        raise HYSPLITTimeoutError("error run timed out", str(tmp_path))

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_write_winderr", lambda: None)
    monkeypatch.setattr(runner, "_write_zierr", lambda: None)
    monkeypatch.setattr(runner, "_write_setup", lambda winderrtf: None)

    result = runner.execute(timeout=5, rm_dat=False)

    assert len(result.particles) == 1
    assert result.error_particles == {}
    assert "=== error run [0] failed ===" in runner.log_path.read_text()


def test_terminate_process_escalates_when_group_kill_does_not_finish(
    tmp_path, point_receptor, monkeypatch
):
    import subprocess

    runner = _make_runner(tmp_path, point_receptor)

    class FakeProc:
        pid = 1234

        def __init__(self) -> None:
            self.kill_calls = 0
            self.wait_calls = 0

        def wait(self, timeout: int) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="hycs_std", timeout=timeout)
            return 0

        def kill(self) -> None:
            self.kill_calls += 1

    proc = FakeProc()
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "stilt.hysplit.driver.os.killpg",
        lambda pid, sig: killpg_calls.append((pid, sig)),
    )

    runner._terminate_process(proc)

    import signal

    assert killpg_calls == [(proc.pid, signal.SIGTERM), (proc.pid, signal.SIGKILL)]
    assert proc.wait_calls == 2


# ---------------------------------------------------------------------------
# HYSPLITDriver._write_setup
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
    assert "seed=" not in content.lower()


def test_write_setup_includes_seed_when_configured(tmp_path, point_receptor):
    runner = HYSPLITDriver(
        directory=tmp_path,
        receptor=point_receptor,
        params=STILTParams(seed=17),
        met_files=[tmp_path / "met" / "dummy"],
        exe_dir=tmp_path,
    )
    runner._write_setup(winderrtf=0)
    content = (tmp_path / "SETUP.CFG").read_text().lower()

    assert "seed=17" in content


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
        hnf_plume=False,
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
    runner = HYSPLITDriver(
        directory=tmp_path,
        receptor=receptor,
        params=params,
        met_files=[tmp_path / "met" / "dummy"],
        exe_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="conflicts with receptor altitude_ref"):
        runner._write_setup(winderrtf=0)


# ---------------------------------------------------------------------------
# HYSPLITDriver._write_winderr / _write_zierr
# ---------------------------------------------------------------------------


def _make_runner_with_xyerr(tmp_path, point_receptor) -> HYSPLITDriver:
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        hnf_plume=False,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        siguverr=1.0,
        tluverr=60.0,
        zcoruverr=500.0,
        horcoruverr=40.0,
    )
    return HYSPLITDriver(
        directory=tmp_path,
        receptor=point_receptor,
        params=params,
        met_files=[],
        exe_dir=tmp_path,
    )


def _make_runner_with_zierr(tmp_path, point_receptor) -> HYSPLITDriver:
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        hnf_plume=False,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        sigzierr=0.6,
        tlzierr=60.0,
        horcorzierr=40.0,
    )
    return HYSPLITDriver(
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
        hnf_plume=False,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        ziscale=[0.8, 0.8, 0.9],
    )
    runner = HYSPLITDriver(
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
        hnf_plume=False,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        ziscale=0.8,
    )
    runner = HYSPLITDriver(
        directory=tmp_path,
        receptor=point_receptor,
        params=params,
        met_files=[],
        exe_dir=tmp_path,
    )

    runner._write_zicontrol()

    lines = (tmp_path / "ZICONTROL").read_text().strip().splitlines()
    assert lines == ["4", "0.8", "0.8", "0.8", "0.8"]


def test_write_zicontrol_skips_file_when_unscaled(tmp_path, point_receptor):
    (tmp_path / "ZICONTROL").write_text("stale")
    runner = _make_runner(tmp_path, point_receptor)

    runner._write_zicontrol()

    assert not (tmp_path / "ZICONTROL").exists()


# ---------------------------------------------------------------------------
# HYSPLITDriver.prepare()
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
        hnf_plume=False,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    runner = HYSPLITDriver(
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
        hnf_plume=False,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
        ziscale=[0.9] * 24,
    )
    runner = HYSPLITDriver(
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


# ---------------------------------------------------------------------------
# Custom HYSPLIT build (STILTParams.exe_dir)
# ---------------------------------------------------------------------------


def _fake_build(path, marker="fake binary"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "hycs_std").write_text(marker)
    return path


def _exe_driver(tmp_path, point_receptor, *, params_exe=None, arg_exe=None, sim="sim"):
    params = STILTParams(
        n_hours=-24,
        numpar=10,
        hnf_plume=False,
        exe_dir=params_exe,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot"],
    )
    return HYSPLITDriver(
        directory=tmp_path / sim,
        receptor=point_receptor,
        params=params,
        met_files=[tmp_path / "met" / "dummy"],
        exe_dir=arg_exe,
    )


def test_exe_dir_from_params_is_used(tmp_path, point_receptor):
    build = _fake_build(tmp_path / "my_build", "patched")
    runner = _exe_driver(tmp_path, point_receptor, params_exe=build)
    runner.prepare()
    assert (tmp_path / "sim" / "hycs_std").read_text() == "patched"


def test_explicit_exe_dir_argument_beats_params(tmp_path, point_receptor):
    from_params = _fake_build(tmp_path / "a", "from params")
    from_arg = _fake_build(tmp_path / "b", "from argument")
    runner = _exe_driver(
        tmp_path, point_receptor, params_exe=from_params, arg_exe=from_arg
    )
    runner.prepare()
    assert (tmp_path / "sim" / "hycs_std").read_text() == "from argument"


def test_default_is_the_bundled_binary(tmp_path, point_receptor):
    runner = _exe_driver(tmp_path, point_receptor)
    assert runner.exe_dir.name in {"linux_x64", "macos_x64"}


def test_exe_dir_without_hycs_std_raises(tmp_path, point_receptor):
    empty = tmp_path / "empty"
    empty.mkdir()
    runner = _exe_driver(tmp_path, point_receptor, params_exe=empty)
    with pytest.raises(FileNotFoundError, match="hycs_std"):
        runner.prepare()


def test_only_hycs_std_is_linked_from_a_custom_build_dir(tmp_path, point_receptor):
    # A real HYSPLIT exec/ directory holds dozens of other programs.
    build = _fake_build(tmp_path / "exec")
    (build / "Makefile").write_text("")
    (build / "concplot").write_text("")
    runner = _exe_driver(tmp_path, point_receptor, params_exe=build)
    runner.prepare()
    sim = tmp_path / "sim"
    assert (sim / "hycs_std").exists()
    assert not (sim / "Makefile").exists()
    assert not (sim / "concplot").exists()


def test_reused_sim_directory_is_relinked_to_the_new_build(tmp_path, point_receptor):
    old = _fake_build(tmp_path / "old", "old build")
    new = _fake_build(tmp_path / "new", "new build")
    _exe_driver(tmp_path, point_receptor, params_exe=old).prepare()
    assert (tmp_path / "sim" / "hycs_std").read_text() == "old build"
    _exe_driver(tmp_path, point_receptor, params_exe=new).prepare()
    assert (tmp_path / "sim" / "hycs_std").read_text() == "new build"


def test_exe_dir_is_not_written_to_setup_cfg(tmp_path, point_receptor):
    build = _fake_build(tmp_path / "my_build")
    runner = _exe_driver(tmp_path, point_receptor, params_exe=build)
    runner.prepare()
    assert "exe_dir" not in (tmp_path / "sim" / "SETUP.CFG").read_text().lower()


# -- error realizations ------------------------------------------------------------

_ERR_VARS = ["time", "indx", "long", "lati", "zagl", "foot"]


def _error_runner(tmp_path, point_receptor, **overrides) -> HYSPLITDriver:
    params = dict(
        n_hours=-24,
        numpar=10,
        rm_dat=False,
        hnf_plume=False,  # the six-column particle rows below carry no plume vars
        siguverr=1.0,
        tluverr=60.0,
        zcoruverr=500.0,
        horcoruverr=40.0,
        varsiwant=_ERR_VARS,
    )
    params.update(overrides)
    return HYSPLITDriver(
        directory=tmp_path,
        receptor=point_receptor,
        params=STILTParams(**params),
        met_files=[tmp_path / "met" / "dummy"],
        exe_dir=tmp_path,
    )


def test_execute_runs_one_error_pass_per_realization(
    tmp_path, point_receptor, monkeypatch
):
    runner = _error_runner(tmp_path, point_receptor, krand=4, error_realizations=3)
    labels: list[str] = []
    setups: list[int] = []

    def fake_run(timeout: int | None, *, label: str = "main") -> None:
        labels.append(label)
        foot = 1e-5 if label == "main" else 1e-5 * (len(labels) + 1)
        _write_particle_dat(
            runner.particle_stilt_path, rows=[[-60, 1, -111.9, 40.7, 10.0, foot]]
        )

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_write_winderr", lambda: None)
    monkeypatch.setattr(runner, "_write_zierr", lambda: None)
    monkeypatch.setattr(
        runner, "_write_setup", lambda winderrtf: setups.append(winderrtf)
    )

    result = runner.execute(timeout=5, rm_dat=False, error_realizations=[0, 1, 2])

    assert labels == ["main", "error[0]", "error[1]", "error[2]"]
    assert setups == [1]  # SETUP.CFG written once for the error passes (XY only)
    assert len(result.particles) == 1
    assert sorted(result.error_particles) == [0, 1, 2]
    foots = [float(result.error_particles[k]["foot"].iloc[0]) for k in (0, 1, 2)]
    assert len(set(foots)) == 3  # each realization parsed its own particle file


def test_one_failed_realization_does_not_lose_the_others(
    tmp_path, point_receptor, monkeypatch
):
    runner = _error_runner(tmp_path, point_receptor, krand=4, error_realizations=3)

    def fake_run(timeout: int | None, *, label: str = "main") -> None:
        if label == "error[1]":
            raise HYSPLITTimeoutError("realization 1 timed out", str(tmp_path))
        _write_particle_dat(
            runner.particle_stilt_path, rows=[[-60, 1, -111.9, 40.7, 10.0, 1e-5]]
        )

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "_write_winderr", lambda: None)
    monkeypatch.setattr(runner, "_write_zierr", lambda: None)
    monkeypatch.setattr(runner, "_write_setup", lambda winderrtf: None)

    result = runner.execute(timeout=5, rm_dat=False, error_realizations=[0, 1, 2])

    assert sorted(result.error_particles) == [0, 2]
    assert "=== error run [1] failed ===" in runner.log_path.read_text()
