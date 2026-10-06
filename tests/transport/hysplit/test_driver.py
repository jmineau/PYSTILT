"""Tests for HYSPLIT's input files, running hycs_std, and reading its particles."""

from pathlib import Path

import pytest

from stilt.exceptions import HYSPLITNotFoundError, SimulationError
from stilt.receptors import ColumnReceptor, MultiPointReceptor
from stilt.transport.hysplit import (
    FailureReason,
    HysplitConfig,
    HysplitModel,
    driver,
    model,
    read_particle_dat,
    write_inputs,
)
from stilt.transport.hysplit.control import ControlFile

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
    assert loaded.receptor.coords() == multipoint_receptor.coords()


def test_control_file_read_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ControlFile.read(tmp_path / "CONTROL")


# ---------------------------------------------------------------------------
# Inputs, runs, and particle files
# ---------------------------------------------------------------------------

VARS = ["time", "indx", "long", "lati", "zagl", "foot"]


def _config(**overrides) -> HysplitConfig:
    """A small config whose particle rows are the six columns of VARS."""
    return HysplitConfig(
        **{"n_hours": -24, "numpar": 10, "hnf_plume": False, "varsiwant": VARS}
        | overrides
    )


def _fake_build(path: Path, marker: str = "fake binary") -> Path:
    """Return a build directory holding a stand-in hycs_std."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "hycs_std").write_text(marker)
    return path


def _inputs(tmp_path, receptor, config=None) -> Path:
    """Write the input files for *receptor* and return the working directory."""
    workdir = tmp_path / "run"
    write_inputs(workdir, receptor, config or _config(), [tmp_path / "met" / "dummy"])
    return workdir


def _write_particle_dat(path: Path, rows: list[list[float]]) -> None:
    """Write a minimal PARTICLE_STILT.DAT with one dummy header line."""
    lines = ["header"]
    lines.extend(" ".join(str(v) for v in row) for row in rows)
    path.write_text("\n".join(lines) + "\n")


class _FakeMet:
    """A met that needs no files."""

    def __init__(self, *args):
        pass

    def files_for(self, window, hour_after=False):
        return []

    def readable(self, files):
        return files


# ---------------------------------------------------------------------------
# read_particle_dat
# ---------------------------------------------------------------------------


def test_read_particle_dat_parses_the_columns(tmp_path):
    dat = tmp_path / "PARTICLE_STILT.DAT"
    _write_particle_dat(
        dat,
        rows=[
            [-60, 1, -111.9, 40.7, 10.0, 1e-5],
            [-120, 1, -112.0, 40.6, 20.0, 2e-5],
        ],
    )

    df = read_particle_dat(dat, VARS)
    assert len(df) == 2
    # HYSPLIT's codes indx, long, lati are the table's particle, lon, lat.
    assert list(df.columns) == ["time", "particle", "lon", "lat", "zagl", "foot"]
    assert df["particle"].iloc[0] == 1
    assert dat.exists()  # the worker removes the working directory, or keeps it


def test_read_particle_dat_rejects_other_columns(tmp_path):
    dat = tmp_path / "PARTICLE_STILT.DAT"
    _write_particle_dat(dat, rows=[[-60, 1, -111.9, 40.7, 10.0, 1e-5]])

    with pytest.raises(ValueError, match="6 columns, expected 2"):
        read_particle_dat(dat, ["time", "indx"])


def test_read_particle_dat_of_an_empty_file_is_an_empty_table(tmp_path):
    dat = tmp_path / "PARTICLE_STILT.DAT"
    _write_particle_dat(dat, rows=[])

    df = read_particle_dat(dat, VARS)
    assert df.empty
    assert list(df.columns) == ["time", "particle", "lon", "lat", "zagl", "foot"]


# ---------------------------------------------------------------------------
# Running hycs_std
# ---------------------------------------------------------------------------


def _script(workdir: Path, text: str) -> None:
    """Put a shell script where hycs_std is run from."""
    workdir.mkdir(parents=True, exist_ok=True)
    exe = workdir / "hycs_std"
    exe.write_text("#!/usr/bin/env bash\n" + text)
    exe.chmod(0o755)


def test_run_keeps_fortran_runtime_output_on_failure(tmp_path):
    _script(
        tmp_path,
        "echo 'Fortran runtime error: File already opened in another unit'\nexit 2\n",
    )

    with pytest.raises(SimulationError) as caught:
        driver._run_hycs_std(tmp_path, timeout=5)
    assert caught.value.reason == FailureReason.FORTRAN_RUNTIME_ERROR
    assert "Fortran runtime error" in (tmp_path / "stilt.log").read_text()
    # The message is HYSPLIT's own line, so a record says what went wrong.
    assert str(caught.value) == (
        "HYSPLIT: Fortran runtime error: File already opened in another unit"
    )


def test_run_times_out_and_keeps_log_output(tmp_path):
    _script(tmp_path, "echo 'starting hycs_std'\nsleep 30\n")

    with pytest.raises(SimulationError, match="timeout") as caught:
        driver._run_hycs_std(tmp_path, timeout=1)
    assert caught.value.reason == FailureReason.TIMEOUT
    assert "starting hycs_std" in (tmp_path / "stilt.log").read_text()


def test_terminate_escalates_when_group_kill_does_not_finish(monkeypatch):
    import signal
    import subprocess

    class FakeProc:
        pid = 1234

        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self, timeout: int) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="hycs_std", timeout=timeout)
            return 0

    proc = FakeProc()
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        driver.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig))
    )

    driver._terminate(proc)  # type: ignore[arg-type]

    assert killpg_calls == [(proc.pid, signal.SIGTERM), (proc.pid, signal.SIGKILL)]
    assert proc.wait_calls == 2


# ---------------------------------------------------------------------------
# HysplitModel.run: what a finished hycs_std leaves behind
# ---------------------------------------------------------------------------


def _fake_hysplit(monkeypatch, *, log: str, rows: list[list[float]] | None) -> list:
    """Replace hycs_std with one that writes *log* and particle *rows* (none: no file)."""
    calls: list[int | None] = []

    def fake_run(workdir: Path, timeout: int | None) -> None:
        calls.append(timeout)
        (workdir / "stilt.log").write_text(log)
        if rows is not None:
            _write_particle_dat(workdir / "PARTICLE_STILT.DAT", rows)

    monkeypatch.setattr(model, "_run_hycs_std", fake_run)
    monkeypatch.setattr(model, "Met", _FakeMet)
    return calls


def _ending_at(minute: int) -> list[list[float]]:
    """Particle rows of one particle whose last row is *minute* before release."""
    return [[-1, 1, -111.9, 40.7, 10.0, 0.0], [-minute, 1, -112.0, 40.6, 20.0, 0.0]]


def _run(tmp_path, receptor, config=None):
    from stilt.meteorology import MetConfig, run_window

    config = config or _config()
    met = MetConfig(directory=tmp_path, file_format="%Y%m%d_%H", file_tres="1h")
    window = run_window(receptor.time, config.n_hours)
    (tmp_path / "run").mkdir(exist_ok=True)
    return HysplitModel().run(
        receptor, config, met, window, tmp_path / "run", timeout=5
    )


def test_run_fails_when_the_met_is_cut_short(tmp_path, point_receptor, monkeypatch):
    _fake_hysplit(
        monkeypatch,
        log=" WARNING metset: Only one time period of meteo data\n",
        rows=_ending_at(13 * 60),
    )

    with pytest.raises(SimulationError) as caught:
        _run(tmp_path, point_receptor)

    assert caught.value.reason == FailureReason.MET_TRUNCATED
    log = (tmp_path / "run" / "stilt.log").read_text()
    assert "particles stop 13 h into a 24 h run" in log


def test_run_keeps_a_run_that_reaches_the_end_past_a_damaged_met_file(
    tmp_path, point_receptor, monkeypatch
):
    _fake_hysplit(
        monkeypatch,
        log=" WARNING metset: Only one time period of meteo data\n",
        rows=_ending_at(24 * 60),
    )

    assert _run(tmp_path, point_receptor).particles["time"].min() == -24 * 60


def test_run_leaves_an_empty_particle_file_to_the_caller(
    tmp_path, point_receptor, monkeypatch
):
    _fake_hysplit(
        monkeypatch,
        log=" WARNING metset: Only one time period of meteo data\n",
        rows=[],
    )

    assert _run(tmp_path, point_receptor).particles.empty


def test_run_keeps_particles_that_left_the_met_domain(
    tmp_path, point_receptor, monkeypatch
):
    _fake_hysplit(monkeypatch, log="", rows=_ending_at(13 * 60))

    assert _run(tmp_path, point_receptor).particles["time"].min() == -13 * 60


def test_run_without_a_particle_file_fails(tmp_path, point_receptor, monkeypatch):
    _fake_hysplit(monkeypatch, log="", rows=None)

    with pytest.raises(SimulationError, match="PARTICLE_STILT.DAT") as caught:
        _run(tmp_path, point_receptor)
    assert caught.value.reason == FailureReason.NO_PARTICLE_DATA


def test_a_perturbed_run_is_one_hycs_std_call_with_winderr(
    tmp_path, point_receptor, monkeypatch
):
    calls = _fake_hysplit(
        monkeypatch, log="", rows=[[-60, 1, -111.9, 40.7, 10.0, 1e-5]]
    )
    config = _config(siguverr=1.0, tluverr=60.0, zcoruverr=500.0, horcoruverr=40.0)

    result = _run(tmp_path, point_receptor, config)

    workdir = tmp_path / "run"
    assert calls == [5]
    assert (workdir / "WINDERR").exists()
    assert not (workdir / "ZIERR").exists()
    assert "winderrtf=1" in (workdir / "SETUP.CFG").read_text().lower()
    assert len(result.particles) == 1


# ---------------------------------------------------------------------------
# write_inputs: SETUP.CFG
# ---------------------------------------------------------------------------


def _setup_cfg(tmp_path, receptor, config=None) -> str:
    return (_inputs(tmp_path, receptor, config) / "SETUP.CFG").read_text().lower()


def test_setup_cfg_holds_the_settings_and_kmsl(tmp_path, point_receptor):
    content = _setup_cfg(tmp_path, point_receptor)
    assert "numpar" in content
    assert "varsiwant" in content
    assert "kmsl=0" in content
    assert "seed=" not in content


def test_setup_cfg_includes_the_seed_when_configured(tmp_path, point_receptor):
    content = _setup_cfg(tmp_path, point_receptor, HysplitConfig(seed=17, krand=2))
    assert "seed=-18" in content  # -(|seed|+1): the value HYSPLIT honours
    assert "winderrtf=0" in content


def test_setup_cfg_sets_winderrtf_from_error_settings(tmp_path, point_receptor):
    config = _config(siguverr=1.0, tluverr=60.0, zcoruverr=500.0, horcoruverr=40.0)
    assert "winderrtf=1" in _setup_cfg(tmp_path, point_receptor, config)


def test_setup_cfg_takes_kmsl_from_an_msl_receptor(tmp_path, point_receptor):
    receptor = point_receptor.model_copy(
        update={"altitude": 1500.0, "altitude_ref": "msl"}
    )
    assert "kmsl=1" in _setup_cfg(tmp_path, receptor)


def test_kmsl_comes_only_from_the_receptor():
    """KMSL follows each receptor's altitude_ref; there is no setting to contradict it."""
    with pytest.raises(ValueError, match="kmsl"):
        HysplitConfig(kmsl=0)


# ---------------------------------------------------------------------------
# write_inputs: CONTROL and the error and mixed-layer files
# ---------------------------------------------------------------------------


def test_inputs_hold_control_setup_and_the_binary(tmp_path, point_receptor):
    workdir = _inputs(tmp_path, point_receptor)

    assert (workdir / "CONTROL").exists()
    assert (workdir / "SETUP.CFG").exists()
    assert (workdir / "hycs_std").is_symlink()
    for name in ("WINDERR", "ZIERR", "ZICONTROL"):
        assert not (workdir / name).exists()


def test_winderr_lines_follow_the_settings_order(tmp_path, point_receptor):
    config = _config(siguverr=1.0, tluverr=60.0, zcoruverr=500.0, horcoruverr=40.0)
    lines = (_inputs(tmp_path, point_receptor, config) / "WINDERR").read_text()
    assert lines.split() == ["1.0", "60.0", "500.0", "40.0"]


def test_zierr_lines_follow_the_settings_order(tmp_path, point_receptor):
    config = _config(sigzierr=0.6, tlzierr=60.0, horcorzierr=40.0)
    lines = (_inputs(tmp_path, point_receptor, config) / "ZIERR").read_text()
    assert lines.split() == ["0.6", "60.0", "40.0"]


def test_zicontrol_holds_the_hourly_factors(tmp_path, point_receptor):
    config = _config(ziscale=[0.8, 0.8, 0.9])
    zicontrol = _inputs(tmp_path, point_receptor, config) / "ZICONTROL"
    assert zicontrol.read_text().strip().splitlines() == ["3", "0.8", "0.8", "0.9"]


def test_zicontrol_repeats_a_scalar_over_the_run(tmp_path, point_receptor):
    config = _config(n_hours=-4, ziscale=0.8)
    lines = (_inputs(tmp_path, point_receptor, config) / "ZICONTROL").read_text()
    assert lines.strip().splitlines() == ["4", "0.8", "0.8", "0.8", "0.8"]


# ---------------------------------------------------------------------------
# write_inputs: the HYSPLIT build and its data tables
# ---------------------------------------------------------------------------


def test_exe_dir_is_used(tmp_path, point_receptor):
    build = _fake_build(tmp_path / "my_build", "patched")
    workdir = _inputs(tmp_path, point_receptor, _config(exe_dir=build))
    assert (workdir / "hycs_std").read_text() == "patched"


def test_default_is_the_bundled_binary(tmp_path, point_receptor):
    workdir = _inputs(tmp_path, point_receptor)
    assert (workdir / "hycs_std").resolve().parent.name in {"linux_x64", "macos_x64"}


def test_exe_dir_without_hycs_std_raises(tmp_path, point_receptor):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(HYSPLITNotFoundError, match="hycs_std"):
        _inputs(tmp_path, point_receptor, _config(exe_dir=empty))


def test_only_hycs_std_is_linked_from_a_custom_build_dir(tmp_path, point_receptor):
    # A real HYSPLIT exec/ directory holds dozens of other programs.
    build = _fake_build(tmp_path / "exec")
    (build / "Makefile").write_text("")
    (build / "concplot").write_text("")
    workdir = _inputs(tmp_path, point_receptor, _config(exe_dir=build))
    assert (workdir / "hycs_std").exists()
    assert not (workdir / "Makefile").exists()
    assert not (workdir / "concplot").exists()
    assert "exe_dir" not in (workdir / "SETUP.CFG").read_text().lower()


def test_data_dir_tables_replace_the_bundled_ones(tmp_path, point_receptor):
    """A table in data_dir is linked in place of the bundled one; the rest are bundled."""
    data = tmp_path / "tables"
    data.mkdir()
    (data / "ROUGLEN.ASC").write_text("custom roughness\n")

    workdir = _inputs(tmp_path, point_receptor, _config(data_dir=data))

    assert (workdir / "ROUGLEN.ASC").resolve() == (data / "ROUGLEN.ASC").resolve()
    bundled = driver._bundled_data_dir()
    assert (workdir / "LANDUSE.ASC").resolve() == (bundled / "LANDUSE.ASC").resolve()


# ---------------------------------------------------------------------------
# Bundled binary selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("system", "machine", "subdir"),
    [
        ("Linux", "x86_64", "linux_x64"),
        ("Darwin", "x86_64", "macos_x64"),
        ("Darwin", "arm64", "macos_x64"),  # runs through Rosetta
    ],
)
def test_bundled_exe_dir_picks_the_build_for_the_platform(
    monkeypatch, system, machine, subdir
):
    monkeypatch.setattr(driver.platform, "system", lambda: system)
    monkeypatch.setattr(driver.platform, "machine", lambda: machine)
    assert driver._bundled_exe_dir().name == subdir


@pytest.mark.parametrize(
    ("system", "machine"),
    [("Linux", "aarch64"), ("Windows", "AMD64"), ("Linux", "ppc64le")],
)
def test_bundled_exe_dir_rejects_platforms_without_a_build(
    monkeypatch, system, machine
):
    """An aarch64 Linux machine must not be handed the x86-64 binary (#61)."""
    monkeypatch.setattr(driver.platform, "system", lambda: system)
    monkeypatch.setattr(driver.platform, "machine", lambda: machine)
    with pytest.raises(HYSPLITNotFoundError, match="exe_dir"):
        driver._bundled_exe_dir()


def test_bundled_exe_dir_rejects_an_install_without_the_binary(monkeypatch, tmp_path):
    """The source archive carries no hycs_std; say so rather than fail later."""
    (tmp_path / "bin" / "linux_x64").mkdir(parents=True)
    monkeypatch.setattr(driver.platform, "system", lambda: "Linux")
    monkeypatch.setattr(driver.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(driver, "pkg_files", lambda _package: tmp_path)
    with pytest.raises(HYSPLITNotFoundError, match="exe_dir"):
        driver._bundled_exe_dir()
