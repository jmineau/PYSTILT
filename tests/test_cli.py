"""Tests for runtraj.cli - Typer command-line interface."""

import errno
import json
from dataclasses import replace

import pandas as pd
import pytest
import yaml
from runtraj.config import ProjectConfig
from runtraj.execution.config import ExecutionConfig
from runtraj.project import Project
from runtraj.spatial import Grid
from typer.testing import CliRunner

import stilt.__main__
import stilt.cli
from stilt.cli import app
from stilt.project import STILT_R_HYSPLIT
from tests.fixtures.factories import blocks

from .fixtures.factories import make_hysplit_dir, make_met_config, make_project_config

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path):
    """Write a minimal config.yaml + releases.csv so _resolve_project succeeds."""
    cfg = make_project_config(
        tmp_path,
        # Local, with the time limit a Slurm run of it needs.
        execution={"slurm": {"time": "01:00:00"}},
    )
    cfg.to_yaml(tmp_path / "config.yaml")
    (tmp_path / "releases.csv").write_text(
        "time,longitude,latitude,altitude\n2023-01-01 12:00:00,-111.85,40.77,5.0\n"
    )


class _Calls(list):
    """What `stilt run` asked of the project; ``states`` is the status the fake run returns."""

    states: list[str] = []


@pytest.fixture
def calls(monkeypatch):
    """Record what `stilt run` asks of the project, and run nothing."""
    recorded = _Calls()

    def run(self, **kwargs):
        recorded.append(("run", kwargs))
        n = len(recorded.states)
        return pd.DataFrame(
            {
                "release": [f"r{i}" for i in range(n)],
                "variant": ["hrrr"] * n,
                "state": recorded.states,
                "reason": [
                    "met_missing" if s == "failed" else None for s in recorded.states
                ],
            }
        )

    def submit(self, **kwargs):
        recorded.append(("submit", kwargs))
        return "12345"

    monkeypatch.setattr("runtraj.cli.Project.run", run)
    monkeypatch.setattr("runtraj.cli.Project.submit", submit)
    return recorded


def test_python_m_entrypoint_invokes_cli(monkeypatch):
    called = False

    def fake_app() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(stilt.__main__, "app", fake_app)

    stilt.__main__.main()

    assert called is True


# ---------------------------------------------------------------------------
# _resolve_project helper
# ---------------------------------------------------------------------------


def test_resolve_project_exits_when_no_config_yaml(tmp_path):
    """Exits with code 2 when no config.yaml is found: the command line named no project."""
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def test_status_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 2


@pytest.mark.parametrize("command", ["check", "status", "run", "fetch"])
def test_commands_report_invalid_config_yaml(tmp_path, command):
    """A malformed project config is a usage error, not an internal traceback (#364)."""
    (tmp_path / "config.yaml").write_text("n_hours: [\n")

    result = runner.invoke(app, [command, str(tmp_path)])

    assert result.exit_code == 2
    assert result.output == (
        "Error: config.yaml is not valid YAML at line 2, column 1: "
        "expected the node content, but found '<stream end>'.\n"
    )


@pytest.mark.parametrize("command", ["check", "status", "run", "fetch"])
@pytest.mark.parametrize(
    ("table", "message"),
    [
        (
            "time,longitude,latitude,altitude\n2024-07-01 12:00:00,-111.85,140.77,10\n",
            "Error: row 0: latitude must be a number within [-90, 90].",
        ),
        (
            "time,longitude,latitude,altitude\nyesterday,-111.85,40.77,10\n",
            "Error: ",
        ),
        (
            "time,longitude,latitude\n2024-07-01 12:00:00,-111.85,40.77\n",
            "Error: Release table must contain columns",
        ),
    ],
    ids=["latitude", "time", "columns"],
)
def test_commands_report_a_bad_releases_csv(tmp_path, command, table, message):
    """A bad row in releases.csv is a usage error for every command, as for check (#364)."""
    _write_minimal_config(tmp_path)
    (tmp_path / "releases.csv").write_text(table)

    result = runner.invoke(app, [command, str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert result.output.startswith(message)
    assert result.output.count("\n") == 1


def test_status_prints_project_info(tmp_path):
    """``status`` prints one summary line with the project root and counts."""
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 0
    assert (
        f"Project: {tmp_path.resolve()}  total=1  complete=0  failed=0  "
        "interrupted=0  pending=1" in result.output
    )


def test_status_counts_full_simulation_completion(tmp_path):
    """A complete trajectory without required footprints is not done yet."""
    from runtraj.project import Project
    from runtraj.releases import PointRelease

    cfg = make_project_config(
        tmp_path,
        grid=Grid(
            xmin=-114.0,
            xmax=-113.0,
            ymin=39.0,
            ymax=40.0,
            xres=0.1,
            yres=0.1,
        ),
        variants={"hrrr": {}},
    )

    release = PointRelease(
        time="2023-01-01 12:00:00",
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    project = Project.init(tmp_path, config=cfg, releases=[release])

    # Particles exist but the required footprint does not: not complete.
    sim = project.simulation(release.id, "hrrr")
    particles = pd.DataFrame(
        {
            "age": [-60.0],
            "particle": [1.0],
            "lon": [-113.5],
            "lat": [39.5],
            "zagl": [10.0],
            "foot": [1e-5],
        }
    )
    sim.output.write_particles(sim.variant, release, particles, [])

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=1  complete=0  failed=0  interrupted=0  pending=1" in result.output

    # Once the footprint is present too, the simulation counts as complete.
    from .fixtures.footprints import run_footprint

    sim = project.simulation(release.id, "hrrr")  # a fresh value
    run_footprint(sim, sim.particles)

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0
    assert "total=1  complete=1  failed=0  interrupted=0  pending=0" in result.output


def test_status_counts_each_state_per_variant(tmp_path, monkeypatch):
    """Failed simulations are counted as failed, not pending (#192)."""
    make_project_config(
        tmp_path, variants={"hrrr": {}, "fine": {"numpar": 50}}
    ).to_yaml(tmp_path / "config.yaml")
    (tmp_path / "releases.csv").write_text("time,longitude,latitude,altitude\n")
    table = pd.DataFrame(
        {
            "variant": ["hrrr", "hrrr", "hrrr", "fine", "fine", "fine"],
            "state": ["complete", "failed", "pending"] + ["interrupted"] * 3,
            "reason": [None, "MET_COVERAGE", None, None, None, None],
        }
    )
    monkeypatch.setattr(Project, "status", lambda self, sel=None: table)
    monkeypatch.setattr("runtraj.cli._print_folders", lambda project, folders: None)

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "total=6  complete=1  failed=1  interrupted=3  pending=1" in result.output
    assert (
        "  hrrr: total=3  complete=1  failed=1  interrupted=0  pending=1"
        in result.output
    )
    assert (
        "  fine: total=3  complete=0  failed=0  interrupted=3  pending=0"
        in result.output
    )


def test_cli_help_lists_current_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    expected = {
        "check",
        "fetch",
        "init",
        "run",
        "status",
        "submit",
    }
    for command in expected:
        assert command in result.output

    registered = {
        cmd.name or cmd.callback.__name__.replace("_", "-")  # type: ignore[union-attr]
        for cmd in app.registered_commands
    }
    assert registered == expected


def test_the_particle_commands_are_the_stilt_commands_but_init():
    # stilt init is STILT's: its starter config has STILT-R's values and the footprint.
    from runtraj import cli

    particle = [c.callback for c in cli.app.registered_commands]
    assert [c.callback for c in app.registered_commands] == [
        stilt.cli.init,
        *particle,
    ]
    assert stilt.cli.init not in particle


# ---------------------------------------------------------------------------
# check command
# ---------------------------------------------------------------------------


def _write_checkable_config(tmp_path, n_releases=1, files=True):
    """Write a project of hourly releases from noon, and met files for them unless ``files`` is false."""
    import datetime as dt

    from .fixtures.factories import make_met_files

    _write_minimal_config(tmp_path)
    times = [
        dt.datetime(2023, 1, 1, 12) + dt.timedelta(hours=h) for h in range(n_releases)
    ]
    (tmp_path / "releases.csv").write_text(
        "time,longitude,latitude,altitude\n"
        + "".join(f"{t:%Y-%m-%d %H:%M:%S},-111.85,40.77,5.0\n" for t in times)
    )
    if files:
        for time in times:
            make_met_files(tmp_path / "met", time, -24)


def _footprint_block(grid):
    """Return a ``stages`` block whose footprint has *grid*."""
    return {"footprint": {"class": "stilt.footprint.FootprintStage", "grid": grid}}


def test_check_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 2


def test_check_finding_nothing_exits_0(tmp_path):
    _write_checkable_config(tmp_path)

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f"Project: {tmp_path.resolve()}  releases=1" in result.output
    assert "Variants: hrrr (met hrrr, particles only)" in result.output
    assert "No problems found." in result.output


def test_check_lists_what_would_stop_a_run_and_exits_1(tmp_path, calls):
    _write_checkable_config(tmp_path, files=False)

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 1, result.output
    lines = result.output.splitlines()
    header = next(line for line in lines if line.startswith("check"))
    assert header.split() == ["check", "variant", "release", "message"]
    row = next(line for line in lines if line.startswith("met_files"))
    assert "202301011200_-111.85_40.77_5" in row
    assert "no met file for 2022-12-31 12:00" in row
    assert "Found 1 problem: met_files 1" in result.output
    assert calls == []  # nothing ran


def test_check_tells_a_release_above_sea_level_to_add_zsfc_and_exits_1(tmp_path):
    _write_checkable_config(tmp_path)
    # A footprint, whose near-field correction reads zsfc for this release,
    # and STILT-R's varsiwant, which has no zsfc (HYSPLIT's own list has it).
    config = tmp_path / "config.yaml"
    settings = yaml.safe_load(config.read_text())
    settings["models"]["hysplit"]["varsiwant"] = STILT_R_HYSPLIT["varsiwant"]
    settings["stages"] = {
        "footprint": {
            "class": "stilt.footprint.FootprintStage",
            "grid": {
                "xmin": -113,
                "xmax": -110,
                "ymin": 40,
                "ymax": 42,
                "xres": 0.1,
                "yres": 0.1,
            },
        }
    }
    config.write_text(yaml.safe_dump(settings))
    (tmp_path / "releases.csv").write_text(
        "time,longitude,latitude,altitude,altitude_ref\n"
        "2023-01-01 12:00:00,-111.85,40.77,1500.0,msl\n"
    )

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 1, result.output
    row = next(line for line in result.output.splitlines() if line.startswith("zsfc"))
    assert "zsfc is not in varsiwant" in row
    assert "varsiwant: [time, indx, long, lati, zagl, foot, mlht" in row
    assert "Found 1 problem: zsfc 1" in result.output


def test_check_shows_the_first_rows_of_each_check_and_counts_the_rest(tmp_path):
    _write_checkable_config(tmp_path, n_releases=12, files=False)

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 1, result.output
    shown = [line for line in result.output.splitlines() if "no met file" in line]
    assert len(shown) == 10
    assert "... 2 more (project.check() lists them all)" in result.output
    assert "Found 12 problems: met_files 12" in result.output


def test_check_exits_2_when_the_project_files_are_invalid(tmp_path):
    _write_checkable_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(config.read_text() + "nonsense: 1\n")

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 2
    assert "nonsense" in result.output


@pytest.mark.parametrize("command", ["check", "status", "run", "submit", "fetch"])
@pytest.mark.parametrize(
    ("block", "message"),
    [
        (
            {"stages": {"wind": {"class": "nopkg_for_this_test.stages.WindRose"}}},
            "Error: config.yaml names nopkg_for_this_test.stages.WindRose, but "
            "nopkg_for_this_test is not installed here. Install the package "
            "that provides nopkg_for_this_test and run this again.",
        ),
        (
            {"models": {"wind": {"class": "nopkg_for_this_test.models.Wind"}}},
            "Error: config.yaml names nopkg_for_this_test.models.Wind, but "
            "nopkg_for_this_test is not installed here.",
        ),
    ],
)
def test_a_class_that_is_not_installed_exits_2_naming_its_package(
    tmp_path, command, block, message
):
    """A project does not open where a class its config names cannot be imported."""
    import yaml

    _write_checkable_config(tmp_path)
    config = tmp_path / "config.yaml"
    settings = yaml.safe_load(config.read_text())
    settings.update(block)
    config.write_text(yaml.safe_dump(settings))

    result = runner.invoke(app, [command, str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert message in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["check", "status", "run"])
def test_a_stage_that_does_not_fit_its_model_exits_2(tmp_path, command):
    """What a stage refuses when the variants are resolved stops every command the same way."""
    import yaml

    _write_checkable_config(tmp_path)
    config = tmp_path / "config.yaml"
    settings = yaml.safe_load(config.read_text())
    settings["models"]["hysplit"]["varsiwant"] = [
        "time",
        "indx",
        "long",
        "lati",
        "zagl",
    ]
    settings["stages"] = _footprint_block(
        {"xmin": -113, "xmax": -110, "ymin": 39, "ymax": 42, "xres": 0.1, "yres": 0.1}
    )
    config.write_text(yaml.safe_dump(settings))

    result = runner.invoke(app, [command, str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert (
        "Error: Variant 'hrrr', stages: footprint: the near-field correction "
        "(hnf_plume) reads the particle columns"
    ) in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ({"models": {"hysplit": {"numpar": 0}}}, "numpar"),
        ({"models": {"hysplit": {"n_hours": 0}}}, "n_hours must not be 0"),
        (
            {
                "stages": _footprint_block(
                    {
                        "xmin": -111,
                        "xmax": -110,
                        "ymin": 40,
                        "ymax": 41,
                        "xres": 0,
                        "yres": 0.1,
                    }
                )
            },
            "xres",
        ),
        (
            {
                "stages": _footprint_block(
                    {
                        "xmin": -111,
                        "xmax": -110,
                        "ymin": 41,
                        "ymax": 40,
                        "xres": 0.1,
                        "yres": 0.1,
                    }
                )
            },
            "ymin < ymax",
        ),
    ],
)
def test_check_exits_2_for_a_run_setting_or_grid_no_run_can_use(
    tmp_path, setting, message
):
    """A numpar or n_hours of 0 and a grid with no cells were found only after the HYSPLIT run (#366)."""
    import yaml

    _write_checkable_config(tmp_path)
    config = tmp_path / "config.yaml"
    settings = yaml.safe_load(config.read_text())
    settings.update(setting)
    config.write_text(yaml.safe_dump(settings))

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert message in result.output


def test_check_exits_2_for_a_varsiwant_name_hycs_std_stops_on(tmp_path):
    """`zfx1` passed validation and stopped hycs_std; now check names the spelling it reads (#324)."""
    _write_checkable_config(tmp_path)
    config = tmp_path / "config.yaml"
    settings = yaml.safe_load(config.read_text())
    settings["models"]["hysplit"]["varsiwant"] = [
        "time",
        "indx",
        "long",
        "lati",
        "zfx1",
    ]
    config.write_text(yaml.safe_dump(settings))

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 2
    assert "zfx1" in result.output
    assert "zcfx" in result.output


def _point_met_at(tmp_path, directory):
    """Point the project's met at *directory*, as an unedited `stilt init` project's placeholder does."""
    config = tmp_path / "config.yaml"
    text = config.read_text()
    assert str(tmp_path / "met") in text
    config.write_text(text.replace(str(tmp_path / "met"), str(directory)))


@pytest.mark.parametrize("command", [["run"], ["submit"]])
def test_a_met_directory_that_does_not_exist_exits_2_in_one_line(
    tmp_path, calls, command
):
    """The line `stilt check` lists, not a traceback, and 2: nothing ran (#280)."""
    _write_minimal_config(tmp_path)
    missing = tmp_path / "no-such-met"
    _point_met_at(tmp_path, missing)

    result = runner.invoke(app, [*command, str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert result.output == f"Error: Met 'hrrr': {missing} does not exist.\n"
    assert calls == []
    checked = runner.invoke(app, ["check", str(tmp_path)])
    assert checked.exit_code == 1
    assert f"{missing} does not exist." in checked.output


def _make_results_folder(tmp_path, name="hrrr", **transport):
    """Make a particles folder, as a run of variant *name* with *transport* settings, under the project's output."""
    project = Project(tmp_path)
    variant = project.variants["hrrr"]
    if transport:
        variant = replace(
            variant, transport=variant.transport.model_copy(update=transport)
        )
    project.output.write_log(replace(variant, name=name), "202301011200_any", "")


@pytest.mark.parametrize("flags", [[], ["--json"]])
def test_status_with_an_unreadable_met_and_an_empty_output_is_all_pending(
    tmp_path, flags
):
    """A bare project, as pystiltctl asks about it: nothing can be there (pystiltctl#37)."""
    _write_minimal_config(tmp_path)
    _point_met_at(tmp_path, tmp_path / "no-such-met")

    result = runner.invoke(app, ["status", str(tmp_path), *flags])

    assert result.exit_code == 0, result.output
    if flags:
        data = json.loads(result.output)
        assert data["simulations"]["pending"] == data["simulations"]["total"] == 1
        assert data["folders"] == []
    else:
        assert "total=1  complete=0  failed=0  interrupted=0  pending=1" in (
            result.output
        )
        assert "no settings folders yet" in result.output


def test_status_with_an_unreadable_met_lists_folders_of_other_settings(tmp_path):
    """A folder of the variant's name that differs in a setting other than the met."""
    _write_minimal_config(tmp_path)
    _make_results_folder(tmp_path, numpar=7)
    _point_met_at(tmp_path, tmp_path / "no-such-met")

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "pending=1" in result.output
    assert "Used by this project: none yet" in result.output
    assert "settings=hrrr-" in result.output
    assert "numpar" in result.output


@pytest.mark.parametrize("flags", [[], ["--json"]])
@pytest.mark.parametrize("name", ["hrrr", "renamed"])
def test_status_with_an_unreadable_met_and_a_folder_that_may_hold_results_exits_2(
    tmp_path, flags, name
):
    """
    Same settings but the met's: pending could be false, so the met is needed.

    The folder's name is the variant that first made it, so a renamed
    variant, or another project's, owns it under another name.
    """
    _write_minimal_config(tmp_path)
    _make_results_folder(tmp_path, name=name)
    missing = tmp_path / "no-such-met"
    _point_met_at(tmp_path, missing)

    result = runner.invoke(app, ["status", str(tmp_path), *flags])

    assert result.exit_code == 2, result.output
    assert result.output == f"Error: Met 'hrrr': {missing} does not exist.\n"


@pytest.mark.parametrize("command", [["run"], ["submit"]])
def test_without_hycs_std_run_and_submit_exit_2_in_one_line(
    tmp_path, calls, monkeypatch, command
):
    """PYSTILT includes no HYSPLIT: say where to put hycs_std, and run nothing."""
    _write_minimal_config(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "no-hysplit"))

    result = runner.invoke(app, [*command, str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert result.output.startswith("Error: No hycs_std on PATH.")
    assert len(result.output.splitlines()) == 1
    assert calls == []


def test_status_without_hycs_std_exits_2_when_the_variant_ran_two_builds(
    tmp_path, monkeypatch
):
    """Two folders of hrrr that differ only in hycs_std: which is this machine's is unknown."""
    _write_minimal_config(tmp_path)
    build = make_hysplit_dir(tmp_path / "rebuilt", body="# rebuilt\nexit 2\n")
    _make_results_folder(tmp_path)
    _make_results_folder(tmp_path, directory=build)
    monkeypatch.setenv("PATH", str(tmp_path / "no-hysplit"))

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert result.output.startswith("Error: No hycs_std on PATH.")


def test_check_lists_a_missing_hycs_std(tmp_path, monkeypatch):
    _write_checkable_config(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "no-hysplit"))

    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 1, result.output
    row = next(line for line in result.output.splitlines() if line.startswith("model"))
    assert "No hycs_std on PATH" in row


def test_a_met_directory_with_no_arl_file_exits_2_in_one_line(tmp_path, calls):
    _write_minimal_config(tmp_path)
    empty = tmp_path / "empty-met"
    empty.mkdir()
    (empty / "README").write_text("not a met file")
    _point_met_at(tmp_path, empty)

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert result.output.startswith(f"Error: Met 'hrrr': No ARL file under {empty}.")
    assert len(result.output.splitlines()) == 1
    assert calls == []


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


def test_run_exits_when_no_config(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 2


def test_run_runs_the_project_with_its_execution_settings(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    [(verb, kwargs)] = calls
    assert verb == "run"
    # config.yaml's
    assert kwargs["execution"] == ExecutionConfig(slurm={"time": "01:00:00"})
    assert kwargs["skip_existing"] is True
    # Resolved once, so the run uses the directory the banner shows.
    from runtraj.execution.runner import resolve_compute_root
    from runtraj.project import Project

    scratch = resolve_compute_root(Project(tmp_path))
    assert kwargs["compute_root"] == str(scratch)
    assert f"Compute root: {scratch}" in result.output


def test_run_prints_a_startup_summary_and_the_status(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 0
    assert (
        f"Starting run: project={tmp_path.resolve()}  backend=local  "
        "tasks=1  cpus=1  skip=existing"
    ) in result.output
    assert "Compute root:" in result.output
    assert f"Output: {tmp_path.resolve() / 'output'}" in result.output
    assert "Releases loaded: 1" in result.output
    assert "Running here, one line per release" in result.output
    assert f"Project: {tmp_path.resolve()}  total=" in result.output


def test_run_forwards_compute_root_and_shows_it(tmp_path, calls):
    _write_minimal_config(tmp_path)
    compute_root = tmp_path / "scratch"

    result = runner.invoke(
        app, ["run", str(tmp_path), "--compute-root", str(compute_root)]
    )

    assert result.exit_code == 0
    assert f"Compute root: {compute_root.resolve()}" in result.output
    assert calls[0][1]["compute_root"] == str(compute_root)


def test_run_no_skip_passes_false(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path), "--no-skip"])

    assert result.exit_code == 0
    assert calls[0][1]["skip_existing"] is False
    assert "skip=no-skip" in result.output


def test_run_options_override_the_execution_settings(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(
        app, ["run", str(tmp_path), "--backend", "local", "--cpus", "4"]
    )

    assert result.exit_code == 0
    execution = calls[0][1]["execution"]
    assert (execution.backend, execution.cpus) == ("local", 4)
    assert "tasks=1  cpus=4" in result.output


def test_cpus_goes_to_the_block_of_the_backend_that_runs(tmp_path, calls):
    """--cpus sets the Slurm block's cpus on a Slurm run, and leaves the local block's."""
    _write_minimal_config(tmp_path)

    result = runner.invoke(
        app, ["run", str(tmp_path), "--backend", "slurm", "--cpus", "4"]
    )

    assert result.exit_code == 0, result.output
    execution = calls[0][1]["execution"]
    assert (execution.slurm.cpus, execution.local.cpus) == (4, 1)
    assert execution.cpus == 4


def test_run_on_slurm_waits_for_the_job(tmp_path, calls):
    _write_minimal_config(tmp_path)

    result = runner.invoke(
        app, ["run", str(tmp_path), "--backend", "slurm", "--n-workers", "2"]
    )

    assert result.exit_code == 0, result.output
    assert [verb for verb, _ in calls] == ["run"]
    assert calls[0][1]["execution"].slurm.n_workers == 2
    assert "Submitting to Slurm and waiting for the job to finish" in result.output


def test_run_writes_to_the_output_given(tmp_path, calls, monkeypatch):
    _write_minimal_config(tmp_path)
    opened = []
    real = Project.__init__

    def record(self, path, *, output=None):
        opened.append(output)
        real(self, path, output=output)

    monkeypatch.setattr(Project, "__init__", record)

    elsewhere = str(tmp_path / "elsewhere")
    result = runner.invoke(app, ["run", str(tmp_path), "--output", elsewhere])

    assert result.exit_code == 0, result.output
    assert elsewhere in opened
    assert f"Output: {elsewhere}" in result.output


def test_run_reads_execution_settings_from_a_file(tmp_path, calls):
    _write_minimal_config(tmp_path)
    settings = tmp_path / "execution.yaml"
    settings.write_text(
        "backend: slurm\nslurm:\n  cpus: 3\ntimeout: 900\nkeep_workdir: true\n"
    )

    result = runner.invoke(
        app,
        [
            "run",
            str(tmp_path),
            "--execution",
            str(settings),
            "--task",
            "0/2",
            "--cpus",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    execution = calls[0][1]["execution"]
    # The file's settings, then the options, and a task runs here: --cpus is
    # the local run's, as the job script passes the array's.
    assert (execution.backend, execution.cpus, execution.timeout) == ("local", 2, 900)
    assert execution.slurm.cpus == 3
    assert execution.keep_workdir is True


@pytest.mark.parametrize(
    ("states", "code"),
    [
        ([], 0),
        (["complete", "complete"], 0),
        (["complete", "failed"], 1),
        (["failed", "pending"], 3),
        (["complete", "interrupted"], 3),
    ],
)
def test_run_exit_code_says_how_the_run_ended(tmp_path, calls, states, code):
    _write_minimal_config(tmp_path)
    calls.states = states

    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == code, result.output


# ---------------------------------------------------------------------------
# An error the command did not expect: 3 to retry, 4 not to (#310)
# ---------------------------------------------------------------------------


def _error_named(name, *bases):
    """Return an exception of a class called *name*, as a cloud library would raise it."""
    return type(name, bases or (Exception,), {})("it failed")


def _client_error(status, code="Error"):
    """Return what botocore raises for a response with this status and error code."""
    error = _error_named("ClientError")
    error.response = {
        "ResponseMetadata": {"HTTPStatusCode": status},
        "Error": {"Code": code},
    }
    return error


#: Base classes the cloud libraries' errors share, named as botocore names them.
_BOTOCORE_CONNECTION_ERROR = type("ConnectionError", (Exception,), {})


def _raised_from(outer, inner):
    """Return *outer* with *inner* as its ``__cause__``, as ``raise outer from inner`` does."""
    outer.__cause__ = inner
    return outer


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (_error_named("EndpointConnectionError", _BOTOCORE_CONNECTION_ERROR), 3),
        (ConnectionResetError("reset"), 3),
        (OSError(5, "I/O error"), 3),
        (RuntimeError("a bug"), 4),
        (KeyError("k"), 4),
        (FileNotFoundError("no such file"), 4),
    ],
)
def test_run_exits_3_for_a_connection_error_and_4_for_any_other(
    tmp_path, monkeypatch, error, code
):
    """An error that escapes the run is never 1, the code of "some simulations failed"."""
    _write_minimal_config(tmp_path)

    def run(self, **kwargs):
        raise error

    monkeypatch.setattr("runtraj.cli.Project.run", run)
    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == code, result.output
    assert str(error) in result.output
    assert f"runtraj exits with {code}" in result.output
    assert "Traceback (most recent call last)" in result.output


def test_run_says_where_an_unexpected_error_happened(tmp_path, monkeypatch):
    """The traceback is kept, so a bug can be found."""
    _write_minimal_config(tmp_path)

    def run(self, **kwargs):
        raise RuntimeError("a bug somewhere")

    monkeypatch.setattr("runtraj.cli.Project.run", run)
    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 4
    assert "in run" in result.output
    assert "RuntimeError: a bug somewhere" in result.output


def test_run_exits_3_when_ctrl_c_arrives_outside_the_workers(tmp_path, monkeypatch):
    """Ctrl-C while waiting for a Slurm job was "Aborted!" with exit 1."""
    _write_minimal_config(tmp_path)

    def run(self, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("runtraj.cli.Project.run", run)
    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 3, result.output


def test_a_failed_simulation_is_still_1_and_a_wrong_command_line_2(tmp_path, calls):
    _write_minimal_config(tmp_path)
    calls.states = ["complete", "failed"]
    assert runner.invoke(app, ["run", str(tmp_path)]).exit_code == 1
    assert runner.invoke(app, ["run", str(tmp_path), "--task", "3/3"]).exit_code == 2
    assert runner.invoke(app, ["run", str(tmp_path / "no_project")]).exit_code == 2
    releases = tmp_path / "missing.txt"
    assert (
        runner.invoke(app, ["run", str(tmp_path), "--releases", str(releases)])
    ).exit_code == 2


def test_a_value_error_of_the_run_is_still_2(tmp_path, monkeypatch):
    """The run's own refusals (a release that is not in the project) are the command line's."""
    _write_minimal_config(tmp_path)

    def run(self, **kwargs):
        raise ValueError("no such release")

    monkeypatch.setattr("runtraj.cli.Project.run", run)
    result = runner.invoke(app, ["run", str(tmp_path)])

    assert result.exit_code == 2
    assert "Error: no such release" in result.output


@pytest.mark.parametrize(
    ("command", "method"),
    [("submit", "submit"), ("status", "status"), ("check", "check")],
)
def test_the_other_commands_exit_4_on_an_unexpected_error(
    tmp_path, monkeypatch, command, method
):
    """Python's own 1 for `status` and `submit` was not a code anyone defined; `check`'s 1 is "found problems"."""
    _write_minimal_config(tmp_path)

    def broken(self, *args, **kwargs):
        raise RuntimeError("a bug somewhere")

    monkeypatch.setattr(f"runtraj.cli.Project.{method}", broken)
    result = runner.invoke(app, [command, str(tmp_path)])

    assert result.exit_code == 4, result.output


def test_status_exits_3_on_a_connection_error(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)

    def broken(self, *args, **kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr("runtraj.cli.Project.status", broken)

    assert runner.invoke(app, ["status", str(tmp_path)]).exit_code == 3


def test_a_closed_stdout_ends_quietly_with_1_as_before(tmp_path, monkeypatch):
    """`stilt status | head -0` raises BrokenPipeError: no traceback and no "retry" (#310)."""
    _write_minimal_config(tmp_path)

    def closed(self, *args, **kwargs):
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    monkeypatch.setattr("runtraj.cli.Project.status", closed)
    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "runtraj exits with" not in result.output


@pytest.mark.parametrize("args", [["run", "--bogus-flag"], ["nosuchcmd"]])
def test_a_wrong_command_line_exits_2_not_interrupted(tmp_path, calls, args):
    """Click exits 2 on a typo, so 2 means a wrong command line and never "retry"."""
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, [*args, str(tmp_path)])

    assert result.exit_code == 2
    assert calls == []


def test_a_task_takes_its_releases_from_the_submissions_rows(tmp_path, monkeypatch):
    """A task reads releases.parquet, not all of releases.csv (#191)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _write_minimal_config(tmp_path)
    project = Project(tmp_path)
    rows = project._rows
    pq.write_table(
        pa.Table.from_pandas(rows, preserve_index=False), tmp_path / "r.parquet"
    )
    (tmp_path / "releases.csv").write_text("not a release table\n")
    seen = []

    def run(self, **kwargs):
        seen.append((list(self.releases["release"]), kwargs["releases"]))
        return pd.DataFrame({"release": [], "variant": [], "state": [], "reason": []})

    monkeypatch.setattr("runtraj.cli.Project.run", run)
    result = runner.invoke(
        app,
        [
            "run",
            str(tmp_path),
            "--releases",
            str(tmp_path / "r.parquet"),
            "--task",
            "0/1",
        ],
    )

    assert result.exit_code == 0, result.output
    ids = list(rows["release"])
    assert seen == [(ids, ids)]


def test_run_task_runs_its_share_here(tmp_path, calls):
    _write_minimal_config(tmp_path)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    config["execution"] = {
        "backend": "slurm",
        "slurm": {"n_workers": 8, "time": "01:00:00"},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    calls.states = ["complete", "failed"]

    result = runner.invoke(app, ["run", str(tmp_path), "--task", "1/3"])

    assert result.exit_code == 1, result.output
    [(verb, kwargs)] = calls
    assert kwargs["task"] == (1, 3)
    assert kwargs["releases"] is None
    assert kwargs["execution"].backend == "local"
    # The banner names the task, not the local backend the task runs on.
    assert "task 1 of 3  cpus=1" in result.output
    assert "backend=" not in result.output
    assert "Releases of this task: 1, 4, 7, ..." in result.output
    # A task sums up what it ran, not the whole project.
    assert (
        "This run:  total=2  complete=1  failed=1  interrupted=0  pending=0"
        in result.output
    )
    assert "failed: met_missing 1" in result.output
    assert "Project:" not in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--task", "3/3"], "0 <= I < N"),
        (["--task", "two/3"], "takes I/N"),
        (["--task", "0/2", "--backend", "slurm"], "drop --backend slurm"),
        (["--releases", "missing.txt"], "cannot read"),
    ],
)
def test_run_task_and_releases_refuse_bad_values(tmp_path, calls, args, message):
    _write_minimal_config(tmp_path)

    result = runner.invoke(app, ["run", str(tmp_path), *args])

    assert result.exit_code == 2
    assert message in result.output
    assert calls == []


def test_run_releases_reads_ids_from_a_file(tmp_path, calls):
    _write_minimal_config(tmp_path)
    listed = tmp_path / "ids.txt"
    listed.write_text("# failed last time\nr1\n\n  r2  \n")

    result = runner.invoke(app, ["run", str(tmp_path), "--releases", str(listed)])

    assert result.exit_code == 0, result.output
    assert calls[0][1]["releases"] == ["r1", "r2"]
    assert "Releases listed: 2" in result.output


def test_run_reports_releases_not_in_the_project(tmp_path, monkeypatch):
    _write_minimal_config(tmp_path)
    listed = tmp_path / "ids.txt"
    listed.write_text("nope\n")

    result = runner.invoke(app, ["run", str(tmp_path), "--releases", str(listed)])

    assert result.exit_code == 2
    assert "not in this project" in result.output


def test_submit_submits_to_slurm_and_returns(tmp_path, calls):
    _write_minimal_config(tmp_path)  # its backend is local

    result = runner.invoke(app, ["submit", str(tmp_path), "--n-workers", "2"])

    assert result.exit_code == 0, result.output
    assert [verb for verb, _ in calls] == ["submit"]
    execution = calls[0][1]["execution"]
    assert (execution.backend, execution.slurm.n_workers) == ("slurm", 2)
    assert "Submitting to Slurm and returning" in result.output
    assert "Submitted job: 12345" in result.output


def test_submit_without_a_time_limit_says_so(tmp_path, calls):
    _write_minimal_config(tmp_path)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    del config["execution"]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    result = runner.invoke(app, ["submit", str(tmp_path)])

    assert result.exit_code == 2
    assert "execution.slurm.time is required with backend: slurm" in result.output
    assert "Traceback" not in result.output
    assert calls == []


def test_submit_releases_reads_ids_from_a_file(tmp_path, calls):
    _write_minimal_config(tmp_path)
    listed = tmp_path / "ids.txt"
    listed.write_text("r1\nr2\n")

    result = runner.invoke(app, ["submit", str(tmp_path), "--releases", str(listed)])

    assert result.exit_code == 0, result.output
    assert calls[0][1]["releases"] == ["r1", "r2"]
    assert "Releases listed: 2" in result.output


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


def test_init_creates_config_yaml(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0
    assert (project / "config.yaml").exists()


def test_init_creates_releases_csv(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0
    assert (project / "releases.csv").exists()


def test_init_releases_csv_reads_after_appending_a_row(tmp_path):
    """The starter releases.csv has no line that parses as a bad release (#49)."""
    from runtraj.releases import read_releases

    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0

    csv = project / "releases.csv"
    assert read_releases(csv) == []
    with csv.open("a") as f:
        f.write("2023-01-01 12:00:00,-111.85,40.77,5\n")
    releases = read_releases(csv)
    assert len(releases) == 1
    assert releases[0].altitude == 5


def test_init_prints_confirmation(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0
    assert "Initialized STILT project" in result.output


def test_init_writes_science_first_commented_config(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0

    text = (project / "config.yaml").read_text()
    parsed = yaml.safe_load(text)

    assert "#" in text
    assert ProjectConfig.from_yaml(project / "config.yaml")
    assert list(parsed) == ["mets", "models", "stages", "variants", "output"]
    assert parsed["output"] == "./output"
    assert list(parsed["models"]) == ["hysplit"]
    assert parsed["models"]["hysplit"] == {"n_hours": -24, **STILT_R_HYSPLIT}
    footprint = parsed["stages"]["footprint"]
    assert footprint["class"] == "stilt.footprint.FootprintStage"
    assert footprint["grid"]["xmin"] == -113.0
    assert parsed["variants"] == {"hrrr": {}}
    assert text.index("output:") < text.index("# execution:")


def test_init_config_omits_advanced_and_internal_knobs(tmp_path):
    project = tmp_path / "new_project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0

    text = (project / "config.yaml").read_text()

    assert "ichem" not in text
    assert "idsp" not in text
    assert "kagl" not in text


def test_init_aborts_when_config_exists(tmp_path):
    project = tmp_path / "existing"
    project.mkdir()
    (project / "config.yaml").write_text("mets:\n  hrrr: /data/hrrr\n")
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 2


def test_status_lists_output_folders_no_variant_uses(tmp_path):
    from runtraj.project import Project

    _write_minimal_config(tmp_path)
    project = Project(tmp_path)
    # A run made under settings the config no longer has.
    variant = project.variants["hrrr"]
    stale = replace(
        variant,
        name="old",
        transport=variant.transport.model_copy(update={"numpar": 7}),
    )
    project.output.write_log(stale, str(project.releases["release"].iloc[0]), "")

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Not used by this project" in result.output
    assert result.output.index("Used by this project") < result.output.index(
        "Not used by this project"
    )
    assert "settings=old-" in result.output
    assert "never deletes" in result.output


def test_status_counts_failures_by_reason(tmp_path):
    from runtraj.project import Project
    from runtraj.releases import PointRelease

    releases = [
        PointRelease(
            time=f"2023-01-01 {h}:00", longitude=-111.85, latitude=40.77, altitude=5
        )
        for h in (12, 13, 14)
    ]
    project = Project.init(
        tmp_path,
        mets={"hrrr": make_met_config(tmp_path / "met")},
        releases=releases,
        models={"hysplit": {"n_hours": -24}},
        variants={"hrrr": {}},
    )
    reasons = ["MET_COVERAGE", "MET_COVERAGE", None]
    for release, reason in zip(releases, reasons, strict=True):
        sim = project.simulation(str(release.id), "hrrr")
        sim.output.record_failure(
            "particles",
            sim.variant,
            sim.release.id,
            {"stage": "particles", "reason": reason or "ValueError", "message": "m"},
        )

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "failed: MET_COVERAGE 2, ValueError 1" in result.output


def test_status_json_is_the_same_as_data(tmp_path):

    from runtraj.project import Project
    from runtraj.releases import PointRelease

    _write_minimal_config(tmp_path)
    project = Project(tmp_path)
    release = PointRelease(
        time="2023-01-01 12:00:00", longitude=-111.85, latitude=40.77, altitude=5.0
    )
    variant = project.variants["hrrr"]
    project.output.record_failure(
        "particles", variant, release.id, {"reason": "MET_COVERAGE"}
    )

    result = runner.invoke(app, ["status", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["project"] == str(tmp_path.resolve())
    assert data["simulations"] == {
        "total": 1,
        "complete": 0,
        "failed": 1,
        "interrupted": 0,
        "pending": 0,
    }
    assert data["variants"]["hrrr"]["failed"] == 1
    assert data["failed"] == {"MET_COVERAGE": 1}
    [folder] = data["folders"]
    assert (folder["kind"], folder["variants"], folder["files"]) == (
        "particles",
        ["hrrr"],
        0,
    )


def test_status_lists_folders_and_what_differs(tmp_path):
    import datetime as dt

    from runtraj.config import ProjectConfig
    from runtraj.project import Project
    from runtraj.releases import PointRelease

    from .fixtures.particles import finished

    release = PointRelease(
        time=dt.datetime(2023, 1, 1, 12),
        longitude=-111.85,
        latitude=40.77,
        altitude=5.0,
    )
    met = make_met_config(tmp_path / "met")
    old = Project.init(
        tmp_path / "old",
        config=ProjectConfig(
            **blocks(
                mets={"hrrr": met},
                variants={"hrrr": {}},
                numpar=500,
                output=str(tmp_path / "out"),
            )
        ),
        releases=[release],
    )
    sim = old.simulation(release.id, "hrrr")
    particles = pd.DataFrame(
        {
            "age": [-60.0],
            "particle": [1.0],
            "lon": [-111.9],
            "lat": [40.7],
            "zagl": [10.0],
            "foot": [1e-5],
        }
    )
    sim.output.write_particles(sim.variant, release, finished(particles, release), [])
    new = Project.init(
        tmp_path / "new",
        config=ProjectConfig(
            **blocks(
                mets={"hrrr": met},
                variants={"hrrr": {}},
                numpar=1000,
                output=str(tmp_path / "out"),
            )
        ),
        releases=[release],
    )

    result = runner.invoke(app, ["status", str(new.directory)])

    assert result.exit_code == 0, result.output
    assert f"Output: {tmp_path / 'out'}" in result.output
    assert "Not used by this project" in result.output
    assert "1 file  numpar: 500 (config: 1000)" in result.output
    assert "numpar: 500 (config: 1000)" in result.output
    assert "runtraj never deletes one" in result.output
    table = old.folders()
    assert table[["kind", "name", "files", "variant", "differs"]].values.tolist() == [
        ["particles", "hrrr", 1, "hrrr", ""]
    ]
