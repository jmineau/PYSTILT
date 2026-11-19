"""Tests for HYSPLIT's input files, running hycs_std, and reading its particles."""

from pathlib import Path

import pytest
from runtraj.exceptions import SimulationError
from runtraj.transport.hysplit import (
    FailureReason,
    HysplitConfig,
    HysplitModel,
    model,
)
from runtraj.transport.hysplit.driver import (
    MET_STOP_CELLS,
    check_met_stop,
)

from ...fixtures.factories import make_met_config, make_met_files

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ControlFile.write / .read roundtrip - single-point release
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inputs, runs, and particle files
# ---------------------------------------------------------------------------

VARS = ["time", "indx", "long", "lati", "zagl", "foot"]


def _config(**overrides) -> HysplitConfig:
    """A small config whose particle rows are the six columns of VARS."""
    return HysplitConfig(
        **{"n_hours": -24, "numpar": 10, "varsiwant": VARS} | overrides
    )


def _write_particle_dat(path: Path, rows: list[list[float]]) -> None:
    """Write a minimal PARTICLE_STILT.DAT with one dummy header line."""
    lines = ["header"]
    lines.extend(" ".join(str(v) for v in row) for row in rows)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# read_particle_dat
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Running hycs_std
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HysplitModel.run: what a finished hycs_std leaves behind
# ---------------------------------------------------------------------------


def _fake_hysplit(
    monkeypatch, *, log: str, rows: list[list[float]] | None, warning: str = ""
) -> list:
    """Replace hycs_std with one that writes *log*, *warning*, and particle *rows* (none: no file)."""
    calls: list[int | None] = []

    def fake_run(workdir: Path, timeout: int | None) -> None:
        calls.append(timeout)
        (workdir / "stilt.log").write_text(log)
        if warning:
            (workdir / "WARNING").write_text(warning)
        if rows is not None:
            _write_particle_dat(workdir / "PARTICLE_STILT.DAT", rows)

    monkeypatch.setattr(model, "run_hycs_std", fake_run)
    return calls


def _run(tmp_path, release, config=None):
    from runtraj.meteorology import run_window

    config = config or _config()
    make_met_files(tmp_path / "met", release.time, config.n_hours)
    met = make_met_config(tmp_path / "met")
    window = run_window(release.time, config.n_hours)
    (tmp_path / "run").mkdir(exist_ok=True)
    return HysplitModel().run(release, config, met, window, tmp_path / "run", timeout=5)


# ---------------------------------------------------------------------------
# check_met_stop: met that ran out, told from a domain exit (#213)
# ---------------------------------------------------------------------------


def _grid(nx: int = 50, ny: int = 50, west: float = 20.0, south: float = -10.0):
    """A 1 degree lat/lon grid of *nx* by *ny* points from (*west*, *south*)."""
    from arlmet import Grid, Projection

    projection = Projection(
        pole_lat=90.0,
        pole_lon=0.0,
        tangent_lat=1.0,
        tangent_lon=1.0,
        grid_size=0.0,
        orientation=0.0,
        cone_angle=0.0,
        sync_x=1.0,
        sync_y=1.0,
        sync_lat=south,
        sync_lon=west,
    )
    return Grid(projection=projection, nx=nx, ny=ny)


#: The middle of :func:`_grid`, 24.5 cells from each side.
MIDDLE = (44.5, 14.5)


def _stopped_at(minute: float, *positions: tuple[float, float, float]):
    """
    Particles that each stop *minute* minutes back, at a (lon, lat, zagl) of *positions*.

    Each starts at the grid's middle and has a row at the release and one at
    the stop, as HYSPLIT writes every time step of each.
    """
    import pandas as pd

    rows = []
    for i, (lon, lat, zagl) in enumerate(positions, start=1):
        rows.append((i, -1.0, *MIDDLE, 10.0))
        rows.append((i, -float(minute), lon, lat, zagl))
    return pd.DataFrame(rows, columns=["particle", "age", "lon", "lat", "zagl"])


def _check(particles, n_hours: int = -24, grid=None) -> None:
    check_met_stop(particles, n_hours, grid or _grid())


def test_a_run_the_met_stopped_inside_the_domain_fails():
    """HYSPLIT can stop every particle where the met ran out and report a complete run."""
    particles = _stopped_at(360, (*MIDDLE, 100.0), (40.0, 10.0, 500.0))

    with pytest.raises(SimulationError, match="the met ran out") as caught:
        _check(particles)

    assert caught.value.reason == FailureReason.MET_COVERAGE
    message = str(caught.value)
    assert "6.0 h into the 24 h run" in message
    assert "up to 24.5 grid cells inside" in message


@pytest.mark.parametrize(
    "position",
    [
        (21.5, 14.5, 10.0),  # 1.5 cells from the west side
        (67.5, 14.5, 10.0),  # 1.5 from the east side
        (44.5, -8.5, 10.0),  # 1.5 from the south side
        (44.5, 37.0, 10.0),  # 2.0 from the north side
    ],
)
def test_particles_that_stop_within_two_cells_of_an_edge_left_the_domain(position):
    """A particle that leaves the domain ends 1 to 2 cells from its edge."""
    _check(_stopped_at(13 * 60, position))


def test_one_particle_more_than_the_limit_inside_fails_the_run():
    inside = MET_STOP_CELLS + 1
    particles = _stopped_at(
        13 * 60, (21.5, 14.5, 10.0), (20.0 + inside, 14.5, 10.0), (67.5, 14.5, 10.0)
    )

    with pytest.raises(SimulationError) as caught:
        _check(particles)
    assert caught.value.reason == FailureReason.MET_COVERAGE


def test_a_particle_near_the_domain_top_still_fails_the_run():
    """HYSPLIT keeps particles from rising above the top, so the top cannot end a run."""
    with pytest.raises(SimulationError):
        _check(_stopped_at(13 * 60, (*MIDDLE, 24900.0)))


def test_only_the_particles_at_the_last_time_step_count():
    """A particle that stopped earlier inside the domain is not where the run stopped."""
    import pandas as pd

    at_the_edge = _stopped_at(13 * 60, (21.5, 14.5, 10.0))
    earlier_inside = _stopped_at(6 * 60, (*MIDDLE, 10.0)).assign(particle=2)

    _check(pd.concat([at_the_edge, earlier_inside]))


def test_a_run_that_reaches_n_hours_is_not_checked():
    _check(_stopped_at(24 * 60, (*MIDDLE, 10.0)))


def test_a_forward_run_is_checked_the_same():
    particles = _stopped_at(360, (*MIDDLE, 10.0)).assign(age=lambda df: df["age"].abs())

    with pytest.raises(SimulationError):
        _check(particles, n_hours=24)


def test_particles_without_positions_are_not_checked():
    _check(_stopped_at(360, (*MIDDLE, 10.0)).drop(columns="lat"))


# ---------------------------------------------------------------------------
# write_inputs: SETUP.CFG
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# write_inputs: CONTROL and the error and mixed-layer files
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# write_inputs: the HYSPLIT build and its data tables
# ---------------------------------------------------------------------------


# Fortran type of every SETUP.CFG entry PYSTILT writes, from the SETUP
# namelist declarations in HYSPLIT's hysetup.f. STILT-R's v5.1.0 build
# reads the same types.
