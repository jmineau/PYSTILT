"""
``foot`` computed by PYSTILT (:func:`stilt.particles.add_foot`) against the ``foot`` HYSPLIT wrote.

Both compute STILT's footprint: the minutes a particle spends below ``footprint_depth``
of the mixed layer, over the air below that height. HYSPLIT adds up the dry
air over its levels in each turbulence step, with 28.84 g/mol; ``add_foot``
takes the air from the drop in pressure between the ground and the depth in
the met files, with 28.97 g/mol. With HYSPLIT's own ``samt`` and ``mlht``
the two agree closely.

A transport model that writes only positions has neither ``samt`` nor
``mlht``: a particle counts for the whole step when it ends the step below
the layer, and ``mlht`` is the met's ``PBLH``, where HYSPLIT's default finds
the mixed layer from the profile (``kmixd: 3``), so the footprint moves
more. ``pbl_height: richardson`` finds the mixed layer as HYSPLIT's
default does (:func:`stilt.sampling.calc_pbl_height`). Run with ``-s`` to
see each step's numbers; the validation page reports them.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.footprint import calc_footprint
from stilt.particles import PARTICLE_SCHEMA, add_foot
from stilt.receptors import PointReceptor
from stilt.sampling import calc_pbl_height, sample_met
from stilt.transport import ModelRun, TransportConfig, _finish, run_model
from stilt.transport.hysplit import HysplitConfig

from ..conftest import integration
from ..fixtures.r_stilt_reference import (
    REFERENCE_SUMMER_TIME,
    REFERENCE_TIME,
    reference_grid,
)
from .conftest import reference_met

VEGHT = 0.5
TOTAL_TOL = 0.01  # the total and the median of foot against HYSPLIT's
ROW_TOL = 0.015  # 90 % of rows
KMIX0 = 150  # HYSPLIT's default floor on the mixed layer, in m


def _run(
    met_dir: Path, workdir: Path, time: dt.datetime, kmixd: int = 3
) -> tuple[ModelRun, PointReceptor]:
    """Run HYSPLIT from the WBB tower for 6 hours, writing the columns foot is made from."""
    receptor = PointReceptor(time=time, longitude=-112.0, latitude=40.5, altitude=5.0)
    config = HysplitConfig(
        n_hours=-6,
        numpar=200,
        krand=2,
        seed=42,
        footprint_depth=VEGHT,
        kmixd=kmixd,
        kmix0=KMIX0,
        hnf_plume=False,
        varsiwant=[
            "time",
            "indx",
            "long",
            "lati",
            "zagl",
            "foot",
            "samt",
            "mlht",
            "dens",
        ],
    )
    run = run_model("hysplit", receptor, config, reference_met(met_dir), workdir, 900)
    return run, receptor


def _timed(particles: pd.DataFrame, receptor: PointReceptor) -> pd.DataFrame:
    """Return *particles* with ``time``, the receptor time plus ``age``."""
    return particles.assign(
        time=pd.Timestamp(receptor.time) + pd.to_timedelta(particles["age"], unit="min")
    )


def _total(particles: pd.DataFrame, receptor: PointReceptor) -> xr.DataArray:
    """Return the footprint of *particles*, summed over time."""
    return calc_footprint(particles, receptor, reference_grid()).sum("time")


def _moved(foot: xr.DataArray, hysplit: xr.DataArray) -> tuple[float, float]:
    """Return a footprint's total over HYSPLIT's, and its summed cell differences over HYSPLIT's total."""
    total = float(hysplit.sum())
    return float(foot.sum()) / total, float(abs(foot - hysplit).sum()) / total


#: A winter afternoon, a winter night, and a summer evening in Salt Lake City.
TIMES = [dt.datetime(2021, 1, 14, 22), REFERENCE_TIME, REFERENCE_SUMMER_TIME]


@pytest.fixture(
    scope="module", params=TIMES, ids=["winter-day", "winter-night", "summer"]
)
def hysplit_run(request, met_dir, tmp_path_factory):
    return _run(met_dir, tmp_path_factory.mktemp("run"), request.param)


@integration
def test_add_foot_with_hysplits_samt_and_mlht_reproduces_its_foot(hysplit_run):
    run, receptor = hysplit_run
    hysplit = run.particles
    ours = add_foot(
        _timed(hysplit.drop(columns="foot"), receptor), run.met_files, VEGHT
    )["foot"]

    positive = hysplit["foot"] > 0
    ratio = ours[positive] / hysplit["foot"][positive]
    print(
        f"\nadd_foot / HYSPLIT's foot with samt and mlht: total "
        f"{ours.sum() / hysplit['foot'].sum():.4f}, median {ratio.median():.4f}, "
        f"5-95 % {ratio.quantile(0.05):.4f} to {ratio.quantile(0.95):.4f}"
    )
    assert ((ours > 0) == positive).all()
    assert ours.sum() / hysplit["foot"].sum() == pytest.approx(1.0, abs=TOTAL_TOL)
    assert ratio.median() == pytest.approx(1.0, abs=TOTAL_TOL)
    assert ratio.quantile(0.05) > 1 - ROW_TOL
    assert ratio.quantile(0.95) < 1 + ROW_TOL


@integration
def test_a_positions_only_run_gets_a_foot_that_makes_a_footprint_near_hysplits(
    hysplit_run,
):
    run, receptor = hysplit_run
    hysplit = run.particles
    positions = hysplit[PARTICLE_SCHEMA.names]

    # What a transport model that writes only positions gets after its run.
    finished = _finish(
        ModelRun(positions, met_files=run.met_files),
        receptor,
        TransportConfig(footprint_depth=VEGHT, hnf_plume=False),
    )
    assert "from the met's PBLH" in finished.log
    assert finished.particles["foot"].notna().all()

    whole_step = add_foot(
        _timed(hysplit.drop(columns=["foot", "samt"]), receptor), run.met_files, VEGHT
    )
    reference = _total(hysplit, receptor)
    print()
    steps = {
        "whole step": whole_step,
        "whole step, the met's PBLH": finished.particles,
    }
    moved = {}
    for name, particles in steps.items():
        moved[name] = _moved(_total(particles, receptor), reference)
        print(
            f"{name}: footprint total / HYSPLIT's {moved[name][0]:.4f}, "
            f"summed cell differences / HYSPLIT's total {moved[name][1]:.4f}"
        )

    total, cells = moved["whole step"]
    assert total == pytest.approx(1.0, abs=0.03)
    assert cells < 0.05
    # The met's PBLH is not HYSPLIT's mixed layer: a bound, not a match.
    total, cells = moved["whole step, the met's PBLH"]
    assert 0.5 < total < 2.0
    assert np.isfinite(cells)


@integration
@pytest.mark.parametrize("time", TIMES, ids=["winter-day", "winter-night", "summer"])
def test_against_hysplit_on_the_mets_mixed_layer_the_footprint_moves_less(
    met_dir, tmp_path, time
):
    """With ``kmixd: 0`` HYSPLIT's mixed layer is the met's PBLH, at least ``kmix0``."""
    run, receptor = _run(met_dir, tmp_path / "run", time, kmixd=0)
    hysplit = run.particles
    timed = _timed(hysplit[PARTICLE_SCHEMA.names], receptor)
    sampled = sample_met(timed, run.met_files, ["mlht"], z_top=None)
    floored = sampled.assign(mlht=np.maximum(sampled["mlht"], KMIX0))
    reference = _total(hysplit, receptor)
    print()
    moved = {}
    for name, particles in {
        "sampled": sampled,
        f"floored at {KMIX0} m": floored,
    }.items():
        foot = add_foot(particles, run.met_files, VEGHT)
        moved[name] = _moved(_total(foot, receptor), reference)
        print(
            f"kmixd 0, mlht {name}: footprint total / HYSPLIT's {moved[name][0]:.4f}, "
            f"summed cell differences / HYSPLIT's total {moved[name][1]:.4f}"
        )
    total, cells = moved[f"floored at {KMIX0} m"]
    assert total == pytest.approx(1.0, abs=0.1)
    assert cells <= moved["sampled"][1] + 0.01


@integration
def test_calc_pbl_height_finds_the_mixed_layer_hysplit_finds(hysplit_run):
    """HYSPLIT's ``mlht`` with ``kmixd: 3`` is the Richardson-number height, at least ``kmix0``."""
    run, receptor = hysplit_run
    hysplit = run.particles["mlht"].to_numpy()
    ours = np.maximum(
        calc_pbl_height(_timed(run.particles, receptor), run.met_files), KMIX0
    )

    ratio = ours / hysplit
    close = np.mean(np.abs(ratio - 1) < 0.02)
    print(
        f"\ncalc_pbl_height / HYSPLIT's mlht: median {np.median(ratio):.4f}, "
        f"5-95 % {np.quantile(ratio, 0.05):.3f} to {np.quantile(ratio, 0.95):.3f}, "
        f"rows within 2 % {close:.3f}"
    )
    assert np.isfinite(ours).all()
    assert np.median(ratio) == pytest.approx(1.0, abs=0.01)
    assert close > 0.6


@integration
def test_each_pbl_height_moves_a_positions_only_footprint_from_hysplits(hysplit_run):
    run, receptor = hysplit_run
    positions = run.particles[PARTICLE_SCHEMA.names]
    reference = _total(run.particles, receptor)
    sources = {
        "met": ("met", 0.0),
        f"met, at least {KMIX0} m": ("met", KMIX0),
        f"richardson, at least {KMIX0} m": ("richardson", KMIX0),
        "a fixed 1000 m": (1000.0, 0.0),
    }
    print()
    moved = {}
    for name, (pbl_height, pbl_height_min) in sources.items():
        finished = _finish(
            ModelRun(positions, met_files=run.met_files),
            receptor,
            TransportConfig(
                footprint_depth=VEGHT,
                hnf_plume=False,
                pbl_height=pbl_height,
                pbl_height_min=pbl_height_min,
            ),
        )
        moved[name] = _moved(_total(finished.particles, receptor), reference)
        print(
            f"pbl_height {name}: footprint total / HYSPLIT's {moved[name][0]:.4f}, "
            f"summed cell differences / HYSPLIT's total {moved[name][1]:.4f}"
        )

    # HYSPLIT's own mixed layer leaves only the whole-step counting.
    total, cells = moved[f"richardson, at least {KMIX0} m"]
    assert total == pytest.approx(1.0, abs=0.03)
    assert cells < 0.1


@integration
def test_pbl_height_set_replaces_hysplits_own_foot(hysplit_run):
    run, receptor = hysplit_run
    finished = _finish(
        ModelRun(run.particles, met_files=run.met_files),
        receptor,
        TransportConfig(
            footprint_depth=VEGHT,
            hnf_plume=False,
            pbl_height="richardson",
            pbl_height_min=KMIX0,
        ),
    )
    total, cells = _moved(
        _total(finished.particles, receptor), _total(run.particles, receptor)
    )
    print(
        f"\nHYSPLIT's particles, foot from pbl_height richardson: footprint total / "
        f"HYSPLIT's {total:.4f}, summed cell differences / HYSPLIT's total {cells:.4f}"
    )
    assert "in place of the model's foot and mlht" in finished.log
    assert total == pytest.approx(1.0, abs=0.03)
