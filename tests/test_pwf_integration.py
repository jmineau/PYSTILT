"""
Integration tests for particle-derived pressure weighting.

The unit tests in ``test_apply_vertical_operator.py`` use an isothermal
reference atmosphere, where ``ln p`` is exactly linear in height and the
hypsometric fit is perfect by construction.  These tests run real HYSPLIT
against real HRRR fields to check the two things that synthetic data cannot
show: how well the fit holds on a real (non-isothermal) profile, and whether
the weights stay physical once particles have been scattered by one timestep
of turbulent transport.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from stilt.config import STILTParams
from stilt.hysplit.driver import HYSPLITDriver
from stilt.meteorology import MetStream
from stilt.observations.apply import apply_vertical_operator
from stilt.observations.operators import VerticalOperator
from stilt.observations.weighting import _particle_pwf, _release_coordinate
from stilt.receptors import ColumnReceptor
from stilt.trajectory import Trajectories

from .conftest import integration
from .fixtures.r_stilt_reference import (
    REFERENCE_MET_FILE_FORMAT,
    REFERENCE_TIME,
)

# WBB sits at ~1480 m, so surface pressure is ~850 hPa rather than ~1000.
WBB_LON = -111.8479
WBB_LAT = 40.7660
SCALE_HEIGHT_RANGE = (6500.0, 9000.0)


def _column_trajectory(
    met_dir: Path,
    tmp_path: Path,
    label: str,
    *,
    numpar: int = 100,
    top: float = 3000.0,
    time: dt.datetime = REFERENCE_TIME,
):
    """Run a short WBB column simulation and return (receptor, particles)."""
    receptor = ColumnReceptor(time, WBB_LON, WBB_LAT, 0.0, top)
    params = STILTParams(n_hours=-2, numpar=numpar, hnf_plume=False, rm_dat=True)
    met_files = MetStream(
        "hrrr",
        directory=met_dir,
        file_format=REFERENCE_MET_FILE_FORMAT,
        file_tres="6h",
    ).required_files(r_time=receptor.time, n_hours=params.n_hours)

    driver = HYSPLITDriver(
        receptor=receptor,
        params=params,
        met_files=met_files,
        directory=Path(tmp_path) / label,
    )
    driver.prepare()
    result = driver.execute(timeout=900, rm_dat=True)
    trajectory = Trajectories.from_particles(
        particles=result.particles,
        receptor=receptor,
        params=params,
        met_files=met_files,
    )
    return receptor, trajectory.data


@integration
def test_hypsometric_fit_holds_on_a_real_winter_profile(met_dir, tmp_path):
    """
    A Salt Lake winter inversion is the least isothermal case available, and
    the log-linear pressure fit still explains essentially all the variance.
    """
    _, particles = _column_trajectory(met_dir, tmp_path, "pwf_fit_winter")

    pres = _release_coordinate(particles, "pres").to_numpy()
    zagl = _release_coordinate(particles, "zagl").to_numpy()
    slope, intercept = np.polyfit(zagl, np.log(pres), 1)

    residual = pres - np.exp(intercept + slope * zagl)
    r_squared = 1 - (residual**2).sum() / ((pres - pres.mean()) ** 2).sum()
    assert r_squared > 0.99

    scale_height = -1 / slope
    assert SCALE_HEIGHT_RANGE[0] < scale_height < SCALE_HEIGHT_RANGE[1]
    # WBB is at ~1480 m elevation: the fitted surface pressure must reflect
    # that, not sea level.
    assert 800.0 < float(np.exp(intercept)) < 900.0


@integration
def test_pwf_weights_are_physical_on_real_trajectories(met_dir, tmp_path):
    """Every particle carries positive weight spanning the hydrostatic ratio."""
    receptor, particles = _column_trajectory(met_dir, tmp_path, "pwf_physical")
    _, pwf = _particle_pwf(particles, None)

    assert (pwf > 0).all()
    # A 0-3 km column holds roughly a third of the atmosphere above WBB.
    assert 0.25 < pwf.sum() < 0.40
    # Across 3 km the air density ratio is exp(3000/H) ~ 1.5, so the spread of
    # weights must be close to that and nowhere near the ~100x that raw
    # first-step pressures would produce.
    assert 1.2 < pwf.max() / pwf.min() < 2.0


@integration
def test_pwf_fit_absorbs_first_step_particle_scatter(met_dir, tmp_path):
    """
    HYSPLIT's first output step is not the release state: turbulence has
    already scattered particles several slab widths, leaving raw pressures
    non-monotone in release height.  Weighting on those raw pressures directly
    would hand neighbouring particles wildly different weights; the fit is
    what keeps the profile physical.
    """
    _, particles = _column_trajectory(met_dir, tmp_path, "pwf_scatter", numpar=1000)

    xhgt = _release_coordinate(particles, "xhgt").to_numpy()
    raw_pres = _release_coordinate(particles, "pres").to_numpy()

    # Establish the premise: raw pressure really is non-monotone in height.
    ascending = np.argsort(xhgt)
    inversions = int((np.diff(raw_pres[ascending]) > 0).sum())
    assert inversions > len(xhgt) // 20

    levels = np.sort(raw_pres)[::-1]
    mids = (levels[:-1] + levels[1:]) / 2.0
    raw_pwf = (
        np.concatenate(([levels[0]], mids))
        - np.concatenate((mids, [2 * levels[-1] - mids[-1]]))
    ) / levels[0]

    _, fitted_pwf = _particle_pwf(particles, None)

    # Both integrate to about the same column mass ...
    assert fitted_pwf.sum() == pytest.approx(raw_pwf.sum(), rel=0.05)
    # ... but only the fitted weights are smooth from particle to particle.
    assert fitted_pwf.max() / fitted_pwf.min() < 2.0
    assert raw_pwf.max() / raw_pwf.min() > 10.0


@integration
def test_pwf_is_independent_of_numpar_on_real_trajectories(met_dir, tmp_path):
    """
    The regression behind the 0.1.0a7 bug report, on real trajectories.

    Note what is and is not asserted here.  A STILT footprint's absolute
    magnitude is *not* numpar-independent: two ensembles of different size
    sample different turbulent paths, and the unweighted per-particle mean
    routinely differs by tens of percent between a 100- and a 1000-particle
    run.  What must be numpar-independent is the weighting's own effect, so
    the comparison is of the weighted-to-unweighted ratio, which divides that
    ensemble noise out.  Comparing raw magnitudes here would make the test
    flaky for a reason that has nothing to do with weighting.
    """
    _, few = _column_trajectory(met_dir, tmp_path, "pwf_n100", numpar=100)
    _, many = _column_trajectory(met_dir, tmp_path, "pwf_n1000", numpar=1000)

    _, pwf_few = _particle_pwf(few, None)
    _, pwf_many = _particle_pwf(many, None)
    assert pwf_few.sum() == pytest.approx(pwf_many.sum(), rel=0.02)

    ratio_few = (
        apply_vertical_operator(few, VerticalOperator(mode="pwf"))["foot"].sum()
        / few["foot"].sum()
    )
    ratio_many = (
        apply_vertical_operator(many, VerticalOperator(mode="pwf"))["foot"].sum()
        / many["foot"].sum()
    )
    assert ratio_few == pytest.approx(ratio_many, rel=0.05)


@integration
def test_pwf_keeps_footprint_magnitude_comparable_to_unweighted(met_dir, tmp_path):
    """Weighting redistributes influence; it must not rescale it by numpar."""
    _, particles = _column_trajectory(met_dir, tmp_path, "pwf_magnitude")
    weighted = apply_vertical_operator(particles, VerticalOperator(mode="pwf"))
    ratio = weighted["foot"].sum() / particles["foot"].sum()
    assert 0.1 < ratio < 10.0


@integration
def test_pwf_deeper_column_covers_more_mass(met_dir, tmp_path):
    """A 0-8 km column holds about twice the air mass of a 0-3 km one."""
    _, shallow = _column_trajectory(met_dir, tmp_path, "pwf_3km", top=3000.0)
    _, deep = _column_trajectory(met_dir, tmp_path, "pwf_8km", top=8000.0)

    _, pwf_shallow = _particle_pwf(shallow, None)
    _, pwf_deep = _particle_pwf(deep, None)
    assert pwf_deep.sum() > pwf_shallow.sum()
    assert 0.55 < pwf_deep.sum() < 0.80
