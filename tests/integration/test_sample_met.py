"""
Met sampled at particle positions with arlmet, against the columns HYSPLIT wrote.

HYSPLIT writes ``mlht``, ``dens``, ``pres`` and ``zsfc`` from its own copy of
the met. :func:`stilt.sampling.sample_met` reads the same met files with
arlmet at each particle's position and time. Comparing the two checks
arlmet's interpolation against an independent reference. Run with ``-s`` to
see the spread of each column. The bounds hold the spread measured on these
two runs with a margin, so a wrong variable, unit, or time fails and the
known differences pass:

``zsfc``
    Both interpolate the terrain bilinearly. HYSPLIT does it where the
    particle was before its turbulent step and writes ``lon`` and ``lat`` to
    four decimals, so a few meters differ on steep terrain (bias under
    0.2 m, 95 % within 3 m).
``mlht``
    With ``kmixd: 0`` HYSPLIT uses the met's ``PBLH``, at least ``kmix0``,
    at the grid point nearest the particle, averaged over the start and end
    of its step. ``sample_met`` interpolates it bilinearly (median
    difference 16 m in winter, 30 m in summer).
``pres`` and ``dens``
    HYSPLIT finds each met level's height from the ``PRES`` field and the
    virtual temperature, from the ground up. arlmet 0.1.0b3 finds a sigma
    file's level heights from the header's sigma values times ``PRSS``,
    which for HRRR are up to about 50 hPa off ``PRES`` aloft. Near the
    ground the two agree to 0.3 hPa; higher up the sampled pressure is too
    high, by about 1 hPa at 100 to 500 m, 6 to 8 hPa at 1 to 2 km, and
    over 20 hPa above 2 km in the summer run. The
    sampled density is also 0.1 to 0.6 % higher near the ground, more in
    moister air, which is not traced.
Below the lowest level
    arlmet gives NaN below the met's lowest level (8 to 9 m above ground
    for HRRR), where HYSPLIT interpolates from the surface values.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stilt.receptors import PointReceptor
from stilt.sampling import sample_met
from stilt.transport import run_model
from stilt.transport.hysplit import HysplitConfig

from ..conftest import integration
from ..fixtures.r_stilt_reference import REFERENCE_SUMMER_TIME, REFERENCE_TIME
from .conftest import reference_met

COLUMNS = ["mlht", "dens", "pres", "zsfc"]
KMIX0 = 150


def _run(met_dir: Path, workdir: Path, time: dt.datetime) -> tuple[pd.DataFrame, list]:
    """Run HYSPLIT from the WBB tower for 6 hours and return its particles and met files."""
    receptor = PointReceptor(
        time=time, longitude=-111.8479, latitude=40.7660, altitude=5.0
    )
    config = HysplitConfig(
        n_hours=-6,
        numpar=200,
        hnf_plume=False,
        kmixd=0,  # the met's PBLH, so HYSPLIT's mlht can be compared
        kmix0=KMIX0,
        varsiwant=["time", "indx", "long", "lati", "zagl", "foot", *COLUMNS],
    )
    run = run_model("hysplit", receptor, config, reference_met(met_dir), workdir, 900)
    particles = run.particles.copy()
    particles["time"] = pd.Timestamp(time) + pd.to_timedelta(
        particles["age"], unit="min"
    )
    return particles, run.met_files


def _spread(hysplit: pd.Series, sampled: pd.Series) -> dict[str, float]:
    """Return the statistics of ``sampled - hysplit``."""
    d = (sampled - hysplit).to_numpy(dtype=float)
    ok = np.isfinite(d)
    a = np.abs(d[ok])
    return {
        "n": int(ok.sum()),
        "nan": int((~ok).sum()),
        "bias": float(d[ok].mean()),
        "sd": float(d[ok].std()),
        "median_abs": float(np.median(a)),
        "p95_abs": float(np.percentile(a, 95)),
        "max_abs": float(a.max()),
        "rel_bias": float((d[ok] / hysplit.to_numpy(dtype=float)[ok]).mean()),
    }


@integration
@pytest.mark.parametrize(
    "time", [REFERENCE_TIME, REFERENCE_SUMMER_TIME], ids=["winter", "summer"]
)
def test_sampled_met_matches_hysplit_columns(met_dir, tmp_path, time):
    hysplit, met_files = _run(met_dir, tmp_path / "run", time)

    sampled = sample_met(hysplit.drop(columns=COLUMNS), met_files)
    sampled["mlht"] = np.maximum(sampled["mlht"], KMIX0)

    spread = pd.DataFrame({c: _spread(hysplit[c], sampled[c]) for c in COLUMNS}).T
    print(f"\nsample_met minus HYSPLIT, {time:%Y-%m-%d %H:%M}, {len(hysplit)} rows:")
    print(spread.to_string(float_format=lambda v: f"{v:.4g}"))
    zagl = hysplit["zagl"]
    for lo, hi in [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 1e5)]:
        rows = (zagl >= lo) & (zagl < hi)
        if rows.any():
            d = (sampled["pres"] - hysplit["pres"])[rows]
            print(
                f"pres, zagl {lo:g} to {hi:g} m: {int(rows.sum())} rows, "
                f"bias {d.mean():+.3g} hPa, sd {d.std():.3g}"
            )

    # Only rows below the lowest HRRR level (8 to 9 m above ground) are NaN.
    nan_rows = sampled[COLUMNS].isna().any(axis=1)
    assert (zagl[nan_rows] < 20.0).all()
    assert abs(spread.loc["zsfc", "bias"]) < 1.0
    assert spread.loc["zsfc", "p95_abs"] < 6.0
    assert abs(spread.loc["mlht", "bias"]) < 30.0
    assert spread.loc["mlht", "median_abs"] < 50.0
    near_ground = zagl < 100.0
    assert abs((sampled["pres"] - hysplit["pres"])[near_ground].mean()) < 0.5
    assert abs(spread.loc["pres", "bias"]) < 6.0
    assert spread.loc["pres", "p95_abs"] < 20.0
    assert abs(spread.loc["dens", "rel_bias"]) < 0.015
