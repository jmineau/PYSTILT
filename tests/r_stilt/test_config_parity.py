"""
Verify PYSTILT's SETUP.CFG matches calc_trajectory.r's physics namelist.

The R helper calc_trajectory.r passes an explicit 40-parameter namelist to
R-STILT's write_setup().  PYSTILT builds its SETUP.CFG from TransportParams
defaults.  This module checks that every physics-affecting entry agrees, so
that a silent default divergence is caught before it can affect trajectories.

Administrative entries (cmass, conage, ncycl, ndump, pinbc, etc.) are omitted
because they do not influence particle motion or footprint sensitivity.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from ..conftest import integration

pytestmark = [pytest.mark.fidelity]

# ---------------------------------------------------------------------------
# Reference values from tests/fixtures/r_helpers/calc_trajectory.r
# ---------------------------------------------------------------------------

# Every key here corresponds to a parameter that affects particle trajectories
# or footprint sensitivity.  Value is what calc_trajectory.r passes to R-STILT.
_R_PHYSICS_NAMELIST: dict[str, float] = {
    "capemin": -1.0,
    "delt": 1.0,
    "hscale": 10800.0,
    "ichem": 8.0,
    "initd": 0.0,
    "k10m": 1.0,
    "kbls": 1.0,
    "kblt": 5.0,
    "kdef": 0.0,
    "kmix0": 150.0,
    "kmixd": 3.0,
    "kpuff": 0.0,
    "kwet": 1.0,
    "kzmix": 0.0,
    "nturb": 0.0,
    "tkerd": 0.18,
    "tkern": 0.18,
    "tlfrac": 0.1,
    "tratio": 0.75,
    "tvmix": 1.0,
    "veght": 0.5,
    "vscale": 200.0,
    "vscaleu": 200.0,
    "vscales": -1.0,
}

# Columns R-STILT's varsiwant always requests.  The trajectory parquet must
# contain all of these; PYSTILT may add extras (e.g. pres) without consequence.
_R_VARSIWANT: list[str] = [
    "time",
    "indx",
    "long",
    "lati",
    "zagl",
    "foot",
    "mlht",
    "dens",
    "samt",
    "sigw",
    "tlgr",
]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_setup_cfg(path: Path) -> dict[str, str]:
    """Parse SETUP.CFG key=value pairs, case-insensitive."""
    result: dict[str, str] = {}
    for line in path.read_text().lower().splitlines():
        for token in line.replace(",", " ").split():
            if "=" in token:
                k, _, v = token.partition("=")
                result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@integration
def test_setup_cfg_physics_matches_r_namelist(scenario_outputs: dict) -> None:
    """
    Every physics-relevant SETUP.CFG entry matches calc_trajectory.r's namelist.

    Catches silent default divergence that would not trigger a trajectory-level
    failure until a parameter crosses a physics discontinuity (e.g. kblt
    selecting a different turbulence scheme).
    """
    parsed = _parse_setup_cfg(scenario_outputs["setup"])
    mismatches: list[str] = []
    for key, r_val in _R_PHYSICS_NAMELIST.items():
        py_str = parsed.get(key)
        if py_str is None:
            mismatches.append(f"  {key}: absent from SETUP.CFG (R wants {r_val})")
            continue
        try:
            py_val = float(py_str)
        except ValueError:
            mismatches.append(f"  {key}: unparseable value {py_str!r}")
            continue
        if py_val != r_val:
            mismatches.append(f"  {key}: PYSTILT={py_val}, R={r_val}")
    assert not mismatches, (
        "SETUP.CFG physics entries diverge from R namelist:\n" + "\n".join(mismatches)
    )


@integration
def test_trajectory_contains_all_r_varsiwant_columns(scenario_outputs: dict) -> None:
    """
    Trajectory parquet exposes every column R-STILT's varsiwant requests.

    PYSTILT may output additional columns beyond R's list without issue, but
    the R-required columns must all be present so the trajectory comparison
    tests have their expected inputs.
    """
    traj = pd.read_parquet(scenario_outputs["traj"])
    missing = [c for c in _R_VARSIWANT if c not in traj.columns]
    assert not missing, f"Trajectory parquet missing R-required columns: {missing}"


@integration
def test_setup_cfg_ivmax_at_least_r_varsiwant_count(scenario_outputs: dict) -> None:
    """
    ivmax in SETUP.CFG is at least as large as R's varsiwant column count.

    ivmax tells HYSPLIT how many output columns to write.  If it were smaller
    than R's varsiwant length, the particle file would be truncated and column
    parsing would be misaligned.
    """
    content = scenario_outputs["setup"].read_text().lower()
    m = re.search(r"ivmax\s*=\s*(\d+)", content)
    assert m, "ivmax not found in SETUP.CFG"
    ivmax = int(m.group(1))
    assert ivmax >= len(_R_VARSIWANT), (
        f"ivmax={ivmax} < R's varsiwant count ({len(_R_VARSIWANT)}); "
        "HYSPLIT would produce too few output columns."
    )
