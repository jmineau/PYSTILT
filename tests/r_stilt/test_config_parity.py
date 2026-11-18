"""
Check that a new project's SETUP.CFG is the one STILT-R writes.

``HysplitConfig`` has HYSPLIT's own defaults; ``stilt init`` and
``Project.init`` write STILT-R's values where they differ
(:data:`stilt.config.STILT_R_HYSPLIT`). The first test has STILT-R's own
``write_setup()`` write ``SETUP.CFG`` from the values its ``run_stilt.r``
sets, and compares it entry by entry with what PYSTILT writes from the
starter config. It needs R and a STILT-R checkout, not HYSPLIT, so it runs
with ``-m r_only``. The others read the particles of the fidelity runs.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from stilt.stilt_project import STARTER_CONFIG
from stilt.transport.hysplit import HysplitConfig
from stilt.transport.hysplit.driver import setup_entries, winderrtf
from stilt.transport.hysplit.namelist import NameList, read_namelist

from ..fixtures.r_stilt_reference import STILT_R_NAMES

_R_HELPERS = Path(__file__).parents[1] / "fixtures" / "r_helpers"

#: Entries the driver writes for each run, from its release and error
#: settings, rather than from the config (``write_inputs``).
_PER_RUN = {"ivmax", "kmsl", "maxpar", "winderrtf"}

# Columns STILT-R's varsiwant always requests.  The trajectory parquet must
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


@pytest.mark.r_only
def test_a_new_projects_setup_cfg_is_the_one_stilt_r_writes(
    rscript, r_stilt_dir, tmp_path
) -> None:
    """
    Every SETUP.CFG entry of the starter config has STILT-R's value.

    PYSTILT adds ``pres`` to ``varsiwant``, which changes no particle. A
    setting left out of the starter would run HYSPLIT's own default and
    show here as missing from PYSTILT's file or as another value.
    """
    r_file = tmp_path / "r" / "SETUP.CFG"
    r_file.parent.mkdir()
    subprocess.run(
        [rscript, str(_R_HELPERS / "write_setup.r"), str(r_stilt_dir), str(r_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    r_entries = read_namelist(r_file)

    config = HysplitConfig(**yaml.safe_load(STARTER_CONFIG)["models"]["hysplit"])
    py_file = tmp_path / "SETUP.CFG"
    namelist = NameList("SETUP")
    namelist.update(setup_entries(config))
    namelist.write(py_file)
    py_entries = read_namelist(py_file)

    assert [v for v in py_entries.pop("varsiwant") if v != "pres"] == r_entries.pop(
        "varsiwant"
    )
    # One release location gets maxpar = numpar, as in STILT-R.
    assert config.maxpar_for(1) == r_entries["maxpar"]
    assert winderrtf(config) == r_entries["winderrtf"]
    differ = {
        name: (py_entries.get(name, "absent"), r_entries.get(name, "absent"))
        for name in (py_entries.keys() | r_entries.keys()) - _PER_RUN
        if py_entries.get(name) != r_entries.get(name)
    }
    assert not differ, f"(PYSTILT, STILT-R) values that differ: {differ}"


@pytest.mark.fidelity
def test_trajectory_contains_all_r_varsiwant_columns(scenario_outputs: dict) -> None:
    """
    Trajectory parquet exposes every column STILT-R's varsiwant requests.

    PYSTILT may output additional columns beyond R's list without issue, but
    the R-required columns must all be present so the trajectory comparison
    tests have their expected inputs.
    """
    # In STILT-R's names, so they read against its varsiwant.
    traj = pd.read_parquet(scenario_outputs["traj"]).rename(columns=STILT_R_NAMES)
    missing = [c for c in _R_VARSIWANT if c not in traj.columns]
    assert not missing, f"Trajectory parquet missing R-required columns: {missing}"


@pytest.mark.fidelity
def test_setup_cfg_ivmax_at_least_r_varsiwant_count(scenario_outputs: dict) -> None:
    """
    ``ivmax`` in SETUP.CFG is at least as large as R's varsiwant column count.

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
