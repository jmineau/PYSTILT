"""
Single source of truth for the seeded R-STILT reference case.

This module defines the canonical receptor, grid, transport controls, and
fixture paths used by both the R-STILT fixture generator and the PYSTILT
fidelity integration test.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stilt.config import FootprintConfig, Grid, ModelConfig
    from stilt.receptor import Receptor

REFERENCE_FIXTURES_DIR = Path(__file__).with_name("r_ref")
REFERENCE_TRAJECTORY_PATH = REFERENCE_FIXTURES_DIR / "r_traj.parquet"
REFERENCE_FOOTPRINT_PATH = REFERENCE_FIXTURES_DIR / "r_foot.nc"

REFERENCE_MET = "hrrr"
REFERENCE_TIME = dt.datetime(2015, 12, 10, 0, 0)
REFERENCE_LONGITUDE = -112.0
REFERENCE_LATITUDE = 40.5
REFERENCE_ALTITUDE = 5.0

REFERENCE_N_HOURS = -6
REFERENCE_NUMPAR = 1000
REFERENCE_KRAND = 2
REFERENCE_SEED = 42
REFERENCE_MET_FILE_FORMAT = "%Y%m%d.%Hz.hrrra"
REFERENCE_MET_FILE_INTERVAL_HOURS = 6

REFERENCE_XMIN = -113.0
REFERENCE_XMAX = -111.0
REFERENCE_YMIN = 39.5
REFERENCE_YMAX = 41.5
REFERENCE_XRES = 0.01
REFERENCE_YRES = 0.01

REFERENCE_VARSIWANT = [
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

REFERENCE_TRAJECTORY_COMPARE_COLUMNS = [
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
    "foot_no_hnf_dilution",
]


def reference_r_sim_id() -> str:
    """Return the canonical R-STILT simulation id for the seeded-fidelity case."""
    return (
        f"{REFERENCE_TIME.strftime('%Y%m%d%H%M')}_"
        f"{REFERENCE_LONGITUDE:g}_{REFERENCE_LATITUDE:g}_{REFERENCE_ALTITUDE:g}"
    )


def reference_time_str() -> str:
    """Return the canonical receptor timestamp for shell and R usage."""
    return REFERENCE_TIME.strftime("%Y-%m-%d %H:%M:%S")


def reference_python_met_file_tres() -> str:
    """Return the Python-side met file interval string."""
    return f"{REFERENCE_MET_FILE_INTERVAL_HOURS}h"


def reference_r_met_file_tres() -> str:
    """Return the R-STILT met file interval string."""
    return f"{REFERENCE_MET_FILE_INTERVAL_HOURS} hours"


def reference_receptor() -> Receptor:
    """Return the canonical seeded-fidelity receptor."""
    from stilt.receptor import Receptor

    return Receptor(
        time=REFERENCE_TIME,
        longitude=REFERENCE_LONGITUDE,
        latitude=REFERENCE_LATITUDE,
        altitude=REFERENCE_ALTITUDE,
    )


def reference_grid() -> Grid:
    """Return the canonical seeded-fidelity footprint grid."""
    from stilt.config import Grid

    return Grid(
        xmin=REFERENCE_XMIN,
        xmax=REFERENCE_XMAX,
        ymin=REFERENCE_YMIN,
        ymax=REFERENCE_YMAX,
        xres=REFERENCE_XRES,
        yres=REFERENCE_YRES,
    )


def reference_footprint_config() -> FootprintConfig:
    """Return the canonical seeded-fidelity footprint config."""
    from stilt.config import FootprintConfig

    return FootprintConfig(grid=reference_grid())


def reference_model_config(met_dir: Path) -> ModelConfig:
    """Return the canonical seeded-fidelity PYSTILT config."""
    from stilt.config import ModelConfig

    return ModelConfig.model_validate(
        {
            "mets": {
                REFERENCE_MET: {
                    "directory": met_dir,
                    "file_format": REFERENCE_MET_FILE_FORMAT,
                    "file_tres": reference_python_met_file_tres(),
                }
            },
            "n_hours": REFERENCE_N_HOURS,
            "numpar": REFERENCE_NUMPAR,
            "krand": REFERENCE_KRAND,
            "seed": REFERENCE_SEED,
            "varsiwant": REFERENCE_VARSIWANT,
            "footprints": {"default": reference_footprint_config()},
        }
    )


def reference_python_sim_id() -> str:
    """Return the canonical PYSTILT simulation id for the seeded-fidelity case."""
    from stilt.simulation import SimID

    return str(SimID.from_parts(REFERENCE_MET, reference_receptor()))


def shell_exports() -> str:
    """Return shell-safe variable assignments for the reference case."""
    values = {
        "STILT_REFERENCE_R_SIM_ID": reference_r_sim_id(),
        "STILT_REFERENCE_PYTHON_SIM_ID": reference_python_sim_id(),
        "STILT_REFERENCE_TIME": reference_time_str(),
        "STILT_REFERENCE_LONGITUDE": REFERENCE_LONGITUDE,
        "STILT_REFERENCE_LATITUDE": REFERENCE_LATITUDE,
        "STILT_REFERENCE_ALTITUDE": REFERENCE_ALTITUDE,
        "STILT_REFERENCE_N_HOURS": REFERENCE_N_HOURS,
        "STILT_REFERENCE_NUMPAR": REFERENCE_NUMPAR,
        "STILT_REFERENCE_KRAND": REFERENCE_KRAND,
        "STILT_REFERENCE_SEED": REFERENCE_SEED,
        "STILT_REFERENCE_MET": REFERENCE_MET,
        "STILT_REFERENCE_MET_FILE_FORMAT": REFERENCE_MET_FILE_FORMAT,
        "STILT_REFERENCE_R_MET_FILE_TRES": reference_r_met_file_tres(),
        "STILT_REFERENCE_XMIN": REFERENCE_XMIN,
        "STILT_REFERENCE_XMAX": REFERENCE_XMAX,
        "STILT_REFERENCE_YMIN": REFERENCE_YMIN,
        "STILT_REFERENCE_YMAX": REFERENCE_YMAX,
        "STILT_REFERENCE_XRES": REFERENCE_XRES,
        "STILT_REFERENCE_YRES": REFERENCE_YRES,
    }
    return "\n".join(
        f"{key}={shlex.quote(str(value))}" for key, value in values.items()
    )


def main() -> None:
    """Emit machine-readable exports for shell consumers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=["shell"], nargs="?", default="shell")
    args = parser.parse_args()

    if args.format == "shell":
        print(shell_exports())


if __name__ == "__main__":
    main()
