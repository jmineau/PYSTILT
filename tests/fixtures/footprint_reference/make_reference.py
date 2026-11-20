"""
Write the particles and STILT-R's footprint of them that the unit suite compares against.

The unit tests in ``tests/footprint/test_stilt_r_reference.py`` grid
``particles.csv`` with PYSTILT and compare it with ``stilt_r_foot.nc`` at
``rtol=1e-7``, so a change to the smoothing kernel fails ``just test``
without R. Rerun this only when ``STILT_R_SHA`` (``.github/workflows/stilt-r.yml``)
moves and the ``r_only`` test of the reference fails::

    STILT_R_DIR=<uataq/stilt at STILT_R_SHA, built> python tests/fixtures/footprint_reference/make_reference.py

The particles are synthetic and seeded: 30 particles leave one release point
and spread for six hours, with two rows each in the first 100 minutes (so
STILT-R's helper adds no anchor rows) and a ``foot`` on every row after the
release, so the kernel width is computed at many ages and spreads.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R_HELPER = HERE.parent / "r_helpers" / "calc_footprint.r"

#: The release point, and the grid the footprint is made on.
LONGITUDE, LATITUDE = -112.0, 40.5
GRID = {
    "xmin": -112.2,
    "xmax": -111.7,
    "ymin": 40.3,
    "ymax": 40.7,
    "xres": 0.01,
    "yres": 0.01,
}


def make_particles(n: int = 30, seed: int = 467) -> pd.DataFrame:
    """
    Return the reference particles, with STILT-R's column names.

    Rows at -1 and -2 minutes at the release point, then every 20 minutes to
    six hours back, each particle drifting east and north with a random walk.
    """
    rng = np.random.default_rng(seed)
    ages = np.array([-1.0, -2.0, *np.arange(-20.0, -361.0, -20.0)])
    steps = rng.normal(0.0, 0.008, (len(ages), n, 2))
    steps[:2] = 0.0
    drift = np.array([0.006, 0.002])
    position = np.cumsum(
        steps + drift * (np.arange(len(ages)) >= 2)[:, None, None], axis=0
    )
    rows = [
        pd.DataFrame(
            {
                "time": age,
                "indx": np.arange(1.0, n + 1.0),
                "long": np.round(LONGITUDE + position[i, :, 0], 5),
                "lati": np.round(LATITUDE + position[i, :, 1], 5),
                "zagl": 5.0,
                "foot": 0.0 if i < 2 else np.round(rng.uniform(5e-4, 2e-3, n), 7),
            }
        )
        for i, age in enumerate(ages)
    ]
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Write ``particles.csv`` and ``stilt_r_foot.nc`` beside this script."""
    stilt_r_dir = os.environ.get("STILT_R_DIR")
    if not stilt_r_dir:
        sys.exit("Set STILT_R_DIR to a built uataq/stilt checkout at STILT_R_SHA.")
    particles = make_particles()
    particles.to_csv(HERE / "particles.csv", index=False)
    parquet = HERE / "particles.parquet"
    particles.to_parquet(parquet, index=False)
    try:
        subprocess.run(
            [
                "Rscript",
                str(R_HELPER),
                str(parquet),
                str(HERE / "stilt_r_foot.nc"),
                *(
                    str(GRID[k])
                    for k in ("xmin", "xmax", "ymin", "ymax", "xres", "yres")
                ),
                "1.0",
                "FALSE",
                stilt_r_dir,
                "+proj=longlat",
            ],
            check=True,
        )
    finally:
        parquet.unlink()


if __name__ == "__main__":
    main()
