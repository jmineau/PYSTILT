"""
``stilt.migration.read_stilt_r`` on a ``by-id`` folder the pinned STILT-R writes.

``fixtures/r_helpers/write_by_id.r`` writes the folder with STILT-R's own
``saveRDS`` and ``calc_footprint``, as it wrote the committed sample, and R
reads it back with ``readRDS``. The reader must give the same receptor time
and the same particles as R, and the same footprint as the file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stilt.migration import read_stilt_r

pytestmark = [pytest.mark.r_only]
pytest.importorskip("rdata")

_R_HELPERS = Path(__file__).parents[1] / "fixtures" / "r_helpers"

# R writes each simulation's receptor time (seconds since 1970) and its
# particles, all digits kept, with STILT-R's column names.
_DUMP = r"""
args <- commandArgs(trailingOnly = TRUE)
for (dir in list.dirs(args[1], recursive = FALSE)) {
  out <- readRDS(file.path(dir, paste0(basename(dir), '_traj.rds')))
  writeLines(sprintf('%.17g', as.numeric(out$receptor$run_time)),
             file.path(args[2], paste0(basename(dir), '_time.txt')))
  p <- as.data.frame(out$particle)
  p[] <- lapply(p, function(x) sprintf('%.17g', x))
  write.csv(p, file.path(args[2], paste0(basename(dir), '_particles.csv')),
            row.names = FALSE, quote = FALSE)
}
"""

_PYSTILT_NAMES = {
    "indx": "particle",
    "time": "age",
    "long": "lon",
    "lati": "lat",
    "xhgt": "release_height",
}


@pytest.fixture(scope="module")
def written(rscript, r_stilt_dir, tmp_path_factory) -> tuple[Path, Path]:
    """Return a by-id folder STILT-R wrote, and a folder of what R reads from it."""
    root = tmp_path_factory.mktemp("stilt_r_by_id")
    by_id, dumped = root / "by-id", root / "dumped"
    dumped.mkdir()
    script = root / "dump.r"
    script.write_text(_DUMP)
    for command in (
        [rscript, str(_R_HELPERS / "write_by_id.r"), str(r_stilt_dir), str(by_id)],
        [rscript, str(script), str(by_id), str(dumped)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{command[1]} failed:\n{result.stderr}")
    return by_id, dumped


def test_receptor_time_and_particles_are_what_r_reads(written):
    by_id, dumped = written
    sims = list(read_stilt_r(by_id))
    names = sorted(p.name for p in by_id.iterdir())
    assert len(sims) == len(names) == 2
    for name, (receptor, particles, _) in zip(names, sims, strict=True):
        seconds = float((dumped / f"{name}_time.txt").read_text())
        assert pd.Timestamp(receptor.time) == pd.Timestamp(seconds, unit="s")
        from_r = pd.read_csv(
            dumped / f"{name}_particles.csv", float_precision="round_trip"
        ).rename(columns=_PYSTILT_NAMES)
        mine = particles.drop(columns="time")
        assert list(mine.columns) == list(from_r.columns)
        np.testing.assert_array_equal(mine.to_numpy(), from_r.to_numpy())


def test_footprint_is_the_file_written(written):
    by_id, _ = written
    names = sorted(p.name for p in by_id.iterdir())
    for name, (_, _, foot) in zip(names, read_stilt_r(by_id), strict=True):
        path = by_id / name / f"{name}_foot.nc"
        with xr.open_dataset(path, decode_times=False) as ds:
            raw = ds["foot"].load().transpose("time", "lat", "lon")
        stamps = pd.to_datetime(raw["time"].to_numpy(), unit="s")
        np.testing.assert_array_equal(foot["time"].to_numpy(), stamps.to_numpy())
        np.testing.assert_array_equal(foot.to_numpy(), raw.to_numpy().astype(float))
