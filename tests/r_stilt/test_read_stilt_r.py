"""
``stilt.migration.read_stilt_r`` on a ``by-id`` folder the pinned STILT-R writes.

``fixtures/r_helpers/write_by_id.r`` writes a project with STILT-R's own
``write_control``, ``saveRDS`` and ``calc_footprint``, as it wrote the
committed sample, and R reads it back with ``readRDS``. The reader must give
the same receptor time and the same particles as R, the release points of
``CONTROL``, and the same footprint as the file, made with the
``smooth_factor`` of ``params.RDS``, and with ``error=True`` the
transport-error particles R reads.
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
  sets <- list(particles = out$particle, error = out$particle_error)
  for (set in names(sets)) {
    if (is.null(sets[[set]])) next
    p <- as.data.frame(sets[[set]])
    p[] <- lapply(p, function(x) sprintf('%.17g', x))
    write.csv(p, file.path(args[2], paste0(basename(dir), '_', set, '.csv')),
              row.names = FALSE, quote = FALSE)
  }
}
"""

_PYSTILT_NAMES = {
    "indx": "particle",
    "time": "age",
    "long": "lon",
    "lati": "lat",
    "xhgt": "release_height",
}


def _read_dump(path: Path) -> pd.DataFrame:
    """
    Read a CSV R dumped, every digit exact.

    ``float_precision="round_trip"`` did this, but pandas 3.1 deprecates it
    (every mode now uses one converter), and pandas 3.0's default parser is
    off in the last digit for some ``%.17g`` values. Casting the text with
    ``astype(float)`` is exact on both, and ``dtype=str`` keeps ``read_csv``
    from parsing the numbers first.
    """
    return pd.read_csv(path, dtype=str).astype(float)


@pytest.fixture(scope="module")
def written(rscript, r_stilt_dir, tmp_path_factory) -> tuple[Path, Path]:
    """Return a by-id folder STILT-R wrote, and a folder of what R reads from it."""
    root = tmp_path_factory.mktemp("stilt_r")
    project, dumped = root / "project", root / "dumped"
    by_id = project / "out" / "by-id"
    dumped.mkdir()
    script = root / "dump.r"
    script.write_text(_DUMP)
    for command in (
        [rscript, str(_R_HELPERS / "write_by_id.r"), str(r_stilt_dir), str(project)],
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
        lines = (by_id / name / "CONTROL").read_text().splitlines()
        points = [
            tuple(map(float, line.split())) for line in lines[2 : 2 + int(lines[1])]
        ]
        assert receptor.coords() == points
        from_r = _read_dump(dumped / f"{name}_particles.csv").rename(
            columns=_PYSTILT_NAMES
        )
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
        assert foot.stilt.config.smooth_factor == 0.5


def test_error_particles_are_what_r_reads(written):
    by_id, dumped = written
    # Only the point receptor, the first folder, has a transport-error run.
    (path,) = dumped.glob("*_error.csv")
    receptor, particles, foot = next(read_stilt_r(by_id, error=True))
    assert f"{receptor.id}_error.csv" == path.name
    from_r = _read_dump(path).rename(columns=_PYSTILT_NAMES)
    mine = particles.drop(columns="time")
    assert list(mine.columns) == list(from_r.columns)
    np.testing.assert_array_equal(mine.to_numpy(), from_r.to_numpy())
    assert foot is None
