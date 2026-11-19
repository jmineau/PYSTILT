"""Particle tables as a transport model run leaves them, for tests that start from raw particles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from runtraj.particles import add_release_heights
from runtraj.releases import PointRelease, Release


def finished(particles: pd.DataFrame, release: Release) -> pd.DataFrame:
    """
    Return raw particles with the release heights PYSTILT adds after any model run.

    The near-field correction is not applied: the footprint applies it, as
    :func:`stilt.calc_footprint` does.
    """
    return add_release_heights(particles, release)


def point_at(altitude: float) -> PointRelease:
    """Return a point release at *altitude* m above ground, for the near-field correction."""
    return PointRelease(
        time="2023-01-01 12:00", longitude=-111.85, latitude=40.77, altitude=altitude
    )


def random_walk(
    n_particles: int = 20,
    n_steps: int = 48 * 6,
    step_minutes: int = 10,
    seed: int = 0,
    release_time: str = "2024-01-15 18:00",
) -> pd.DataFrame:
    """
    Return a backward particle table: a seeded random walk from a release at -111.85, 40.77.

    Particle ``k`` stops after ``n_steps - k`` steps, so the trajectories
    have different lengths, as particles that leave the meteorology domain
    do. The table has a ``time`` column, as one read from a particle file does.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for particle in range(1, n_particles + 1):
        steps = n_steps - particle
        age = -step_minutes * np.arange(1, steps + 1, dtype=float)
        lon = -111.85 + np.cumsum(rng.normal(-0.004, 0.01, steps))
        lat = 40.77 + np.cumsum(rng.normal(0.002, 0.01, steps))
        zagl = np.abs(5 + np.cumsum(rng.normal(2, 20, steps)))
        foot = np.where(zagl < 200, rng.uniform(0, 0.01, steps), 0.0)
        rows.append(
            pd.DataFrame(
                {
                    "particle": float(particle),
                    "age": age,
                    "lon": lon,
                    "lat": lat,
                    "zagl": zagl,
                    "foot": foot,
                    "mlht": 600.0,
                }
            )
        )
    table = pd.concat(rows, ignore_index=True)
    table["time"] = pd.Timestamp(release_time) + pd.to_timedelta(
        table["age"], unit="min"
    )
    return table
