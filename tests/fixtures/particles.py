"""Particle tables as a transport model run leaves them, for tests that start from raw particles."""

from __future__ import annotations

from typing import Any

import pandas as pd

from stilt.particles import add_release_heights, correct_near_field
from stilt.receptors import PointReceptor, Receptor


def finished(particles: pd.DataFrame, receptor: Receptor, config: Any) -> pd.DataFrame:
    """
    Return raw particles with the steps PYSTILT applies after any model run.

    The release heights, then the near-field correction when
    ``config.hnf_plume`` is set, as :func:`stilt.transport.run_model` does.
    """
    particles = add_release_heights(particles, receptor)
    if config.hnf_plume:
        particles = correct_near_field(particles, receptor, config.veght)
    return particles


def point_at(altitude: float) -> PointReceptor:
    """Return a point receptor at *altitude* m above ground, for the near-field correction."""
    return PointReceptor(
        time="2023-01-01 12:00", longitude=-111.85, latitude=40.77, altitude=altitude
    )
