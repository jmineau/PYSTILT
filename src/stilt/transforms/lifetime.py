"""
The first-order lifetime transform: decay each particle's ``foot`` with its age.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from stilt.receptors import Receptor


class FirstOrderLifetime(BaseModel):
    """
    Transform that decays each particle's ``foot`` by ``exp(-age / lifetime)``.

    ``age`` is the particle's time since release, its ``age`` column
    (minutes), and the lifetime is the species' e-folding lifetime.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["first_order_lifetime"] = "first_order_lifetime"
    lifetime_hours: float = Field(gt=0, description="E-folding lifetime, in hours.")

    def apply(
        self,
        particles: pd.DataFrame,
        receptor: Receptor | None = None,
        directory: str | Path | None = None,
    ) -> pd.DataFrame:
        """Return the particles with ``foot`` decayed by age."""
        if "age" not in particles.columns:
            raise ValueError(
                "Particle DataFrame has no 'age' column, required for "
                "first_order_lifetime."
            )
        age_hours = np.abs(particles["age"].to_numpy(dtype=float)) / 60.0
        out = particles.copy()
        out["foot"] = out["foot"] * np.exp(-age_hours / self.lifetime_hours)
        return out
