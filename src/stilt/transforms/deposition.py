"""
The dry deposition transform: weigh each particle's ``foot`` by the tracer that survives to the receptor.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from stilt.receptors import Receptor

#: The longest time step, in minutes, the surviving fraction is computed over.
#: A particle can cross the surface layer within a few minutes, so rows
#: further apart than HYSPLIT's one-minute step do not say how long it
#: stayed below the layer top.
MAX_STEP_MINUTES = 1.0


def dry_deposition_survival(
    particles: pd.DataFrame,
    velocity: float,
    surface_layer_depth: float = 50.0,
) -> pd.Series:
    r"""
    Return the fraction of a depositing tracer that survives from each row to the receptor.

    A particle below ``surface_layer_depth`` loses tracer at the rate
    ``velocity / surface_layer_depth``, HYSPLIT's dry deposition rule.
    Each row stands for the time step that ends at it, so its fraction
    includes its own step and every step between it and the release.

    Parameters
    ----------
    particles : pandas.DataFrame
        The particle table, with ``particle``, ``age`` (minutes) and
        ``zagl`` (m) columns and a row at every time step (``outdt: 0``).
    velocity : float
        The dry deposition velocity, in m/s.
    surface_layer_depth : float, default 50.0
        The depth of the layer the tracer deposits from, in meters above
        ground.

    Returns
    -------
    pandas.Series
        The surviving fraction, between 0 and 1, aligned with
        ``particles``' index.

    Raises
    ------
    ValueError
        If a column is missing or a particle's rows are more than
        :data:`MAX_STEP_MINUTES` apart.

    Notes
    -----
    The fraction is :math:`\exp(-\sum v_d \, \Delta t / H)` over the
    steps spent below :math:`H`, where a step's height is its row's
    ``zagl``. It works for any species with a constant deposition
    velocity; use it directly when you want the particles, not a
    footprint.

    Examples
    --------
    >>> import pandas as pd
    >>> from stilt.transforms import dry_deposition_survival
    >>> p = pd.DataFrame(
    ...     {
    ...         "particle": [1, 1, 1],
    ...         "age": [-1.0, -2.0, -3.0],
    ...         "zagl": [10.0, 100.0, 10.0],
    ...     }
    ... )
    >>> dry_deposition_survival(p, velocity=0.01).round(4).tolist()
    [0.9881, 0.9881, 0.9763]
    """
    missing = [c for c in ("particle", "age", "zagl") if c not in particles.columns]
    if missing:
        raise ValueError(
            f"Particle DataFrame has no {missing} column(s), required for "
            "dry deposition."
        )
    elapsed = particles["age"].abs().astype(float)
    order = elapsed.groupby(particles["particle"]).rank(method="first")
    sorted_rows = pd.DataFrame(
        {"particle": particles["particle"], "elapsed": elapsed, "order": order}
    ).sort_values(["particle", "order"])
    step = sorted_rows.groupby("particle")["elapsed"].diff()
    # The first row's step runs from the release (age 0).
    step = step.fillna(sorted_rows["elapsed"])
    if (step > MAX_STEP_MINUTES + 1e-6).any():
        raise ValueError(
            "Dry deposition needs a particle row at every one-minute time step; "
            f"these rows are up to {step.max():g} minutes apart. Run with "
            "outdt: 0."
        )
    below = particles.loc[sorted_rows.index, "zagl"].to_numpy(dtype=float) < (
        surface_layer_depth
    )
    loss = np.where(below, velocity * step.to_numpy() * 60.0 / surface_layer_depth, 0.0)
    cumulative = (
        pd.Series(loss, index=sorted_rows.index)
        .groupby(sorted_rows["particle"])
        .cumsum()
    )
    survival = pd.Series(np.exp(-cumulative.to_numpy()), index=cumulative.index)
    return survival.reindex(particles.index)


class DryDeposition(BaseModel):
    """
    Transform that weighs each particle's ``foot`` by the tracer that survives dry deposition.

    Use it for a species that deposits to the ground, such as a soluble gas
    or a fine aerosol. Each row's ``foot`` is multiplied by
    :func:`dry_deposition_survival`, the fraction of tracer emitted there
    that reaches the receptor.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["dry_deposition"] = "dry_deposition"
    velocity: float = Field(gt=0, description="Dry deposition velocity, in m/s.")
    surface_layer_depth: float = Field(
        50.0,
        gt=0,
        description=(
            "Depth of the layer the tracer deposits from, in meters above "
            "ground. HYSPLIT uses its lowest internal level at or below 75 m, "
            "50 m on HRRR."
        ),
    )

    def apply(
        self,
        particles: pd.DataFrame,
        receptor: Receptor | None = None,
        directory: str | Path | None = None,
    ) -> pd.DataFrame:
        """Return the particles with ``foot`` scaled by the surviving fraction."""
        survival = dry_deposition_survival(
            particles, self.velocity, self.surface_layer_depth
        )
        out = particles.copy()
        out["foot"] = out["foot"] * survival.to_numpy()
        return out
