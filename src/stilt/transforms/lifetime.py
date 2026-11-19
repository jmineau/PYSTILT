"""
Loss along the trajectory: the surviving fraction for any rate, and the first-order lifetime transform.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field

from stilt.transforms._common import step_minutes

if TYPE_CHECKING:
    from runtraj.releases import Release


def survival(particles: pd.DataFrame, rate: ArrayLike) -> pd.Series:
    r"""
    Return the fraction of a tracer that survives from each row to the release.

    Use it for a loss whose rate is known at every row and does not depend
    on how much tracer the particle carries: a reaction with an oxidant,
    deposition, rain-out, or a lifetime from your own chemistry model. The
    fraction emitted at a row is the fraction that reaches the release, so
    ``foot`` times it is still a footprint.

    Parameters
    ----------
    particles : pandas.DataFrame
        The particle table, with ``particle`` and ``age`` (minutes) columns.
    rate : array-like
        The loss rate at each row, in s⁻¹, aligned with ``particles``. Zero
        where there is no loss.

    Returns
    -------
    pandas.Series
        The surviving fraction, between 0 and 1, aligned with
        ``particles``' index.

    Raises
    ------
    ValueError
        If a column is missing, ``rate`` has the wrong length, or a rate is
        negative or missing.

    Notes
    -----
    The fraction at a row is :math:`\exp(-\sum k \, \Delta t)` over the
    steps from the release to that row, where each row stands for the step
    that ends at it (a particle's first step runs from the release) and :math:`k` is that row's
    rate. A rate that changes slowly between rows, such as one set by
    temperature or an oxidant field, needs no particular step; a rate that
    switches on and off within a step, such as deposition below a fixed
    height, needs short steps (:func:`dry_deposition_survival` asks for
    one-minute rows).

    Examples
    --------
    >>> import pandas as pd
    >>> from stilt.transforms import survival
    >>> p = pd.DataFrame({"particle": [1, 1, 1], "age": [-60.0, -120.0, -180.0]})
    >>> survival(p, rate=[1e-4, 1e-4, 0.0]).round(4).tolist()
    [0.6977, 0.4868, 0.4868]
    """
    k = np.asarray(rate, dtype=float).ravel()
    if k.shape != (len(particles),):
        raise ValueError(
            f"rate has {k.size} values for {len(particles)} particle rows."
        )
    if np.isnan(k).any() or (k < 0).any():
        raise ValueError("rate must be zero or positive at every row, with no NaN.")
    step = step_minutes(particles)
    loss = pd.Series(k * step.to_numpy() * 60.0, index=particles.index)
    # Add the losses up from the release, one particle at a time.
    elapsed = particles["age"].abs().to_numpy(dtype=float)
    order = np.lexsort((elapsed, particles["particle"].to_numpy()))
    cumulative = loss.iloc[order].groupby(particles["particle"].iloc[order]).cumsum()
    return pd.Series(np.exp(-cumulative.to_numpy()), index=cumulative.index).reindex(
        particles.index
    )


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
        release: Release | None = None,
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
