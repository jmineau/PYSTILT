"""Apply observation operators to particle trajectories."""

from __future__ import annotations

import pandas as pd

from .operators import VerticalOperator
from .weighting import VerticalOperatorWeighting, WeightingContext


def apply_vertical_operator(
    particles: pd.DataFrame,
    operator: VerticalOperator,
    *,
    coordinate: str = "xhgt",
) -> pd.DataFrame:
    """
    Weight particle footprint values by a vertical observation operator.

    This ports the core X-STILT weighting step (``wgt.trajec.foot``) into a
    generic, sensor-independent function.  The profile stored in
    ``VerticalOperator`` is mapped to each particle's release coordinate
    (linear interpolation for ``ak``; nearest level for the layer-weight
    modes listed below) and multiplied into the ``foot`` column before
    spatial aggregation by :func:`~stilt.footprint.Footprint.calculate`.

    Parameters
    ----------
    particles:
        Particle DataFrame from ``Trajectories.from_particles()``.  Must
        contain columns ``indx``, ``foot``, and the column named by
        *coordinate*.
    operator:
        Vertical operator to apply.  ``operator.levels`` and
        ``operator.values`` define the profile; ``operator.mode`` controls
        the scaling behaviour.
    coordinate:
        Name of the column in *particles* to use as the interpolation
        coordinate.  Defaults to ``"xhgt"`` (release height AGL in metres,
        assigned by ``Trajectories.from_particles`` for column and multipoint
        receptors).  Pass ``"pres"`` to interpolate on pressure (hPa);
        levels are internally sorted to ascending order before interpolation.

    Returns
    -------
    pd.DataFrame
        Copy of *particles* with ``foot`` replaced by the operator-weighted
        footprint.  The unweighted original is preserved in
        ``foot_before_weight``.

    Notes
    -----
    Re-weighting is idempotent: if ``foot_before_weight`` already exists in
    *particles* (from a prior call), the original unweighted ``foot`` is
    restored before applying the new operator.

    For modes that include pressure weighting (``pwf``, ``ak_pwf``,
    ``integration``, ``tccon``), ``operator.values`` are treated as *layer
    weights* (a PWF sums to 1 over the column).  Each particle takes the value
    of its nearest level, shared among the particles at that level:
    ``weight = value_level × n_particles / n_particles_at_level``.  This
    compensates for the per-particle normalization applied in
    ``Footprint.calculate``, so the column-integrated weight equals the sum
    of the level values that received particles and the weighted footprint
    keeps the same magnitude regardless of ``numpar``.  When the profile is
    supplied once per particle (the X-STILT convention) every level count is
    one and this reduces to ``value × n_particles``.

    Operator modes
    --------------
    ``"none"``
        Return *particles* unchanged — no copy is made.
    ``"uniform"``
        Return a copy of *particles* unchanged (equal contribution from every
        particle, which is the default STILT behaviour).
    ``"ak"``
        Weight by the normalized averaging kernel only.
        ``weight = AK_norm``
    ``"pwf"``
        Weight by the pressure weighting function only.
        ``weight = PWF × n_particles / n_at_level``
    ``"ak_pwf"``
        Weight by AK × PWF (the standard X-STILT OCO weighting).
        ``weight = AK_norm × PWF × n_particles / n_at_level``
    ``"integration"``
        Pure vertical integration (equivalent to PWF only).
        ``weight = values × n_particles / n_at_level``
    ``"tccon"``
        TCCON-specific AK × PWF × wet-air scaling factor.
        ``weight = AK_norm × PWF × sf_wet × n_particles / n_at_level``
        The combined profile (already including sf_wet) should be stored in
        ``operator.values`` by the TCCON sensor adapter.
    """
    weighting = VerticalOperatorWeighting(operator)
    return weighting.apply(
        particles,
        context=WeightingContext(coordinate=coordinate, operator=operator),
    )
