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

    This ports the core X-STILT weighting step (``wgt.trajec.foot`` and
    ``get.wgt.*.func``) into a generic, sensor-independent function.  Each
    particle's ``foot`` is multiplied by a weight before spatial aggregation by
    :func:`~stilt.footprint.Footprint.calculate`.

    Parameters
    ----------
    particles:
        Particle DataFrame from ``Trajectories.from_particles()``.  Must
        contain ``indx`` and ``foot``; averaging-kernel modes also need the
        column named by *coordinate*, and pressure-weighting modes need
        ``pres`` and ``zagl`` (both in the default ``varsiwant``).
    operator:
        Vertical operator to apply.  ``operator.mode`` selects the weighting;
        ``operator.levels`` / ``operator.values`` hold the averaging kernel;
        ``operator.surface_pressure`` optionally closes the column bottom.
    coordinate:
        Particle column on which the averaging kernel ``levels`` are defined.
        Defaults to ``"xhgt"`` (release height AGL in metres, assigned by
        ``Trajectories.from_particles`` for column and multipoint receptors).
        Pass ``"pres"`` for kernels on pressure levels (hPa).  The value is
        taken from each particle's release row and applied along its whole
        trajectory.

    Returns
    -------
    pd.DataFrame
        Copy of *particles* with ``foot`` replaced by the operator-weighted
        footprint.  The unweighted original is preserved in
        ``foot_before_weight``.  Pressure-weighting modes also add ``xpres``
        (release pressure, hPa) and ``pwf`` (the particle's mass fraction).

    Notes
    -----
    Re-weighting is idempotent: if ``foot_before_weight`` already exists in
    *particles* (from a prior call), the original unweighted ``foot`` is
    restored before applying the new operator.

    **Pressure weighting.**  A column instrument averages over air mass, but
    HYSPLIT releases column particles uniformly in *height*, so a plain mean
    over particles over-weights the thin upper layers.  Following X-STILT, the
    pressure weighting function is built from the particles themselves: a
    hypsometric curve is fit to their first-step heights and pressures and
    evaluated at each release height, and each particle is given the slab of
    air centred on it (edges midway between adjacent release pressures, the
    surface closing the bottom).  The weights sum to the fraction of the
    atmosphere's mass the column covers, ``(p_sfc - p_top) / p_sfc`` — air
    above the column top has no surface influence within the back-trajectory,
    so nothing is missing from the footprint.  Because ``Footprint.calculate``
    divides by the particle count, the weights are multiplied by
    ``n_particles`` so the result does not scale with ``numpar``.

    Operator modes
    --------------
    ``"none"``
        Return *particles* unchanged — no copy is made.
    ``"uniform"``
        Return a copy of *particles* unchanged (equal contribution from every
        particle, which is the default STILT behaviour).
    ``"ak"``
        Weight by the normalized averaging kernel only.
        ``weight = AK_norm(z)``
    ``"pwf"``
        Weight by the particle-derived pressure weighting function only.
        ``weight = PWF_i × n_particles``
    ``"ak_pwf"``
        Both (the standard X-STILT column weighting).
        ``weight = AK_norm(z) × PWF_i × n_particles``
        Instrument-specific factors such as TCCON's wet-air scaling are folded
        into ``values`` by the caller.
    """
    weighting = VerticalOperatorWeighting(operator)
    return weighting.apply(
        particles,
        context=WeightingContext(coordinate=coordinate, operator=operator),
    )
