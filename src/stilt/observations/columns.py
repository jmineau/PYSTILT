"""
The column a retrieval would report for the modelled air.

A retrieval is not equally sensitive at every level, and where it is not,
it reports its prior. To compare a modelled column with a retrieved one,
weight the modelled profile by the averaging kernel and add the prior
where the retrieval did not see (Wu et al., 2018, GMD,
doi:10.5194/gmd-11-4843-2018, as in X-STILT).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def modelled_column(
    enhancement: float,
    background: float,
    *,
    ak: ArrayLike | None = None,
    pressure_weight: ArrayLike | None = None,
    prior: ArrayLike | None = None,
) -> float:
    """
    Return the column-average mole fraction a retrieval would report for the modelled air.

    With the retrieval's averaging kernel ``A``, pressure weights ``w``, and
    prior profile ``x_prior`` on the same levels, it is::

        enhancement + background + Σ_l w_l (1 − A_l) x_prior,l

    The enhancement and the background must already be weighted by the
    averaging kernel and the pressure weights, as they are when the
    footprint's transforms include ``averaging_kernel`` and
    ``pressure_weighting`` (``sim.footprint`` and ``sim.background``). The
    last term is the prior the retrieval carries where it is not
    sensitive. Without ``ak``, ``pressure_weight``, and ``prior`` the result
    is ``enhancement + background``.

    Parameters
    ----------
    enhancement : float
        The modelled enhancement of the column, weighted by the kernel and
        the pressure weights, in the units of ``prior``. A flux in
        µmol m⁻² s⁻¹ gives ppm; multiply by 1000 for ppb.
    background : float
        The weighted background (``sim.background(field).value``). Over a
        receptor that stops below the top of the atmosphere it covers the
        receptor's levels only; add the weighted field above the top first.
    ak : array-like, optional
        The column averaging kernel on the retrieval's levels (the
        soundings' ``ak``).
    pressure_weight : array-like, optional
        Each level's share of the column's dry air, summing to 1. For a
        layer product with ``pressure_levels`` ``p`` (surface first), that
        is ``-np.diff(p) / (p[0] - p[-1])``.
    prior : array-like, optional
        The retrieval's prior profile on the same levels (the soundings'
        ``apriori``).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If only some of ``ak``, ``pressure_weight``, and ``prior`` are
        given, or they differ in length.

    Examples
    --------
    >>> bg = sim.background(cams).value
    >>> dxch4 = 1000 * float(sim.footprint.stilt.enhancement(flux).sum())  # ppb
    >>> p = row.pressure_levels
    >>> modelled_column(
    ...     dxch4,
    ...     bg,
    ...     ak=row.ak,
    ...     prior=row.apriori,
    ...     pressure_weight=-np.diff(p) / (p[0] - p[-1]),
    ... )
    """
    given = [ak is not None, pressure_weight is not None, prior is not None]
    if any(given) and not all(given):
        raise ValueError(
            "modelled_column needs ak, pressure_weight, and prior together, or none."
        )
    total = float(enhancement) + float(background)
    if not any(given):
        return total
    a = np.asarray(ak, dtype=float).ravel()
    w = np.asarray(pressure_weight, dtype=float).ravel()
    x = np.asarray(prior, dtype=float).ravel()
    if not a.size == w.size == x.size:
        raise ValueError(
            f"ak, pressure_weight, and prior differ in length: {a.size}, {w.size}, {x.size}."
        )
    return total + float(np.sum(w * (1.0 - a) * x))


__all__ = ["modelled_column"]
