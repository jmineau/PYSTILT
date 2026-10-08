"""
What the transforms share: a particle's value of a column at release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


def release_coordinate(particles: pd.DataFrame, coordinate: str) -> pd.Series:
    """
    Return each particle's value of ``coordinate`` at release, indexed by ``particle``.

    Some columns, such as ``release_height``, are constant along a trajectory. Others,
    such as ``pres``, change at every time step. The release value is taken
    from the row nearest the receptor time (smallest ``|time|``), or from the
    first row when there is no numeric ``age`` column.
    """
    p = particles
    if "age" in p.columns and pd.api.types.is_numeric_dtype(p["age"]):
        ordered = p.assign(_age=p["age"].abs()).sort_values("_age", kind="stable")
    else:
        ordered = p
    first = ordered.drop_duplicates(subset="particle")
    return pd.Series(
        first[coordinate].to_numpy(dtype=float), index=first["particle"].to_numpy()
    )
