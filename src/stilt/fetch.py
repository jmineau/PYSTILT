"""
The fetch before a run: the met files a project's unfinished simulations need, got once.

:func:`fetch_project` asks each variant's transport model for the files its
runs would ask for, so the runs that follow find them and download nothing.
``Project.fetch`` and ``stilt fetch`` call it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from stilt.meteorology import run_window
from stilt.transport import get_model

if TYPE_CHECKING:
    from stilt.project import Project

#: The columns of the table :func:`fetch_project` returns.
FETCH_COLUMNS = ("met", "file", "message")


def fetch_project(project: Project, unrun: pd.DataFrame) -> pd.DataFrame:
    """
    Return the met files the simulations in *unrun* need, after getting them.

    *unrun* is the simulations whose transport model has yet to run (rows of
    ``project.simulations`` with ``receptor`` and ``variant`` columns). For
    each met they use, the model gets the files for the time each run
    covers, as a run would (``fetch_met`` of the transport model, see
    :class:`stilt.transport.TransportModel`). A model without ``fetch_met``
    has no files to get and adds no row.

    Returns
    -------
    pandas.DataFrame
        ``met`` (its name under ``mets:``), ``file``, and ``message`` columns:
        a row for each file, with ``message`` empty, then a row for each
        window whose files could not be had, with ``file`` empty.
    """
    start = project._rows.drop_duplicates("receptor").set_index("receptor")["time"]
    needed: dict[tuple[str, str], tuple[Any, set[tuple[Any, Any]]]] = {}
    for name, group in unrun.groupby("variant", sort=False):
        variant = project.variants[str(name)]
        times = start.loc[group["receptor"].astype(str).unique()].drop_duplicates()
        windows = {run_window(t, variant.transport.n_hours) for t in times}
        key = (variant.met, variant.model.name)
        needed.setdefault(key, (variant.met_config, set()))[1].update(windows)
    tables = []
    for (met, model), (config, windows) in needed.items():
        fetch_met = getattr(get_model(model), "fetch_met", None)
        if fetch_met is not None:
            tables.append(fetch_met(config, windows).assign(met=met))
    if not tables:
        return pd.DataFrame(columns=list(FETCH_COLUMNS)).astype("string")
    table = pd.concat(tables, ignore_index=True)
    return table.loc[:, list(FETCH_COLUMNS)].astype("string")


__all__ = ["FETCH_COLUMNS", "fetch_project"]
