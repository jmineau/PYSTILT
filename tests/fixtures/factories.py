"""
Small objects many tests need, built in one place.

A test that needs a met config, a project config, a resolved variant, or a
receptor calls these, so a change to :class:`stilt.meteorology.MetConfig`,
:class:`stilt.config.Variant`, or the receptor types is one edit here, not
one in every test module.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from stilt.config import ProjectConfig, Variant
from stilt.footprint.config import FootprintConfig
from stilt.meteorology import MetConfig
from stilt.receptors import PointReceptor
from stilt.transport import ModelInfo, TransportConfig
from stilt.transport.hysplit import HysplitConfig


def make_met_config(
    directory: str | Path,
    *,
    file_format: str = "%Y%m%d_%H",
    file_tres: str = "1h",
    **fields: Any,
) -> MetConfig:
    """Return a met config of files named by hour (``%Y%m%d_%H``) under *directory*."""
    return MetConfig(
        directory=directory, file_format=file_format, file_tres=file_tres, **fields
    )


def make_met_files(directory: Path, time: Any, n_hours: int) -> list[Path]:
    """
    Write empty hourly met files that cover a run of *n_hours* from *time*, and return them.

    They are named as :func:`make_met_config` expects, one hour beyond the
    run on each side, so :class:`stilt.meteorology.Met` finds them. Nothing
    can read them: use them where the transport model does not run.
    """
    import pandas as pd

    from stilt.meteorology import run_window

    start, end = run_window(time, n_hours)
    hours = pd.date_range(
        pd.Timestamp(start).floor("h") - pd.Timedelta(hours=1),
        pd.Timestamp(end).ceil("h") + pd.Timedelta(hours=1),
        freq="h",
    )
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for hour in hours:
        path = directory / f"{hour:%Y%m%d_%H}"
        path.touch()
        files.append(path)
    return files


def make_project_config(tmp_path: Path, **settings: Any) -> ProjectConfig:
    """
    Return a project config with one met, ``hrrr``, and one variant of the defaults.

    The met is :func:`make_met_config` under ``tmp_path / "met"``. *settings*
    are other :class:`~stilt.ProjectConfig` settings, and replace the met
    and the variant when they name ``mets`` or ``variants``.
    """
    settings.setdefault("mets", {"hrrr": make_met_config(tmp_path / "met")})
    settings.setdefault("variants", {"hrrr": {}})
    return ProjectConfig(**settings)


#: A met that names files nobody reads: enough for a variant's settings.
MET = make_met_config("/data/hrrr", file_tres="6h")


def make_variant(
    name: str = "hrrr",
    *,
    transport: TransportConfig | None = None,
    met_config: MetConfig | None = None,
    footprint: FootprintConfig | None = None,
    realizations: int | None = None,
    **fields: Any,
) -> Variant:
    """
    Return a resolved variant without a project.

    *fields* are HYSPLIT parameters over ``n_hours=-24, numpar=100``, used
    when *transport* is not given. The met is :data:`MET` unless given.
    """
    if transport is None:
        transport = HysplitConfig(**{"n_hours": -24, "numpar": 100, **fields})
    elif fields:
        raise TypeError("Give HYSPLIT fields or a transport config, not both.")
    return Variant(
        name=name,
        met="hrrr",
        met_config=met_config if met_config is not None else MET,
        transport=transport,
        model=ModelInfo(name="hysplit", version="v5.1.0"),
        footprint=footprint,
        realizations=realizations,
    )


def make_receptor(
    time: dt.datetime = dt.datetime(2023, 1, 1, 12),
    longitude: float = -111.85,
    latitude: float = 40.77,
    altitude: float = 5.0,
    **attrs: Any,
) -> PointReceptor:
    """Return a point receptor at the University of Utah's WBB site; *attrs* become its extra columns."""
    return PointReceptor(
        time=time,
        longitude=longitude,
        latitude=latitude,
        altitude=altitude,
        attrs=attrs,
    )
