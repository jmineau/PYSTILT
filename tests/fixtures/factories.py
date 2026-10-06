"""
Small objects many tests need, built in one place.

A test that needs a resolved variant or a receptor calls these, so a change
to :class:`stilt.config.Variant` or the receptor types is one edit here, not
one in every test module.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from stilt.config import Variant
from stilt.footprint.config import FootprintConfig
from stilt.meteorology import MetConfig
from stilt.receptors import PointReceptor
from stilt.transport import ModelInfo, TransportConfig
from stilt.transport.hysplit import HysplitConfig

#: A met that names files nobody reads: enough for a variant's settings.
MET = MetConfig(directory="/data/hrrr", file_format="%Y%m%d_%H", file_tres="6h")


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
        model=ModelInfo(version="v5.1.0"),
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
