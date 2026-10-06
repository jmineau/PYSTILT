"""A footprint array made from given values, for tests that need one without particles."""

from __future__ import annotations

import xarray as xr

from stilt.footprint.config import FootprintConfig
from stilt.footprint.io import _describe
from stilt.receptors import Receptor


def as_footprint(
    data: xr.DataArray,
    receptor: Receptor,
    config: FootprintConfig,
    name: str = "",
    geometry_hash: str | None = None,
) -> xr.DataArray:
    """
    Return *data* as the footprint of *receptor*, with the attributes ``calc_footprint`` gives one.

    PYSTILT has no public footprint constructor: a footprint is made from
    particles. Tests that need one with chosen values build it here, the one
    test module that reaches the private helper.
    """
    return _describe(data, receptor, config, name, geometry_hash)
