"""
Readers for column retrieval products, one module per instrument.

Each reader opens one product file and returns a :class:`pandas.DataFrame`
with one row per sounding and the same set of columns, so the rest of
:mod:`stilt.observations` works the same for every instrument. The
*Reading Retrieval Products* guide lists the columns and shows how to add a
reader for another product.

In every reader, vertical arrays run from the surface upward, azimuths are
degrees clockwise from north toward the instrument or the sun, pressures
are in hPa, and altitudes are in meters above sea level.
"""

from .ggg import read_ggg_netcdf, read_ggg_oof
from .oco import read_oco2
from .tccon import read_tccon
from .tropomi import read_tropomi_ch4

__all__ = [
    "read_ggg_netcdf",
    "read_ggg_oof",
    "read_oco2",
    "read_tccon",
    "read_tropomi_ch4",
]
