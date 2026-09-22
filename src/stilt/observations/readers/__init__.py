"""
Readers for column retrieval products, one module per instrument.

Each reader opens one product file and returns a :class:`pandas.DataFrame`
with one row per sounding and a shared set of columns, so the rest of
:mod:`stilt.observations` (selection, receptors, kernel tables, slant
geometry) works the same whichever instrument the data came from. The
*Reading Retrieval Products* guide lists the columns. Vertical arrays run from the surface
upward, azimuths are degrees clockwise from north toward the instrument or
the sun, pressures are hPa, and altitudes are metres above mean sea level.

A reader keeps its product's conventions in one place: which variable holds
the retrieval, its fill values and quality flag, how the pressure grid is
rebuilt, which way the layers run. A product not covered here is one more
module of the same shape; see that guide.
"""

from .oco import read_oco2
from .tccon import read_tccon
from .tropomi import read_tropomi_ch4

__all__ = ["read_oco2", "read_tccon", "read_tropomi_ch4"]
