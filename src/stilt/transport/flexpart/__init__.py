"""
FLEXPART 11 as a transport model.

``model: flexpart`` runs the executable of a FLEXPART 11 checkout you built
(:class:`FlexpartConfig`'s ``flexpart_dir``) on GFS GRIB files
(:class:`FlexpartMetConfig`). PYSTILT never ships FLEXPART, which is GPL-3.
:func:`read_partoutput` reads FLEXPART's particle output on its own, for a
run made outside PYSTILT.
"""

from stilt.transport.flexpart.config import (
    FlexpartConfig,
    FlexpartMetConfig,
    flexpart_version,
)
from stilt.transport.flexpart.model import FlexpartModel
from stilt.transport.flexpart.output import PARTICLE_COLUMNS, read_partoutput

__all__ = [
    "PARTICLE_COLUMNS",
    "FlexpartConfig",
    "FlexpartMetConfig",
    "FlexpartModel",
    "flexpart_version",
    "read_partoutput",
]
