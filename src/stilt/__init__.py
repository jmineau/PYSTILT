"""
PYSTILT, a Python implementation of the STILT transport model.

Start with :class:`Project`, a directory of receptors and settings that
runs them through HYSPLIT and loads their particles and footprints.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .config import ProjectConfig
from .exceptions import StiltError
from .execution import ExecutionConfig
from .footprint import FootprintConfig, Geometry, Mesh, Zones, read_footprint
from .meteorology import Met, MetConfig
from .output import Output
from .particles import particles_metadata, read_particles, write_particles
from .project import Project, Simulations
from .receptors import (
    ColumnReceptor,
    MultiPointReceptor,
    PointReceptor,
    Receptor,
    read_receptors,
)
from .simulation import SimID, Simulation
from .spatial import Bounds, Grid
from .variants import Variant

try:
    __version__ = _version("pystilt")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    # Core
    "Project",
    "Simulations",
    "Output",
    # Configuration
    "ProjectConfig",
    "ExecutionConfig",
    "FootprintConfig",
    "Grid",
    "Bounds",
    "MetConfig",
    # Simulations and their results
    "Variant",
    "Simulation",
    "SimID",
    "read_particles",
    "particles_metadata",
    "write_particles",
    "read_footprint",
    # Receptors
    "Receptor",
    "ColumnReceptor",
    "MultiPointReceptor",
    "PointReceptor",
    "read_receptors",
    # Spatial geometries (state geometry for aggregation)
    "Geometry",
    "Mesh",
    "Zones",
    # Meteorology
    "Met",
    # Exceptions (all of them live in stilt.exceptions)
    "StiltError",
    # Transforms (the interface; built-ins live in stilt.transforms)
]
