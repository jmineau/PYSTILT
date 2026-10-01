"""
PYSTILT, a Python implementation of the STILT transport model.

Start with :class:`Project`, a directory of receptors and settings that
runs them through HYSPLIT and loads their trajectories and footprints.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .config import (
    Bounds,
    FootprintConfig,
    Grid,
    MetConfig,
    ProjectConfig,
    RuntimeSettings,
    VariantConfig,
)
from .exceptions import StiltError
from .footprint import Footprint
from .geometry import Geometry, Mesh, SpatialTarget, Zones
from .meteorology import Met
from .output import Output
from .project import Project
from .receptors import (
    ColumnReceptor,
    MultiPointReceptor,
    PointReceptor,
    Receptor,
    read_receptors,
)
from .simulation import SimID, Simulation
from .trajectory import Trajectories
from .transforms import ParticleTransform, TransformContext

try:
    __version__ = _version("pystilt")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    # Core
    "Project",
    "Output",
    # Configuration
    "ProjectConfig",
    "FootprintConfig",
    "Grid",
    "Bounds",
    "MetConfig",
    "RuntimeSettings",
    "VariantConfig",
    # Data objects (returned by Project methods)
    "Simulation",
    "SimID",
    "Footprint",
    "Trajectories",
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
    "SpatialTarget",
    # Meteorology
    "Met",
    # Exceptions (all of them live in stilt.exceptions)
    "StiltError",
    # Transforms (the interface; built-ins live in stilt.transforms)
    "ParticleTransform",
    "TransformContext",
]
