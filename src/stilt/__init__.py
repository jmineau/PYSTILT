"""
PYSTILT, a Python implementation of the STILT transport model.

Start with :class:`Model`, which runs receptors through HYSPLIT and loads
their trajectories and footprints.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .config import (
    Bounds,
    FootprintConfig,
    Grid,
    MetConfig,
    ModelConfig,
    RuntimeSettings,
    VariantConfig,
)
from .footprint import Footprint
from .geometry import Geometry, Mesh, SpatialTarget, Zones
from .meteorology import Met
from .model import Model
from .output import Output
from .receptors import (
    ColumnReceptor,
    LocationID,
    MultiPointReceptor,
    PointReceptor,
    Receptor,
    ReceptorID,
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
    "Model",
    "Output",
    # Configuration
    "ModelConfig",
    "FootprintConfig",
    "Grid",
    "Bounds",
    "MetConfig",
    "RuntimeSettings",
    "VariantConfig",
    # Data objects (returned by Model methods)
    "Simulation",
    "SimID",
    "Footprint",
    "Trajectories",
    # Receptors
    "Receptor",
    "ColumnReceptor",
    "LocationID",
    "MultiPointReceptor",
    "PointReceptor",
    "ReceptorID",
    "read_receptors",
    # Spatial geometries (state geometry for aggregation)
    "Geometry",
    "Mesh",
    "Zones",
    "SpatialTarget",
    # Meteorology
    "Met",
    # Transforms (the interface; built-ins live in stilt.transforms)
    "ParticleTransform",
    "TransformContext",
]
