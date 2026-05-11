"""PYSTILT public package surface."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .config import (
    Bounds,
    FootprintConfig,
    Grid,
    MetConfig,
    ModelConfig,
    RuntimeSettings,
)
from .footprint import Footprint
from .meteorology import MetStream
from .model import Model
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
from .transforms import ParticleTransform

try:
    __version__ = _version("pystilt")
except PackageNotFoundError:
    __version__ = "0+unknown"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"

__all__ = [
    # Core
    "Model",
    # Configuration
    "ModelConfig",
    "FootprintConfig",
    "Grid",
    "Bounds",
    "MetConfig",
    "RuntimeSettings",
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
    # Meteorology
    "MetStream",
    # Transforms
    "ParticleTransform",
]
