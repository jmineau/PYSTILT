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
from .meteorology import MetSource
from .model import Model
from .receptor import LocationID, Receptor, ReceptorID, ReceptorKind, read_receptors
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
    "LocationID",
    "Receptor",
    "ReceptorID",
    "ReceptorKind",
    "read_receptors",
    # Meteorology
    "MetSource",
    # Transforms
    "ParticleTransform",
]
