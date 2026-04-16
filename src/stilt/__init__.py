"""PYSTILT

A Python implementation of the STILT Lagrangian atmospheric transport model.
"""

from importlib.metadata import version as _version

__version__ = _version("pystilt")
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .artifacts import ArtifactStore, FsspecArtifactStore
from .config import (
    Bounds,
    FootprintConfig,
    Grid,
    MetConfig,
    ModelConfig,
)
from .footprint import Footprint
from .meteorology import MetArchive, MetStream
from .model import Model
from .receptor import Receptor, read_receptors
from .runtime import RuntimeSettings
from .service import BatchStatus, QueueStatus, Service
from .simulation import SimID, Simulation
from .trajectory import Trajectories

__all__ = [
    # Core
    "Model",
    # Configuration
    "ModelConfig",
    "FootprintConfig",
    "Grid",
    "Bounds",
    "MetConfig",
    # Durable artifact access
    "ArtifactStore",
    "FsspecArtifactStore",
    # Data objects (returned by Model methods)
    "Simulation",
    "SimID",
    "Footprint",
    "Trajectories",
    # Receptors
    "Receptor",
    "read_receptors",
    # Runtime settings
    "RuntimeSettings",
    # Service facade
    "Service",
    "QueueStatus",
    "BatchStatus",
    # Meteorology
    "MetStream",
    "MetArchive",
]
