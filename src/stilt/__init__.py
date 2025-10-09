"""PYSTILT

A python implementation of the STILT model.
"""

__version__ = "2025.10.0"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .config import ModelConfig
from .meteorology import Meteorology
from .model import Model, stilt_init
from .receptors import Receptor
from .simulation import Footprint, Simulation, Trajectory

__all__ = [
    "ModelConfig",
    "Meteorology",
    "Model",
    "stilt_init",
    "Receptor",
    "Simulation",
    "Trajectory",
    "Footprint",
]
