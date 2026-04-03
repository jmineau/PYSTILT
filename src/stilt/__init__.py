"""PYSTILT

A python implementation of the STILT model.
"""

__version__ = "2025.10.0"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .config import ModelConfig
from .meteorology import Meteorology
from .model import Model, parse_sim_id, stilt_init
from .receptors import Receptor
from .footprint import Footprint
from .simulation import Simulation
from .trajectory import Trajectory

units = {
    "latex": None  # TODO
}


__all__ = [
    "ModelConfig",
    "Meteorology",
    "Model",
    "parse_sim_id",
    "stilt_init",
    "Receptor",
    "Simulation",
    "Trajectory",
    "Footprint",
]
