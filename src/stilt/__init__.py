"""
PYSTILT, a Python implementation of the STILT transport model.

:func:`run_trajectories` and :func:`calc_footprint` make the particles and
the footprint of one receptor. :class:`Project` runs many receptors from a
directory of receptors and settings, and loads their results.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .config import ProjectConfig
from .exceptions import StiltError
from .execution import ExecutionConfig, task_share
from .footprint import FootprintConfig, Mesh, Zones, calc_footprint, read_footprint
from .meteorology import MetConfig
from .particles import read_particles
from .project import Project
from .receptors import (
    ColumnReceptor,
    MultiPointReceptor,
    PointReceptor,
    Receptor,
    read_receptors,
)
from .simulation import Simulation
from .spatial import Bounds, Grid
from .transforms import averaging_kernel_table
from .transport import run_trajectories

try:
    __version__ = _version("pystilt")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    # A project and its settings
    "Project",
    "ProjectConfig",
    "ExecutionConfig",
    "FootprintConfig",
    "MetConfig",
    # Receptors
    "Receptor",
    "PointReceptor",
    "ColumnReceptor",
    "MultiPointReceptor",
    "read_receptors",
    # One simulation of a project
    "Simulation",
    # Grids, and the cells footprints are summed onto
    "Grid",
    "Bounds",
    "Mesh",
    "Zones",
    # One receptor, without a project
    "run_trajectories",
    "task_share",
    "calc_footprint",
    # Result files
    "read_particles",
    "read_footprint",
    # Satellite columns
    "averaging_kernel_table",
    # Exceptions (all of them live in stilt.exceptions)
    "StiltError",
]
