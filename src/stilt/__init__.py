"""
PYSTILT, a Python implementation of the STILT transport model.

:func:`run_trajectories` and :func:`calc_footprint` make the particles and
the footprint of one release. :class:`Project` runs many releases from a
directory of releases and settings, and loads their results. The particle
runs are `runtraj <https://jmineau.github.io/runtraj/>`_'s: the names here
that make and read them are its own.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from runtraj import (
    Bounds,
    ColumnRelease,
    ExecutionConfig,
    Grid,
    MultiPointRelease,
    PointRelease,
    ProjectConfig,
    Release,
    RuntrajError,
    read_particles,
    read_releases,
    run_trajectories,
    task_share,
)

from .background import Background, background
from .footprint import FootprintConfig, Mesh, Zones, calc_footprint, read_footprint
from .project import Project
from .simulation import Simulation

# Registers particles.stilt, with STILT's methods (stilt.particles.ParticlesAccessor).
from . import particles  # noqa: F401  # isort: skip
from .transforms import averaging_kernel_table
from .transport_error import TransportError, transport_error

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
    # Releases
    "Release",
    "PointRelease",
    "ColumnRelease",
    "MultiPointRelease",
    "read_releases",
    # One simulation of a project
    "Simulation",
    # Grids, and the cells footprints are summed onto
    "Grid",
    "Bounds",
    "Mesh",
    "Zones",
    # One release, without a project
    "run_trajectories",
    "task_share",
    "calc_footprint",
    # What the particles give beyond the footprint
    "background",
    "Background",
    "transport_error",
    "TransportError",
    # Result files
    "read_particles",
    "read_footprint",
    # Satellite columns
    "averaging_kernel_table",
    # Exceptions (runtraj.exceptions has the rest)
    "RuntrajError",
]
