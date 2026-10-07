"""
The particle table: reading and writing particle files, and what is computed from particles.

Besides the table itself (its schema, files, release heights, and the
near-field plume dilution correction, in :mod:`~stilt.particles.table`),
this package holds what a simulation's particles give beyond the
footprint: the background mole fraction at the receptor
(:func:`background`, ``sim.background``) and the transport error of the
modeled enhancement (:func:`transport_error`, ``sim.transport_error``).
Importing it registers the ``.stilt`` accessor on DataFrames
(:class:`ParticlesAccessor`).
"""

from stilt.particles.accessor import ParticlesAccessor
from stilt.particles.background import Background, background
from stilt.particles.table import (
    FOOTPRINT_COLUMNS,
    HNF_PLUME_COLUMNS,
    PARTICLE_SCHEMA,
    ParticleMetadata,
    add_release_heights,
    check_particles,
    correct_near_field,
    particles_from_table,
    particles_metadata,
    read_particles,
    write_particles,
)
from stilt.particles.transport_error import (
    DEFAULT_LENGTH_SCALE,
    TransportError,
    transport_error,
)

__all__ = [
    "DEFAULT_LENGTH_SCALE",
    "FOOTPRINT_COLUMNS",
    "HNF_PLUME_COLUMNS",
    "PARTICLE_SCHEMA",
    "Background",
    "ParticleMetadata",
    "ParticlesAccessor",
    "TransportError",
    "add_release_heights",
    "background",
    "check_particles",
    "correct_near_field",
    "particles_from_table",
    "particles_metadata",
    "read_particles",
    "transport_error",
    "write_particles",
]
