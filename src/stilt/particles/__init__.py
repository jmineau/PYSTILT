"""
The particle table: its schema, particle files, and release heights.

The functions are in :mod:`~stilt.particles.table`. What STILT computes from
a particle table is elsewhere: ``foot`` and the near-field correction
(:mod:`stilt.footprint.foot`), the background mole fraction at the release
(:func:`stilt.background`) and the transport error of the modeled
enhancement (:func:`stilt.transport_error`).
"""

from stilt.particles.accessor import ParticlesAccessor
from stilt.particles.table import (
    FOOTPRINT_COLUMNS,
    PARTICLE_SCHEMA,
    ParticleMetadata,
    add_release_heights,
    check_particles,
    particles_from_table,
    particles_metadata,
    read_particles,
    write_particles,
)

__all__ = [
    "FOOTPRINT_COLUMNS",
    "PARTICLE_SCHEMA",
    "ParticleMetadata",
    "ParticlesAccessor",
    "add_release_heights",
    "check_particles",
    "particles_from_table",
    "particles_metadata",
    "read_particles",
    "write_particles",
]
