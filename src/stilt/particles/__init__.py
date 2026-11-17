"""
The particle table: its schema, particle files, release heights, ``foot``, and the near-field correction.

The functions are in :mod:`~stilt.particles.table`. What a simulation's
particles give beyond the footprint is elsewhere: the background mole
fraction at the receptor (:func:`stilt.background`) and the transport error
of the modeled enhancement (:func:`stilt.transport_error`). Importing this
package registers the ``.stilt`` accessor on DataFrames
(:class:`ParticlesAccessor`).
"""

from stilt.particles.accessor import ParticlesAccessor
from stilt.particles.table import (
    FOOTPRINT_COLUMNS,
    HNF_PLUME_COLUMNS,
    PARTICLE_SCHEMA,
    ParticleMetadata,
    add_foot,
    add_release_heights,
    check_particles,
    correct_near_field,
    particles_from_table,
    particles_metadata,
    read_particles,
    write_particles,
)

__all__ = [
    "FOOTPRINT_COLUMNS",
    "HNF_PLUME_COLUMNS",
    "PARTICLE_SCHEMA",
    "ParticleMetadata",
    "ParticlesAccessor",
    "add_foot",
    "add_release_heights",
    "check_particles",
    "correct_near_field",
    "particles_from_table",
    "particles_metadata",
    "read_particles",
    "write_particles",
]
