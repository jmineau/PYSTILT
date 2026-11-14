"""
FLEXPART 11 as a transport model.

Only the reader is here so far: :func:`read_partoutput` turns FLEXPART 11's
particle output into a particle table. PYSTILT does not run FLEXPART yet.
FLEXPART is GPL-3, so PYSTILT will run an executable the user built and never
ship FLEXPART's code or a binary.
"""

from stilt.transport.flexpart.output import PARTICLE_COLUMNS, read_partoutput

__all__ = ["PARTICLE_COLUMNS", "read_partoutput"]
