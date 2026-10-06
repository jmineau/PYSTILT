The Particle Table
==================

.. currentmodule:: stilt.particles

A transport model run gives one table: a row per particle per output step.
Everything after the run reads it: the footprint, the near-field
correction, the transforms, the plots, and the particle files in the output
directory. ``sim.particles`` and :func:`stilt.run_trajectories` return it as
a pandas DataFrame.

Every table has five columns, :data:`PARTICLE_SCHEMA`:

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Column
     - Units
     - Meaning
   * - ``indx``
     - none
     - Particle number, from 1 to ``numpar``. A particle keeps its number
       for the whole run.
   * - ``time``
     - minutes
     - Time since release. Negative for a backward run, positive for a
       forward one. Stored as whole minutes.
   * - ``long``
     - degrees east
     - Longitude of the particle.
   * - ``lati``
     - degrees north
     - Latitude of the particle.
   * - ``zagl``
     - m
     - Height of the particle above ground.

A footprint also needs ``foot``:

``foot``
   The particle's sensitivity to surface fluxes accumulated over the output
   step, in ppm per (µmol m⁻² s⁻¹). It is zero while the particle is above
   ``veght`` of the mixed layer (or ``veght`` metres when that is more than
   1). HYSPLIT computes it as STILT-R does: the time the particle spent
   in that surface layer, divided by the air density and the depth the
   flux is mixed through. :func:`stilt.calc_footprint` adds each
   particle's ``foot`` into its grid cell and divides by the number of
   particles.

The release row
---------------

A model should write each particle's release as a row at ``time = 0``. That
row is where the particle started, so it says which release point of a
multipoint receptor the particle left from, and which slab of a column
receptor (:func:`add_release_heights`). The HYSPLIT build bundled with
PYSTILT writes no such row: its first row is one time step after release.
For it, PYSTILT matches each particle to a release point from its first
position, and takes a column's particles to be released bottom to top in
``indx`` order. A HYSPLIT build patched to write the release rows
(``exe_dir``) makes the matching exact.

Columns the model adds
----------------------

HYSPLIT writes the variables in ``varsiwant``, under HYSPLIT's names. Some
that PYSTILT reads:

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Column
     - Units
     - Meaning
   * - ``mlht``
     - m
     - Mixed-layer height above ground.
   * - ``dens``
     - kg m⁻³
     - Air density.
   * - ``samt``
     - minutes
     - Time the particle spent in the surface layer during the step.
   * - ``sigw``
     - m s⁻¹
     - Standard deviation of the vertical wind.
   * - ``tlgr``
     - s
     - Lagrangian time scale.
   * - ``pres``
     - hPa
     - Pressure, for pressure weighting.
   * - ``zsfc``
     - m
     - Terrain height above sea level, for receptors given above sea level.

The near-field correction reads ``dens``, ``samt``, ``sigw``, ``tlgr``,
``foot``, and ``mlht`` (:data:`HNF_PLUME_COLUMNS`).

Columns PYSTILT adds
--------------------

``xhgt``
   Release height of each particle, in metres in the receptor's own
   vertical reference, for column and multipoint receptors
   (:func:`add_release_heights`). For a column it is the centre of the
   particle's slab, the part of the column the particle stands for.

``foot_no_hnf_dilution``
   ``foot`` before the near-field correction (:func:`correct_near_field`),
   when ``hnf_plume`` is on.

``datetime``
   The time of each row, in UTC, added when the table is read from a file.

Checking a table
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   PARTICLE_SCHEMA
   FOOTPRINT_COLUMNS
   HNF_PLUME_COLUMNS
   check_particles
