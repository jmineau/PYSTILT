The Particle Table
==================

.. currentmodule:: stilt.particles

A transport model run gives one table: a row per particle per output step.
Everything after the run reads it: the footprint, the near-field
correction, the transforms, the plots, and the particle files in the output
directory. ``sim.particles`` and :func:`stilt.run_trajectories` return it as
a pandas DataFrame. :doc:`layout` says how a particle file stores it.

Every table has five columns, :data:`PARTICLE_SCHEMA`:

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Column
     - Units
     - Meaning
   * - ``particle``
     - none
     - Particle number, from 1 to ``numpar``. A particle keeps its number
       for the whole run. HYSPLIT and STILT-R call it ``indx``. Stored as
       int32 and read back as float64.
   * - ``age``
     - minutes
     - Time since release. Negative for a backward run, positive for a
       forward one. HYSPLIT and STILT-R call it ``time``. Stored as whole
       minutes, int32, and read back as float64.
   * - ``lon``
     - degrees east
     - Longitude of the particle (HYSPLIT's ``long``).
   * - ``lat``
     - degrees north
     - Latitude of the particle (HYSPLIT's ``lati``).
   * - ``zagl``
     - m
     - Height of the particle above ground.

A footprint also needs ``foot``:

``foot``
   The particle's sensitivity to surface fluxes accumulated over the output
   step, in ppm per (µmol m⁻² s⁻¹). It is zero while the particle is above
   ``veght`` of the mixed layer (or ``veght`` meters when that is more than
   1). It is the time the particle spent in that surface layer, divided by
   the depth the flux is mixed through times the mean air density below it
   (Lin et al., 2003).
   :func:`stilt.calc_footprint` adds each particle's ``foot`` into its grid
   cell and divides by the number of particles.

Which models write ``foot``:

- **HYSPLIT** computes it in STILT mode, as STILT-R's runs do, when
  ``varsiwant`` has ``foot`` (the default). PYSTILT keeps it as written.
- **A transport model that writes only positions**, or any model whose
  particles have no ``foot``, gets it from PYSTILT after the run, from the
  met files the run read: :func:`add_foot` computes it from ``zagl``,
  ``mlht``, the air below the depth (the drop in pressure from the ground
  to the depth) and the output step. The run's log says so. A particle
  counts for the whole step when it ends the step below the layer, where
  HYSPLIT counts the minutes it spent there. Without the model's own
  ``mlht``, the met's boundary-layer height is used, not the one HYSPLIT
  finds from the profile. :doc:`/development/validation` says how far a
  footprint moves.

The release row
---------------

A model should write each particle's release as a row at ``age = 0``, with
``foot = 0``: a positive ``foot`` there would add a layer at hour 0 to the
footprint. That row is where the particle started, so it says which release point of a
multipoint receptor the particle left from, and which slab of a column
receptor (:func:`add_release_heights`). The HYSPLIT build bundled with
PYSTILT writes no such row: its first row is one time step after release.
For it, PYSTILT matches each particle to a release point from its first
position, and takes a column's particles to be released bottom to top in
``particle`` order. A HYSPLIT build patched to write the release rows
(``exe_dir``) makes the matching exact.

Columns the model adds
----------------------

HYSPLIT writes the variables in ``varsiwant`` under HYSPLIT's lowercase
four-letter names, except ``indx``, ``time``, ``long``, and ``lati``, which
become the columns above. The allowed names are listed under ``varsiwant`` in
:doc:`configuration`; the config refuses any other name when it is read.
Some that PYSTILT reads:

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
     - Terrain height above sea level. It is not in the default
       ``varsiwant``. Add it for a receptor given above sea level
       (``altitude_ref="msl"``): pressure weighting raises without it, and
       the near-field correction is skipped without it.
   * - ``zcfx``
     - m
     - Vertical displacement by convective flux along the trajectory,
       summed over the steps (the Grell scheme, ``capemin: -2``). HYSPLIT
       reads this name only: ``zfx1`` stops it, and the config refuses it.

The near-field correction reads ``dens``, ``samt``, ``sigw``, ``tlgr``,
``foot``, and ``mlht`` (:data:`HNF_PLUME_COLUMNS`). The plume it models
grows from the release height above ground, so for a receptor above sea
level it also reads ``zsfc``. Without it the run's log says the correction
was skipped.

``mlht``, ``dens``, ``pres``, and ``zsfc`` can also be sampled from the met
files at each particle's position, for a model that writes only positions:
:func:`stilt.sampling.sample_met`. They differ a little from HYSPLIT's own
columns; :doc:`/development/validation` says by how much.

Columns PYSTILT adds
--------------------

``release_height``
   Release height of each particle, in meters in the receptor's own
   vertical reference, for column and multipoint receptors
   (:func:`add_release_heights`). For a column it is the center of the
   particle's slab, the part of the column the particle stands for. STILT-R and
   X-STILT call it ``xhgt``.

``foot``
   When the model wrote none (:func:`add_foot`), as above.

``foot_no_hnf_dilution``
   ``foot`` before the near-field correction (:func:`correct_near_field`),
   when ``hnf_plume`` is on.

``time``
   The time of each row, in UTC: the receptor time plus ``age``. It is
   not stored; reading a particle file adds it, and so does
   :func:`stilt.run_trajectories`. STILT-R's ``time`` is ``age`` here.

Checking a table
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   PARTICLE_SCHEMA
   FOOTPRINT_COLUMNS
   HNF_PLUME_COLUMNS
   check_particles

After a run
-----------

PYSTILT adds the release heights to any model's particles, ``foot`` to
particles that have none, and applies the near-field correction when
``hnf_plume`` is set. A model whose particles lack the columns the
correction reads gets none, and a warning says so. The correction reads
``samt``, ``sigw`` and ``tlgr``, which only HYSPLIT writes, so particles
that got their ``foot`` from PYSTILT are not corrected.

.. autosummary::
   :toctree: _api
   :nosignatures:

   add_release_heights
   add_foot
   correct_near_field
