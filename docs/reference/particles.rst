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
   step, in ppm per (µmol m⁻² s⁻¹): the time the particle spent in the
   surface layer, divided by the depth the flux is mixed through times the
   mean air density below it (Lin et al., 2003). The surface layer is a
   fraction of the mixed layer, or that many meters when the number is more
   than 1: HYSPLIT's ``veght`` for HYSPLIT's own ``foot``, the footprint's
   ``footprint_depth`` for one PYSTILT computes. Above it ``foot`` is zero.
   :func:`stilt.calc_footprint` adds each particle's ``foot`` into its grid
   cell and divides by the number of particles.

Where ``foot`` comes from:

- **HYSPLIT** computes it in STILT mode, as STILT-R's runs do, when
  ``varsiwant`` has ``foot`` (the default), with its ``veght``. PYSTILT
  keeps it as written.
- **Particles with no** ``foot``, such as those of a model that writes only
  positions, get it from the footprint stage when it makes the footprint
  (:func:`stilt.footprint.footprint_particles`, which calls
  :func:`stilt.footprint.add_foot`), from ``zagl``, ``mlht``, the output step, and the
  pressure drop from the ground to the surface layer's top in the met files
  the run read. PYSTILT's log says so. A particle counts for the whole step
  when it ends the step inside the layer, where HYSPLIT counts the minutes
  it spent there. When the model wrote no ``mlht``, the met's
  boundary-layer height (``PBLH``) is used, unless ``pbl_height`` says
  otherwise. :doc:`/development/validation` says how far this moves a
  footprint.

A ``foot`` PYSTILT computes is stored nowhere: the particle file keeps what
the model wrote, and each footprint computes it again. It reads the met
files at the paths the particle file records (``stilt:met_files``), so
they must still be there when the footprint is made, and when
``sim.calc_footprint`` or the transport error makes one later. Keep the
met files where the run read them for as long as you may remake a
footprint from those particles. A footprint made from HYSPLIT's own
``foot`` needs no met files.

So ``sim.particles`` has the model's ``foot``, not the one the footprint
used, whenever PYSTILT computed one:

- From a model that writes only positions, ``sim.particles`` has no
  ``foot``, and ``sim.particles.stilt.enhancement(flux)`` says so.
- With ``pbl_height`` set on a HYSPLIT run,
  ``sim.particles.stilt.enhancement(flux)`` uses HYSPLIT's ``foot`` and
  differs from the enhancement of ``sim.footprint``.

For the ``foot`` the footprint used, pass the particles through
:func:`stilt.footprint.footprint_particles` with the variant's footprint
settings and the met files the run read:

.. code-block:: python

   from stilt.footprint import footprint_particles
   from stilt.particles import particles_metadata

   meta = particles_metadata(sim.particles_path)
   particles = footprint_particles(
       sim.particles,
       sim.variant.stages["footprint"],
       meta.met_files,
       z_top=sim.variant.transport.z_top,  # HYSPLIT's, for pbl_height: richardson
   )
   particles.stilt.enhancement(flux)

The mixed-layer height
~~~~~~~~~~~~~~~~~~~~~~

``foot`` depends on the mixed-layer height, the ``mlht`` column, through the
depth surface fluxes are mixed through (``footprint_depth`` of it). The
``pbl_height`` setting says where PYSTILT takes that height from when it
computes ``foot``; ``pbl_height`` and ``mlht`` name the same quantity. The
three are settings of the footprint stage, so variants that differ only in
them share one transport model run:

.. code-block:: yaml

   stages:
     footprint:
       class: stilt.footprint.FootprintStage
       grid: {xmin: -113.0, xmax: -110.5, ymin: 40.0, ymax: 42.0, xres: 0.01, yres: 0.01}
       pbl_height: richardson   # or met, or a height in meters
       pbl_height_min: 150      # the lowest height, in meters
       footprint_depth: 0.5     # the default

- **Unset** (the default), PYSTILT stays as close to the model as its
  output allows: it keeps the model's ``foot`` (HYSPLIT's, which STILT-R
  parity covers); for a model that writes no ``foot`` it uses the model's
  ``mlht``, or the met's ``PBLH`` when the model wrote none.
- **Set**, PYSTILT computes ``foot`` itself, with that height and
  ``footprint_depth``, for any model's particles, HYSPLIT's included, and
  makes the footprint from it in place of the model's ``foot`` and ``mlht``.
  PYSTILT's log says which height it used. ``met`` is the
  met's ``PBLH``; ``richardson`` finds the height from the met's profile
  with a bulk Richardson number, as HYSPLIT's default ``kmixd: 3`` does
  (:func:`stilt.sampling.calc_pbl_height`), on HYSPLIT's internal levels for
  a HYSPLIT run's own ``z_top`` (for another model, HYSPLIT's default top);
  a number is a fixed height in meters above ground.

``pbl_height_min`` keeps a height PYSTILT finds at least that many meters,
as ``kmix0`` (150 m by default) does for HYSPLIT. It does not change a
model's own ``mlht``.

The mixed layer used for ``foot`` can differ from the one the model moved
its particles in: HYSPLIT mixes them through the layer ``kmixd`` and
``kmix0`` choose. To compare two transport models with the same footprint
layer, so that only their transport differs, set both settings:
``pbl_height: richardson`` with ``pbl_height_min: 150`` matches HYSPLIT's
defaults, and ``pbl_height: met`` with the same minimum matches a HYSPLIT
run with ``kmixd: 0``. :doc:`/development/validation` gives how far each
choice moves a footprint.

The release row
---------------

A model should write each particle's release as a row at ``age = 0``, with
``foot = 0``: a positive ``foot`` there would add a layer at hour 0 to the
footprint. That row is where the particle started, so it says which release point of a
multipoint receptor the particle left from, and which slab of a column
receptor (:func:`add_release_heights`). HYSPLIT 5.1.0, STILT-R's build,
writes no such row: its first row is one time step after release.
For it, PYSTILT matches each particle to a release point from its first
position, and takes a column's particles to be released bottom to top in
``particle`` order. A HYSPLIT build patched to write the release rows
(``directory``) makes the matching exact.

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
``foot``, and ``mlht`` (:data:`stilt.footprint.HNF_PLUME_COLUMNS`). The plume it models
grows from the release height above ground, so for a receptor above sea
level it also reads ``zsfc``. Without it the run's log says the correction
was skipped.

For a model that writes only positions, :func:`stilt.sampling.sample_met`
reads ``mlht``, ``dens``, ``pres`` and ``zsfc`` from the met files at each
particle's position and time. Give it the files the run read and the top of
its domain, which a stored run records:

.. code-block:: python

   from stilt.sampling import sample_met

   meta = stilt.particles.particles_metadata(sim.particles_path)
   sampled = sample_met(
       sim.particles.drop(columns=["mlht", "dens"]),
       meta.met_files,
       columns=["mlht", "dens"],
       z_top=meta.settings["z_top"],
   )

The sampled values differ a little from HYSPLIT's own columns;
:doc:`/development/validation` says by how much.

Columns PYSTILT adds
--------------------

``release_height``
   Release height of each particle, in meters in the receptor's own
   vertical reference, for column and multipoint receptors
   (:func:`add_release_heights`). For a column it is the center of the
   particle's slab, the part of the column the particle stands for. STILT-R and
   X-STILT call it ``xhgt``.

``foot``
   When the model wrote none, or ``pbl_height`` is set (:func:`stilt.footprint.add_foot`),
   as above: only to the particles a footprint is made from, never to a
   stored particle file.

``foot_no_hnf_dilution``
   ``foot`` before the near-field correction, added by
   :func:`stilt.footprint.correct_near_field`. A stored particle file has none: it holds
   ``foot`` uncorrected, and the footprint applies the correction.

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
   check_particles

After a run
-----------

After a run PYSTILT adds the release heights to any model's particles.
That, with the columns the model wrote, is what a particle file stores and
``sim.particles`` returns: a model that writes no ``foot`` has none there.

The near-field correction is not stored. :func:`stilt.calc_footprint`
applies it when ``hnf_plume`` is set (the default), with the footprint's
``footprint_depth``, which for HYSPLIT's own ``foot`` is the run's
``veght``. The transport error does the same, with the same ``foot``, so
its enhancement is the footprint's. Anything else that reads ``foot`` from the
stored particles, such as ``particles.stilt.enhancement``, reads it
uncorrected: call :func:`stilt.footprint.correct_near_field` first to match the footprint.
The correction reads ``samt``, ``sigw`` and ``tlgr``, which only HYSPLIT
writes, so particles that lack them, including those whose ``foot`` came
from PYSTILT, are not corrected, and a warning says so.

.. autosummary::
   :toctree: _api
   :nosignatures:

   add_release_heights

What the footprint stage computes from the particles before gridding them,
``foot`` when it needs one and the near-field correction, is STILT's, in
:mod:`stilt.footprint.foot`.

.. currentmodule:: stilt.footprint

.. autosummary::
   :toctree: _api
   :nosignatures:

   add_foot
   correct_near_field
   HNF_PLUME_COLUMNS

.. currentmodule:: stilt.particles
