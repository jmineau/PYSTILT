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
   mean air density below it (Lin et al., 2003). The surface layer is
   ``footprint_depth`` of the mixed layer, or ``footprint_depth`` meters
   when that is more than 1; above it ``foot`` is zero.
   :func:`stilt.calc_footprint` adds each particle's ``foot`` into its grid
   cell and divides by the number of particles.

Where ``foot`` comes from:

- **HYSPLIT** computes it in STILT mode, as STILT-R's runs do, when
  ``varsiwant`` has ``foot`` (the default). PYSTILT keeps it as written.
- **Any other model whose particles have no** ``foot``, such as one that
  writes only positions, gets it from PYSTILT after the run
  (:func:`add_foot`), from ``zagl``, ``mlht``, the output step, and the
  pressure drop from the ground to the surface layer's top in the met files
  the run read. The run's log says so. A particle counts for the whole step
  when it ends the step inside the layer, where HYSPLIT counts the minutes
  it spent there. When the model wrote no ``mlht``, the met's
  boundary-layer height (``PBLH``) is used, unless ``pbl_height`` says
  otherwise. :doc:`/development/validation` says how far this moves a
  footprint.

The mixed-layer height
~~~~~~~~~~~~~~~~~~~~~~

``foot`` depends on the mixed-layer height, the ``mlht`` column, through the
depth surface fluxes are mixed through (``footprint_depth`` of it). The
``pbl_height`` setting says where PYSTILT takes that height from when it
computes ``foot``; ``pbl_height`` and ``mlht`` name the same quantity.

.. code-block:: yaml

   models:
     hysplit:
       pbl_height: richardson   # or met, or a height in meters
       pbl_height_min: 150      # the lowest height, in meters

- **Unset** (the default), PYSTILT stays as close to the model as its
  output allows: it keeps the model's ``foot`` (HYSPLIT's, which STILT-R
  parity covers); for a model that writes no ``foot`` it uses the model's
  ``mlht``, or the met's ``PBLH`` when the model wrote none.
- **Set**, PYSTILT computes ``foot`` itself, with that height, for any
  model's particles, HYSPLIT's included, and replaces the model's ``foot``
  and ``mlht``. The run's log says which height it used. ``met`` is the
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
``foot``, and ``mlht`` (:data:`HNF_PLUME_COLUMNS`). The plume it models
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
   When the model wrote none (:func:`add_foot`), as above.

``foot_no_hnf_dilution``
   ``foot`` before the near-field correction, added by
   :func:`correct_near_field`. A stored particle file has none: it holds
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
   HNF_PLUME_COLUMNS
   check_particles

After a run
-----------

After a run PYSTILT adds the release heights to any model's particles, and
``foot`` to particles that have none. That is what a particle file stores
and ``sim.particles`` returns.

The near-field correction is not stored. :func:`stilt.calc_footprint`
applies it when ``hnf_plume`` is set (the default), with the run's
``footprint_depth``, and the transport error does the same, so its
enhancement is the footprint's. Anything else that reads ``foot`` from the
stored particles, such as ``particles.stilt.enhancement``, reads it
uncorrected: call :func:`correct_near_field` first to match the footprint.
The correction reads ``samt``, ``sigw`` and ``tlgr``, which only HYSPLIT
writes, so particles that lack them, including those whose ``foot`` came
from PYSTILT, are not corrected, and a warning says so.

.. autosummary::
   :toctree: _api
   :nosignatures:

   add_release_heights
   add_foot
   correct_near_field
