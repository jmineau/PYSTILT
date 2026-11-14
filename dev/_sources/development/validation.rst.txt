Validation
==========

How closely PYSTILT's results match a reference, and how fast its slow
steps are, as measured. The user guides give the setting to use; this page
gives the numbers behind it, each with the test or study that measured it,
so you can judge how far to trust a result and rerun the check yourself.

Footprints against STILT-R
--------------------------

Every footprint cell matches STILT-R to a relative tolerance of 1e-7, for
the same particles, on a pinned STILT-R commit. The ``fidelity`` tests run
both on the same HYSPLIT output in CI. What is compared, and the few
settings that do not match on purpose, are in :doc:`stilt_r_parity`.

Met stops that pass as domain exits
-----------------------------------

A run whose met runs out partway fails as ``MET_COVERAGE`` when a particle
at the last step is more than 5 grid cells from every side of the met's
grid (:doc:`/guides/checking`). Particles that leave the domain end at most
2.0 cells inside it, on HRRR (3 km), NAM (12 km) and NARR (32 km) grids and
their crops.

On domains a hundred cells across or more, every met stop tested was caught:
the particle furthest inside was 34 cells in or more. On crops 45 to 120
cells across, 3 to 9 % of the met stops tested passed as domain exits. None
of them was in the first 6 hours of a run, at most one particle in six was
still in the air, and under 3 % of the footprint was lost. So on a small
crop, a run cut short late by the met can look complete. The rule and its
reasons are in :doc:`architecture` ("Implementation notes").

Met sampled at particle positions
---------------------------------

:func:`stilt.sampling.sample_met` reads ``mlht``, ``dens``, ``pres`` and
``zsfc`` from the met files at each particle's position. On two 6-hour HRRR
runs from Salt Lake City (``tests/integration/test_sample_met.py``), against
HYSPLIT's own columns:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Column
     - Difference from HYSPLIT
   * - ``zsfc``
     - A few meters.
   * - ``mlht``
     - A median of 16 to 30 m against a HYSPLIT run with ``kmixd: 0``
       (floored at ``kmix0``). With the default ``kmixd: 3`` HYSPLIT finds
       its own mixed layer from the profile, so the two are not comparable.
   * - ``pres``
     - A median of 0.02 hPa, 0.23 to 0.30 hPa at the 95th percentile, and
       1.4 to 4.6 hPa at worst.
   * - ``dens``
     - 0.17 to 0.57 % higher.

On one of those runs with a 10 km top, ``z_top=10000`` gives a median
``pres`` difference of 0.024 hPa against 0.032 for the default 25 km, and
about 0.29 hPa at the 95th percentile with either. Terrain-following files
(flag 3) are read on their own levels rather than HYSPLIT's internal ones,
so their values differ more, by an amount not measured.

``foot`` for a model that writes only positions
------------------------------------------------

PYSTILT computes ``foot`` for particles that have none
(:func:`stilt.particles.add_foot`): the time below ``veght`` of the mixed
layer over the air below that depth, which it takes from the drop in
pressure between the ground and the depth in the met files. HYSPLIT
computes its own the same way in its turbulence steps, adding up the dry
air over its levels, with 28.84 g/mol where PYSTILT has 28.97. Three
6-hour HRRR runs from Salt Lake City, a winter afternoon and night and a
summer evening, compare the two, with HYSPLIT's footprint as the reference
(``tests/integration/test_foot.py``; runs of 200 and 1,000 particles gave
the ranges below). "Cells" is the sum of the absolute cell differences over
the reference's total.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - ``foot`` from
     - Footprint total
     - Cells
   * - HYSPLIT's ``samt`` and ``mlht``
     - within 0.4 %
     - 0.2 to 0.4 %
   * - The whole step below the layer, in place of ``samt``
     - within 0.6 %
     - 0.6 to 1.3 %
   * - Then the met's ``mlht``, against HYSPLIT's default ``kmixd: 3``
     - 0.3 to 25 % higher
     - 9 to 61 %
   * - The same, against a HYSPLIT run with ``kmixd: 0`` (the met's ``PBLH``)
     - 0.3 % lower to 10 % higher
     - 1.5 to 20 %
   * - The same, with the met's ``mlht`` kept at least ``kmix0``, as HYSPLIT keeps its own
     - within 1.5 %
     - 1.5 to 6.0 %

Row by row, with HYSPLIT's ``samt`` and ``mlht``, ``foot`` agrees to a
median of 0.4 % and within 0.8 % on nine rows in ten. Counting the whole
step changes little because HYSPLIT's ``samt`` is almost always 0 or the
whole minute.

What is left comes from the mixed-layer height. HYSPLIT's default finds
the mixed layer from the met's profile, by a bulk Richardson number, and
keeps it at least ``kmix0`` (150 m), while the met's ``mlht`` is the met
model's ``PBLH``: at night it was 1.5 to 1.8 times HYSPLIT's, and in the
afternoon 0.84 times. Against a HYSPLIT run that reads the met's ``PBLH``
(``kmixd: 0``) the two heights match to a few percent, and what still moves
the night footprints is the ``kmix0`` floor: below 150 m HYSPLIT mixes the
flux through more air than the met's boundary layer holds. A transport
model that writes only positions has no such floor, so its footprint
follows the met's own boundary layer.

Pressure weighting
------------------

The pressure-weighting transform (:doc:`/guides/transforms`) weights each
particle of a column by the air mass of its release layer, from a fit of
``ln p`` against height. On a winter column at WBB in Salt Lake City, the
least isothermal case in the test data, the fit explains more than 99 % of
the variance in pressure. Its scale height is 6.5 to 9 km, and its surface
pressure is 800 to 900 hPa, as the site's 1,480 m elevation needs. Between
runs of 100 and 1,000 particles, the weights' total agrees to 2 % and the
weighted footprint's ratio to the unweighted one to 5 %
(``tests/integration/test_pressure_weighting.py``).

Reading a large met archive
---------------------------

The first time PYSTILT reads a folder of met files it reads each file's
header into a table. With 16 threads, 16,600 HRRR files on a network
filesystem took about two minutes; one thread took 35. Later runs read the
table.
