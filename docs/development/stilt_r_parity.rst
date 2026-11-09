.. _stilt-r-parity:

STILT-R Parity
==============

PYSTILT is tested against a fixed commit of
`STILT-R <https://github.com/uataq/stilt>`_. This page says what "matches
STILT-R" covers and how to move to a newer STILT-R commit.

Pinned upstream commit
----------------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Upstream
     - https://github.com/uataq/stilt
   * - Pinned SHA
     - ``278b1ef6d542efd21c6ccab2a85b796acd9812a6``
   * - Pin location
     - ``.github/workflows/tests.yml`` (env ``STILT_R_SHA``)
   * - Last verified
     - 2026-11-08

When the fidelity tests pass, PYSTILT matches STILT-R at this commit. It
says nothing about STILT-R's current ``main`` branch.

What matches
------------

PYSTILT's footprints match STILT-R's for the 21 fidelity scenarios in
`tests/fixtures/r_stilt_reference.py <https://github.com/jmineau/PYSTILT/blob/main/tests/fixtures/r_stilt_reference.py>`_,
within these tolerances:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Tolerance
   * - Footprint, per cell
     - ``rtol=1e-7, atol=1e-8``
   * - Footprint total
     - ``rtol=1e-6``
   * - Footprint peak
     - ``rtol=1e-7``
   * - Number of nonzero cells
     - identical
   * - Intermediate tables (interpolation, rtime, gridding)
     - ``rtol=1e-12``
   * - Trajectory positions and ``foot`` after the HNF correction
     - ``rtol=1e-7``
   * - Grid coordinates, longitude/latitude
     - ``atol=1e-12`` degrees
   * - Grid coordinates, projected
     - ``atol=1e-3`` m

The scenarios cover:

- point, column, and three-location multipoint receptors
- 6 h and 24 h backward runs, and a 6 h forward run
- the hyper-near-field (HNF) plume correction on (every scenario but one) and
  off, with ``veght`` at its default of 0.5 (a fraction of the mixed-layer
  height) and at 100 (meters above ground)
- ``smooth_factor`` of 0, 0.5, 1, and 2
- 0.01° and 0.05° longitude/latitude grids and an unsmoothed UTM grid
- a 2° box, a receptor in the domain's corner, a receptor at the edge of
  the met grid, and a small 0.1° domain that particles leave early
- receptor heights above ground, above sea level (``kmsl=1``), and at 0.5 m
- winter (stable) and summer (convective) HRRR
- hourly footprints, and STILT-R's ``time_integrate=TRUE`` against one
  PYSTILT layer as long as the run
- a wind-error run (``siguverr=1`` m/s)
- cells with ``xres`` different from ``yres``, in both orders (synthetic particles)

The synthetic tests in
`tests/r_stilt/test_footprint_synth.py <https://github.com/jmineau/PYSTILT/blob/main/tests/r_stilt/test_footprint_synth.py>`_
feed hand-made particle tables to both implementations to test single code
paths: one Gaussian particle, cells on the grid boundary, the dateline, a
global grid, and the latitude scaling of the kernel width. They need R but
no meteorology or HYSPLIT, so they run anywhere R is installed:

.. code-block:: bash

   STILT_R_DIR=$PWD/stilt-r-src uv run pytest tests/r_stilt -m r_only

What does not match
-------------------

- **The NetCDF layout.** STILT-R writes dimensions ``(x, y, time)`` with a
  fill value of ``-1``. PYSTILT writes CF-1.8 NetCDF with dimensions
  ``(time, lat, lon)``. Read PYSTILT files as ordinary CF NetCDF. Tools that
  expect STILT-R's layout will not read them.
- **Smoothing on a projected grid.** The kernel width,
  ``0.06 * (var(x) + var(y))**0.25 * sqrt(|rtime| / 1440)``, is made for
  coordinates in degrees. STILT-R applies it to the projected coordinates,
  so on a grid in meters the kernel is a few meters wide and the footprint
  is hardly smoothed. On a 24 h run on 1 km UTM cells, STILT-R's footprint
  has a third of the nonzero cells of the same run on a 0.01° grid (9,771
  and 28,706), and PYSTILT's has 26,385. PYSTILT works the width out in
  degrees from the particles' longitudes and latitudes, as on a
  longitude/latitude grid, then multiplies it by the length of one degree
  of longitude at the particles' mean position, measured in the grid's
  projection and units. Each step's longitudes are first taken within 180°
  of that step's circular mean, so a cloud is measured as one wherever it
  is, on both sides of 180° included.
  The kernel is then as wide from east to west, on the ground, as on a
  longitude/latitude grid. On the projected grid it is round. On a
  longitude/latitude grid it is ``1 / cos(lat)`` times wider from north to
  south, because it has one width in degrees on both axes.
  The fidelity suite's UTM scenario compares unsmoothed footprints, and
  ``test_projected_smoothing_intentionally_differs_from_r`` guards the
  difference. Longitude/latitude grids are not affected.
- **The early-time interpolation on a projected grid.** In the first 100
  minutes, when the particles' median step is longer than a cell along
  either axis, both add positions between the output steps. STILT-R
  measures the step in degrees and compares it with ``xres`` and ``yres``
  in the grid's units, so on a grid in meters it never interpolates. On a
  24 h run on 100 m UTM cells, unsmoothed, STILT-R's rule gives a
  footprint with 12,341 nonzero cells, against 15,139 for the same run on
  a 0.001° grid, which does interpolate. PYSTILT measures the step on the
  projected grid, in its units and along its axes, and its footprint has
  14,563. It then
  interpolates as STILT-R does on longitude/latitude cells of the same
  size. The positions added are interpolated in degrees on any grid.
  ``test_utm_early_interpolation_intentionally_differs_from_r`` guards the
  difference. Longitude/latitude grids are not affected.
- **Smoothing a cloud near 180° on a global grid.** On a grid that goes
  all the way around (``xmin=-180``, ``xmax=180``), STILT-R leaves the
  particles' longitudes as HYSPLIT writes them, from -180 to 180. A cloud
  on both sides of 180° then has a variance of thousands of square degrees
  and gets a kernel many times too wide, and what the kernel spreads past
  ±180° is lost. For a cloud of 300 particles (standard deviation 0.3°, a
  day after release) on a 0.25° grid, a cloud at 175° covers 36 cells and
  keeps its whole footprint in both programs; at 179.9° STILT-R covers 1,242
  cells and loses 42 % of the footprint, and PYSTILT covers 34 cells and
  keeps all of it. PYSTILT takes the longitudes of each time step within
  180° of that step's mean longitude (the circular mean, so the seam is on
  the far side of the globe from the cloud) before it measures the spread.
  The cloud is measured as one wherever it is, and a real spread of 10° stays
  10°. PYSTILT also adds the smoothing that passes one edge of such a grid
  to the cells at the other edge, so the footprint of a cloud near 180° is
  the footprint of the same cloud anywhere else, shifted. The longitudes
  are unwrapped this way on every longitude/latitude grid, which also fixes
  a cloud around 0° on a wide grid that crosses 180°, where STILT-R's shift to 0° to 360° cuts the
  cloud in two. A grid whose cells do not fit 360° (0.7° cells leave 0.2°)
  has an edge, and smoothing is cut there as before. A cloud that is not
  cut by the seam is measured as before, bit for bit.
  ``test_global_grid_across_180_intentionally_differs_from_r`` guards the
  difference.
- **Particles across 180° from the edge of a regional grid.** Both programs
  pad the grid by the reach of the largest kernel, so particles just outside
  an edge smooth onto it. On a grid that ends at or near 180° (from -180° to
  -170°, or from 170° to 180°) STILT-R leaves the longitudes as written, so
  a particle at 179.9° is 360° from a grid that starts at -180° and is
  dropped. A cloud only on that side gives no footprint, and a cloud on both
  sides loses what was written on the far side. PYSTILT moves a longitude
  by a whole turn when that puts it on the padded grid, so the footprint is
  that of the same cloud and grid anywhere else. Particles already on the
  padded grid are not moved, so a grid whose padding does not reach 180°
  gets the footprint it got before, bit for bit.
  ``test_grid_edge_at_180_intentionally_differs_from_r`` guards the
  difference.
- **The longitudes of a global grid not written from -180°.** STILT-R
  labels the cells of any grid a whole turn wide from -180° to 180°. PYSTILT
  keeps the longitudes the grid is written with, as on any other grid: 0° to
  360° for a grid from 0° to 360°. The values are the same, a turn apart. A
  grid across 180° (``170`` to ``190``, or ``170`` to ``-170``) has STILT-R's
  longitudes, 170° to 190°: STILT-R shifts it to 0° to 360°, and PYSTILT moves
  the particles onto it by whole turns instead.
- **Grids written past ±180°.** A grid that passes 0° as well as 180° has no
  cells in STILT-R (``-10`` to ``190`` becomes ``350`` to ``190``), and a
  grid west of -180° (``-190`` to ``-170``) is not shifted, so its particles
  are dropped. PYSTILT makes a footprint on these as on any other. A grid
  wider than 360° is refused when it is made.
- **The early-time interpolation across 180°.** HYSPLIT writes longitudes
  from -180° to 180°, and STILT-R interpolates them as written, so a step
  from 179.99° to -179.97° is interpolated through 0°, and the positions
  added put part of that step's ``foot`` on the far side of the earth. On a
  global grid that is a misplaced footprint (0.5 % of it for 40 particles
  crossing 180° at 0.03° a minute on 0.01° cells); on a regional grid at
  180° the part is lost. On a grid STILT-R shifts to 0° to 360° the same
  happens to a particle that crosses 0°, and lands near 180°. PYSTILT interpolates
  each step the short way round and measures it that way when it decides
  whether to interpolate. A particle that takes no step across the seam is
  interpolated as before, bit for bit.
  ``test_early_interpolation_across_180_intentionally_differs_from_r``
  guards the difference.
- **The HNF correction of a receptor above sea level.** PYSTILT grows the
  plume from the release height above ground, so it needs ``zsfc`` in
  ``varsiwant`` for an MSL receptor. STILT-R grows it from the
  receptor height as given, which with ``kmsl=1`` is above sea level, so
  over high terrain its correction does nothing. The ``msl_altitude``
  scenario compares the uncorrected ``foot_no_hnf_dilution``, and
  ``test_msl_hnf_foot_intentionally_differs_from_r`` guards the difference.
- **Untested inputs.** Runs longer than 24 h backward, latitudes above 80°,
  and grids finer than 0.001° have not been compared.
- **Other STILT-R commits.** Nothing detects upstream changes to
  ``calc_footprint.r``, ``permute.f90``, ``calc_trajectory.r``, or
  ``calc_plume_dilution.r``. Bump the pinned SHA by hand (below).
- **STILT-R before the pinned commit.** Older STILT-R accumulated the
  near-field plume from the far end of a forward trajectory, which made it
  widest at the release point
  (`uataq/stilt#144 <https://github.com/uataq/stilt/pull/144>`_), and
  multiplied a ``veght`` above 1 by the mixed-layer height instead of reading
  it as meters above ground, as HYSPLIT does
  (`uataq/stilt#145 <https://github.com/uataq/stilt/pull/145>`_). Its
  ``permute.f90`` also read a smoothing kernel as if it had the same number
  of cells along longitude and latitude, so on cells with ``xres`` different
  from ``yres`` the footprint total came out wrong or the kernel was read past
  its end
  (`uataq/stilt#148 <https://github.com/uataq/stilt/pull/148>`_). A forward
  run with the near-field correction, a ``veght`` above 1, or cells that are
  not square matches PYSTILT only on the pinned commit or a later one.

Large-scale comparison
----------------------

Outside CI, PYSTILT was compared with STILT-R on 200 receptors at the WBB
tower in the Salt Lake Valley, 2016 to 2024 (35 m above ground, 1000
particles, 24 h backward, HRRR, ``krand=2`` and ``seed=42``, the same v5.1.0
``hycs_std`` on both sides). For each receptor, the PYSTILT trajectory was
compared with a separate STILT-R ``calc_trajectory`` run, and the PYSTILT
footprint with STILT-R's ``calc_footprint`` of the same particles, at 0.01°,
0.05°, and 0.1°. All 200 receptors matched.

.. list-table::
   :header-rows: 1

   * - Quantity
     - Agreement (worst of 200 receptors)
   * - Trajectory, per particle
     - absolute ``1.0e-17``, relative ``5.6e-16`` (float64 round-off)
   * - Footprint, per-cell relative difference
     - median ``~2e-8``, max ``~6e-8``
   * - Footprint nonzero cells
     - identical (for example 250,644 cells at 0.01°)
   * - Footprint total
     - relative ``~2e-8``

The trajectories are identical, so the footprint differences come only from
the order in which floating-point sums are added. The relative difference is
the same for small cells (under 0.01 % of the peak) as for large ones.
Footprints are stored as ``float32``, whose precision is about ``1.2e-7``, so
the two outputs agree to the precision the file can hold. The mass-weighted
relative difference, which is what an inversion sums over, is also about
``2e-8``.

Reading the same met files on both sides over the full date range needs the
``find_met_files`` fix in the pinned STILT-R commit. HRRR archives can hold
the same file both at the top level (as a symlink) and in a ``YYYY/MM/``
subdirectory, and older STILT-R listed it twice. The pinned commit also
lists the next met file for a run that ends partway through a file
(`uataq/stilt#149 <https://github.com/uataq/stilt/pull/149>`_), as PYSTILT does.

Moving to a newer STILT-R commit
--------------------------------

#. Change ``STILT_R_SHA`` in ``.github/workflows/tests.yml``.
#. Run the fidelity tests locally:

   .. code-block:: bash

      git clone https://github.com/uataq/stilt stilt-r-src
      git -C stilt-r-src checkout NEW_SHA   # the commit to pin
      STILT_R_DIR=$PWD/stilt-r-src STILT_TEST_MET_DIR=tests/met_cache \
        uv run pytest tests/r_stilt/ -v -m fidelity

#. If every scenario passes at the current tolerances, commit the change and
   update "Last verified" above.
#. If a scenario fails, diff ``calc_footprint.r``, ``permute.f90``, and
   ``calc_trajectory.r`` between the two commits to find the upstream change.
   Then do one of the following:

   - make the same change in PYSTILT,
   - loosen that scenario's tolerance, with a comment saying why, or
   - stay on the old commit.

Files to diff on every bump
---------------------------

These STILT-R files determine the numbers PYSTILT is compared against:

- ``r/src/calc_footprint.r``: footprint kernel and gridding
- ``r/src/calc_trajectory.r``: the HYSPLIT wrapper, random seed, and ``krand``
- ``r/src/calc_plume_dilution.r``: the near-field plume correction (``hnf_plume``)
- ``r/src/permute.f90``: the Fortran loop that adds each particle's kernel to
  the grid

The R helper scripts in ``tests/fixtures/r_helpers/`` call these files for
the synthetic tests. They may need updating if a function signature changes
upstream.
