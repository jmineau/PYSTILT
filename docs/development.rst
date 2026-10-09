Development
===========

Contributions are welcome: bug reports, documentation fixes, and code. See
`CONTRIBUTING.md <https://github.com/jmineau/PYSTILT/blob/main/CONTRIBUTING.md>`_
for conventions, and open issues on
`GitHub <https://github.com/jmineau/PYSTILT/issues>`_.

Use of AI coding agents
-----------------------

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science.

If you contribute with an agent, ``AGENTS.md`` at the repository root is the
orientation file it should read.

Set up a development environment
--------------------------------

PYSTILT uses `uv <https://docs.astral.sh/uv/>`_ and
`just <https://github.com/casey/just>`_. From a clone of the repository:

.. code-block:: bash

   uv sync
   uv run pre-commit install

Common tasks:

.. code-block:: bash

   just test             # run the unit tests
   just quality-check    # lint, type check, import contracts, docstrings, unit tests
   just build-docs       # build this documentation into docs/_build/html
   just docs-serve       # preview it at http://127.0.0.1:8000, rebuilt on every save

``just`` with no arguments lists the rest.

.. _stilt-r-parity:

STILT-R parity
--------------

PYSTILT is tested against a fixed commit of
`STILT-R <https://github.com/uataq/stilt>`_. This section says what "matches
STILT-R" covers and how to move to a newer STILT-R commit.

Pinned upstream commit
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Upstream
     - https://github.com/uataq/stilt
   * - Pinned SHA
     - ``0e290a68730e155bc2156e28af492a463228c88a``
   * - Pin location
     - ``.github/workflows/tests.yml`` (env ``STILT_R_SHA``)
   * - Last verified
     - 2026-06-25

When the fidelity tests pass, PYSTILT matches STILT-R at this commit. It
says nothing about STILT-R's current ``main`` branch.

What matches
^^^^^^^^^^^^

PYSTILT's footprints match STILT-R's for the 20 fidelity scenarios in
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
- the hyper-near-field (HNF) plume correction on (12 scenarios) and off
- ``smooth_factor`` of 0, 0.5, 1, and 2
- 0.01° and 0.05° longitude/latitude grids and a UTM grid
- a 2° box, a receptor in the domain's corner, a receptor at the edge of
  the met grid, and a small 0.1° domain that particles leave early
- receptor heights above ground, above sea level (``kmsl=1``), and at 0.5 m
- winter (stable) and summer (convective) HRRR
- hourly footprints and ``time_integrate=True``
- a wind-error run (``siguverr=1`` m/s)

The synthetic tests in
`tests/r_stilt/test_footprint_synth.py <https://github.com/jmineau/PYSTILT/blob/main/tests/r_stilt/test_footprint_synth.py>`_
feed hand-made particle tables to both implementations to test single code
paths: one Gaussian particle, cells on the grid boundary, the dateline, a
global grid, and the latitude scaling of the kernel width. They need R but
no meteorology or HYSPLIT, so they run anywhere R is installed:

.. code-block:: bash

   STILT_R_DIR=$PWD/stilt-r-src uv run pytest tests/r_stilt -m r_only

What does not match
^^^^^^^^^^^^^^^^^^^

- **The NetCDF layout.** STILT-R writes dimensions ``(x, y, time)`` with a
  fill value of ``-1``. PYSTILT writes CF-1.8 NetCDF with dimensions
  ``(time, lat, lon)``. Read PYSTILT files as ordinary CF NetCDF. Tools that
  expect STILT-R's layout will not read them.
- **Forward runs with the HNF correction.** STILT-R accumulates the plume
  from the far end of a forward trajectory back toward the release point,
  so its plume is widest at release. PYSTILT accumulates outward from the
  release point. The two ``foot`` values differ on purpose, and
  ``test_forward_hnf_foot_intentionally_differs_from_r`` guards the
  difference.
- **The HNF correction with ``veght`` above 1.** HYSPLIT reads such a
  ``veght`` as meters above ground, and so does PYSTILT's correction.
  STILT-R multiplies it by the mixed-layer height, which removes most of
  the footprint after the first hour
  (`uataq/stilt#142 <https://github.com/uataq/stilt/issues/142>`_). No
  fidelity scenario sets ``veght`` above 1.
- **Untested inputs.** Runs longer than 24 h backward, latitudes above 80°,
  and grids finer than 0.001° have not been compared.
- **Other STILT-R commits.** Nothing detects upstream changes to
  ``calc_footprint.r``, ``permute.f90``, or ``calc_trajectory.r``. Bump the
  pinned SHA by hand (below).

Large-scale comparison
^^^^^^^^^^^^^^^^^^^^^^

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
subdirectory, and older STILT-R listed it twice.

Moving to a newer STILT-R commit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These STILT-R files determine the numbers PYSTILT is compared against:

- ``r/src/calc_footprint.r``: footprint kernel and gridding
- ``r/src/calc_trajectory.r``: the HYSPLIT wrapper, random seed, and ``krand``
- ``r/src/permute.f90``: the Fortran loop that adds each particle's kernel to
  the grid

The R helper scripts in ``tests/fixtures/r_helpers/`` call these files for
the synthetic tests. They may need updating if a function signature changes
upstream.
