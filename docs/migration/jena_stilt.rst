Coming From Jena STILT
======================

Jena STILT is the ``stiltR`` code from the Max Planck Institute for
Biogeochemistry, which the ICOS footprint service
(`stilt.icos-cp.eu <https://stilt.icos-cp.eu>`_) runs. It and PYSTILT
drive the same HYSPLIT particle model. They differ in how the particles
become a footprint, and in their defaults. This page lists the
differences, read from the ``stiltR`` code, and the PYSTILT settings that
come closest.

How the footprints differ
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Topic
     - Jena STILT
     - PYSTILT
   * - Resolution
     - 16 grids, from the fine grid to one 32 times coarser in each
       direction. At each step back in time, the extent of the particle
       cloud picks the grid, and the grid never gets finer as the steps go
       further back. A coarse cell's total is spread evenly over the fine
       cells it covers.
     - One grid. Each particle's influence is spread by a Gaussian kernel
       whose width grows with the spread of the particles, as in STILT-R
       (``smooth_factor``).
   * - Near the receptor
     - No correction.
     - The near-field correction of ``foot`` (``hnf_plume``), as in
       STILT-R.
   * - Mass
     - Particles whose ``dmass`` leaves the range 1e-3 to 1e3 are dropped.
       ``foot`` is weighted by ``dmass``, divided by its mean over the
       particles at each step.
     - Not used.
   * - Grid edges
     - A particle that crosses the western edge is dropped from that step
       on, since the background is taken there. Steps outside the other
       edges are left out.
     - A step outside the grid adds nothing. A particle that comes back
       counts again.
   * - Time layers
     - Hourly: layer ``k`` holds the steps from ``k`` to ``k + 1`` hours
       back.
     - Hourly, the same hours: the first hour back is layer -1
       (:doc:`../reference/layout`).
   * - Particle count
     - The total is divided by the particles left after the ``dmass``
       check.
     - The total is divided by ``numpar``.
   * - Defaults
     - 240 hours and 250 particles.
     - 24 hours and 200 particles.
   * - Meteorology
     - ECMWF short-term forecasts every 3 hours at about 35 km, in ARL
       format.
     - Any ARL meteorology, ERA5 included once converted
       (:doc:`../guides/meteorology`).

The settings that come closest
------------------------------

Give the run length and particle count of your Jena runs, turn off the
near-field correction, and use the same grid:

.. code-block:: yaml

   n_hours: -240
   numpar: 250
   hnf_plume: false
   grid:
     xmin: -15.0      # lon_ll
     xmax: 35.0       # lon_ll + numpix_x * lon_res
     ymin: 33.0       # lat_ll
     ymax: 72.0       # lat_ll + numpix_y * lat_res
     xres: 0.2        # lon_res
     yres: 0.1        # lat_res

The coarsening grids, the ``dmass`` weighting, and the western-edge rule
have no PYSTILT setting. Footprints from the two will not match cell by
cell, most of all far from the receptor, where Jena STILT's grids are
coarsest. Compare totals over regions (:doc:`../guides/aggregation`), or
the modelled mole fractions, rather than single cells.
