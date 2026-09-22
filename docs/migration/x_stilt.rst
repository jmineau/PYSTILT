Migrating From X-STILT
======================

X-STILT users usually care about column-aware receptor geometry, averaging
kernel and pressure weighting, and product-driven overpass workflows. PYSTILT
ports those as small objects and functions rather than as scripts, and stays
more generic than X-STILT in this alpha.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - X-STILT concept
     - X-STILT API / file
     - PYSTILT equivalent
   * - Column receptor (``minagl`` / ``maxagl``, ``agl`` levels)
     - ``get.recp.sensorv2.r``
     - :func:`stilt.observations.build_column_receptor` → :class:`stilt.ColumnReceptor`
   * - Slant column (``run_slant``)
     - ``get.recp.sensorv2.r``
     - :func:`stilt.observations.build_slant_receptor` from ``ViewingGeometry`` + ``LineOfSight``
   * - Sounding selection (near-field + background)
     - ``sel.obs4recpv2``
     - :func:`stilt.observations.select_observations_spatial`
   * - Jittered receptors in a pixel (``jitterTF``)
     - ``jitter.obs4recp.r``
     - :func:`stilt.observations.jitter_observation`
   * - Overpass grouping
     - ``get_timestr`` / overpass search
     - :func:`stilt.observations.group_by_overpass` → :class:`stilt.observations.Scene`
   * - Vertical weighting (AK × PWF)
     - ``wgt.trajec.foot*.r``
     - ``averaging_kernel`` + ``pressure_weighting`` transforms (:doc:`/advanced/transforms`)
   * - First-order chemistry
     - ``chem_lifetime``
     - ``first_order_lifetime`` transform
   * - Column footprint outputs
     - X-STILT column products
     - standard PYSTILT footprints from column / slant receptors
   * - Product readers (OCO-2/3, TROPOMI, TCCON)
     - ``column_obs/*``
     - your code, producing :class:`stilt.observations.Observation`
   * - Transport error to XCO2, background methods
     - ``error_functions/``, ``background/``
     - not ported; error trajectories and ``FootprintConfig.error`` are the building block

Practical migration strategy
----------------------------

1. write a reader that turns your product into ``Observation`` objects
   (see *Adding your own instrument* in :doc:`/advanced/observations`)
2. group by overpass and select soundings with the built-in helpers
3. build receptors with a built-in or your own builder
4. put the averaging kernel on each observation's ``transforms`` and
   ``pressure_weighting`` in the footprint config
