Migrating From X-STILT
======================

X-STILT users usually care about column-aware geometry, weighting and chemistry
transforms, and product-driven observation workflows. PYSTILT now has first
building blocks for those, but it remains more generic than X-STILT in this
alpha.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - X-STILT concept
     - X-STILT API / file
     - PYSTILT equivalent
   * - Column receptor definition
     - X-STILT column setup scripts
     - :class:`stilt.observations.ColumnSensor` and receptor builders
   * - Vertical weighting
     - custom AK × PWF logic
     - ``VerticalOperatorTransformSpec`` in ``FootprintConfig.transforms``
   * - First-order chemistry
     - chemistry hooks
     - ``FirstOrderLifetimeTransformSpec``
   * - Scene grouping
     - product-level grouping logic
     - :class:`stilt.observations.Scene` plus grouping helpers
   * - Column footprint outputs
     - X-STILT column products
     - standard PYSTILT footprint outputs from column/slant receptors
   * - Observation normalization
     - built-in product readers
     - generic ``Observation`` and sensor interfaces

Current boundary
----------------

PYSTILT intentionally stops at a generic observation layer. It does not yet
ship a large catalog of product-specific X-STILT file readers or retrieval
pipelines inside the core package.

Practical migration strategy
----------------------------

1. normalize your product data into ``Observation`` objects
2. use ``PointSensor`` or ``ColumnSensor`` to build receptors
3. encode reusable transforms declaratively where possible
4. keep product-specific I/O in your application layer until the alpha API settles
