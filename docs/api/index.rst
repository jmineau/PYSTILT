API Reference
=============

Complete API reference generated from source docstrings.

.. note::

   NumPy-style docstrings are used throughout. Type annotations are rendered
   automatically via ``sphinx-autodoc-typehints``. Click ``[source]`` on any
   member to view the implementation.


Public API
----------

The following names are exported from the top-level ``stilt`` namespace:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symbol
     - Description
   * - :class:`~stilt.Model`
     - Top-level project interface
   * - :class:`~stilt.ModelConfig`
     - Project-level configuration (all STILT + met + footprint + execution)
   * - :class:`~stilt.STILTParams`
     - All HYSPLIT/STILT transport parameters
   * - :class:`~stilt.MetConfig`
     - Meteorology source configuration
   * - :class:`~stilt.MetStream`
     - Runtime meteorology file discovery and selection
   * - :class:`~stilt.FootprintConfig`
     - Footprint output configuration (grid, smoothing, time integration)
   * - :class:`~stilt.Grid`
     - Spatial grid definition (bounds + resolution)
   * - :class:`~stilt.Bounds`
     - Spatial bounding box
   * - :class:`~stilt.Simulation`
     - Container for running and reading a single simulation
   * - :class:`~stilt.SimID`
     - Structured simulation identifier
   * - :class:`~stilt.Trajectories`
     - STILT particle trajectory ensemble
   * - :class:`~stilt.Footprint`
     - STILT footprint container
   * - :class:`~stilt.Receptor`
     - Spatio-temporal receptor definition
   * - :func:`~stilt.read_receptors`
     - Load receptors from a CSV file
   * - :class:`~stilt.Service`
     - Thin service-facing facade over the queue/runtime substrate
   * - :class:`~stilt.QueueStatus`
     - Project-level queue summary returned by ``stilt.service``
   * - :class:`~stilt.BatchStatus`
     - Batch progress summary returned by ``stilt.service``


Full Autodoc
------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   stilt
