Configuration Reference
=======================

The main durable project schema is :class:`stilt.config.ModelConfig`. It layers
named meteorology streams, footprint definitions, and executor settings on top
of the STILT parameter surface. The field tables below are rendered directly
from the live Pydantic model metadata so the parameter descriptions stay in
sync with the code.

PYSTILT intentionally keeps the public configuration surface flat. Common run
controls such as ``n_hours``, ``numpar``, and ``seed`` can be passed directly to
``ModelConfig`` or ``Model`` rather than through nested parameter objects.
Internally, those fields are routed to PYSTILT logic or to HYSPLIT files such
as ``CONTROL``, ``SETUP.CFG``, ``ZICONTROL``, ``WINDERR``, and ``ZIERR``.
Most users do not need to care about that routing unless they are debugging
generated HYSPLIT inputs or adding a new parameter.

Configuration models
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Purpose
   * - :doc:`ModelConfig <_api/stilt.config.ModelConfig>`
     - Project-level config: STILT params plus met and footprint definitions.
   * - :doc:`MetConfig <_api/stilt.config.MetConfig>`
     - Meteorology file discovery and optional subgridding.
   * - :doc:`FootprintConfig <_api/stilt.config.FootprintConfig>`
     - Settings for one named footprint product.
   * - :doc:`Bounds <_api/stilt.config.Bounds>`
     - Immutable bounding box used for spatial subsetting.
   * - :doc:`Grid <_api/stilt.config.Grid>`
     - Gridded spatial domain for footprint computation.
   * - :doc:`RuntimeSettings <_api/stilt.config.RuntimeSettings>`
     - Runtime-only settings shared across CLI, workers, and executors.

Parameter models
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Purpose
   * - :doc:`ModelParams <_api/stilt.config.ModelParams>`
     - Core STILT run controls.
   * - :doc:`TransportParams <_api/stilt.config.TransportParams>`
     - HYSPLIT transport and turbulence parameterization.
   * - :doc:`ErrorParams <_api/stilt.config.ErrorParams>`
     - Transport error trajectory parameters for XY and ZI perturbations.

Transform specifications
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Purpose
   * - :doc:`VerticalOperatorTransformSpec <_api/stilt.config.VerticalOperatorTransformSpec>`
     - Declarative built-in transform for applying a vertical operator.
   * - :doc:`FirstOrderLifetimeTransformSpec <_api/stilt.config.FirstOrderLifetimeTransformSpec>`
     - Declarative built-in transform for first-order lifetime decay.

.. toctree::
   :hidden:

   _api/stilt.config.ModelConfig
   _api/stilt.config.MetConfig
   _api/stilt.config.FootprintConfig
   _api/stilt.config.Bounds
   _api/stilt.config.Grid
   _api/stilt.config.RuntimeSettings
   _api/stilt.config.ModelParams
   _api/stilt.config.TransportParams
   _api/stilt.config.ErrorParams
   _api/stilt.config.VerticalOperatorTransformSpec
   _api/stilt.config.FirstOrderLifetimeTransformSpec
