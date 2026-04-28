Configuration
=============

.. currentmodule:: stilt.config

The main output project schema is :class:`stilt.config.ModelConfig`. It layers
named meteorology streams, footprint definitions, and executor settings on top
of the STILT parameter surface. The field tables below are rendered directly
from the live Pydantic model metadata so the parameter descriptions stay in
sync with the code.

.. tip::
  If you are new to PYSTILT, start with the :doc:`configuration guide <../guides/configuration>`
  before diving into the reference details below. The guide focuses on the most common configuration
  fields. This API reference is comprehensive and includes all fields, but it may be overwhelming
  if you are just getting started.

Config objects
--------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   ModelConfig
   MetConfig
   FootprintConfig
   Bounds
   Grid
   RuntimeSettings

Parameters
----------

.. autosummary::
   :toctree: _api
   :nosignatures:

   ModelParams
   TransportParams
   ErrorParams


Transform specifications
------------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   VerticalOperatorTransformSpec
   FirstOrderLifetimeTransformSpec
