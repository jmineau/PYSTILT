Configuration
=============

.. currentmodule:: stilt.config

Everything you can write in ``config.yaml`` is a field of
:class:`ModelConfig`. That covers the mets
(:class:`MetConfig`), the footprint settings (:class:`FootprintConfig`),
the ``variants``, the ``execution`` section, and the STILT and HYSPLIT
settings. Each page below lists its fields with their defaults and is
generated from the code.

.. tip::
  New to PYSTILT? The :doc:`configuration guide <../guides/configuration>`
  covers the few settings most projects need. This page lists all of them.

Config objects
--------------

:class:`VariantConfig` is one variant with all of its settings filled in,
as PYSTILT runs it. :class:`TransportSettings` is the part of a variant
that decides its particles (its transport fields, the met's
:class:`MetSettings`, and the :class:`EngineInfo`), whose hash names the
run in the output directory. :class:`FootprintConfig` holds the footprint
settings, and each footprint keeps the ones it was calculated with.
:class:`RuntimeSettings` reads the
``PYSTILT_*`` environment variables, which set where work runs and never
change a result.

.. autosummary::
   :toctree: _api
   :nosignatures:

   ModelConfig
   VariantConfig
   TransportSettings
   EngineInfo
   MetConfig
   MetSettings
   FootprintConfig
   Bounds
   Grid
   RuntimeSettings

Parameters
----------

The STILT and HYSPLIT settings, in three groups. :class:`STILTParams`
combines them.

.. autosummary::
   :toctree: _api
   :nosignatures:

   ModelParams
   TransportParams
   ErrorParams
   STILTParams


Geometry specifications
-----------------------

The kinds of area you can name in the ``geometry`` footprint setting. Each
has a ``build()`` method that returns a :class:`stilt.Mesh`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   FileGeometrySpec
   H3GeometrySpec
   WindowsGeometrySpec
