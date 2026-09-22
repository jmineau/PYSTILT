Configuration
=============

.. currentmodule:: stilt.config

Everything that can go in ``config.yaml`` is a field of
:class:`stilt.config.ModelConfig`: named meteorology sources
(:class:`MetConfig`), named footprints (:class:`FootprintConfig`), the
``execution`` section, and the STILT and HYSPLIT settings. The tables below
are generated from the code, so they are always current.

.. tip::
  New to PYSTILT? The :doc:`configuration guide <../guides/configuration>`
  covers the handful of settings most projects need. This page lists all of
  them.

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


Geometry specifications
-----------------------

Declarative state geometries for ``FootprintConfig.geometry``; each has a
``build()`` returning a :class:`stilt.Mesh`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   FileGeometrySpec
   H3GeometrySpec
   WindowsGeometrySpec
