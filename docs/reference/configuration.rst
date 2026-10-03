Configuration
=============

.. currentmodule:: stilt.config

Everything you can write in ``config.yaml`` is a field of
:class:`ProjectConfig`. That covers the mets
(:class:`MetConfig`), the footprint settings (:class:`FootprintConfig`),
the ``variants``, the ``execution`` section, and the STILT and HYSPLIT
settings. Each page below lists its fields with their defaults and is
generated from the code.

.. tip::
  New to PYSTILT? The :doc:`configuration guide <../guides/configuration>`
  covers the few settings most projects need. This page lists all of them.

Config objects
--------------

A config is what you write. :class:`VariantConfig` is one declared
variant merged with the defaults. ``project.variants`` resolves each into a
:class:`stilt.Variant`, which adds what the file alone does not say: the
grid of a footprint given by a geometry, and the transport model build.
The ``PYSTILT_COMPUTE_ROOT`` environment variable sets the scratch
directory HYSPLIT runs in. It never changes a result.

.. autosummary::
   :toctree: _api
   :nosignatures:

   ProjectConfig
   VariantConfig
   MetConfig
   FootprintConfig
   Bounds
   Grid

Parameters
----------

The settings that shape a run's particles belong to the transport model,
whose ``config.yaml`` name is the ``model`` field (``hysplit`` unless set).
They are flat, top-level keys. HYSPLIT's are in
:class:`stilt.transport.hysplit.HysplitConfig`, under HYSPLIT's names (see
:doc:`hysplit`). A variant that names another ``model`` gives that model's
parameters itself and inherits only the met and the footprint settings.


Geometry specifications
-----------------------

The kinds of area you can name in the ``geometry`` footprint setting.
:meth:`stilt.Mesh.from_spec` reads one into a :class:`stilt.Mesh`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   FileGeometrySpec
   H3GeometrySpec
   WindowsGeometrySpec
