Configuration
=============

.. currentmodule:: stilt

Everything you can write in ``config.yaml`` is a field of
:class:`ProjectConfig`. That covers the mets (for HYSPLIT,
:class:`~stilt.transport.hysplit.MetConfig`), the footprint settings
(:class:`FootprintConfig`), the ``variants``, the ``execution`` section
(:class:`ExecutionConfig`), and the STILT and HYSPLIT settings. Each class
lives next to the code that uses it (``stilt.footprint.config``,
``stilt.execution.config``, ``stilt.transport.hysplit``), and all but the
transport model's are importable from ``stilt``.
Each page below lists its fields with their defaults and is generated from
the code.

.. tip::
  New to PYSTILT? The :doc:`Projects and variants guide <../guides/projects>`
  covers the few settings most projects need. This page lists all of them.

Every key
---------

These tables are generated from the code, so they always match the
installed version.

Top level
~~~~~~~~~

A project's own keys. Every key of the transport model and
the footprint below is also top-level, as the defaults each variant starts
from, and a variant may override any of them.

.. config-model:: stilt.config.ProjectConfig
   :declared-only:

HYSPLIT
~~~~~~~

The transport model's parameters, top-level keys of
``config.yaml`` (:doc:`hysplit`).

.. config-model:: stilt.transport.hysplit.HysplitConfig

Footprint
~~~~~~~~~

The footprint's settings, top-level keys too.

.. config-model:: stilt.footprint.config.FootprintConfig

A met
~~~~~

The keys of one entry under ``mets:``, for HYSPLIT
(:doc:`../guides/meteorology`).

.. config-model:: stilt.transport.hysplit.MetConfig

Execution
~~~~~~~~~

The keys under ``execution:`` (:doc:`../guides/running`).

.. config-model:: stilt.execution.config.ExecutionConfig

Config objects
--------------

A config is what you write. ``project.variants`` resolves each declared
variant into a :class:`stilt.config.Variant`: its transport settings checked by the
model's config class, its realizations expanded, and what the file alone
does not say, the grid of a footprint given by a geometry and the transport
model build.
The ``PYSTILT_COMPUTE_ROOT`` environment variable sets the scratch
directory HYSPLIT runs in. It never changes a result.

.. autosummary::
   :toctree: _api
   :nosignatures:

   ProjectConfig
   FootprintConfig
   ExecutionConfig
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

   footprint.config.FileGeometrySpec
   footprint.config.H3GeometrySpec
   footprint.config.WindowsGeometrySpec
