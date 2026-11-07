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

Footprint grids
^^^^^^^^^^^^^^^

Longitudes count modulo 360, so a grid across 180° can be written
``xmin: 170``, ``xmax: 190`` or ``xmin: 170``, ``xmax: -170``, and the
footprint's longitudes are then 170 to 190. A grid spans at most 360°. A
particle cloud on both sides of 180° is measured as one cloud for the
smoothing. A grid that goes all the way around (``xmin: -180``,
``xmax: 180``, with cells that fit 360°) is a circle, so smoothing that
passes the east edge returns at the west edge. On a grid that ends at or
near 180°, particles just across 180° smooth onto the grid, as particles
just past any other edge do.

Projected grids (a ``crs`` such as ``EPSG:32612``) keep their bounds in
degrees and take ``xres`` and ``yres`` in the projection's units. The two
steps that depend on the cell size work in the grid's units. The smoothing
kernel's width is worked out in degrees and converted, so a footprint is
smoothed about as much on 1 km UTM cells as on 0.01° cells. The early-time
interpolation, which adds particle positions between output steps in the
first 100 minutes when the particles move more than a cell per step, runs
on 100 m UTM cells as it does on 0.001° cells. :ref:`stilt-r-parity` says
where STILT-R differs on these grids.

Time layers (``tres``) are recorded one way, so ``60min`` is recorded as
``1h`` and two spellings of one length share a settings folder. Each layer
holds the influence of the particles in its minutes, so the layers sum to
the same total footprint whatever ``tres`` is.

Loading ``config.yaml`` does not read a ``geometry`` file.
PYSTILT reads it once, when it works out the variants' settings, and picks
the grid then (:meth:`stilt.Mesh.to_grid`), so a variant with a geometry
cannot change only part of the grid (``grid: {xres: 0.1}``). A ``grid``
given with a ``geometry`` is used as given. Each footprint file stores a
code for the geometry it was made for, which is how
``foot.stilt.aggregate`` knows when a mesh no longer matches.

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

A config is what you write. ``project.variants`` turns each declared
variant into a :class:`stilt.config.Variant`: its transport settings
checked by the model's config class, and what the file alone does not
say, such as the grid of a footprint given by a geometry and the transport
model build. A variant with ``realizations: N`` stays one variant.

The ``PYSTILT_COMPUTE_ROOT`` environment variable sets the compute root,
where each run's workdir is made. It never changes a result, and projects
can share one.

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
parameters itself, and inherits the met, the footprint settings, and the
parameters PYSTILT reads itself (``n_hours``, ``seed``, ``hnf_plume``, and
``veght``).


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
