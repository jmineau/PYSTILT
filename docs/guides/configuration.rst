Configuration
=============

PYSTILT configuration is meant to describe one project: which meteorology files
to use, which footprints to create, and the common run controls for HYSPLIT.
The output file is ``config.yaml`` in the project root.

Use :doc:`../reference/configuration` when you need every field and default.
Use this guide when you are deciding what to write in ``config.yaml``.

Minimal project config
----------------------

Most projects start with one meteorology stream and one footprint:

.. code-block:: yaml

   mets:
     hrrr:
       directory: /data/arl/hrrr
       file_format: "%Y%m%d_%H"
       file_tres: 1h

   footprints:
     default:
       xmin: -114.0
       xmax: -111.0
       ymin: 39.0
       ymax: 42.0
       xres: 0.01
       yres: 0.01

   n_hours: -24
   numpar: 500
   skip_existing: true


What each section means
-----------------------

``mets``
   Named meteorology streams. The name, such as ``hrrr``, becomes part of each
   simulation ID. Each stream points at ARL files and defines how timestamps map
   to filenames.

``footprints``
   Named footprint products. The simple form shown above is shorthand for a
   ``FootprintConfig`` with an inline grid. You only need a nested ``grid`` key
   when you want to be explicit.

``n_hours`` and ``numpar``
   The most common run controls. ``n_hours`` is negative for backward runs and
   positive for forward runs. ``numpar`` controls particle count.

``execution``
   Optional executor settings for Slurm or Kubernetes. Leave this out for local
   runs.

``skip_existing``
   Whether ``stilt run`` should avoid rerunning simulations whose required
   outputs are already complete.

Footprint grid shorthand
------------------------

The starter YAML uses the short form:

.. code-block:: yaml

   footprints:
     default:
       xmin: -114.0
       xmax: -111.0
       ymin: 39.0
       ymax: 42.0
       xres: 0.01
       yres: 0.01

This is equivalent to:

.. code-block:: yaml

   footprints:
     default:
       grid:
         xmin: -114.0
         xmax: -111.0
         ymin: 39.0
         ymax: 42.0
         xres: 0.01
         yres: 0.01

Use the nested form if it reads better for your workflow or if you are
generating config files programmatically.

Multiple outputs
----------------

You can define more than one named footprint:

.. code-block:: yaml

   footprints:
     near_field:
       xmin: -114.0
       xmax: -111.0
       ymin: 39.0
       ymax: 42.0
       xres: 0.01
       yres: 0.01

     regional:
       xmin: -125.0
       xmax: -100.0
       ymin: 30.0
       ymax: 50.0
       xres: 0.1
       yres: 0.1
       smooth_factor: 1.0

Each named footprint is tracked separately: a simulation is complete only once
every configured footprint exists for it.

Footprints for a state geometry
-------------------------------

When the footprint exists to feed an inversion whose state lives on a
non-rectilinear geometry (a shapefile, H3 hexagons, point-source windows),
name that geometry instead of the raster and let PYSTILT derive a raster
fine enough to resolve it (:meth:`stilt.Grid.from_geometry`):

.. code-block:: yaml

   footprints:
     counties:
       geometry:
         kind: file          # shapefile / GeoPackage / GeoJSON (needs geopandas)
         path: counties.shp
         ids: NAME           # attribute column used as cell ids
     hexes:
       geometry:
         kind: h3            # needs the h3 package
         resolution: 8
         bounds: {xmin: -112.3, xmax: -111.6, ymin: 40.4, ymax: 41.0}
       cells_per_target: 4   # native cells across the smallest hexagon (default)
     sources:
       geometry:
         kind: windows
         coords: [[-111.97, 40.515], [-112.015, 40.779]]
         size: 0.01
         ids: [landfill, wwtp]

The derived ``grid`` is written back into the config, so the geometry object
is never needed to read a stored footprint.  Give both ``grid`` and
``geometry`` to pin the raster explicitly; ``geometry`` is then kept as a
record and ``config.geometry.build()`` returns the :class:`stilt.Mesh` to
aggregate onto.  A content hash of the built geometry (``geometry_hash``) is
stored with the config and in each footprint file; ``Footprint.aggregate``
warns if the mesh it is handed no longer matches, which catches a shapefile
edited after the footprints were computed.

.. note::

   :class:`stilt.Zones` (super-cells) have no YAML form yet; build them in
   code with ``Zones.from_labels(base, labels)``.  A ``kind: zones`` spec
   would need a label source (an attribute column, a CSV keyed by cell id, or
   polygons assigned by cell centre) and will be added once a project needs
   its zoning to live in the config.

Execution examples
------------------

Local execution does not need an ``execution`` section. For Slurm, add one:

.. code-block:: yaml

   execution:
     backend: slurm
     account: lin-group
     partition: lin
     time: "02:00:00"
     memory: 8G

Executor-specific fields are passed to the configured backend. Keep project
science controls, such as ``numpar`` and footprint grids, outside this section.

Python equivalent
-----------------

The same configuration can be built from Python:

.. code-block:: python

   import stilt

   config = stilt.ModelConfig(
       mets={
           "hrrr": stilt.MetConfig(
               directory="/data/arl/hrrr",
               file_format="%Y%m%d_%H",
               file_tres="1h",
           )
       },
       footprints={
           "default": stilt.FootprintConfig(
               grid=stilt.Grid(
                   xmin=-114.0,
                   xmax=-111.0,
                   ymin=39.0,
                   ymax=42.0,
                   xres=0.01,
                   yres=0.01,
               )
           )
       },
       n_hours=-24,
       numpar=500,
       skip_existing=True,
   )

Advanced parameters
-------------------

PYSTILT exposes lower-level HYSPLIT/STILT parameters for compatibility and
experimentation. Keep them out of starter configs unless you know why you are
changing them.

Unknown keys are rejected when YAML is loaded.
