Core Objects
============

.. currentmodule:: stilt

The core reference centers on the project model, receptors, and the simulation
objects returned by transport runs.

Project interface
-----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Model

Receptor objects
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Receptor
   PointReceptor
   ColumnReceptor
   MultiPointReceptor
   ReceptorID
   LocationID
   read_receptors

Simulation objects
------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Simulation
   SimID
   Trajectories
   Footprint

Collections
-----------

``model.simulations`` is a :class:`~stilt.collections.SimulationCollection`
over receptors × variants; :meth:`~stilt.collections.SimulationCollection.sel`
narrows it and ``.trajectories`` / ``.footprint`` give an
:class:`~stilt.collections.OutputCollection` of one output.

.. currentmodule:: stilt.collections

.. autosummary::
   :toctree: _api
   :nosignatures:

   SimulationCollection
   OutputCollection
   ReceptorCollection

.. currentmodule:: stilt

Spatial geometries
------------------

The state geometry a footprint is aggregated onto (see
:meth:`Footprint.aggregate`).  :class:`Grid` doubles as a rectilinear
geometry and is documented under :doc:`configuration`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   Mesh
   Zones
   Geometry
   SpatialTarget

Overlap weights between a footprint raster and a geometry are built once and
cached; these helpers are in :mod:`stilt.geometry`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   geometry.overlap_weights
   geometry.check_resolution
   geometry.same_crs
   geometry.is_longlat_crs

Flux fields
-----------

Sampling a surface flux field under a footprint (:meth:`Footprint.enhancement`)
or along particles, in :mod:`stilt.flux`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   flux.sample_flux
   flux.particle_enhancement
   flux.horizontal_dims
   flux.nearest_cell
