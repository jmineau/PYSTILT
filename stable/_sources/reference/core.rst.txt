Core Objects
============

.. currentmodule:: stilt

The model, its receptors, the simulations it runs, and their outputs.

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
of every receptor under every variant.
:meth:`~stilt.collections.SimulationCollection.sel` narrows it.
``.trajectories`` and ``.footprint`` give an
:class:`~stilt.collections.OutputCollection` for loading one kind of output.

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

The cells :meth:`Footprint.aggregate` sums a footprint into, such as
polygons, hexagons, windows around point sources, or groups of grid cells.
A :class:`Grid` works as well. It is documented under :doc:`configuration`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   Mesh
   Zones
   Geometry
   SpatialTarget

:mod:`stilt.geometry` also has the helpers that compute how much of each
footprint cell falls in each target cell. These overlap weights are cached,
so each geometry's weights are computed once per footprint grid.

.. autosummary::
   :toctree: _api
   :nosignatures:

   geometry.overlap_weights
   geometry.check_resolution
   geometry.same_crs
   geometry.is_longlat_crs

Flux fields
-----------

:mod:`stilt.flux` looks up a surface flux field under a footprint (used by
:meth:`Footprint.enhancement`) or along particle paths.

.. autosummary::
   :toctree: _api
   :nosignatures:

   flux.sample_flux
   flux.sample_field
   flux.vertical_dim
   flux.particle_enhancement
   flux.horizontal_dims
   flux.nearest_cell
