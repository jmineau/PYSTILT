Core Objects
============

.. currentmodule:: stilt

The project, its receptors, the simulations it runs, and their outputs.

Project interface
-----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Project

Receptor objects
----------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Receptor
   PointReceptor
   ColumnReceptor
   MultiPointReceptor
   read_receptors
   receptors.write_receptors
   receptors.receptors_to_frame
   receptors.receptors_from_frame
   receptors.receptor_rows
   receptors.receptor_from_rows
   receptors.read_receptor_frame
   receptors.parse_receptor_id

Simulation objects
------------------

.. autosummary::
   :toctree: _api
   :nosignatures:

   Simulation
   SimID

Particles and footprints
------------------------

A simulation's particles are a pandas DataFrame and its footprint an xarray
DataArray. PYSTILT's methods on them are under ``.stilt``
(:class:`~stilt.particles.ParticlesAccessor`,
:class:`~stilt.footprint.FootprintAccessor`). The functions below read and
write particle and footprint files without a project.

.. autosummary::
   :toctree: _api
   :nosignatures:

   read_particles
   particles_metadata
   write_particles
   read_footprint
   particles.prepare
   particles.ParticlesAccessor
   footprint.calculate
   footprint.FootprintAccessor

Simulation tables
-----------------

``project.receptors`` is a pandas DataFrame with one row per receptor.
``project.simulations`` is a :class:`Simulations`, one row per simulation.
Select its rows as in pandas, then ask the selection for its status or
results.

.. code-block:: python

   sims = project.simulations
   wbb = sims[(sims.variant == "hrrr") & (sims.site == "WBB")]
   feet = wbb.load_footprints()

.. autosummary::
   :toctree: _api
   :nosignatures:

   Simulations

.. currentmodule:: stilt

Spatial geometries
------------------

The cells ``foot.stilt.aggregate`` sums a footprint into, such as
polygons, hexagons, windows around point sources, or groups of grid cells.
A :class:`Grid` works as well. It is documented under :doc:`configuration`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   Mesh
   Zones
   Geometry

:mod:`stilt.spatial` also has the helpers that compute how much of each
footprint cell falls in each target cell. These overlap weights are cached,
so each geometry's weights are computed once per footprint grid.

.. autosummary::
   :toctree: _api
   :nosignatures:

   spatial.overlap_weights
   spatial.check_resolution
   spatial.same_crs
   spatial.is_longlat

Flux fields
-----------

:mod:`stilt.flux` looks up a surface flux field under a footprint (used by
``foot.stilt.enhancement``) or along particle paths.

.. autosummary::
   :toctree: _api
   :nosignatures:

   flux.sample_flux
   flux.sample_field
   flux.vertical_dim
   flux.particle_enhancement
   flux.horizontal_dims
   flux.nearest_cell
