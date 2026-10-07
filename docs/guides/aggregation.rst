Summing Footprints: Areas And The Jacobian
==========================================

Footprints are calculated on the regular grid in your config, as in
STILT-R. An inversion or an emissions comparison needs the influence of
other areas: counties, hexagons, sectors, or small windows around point
sources. This page sums a footprint over such areas, and many footprints
at once into a Jacobian.

One footprint over areas
------------------------

``foot.stilt.aggregate`` adds up the footprint cells in each area, and the
hours in each time bin:

.. code-block:: python

   import pandas as pd
   import stilt

   hours = foot.indexes["time"]   # the start of each footprint hour
   bins = pd.interval_range(
       start=hours.min(), periods=len(hours), freq="1h", closed="left"
   )
   state = stilt.Grid(xmin=-112.3, xmax=-111.6, ymin=40.4, ymax=41.0,
                      xres=0.02, yres=0.02)
   by_cell = foot.stilt.aggregate(state, time_bins=bins)   # index == state.index

The result is a DataFrame with one row per area and one column per time
bin, labelled by the start of the bin. Footprint times are the start of
each hour, so the bins must be closed on the left (``closed="left"``). A footprint cell that straddles two
areas is split between them by area, so the total influence is kept. This
is what you want before multiplying by emissions. Influence that falls
outside every area is dropped.

The target can be:

- a :class:`stilt.Grid`. Rows follow ``grid.index``, one ``(lon, lat)``
  pair per cell.
- a :class:`stilt.Mesh` of polygons with ids: a shapefile
  (``Mesh.from_file``), H3 hexagons (``Mesh.from_h3``), or windows around
  points (``Mesh.from_windows``). Rows are the polygon ids.
- a :class:`stilt.Zones`, which merges the cells of a grid or mesh into
  larger groups by label.

For cells given some other way, such as an xarray grid or a list of cell
centres, build the :class:`stilt.Grid` they lie on and select the rows you
need from the result.

.. code-block:: python

   sources = stilt.Mesh.from_windows(
       [(-111.97, 40.515), (-112.015, 40.779)], 0.01, ids=["landfill", "wwtp"]
   )
   by_source = foot.stilt.aggregate(sources, time_bins=bins)   # index == ["landfill", "wwtp"]

   counties = stilt.Mesh.from_file("counties.shp", ids="NAME")
   by_county = foot.stilt.aggregate(counties, time_bins=bins)

   sectors = stilt.Zones.from_labels(state, labels)       # one label per cell of state
   by_sector = foot.stilt.aggregate(sectors, time_bins=bins)

Polygons in another coordinate system are reprojected onto the footprint
grid. The overlaps between the footprint grid and your areas are worked out
once and reused, so adding up thousands of footprints is fast. Polygon
overlaps use shapely. If
`exactextract <https://github.com/isciences/exactextract>`_ is installed,
as with the ``geometry`` extra, PYSTILT uses it instead. It gives the same
result and is about a hundred times faster on large grids.

The footprint grid must be fine enough to resolve your areas.
``aggregate`` warns when the smallest area spans fewer than two footprint
cells. In that case, calculate the footprint again from its particles on a
finer grid:

.. code-block:: python

   hexes = stilt.Mesh.from_h3(8, bounds=state)
   grid = hexes.to_grid(cells_per_target=4)
   fine = sim.calc_footprint(grid=grid)
   by_hex = fine.stilt.aggregate(hexes, time_bins=bins)

``to_grid`` picks a grid that covers the areas with at least four
cells across the smallest one. ``sim.calc_footprint`` applies the
variant's particle transforms, as the stored footprint did, and does not
overwrite the stored file.

Many footprints: the Jacobian
-----------------------------

An inversion needs every footprint summed onto the flux cells, one row
per receptor: the Jacobian. :meth:`~stilt.Project.jacobian` does this for
a selection of simulations (:doc:`outputs`), all of one variant:

.. code-block:: python

   sims = project.simulations
   july = sims[(sims.variant == "hrrr") & (sims.time.dt.month == 7)]
   bins = pd.interval_range(
       pd.Timestamp("2023-06-30"), pd.Timestamp("2023-08-01"), freq="6h", closed="left"
   )

   H = project.jacobian(july, counties, bins)
   H.to_xarray()             # receptor, time, cell
   H.to_frame(sparse=True)   # receptors by (time, cell), pandas sparse columns

It reads the footprints in batches, in several threads, and returns a
sparse matrix, so its size does not depend on the grid and memory stays
at a few batches of footprints. ``H.to_xarray()`` holds a sparse array
from the optional ``sparse`` package (``pip install pystilt[sparse]``), or
a NumPy array with ``dense=True``. Receptors whose footprint is empty are
in ``H.empty``, and those not run yet in ``H.missing``.

The columns are the time bin and then the target's cells, as named levels:
``time, cell`` for zones or a mesh, and ``time, lon, lat`` for a grid
(``time, x, y`` when it is projected). For a grid,
``H.to_xarray().unstack("cell")`` has ``lon`` and ``lat`` dimensions.
