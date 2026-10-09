Tutorial: A Satellite Column, Start To Finish
=============================================

A tower measurement becomes a receptor directly. A satellite sounding
takes a few more steps: choose which soundings to run, make a receptor for
each, and weight each one's particles with its own averaging kernel. This
tutorial goes from a TROPOMI methane orbit over Salt Lake City to modeled
and observed XCH4 enhancements. The same steps work for OCO-2, TCCON, and
EM27/SUN; only the reader changes.

What you'll learn
-----------------

- how to read a product and choose the soundings to run
- how to make a receptor and a kernel table for each sounding
- how to weight the particles with the kernel and the pressure weights
- how to compare modeled and observed enhancements

Starting point
--------------

A project made with ``stilt init`` or :meth:`stilt.Project.init`, with HRRR
meteorology and a footprint grid over the valley. A Level 2 file becomes
a table with one row per sounding, and PYSTILT works on that table.

Step 1: Read the soundings
--------------------------

:func:`~stilt.observations.read_tropomi_ch4` reads an orbit into a table
with one row per pixel. Its columns are the same for every reader
(:doc:`../guides/readers`). Quality flags are a pandas filter:

.. code-block:: python

   import pandas as pd

   import stilt
   from stilt.observations import read_tropomi_ch4

   project = stilt.Project("./xch4")

   df = read_tropomi_ch4(
       "S5P_OFFL_L2__CH4____20231019T192749_20231019T210918_31176_03_020500_20231021T113213.nc",
       lon_range=(-113.5, -110.5), lat_range=(39.5, 42.0),
   )
   df = df[df.good]

Step 2: Choose the soundings to run
-----------------------------------

A swath over a city has more pixels than you need. X-STILT's sampling keeps
every pixel near the city and a thinner grid of pixels farther out, for
the background. :func:`~stilt.observations.group_by_overpass` labels each
overpass, and :func:`~stilt.observations.select_observations_spatial`
thins each one:

.. code-block:: python

   from stilt.observations import group_by_overpass, select_observations_spatial

   df["overpass"] = group_by_overpass(df["time"])
   chosen = []
   for _, scene in df.groupby("overpass"):
       keep = select_observations_spatial(
           scene.longitude, scene.latitude,
           site_longitude=-111.85, site_latitude=40.77,
           near_field_dlon=0.3, near_field_dlat=0.3,
           near_field_cols=20, near_field_rows=20,
           background_cols=10, background_rows=10,
           domain_lon_range=(-113.5, -110.5), domain_lat_range=(39.5, 42.0),
       )
       chosen.append(scene.iloc[keep])
   df = pd.concat(chosen)

Step 3: Make a receptor and a kernel for each sounding
------------------------------------------------------

:func:`~stilt.observations.receptors_from_soundings` makes one receptor per
row and the table of their averaging kernels. ``"slant"`` releases the
particles along the satellite's line of sight, at the retrieval's own
levels up to 3 km above the surface (:doc:`../guides/slant_columns`).
``"column"`` releases them straight up instead.

.. code-block:: python

   from stilt.observations import receptors_from_soundings

   receptors, kernels = receptors_from_soundings(df, "slant", top=3000)
   project.add_receptors(receptors)
   project.add_table("kernels", kernels)    # tables/kernels.parquet

Each receptor keeps its ``sounding_id`` as a column of ``receptors.csv``,
so the results join back to the soundings. The kernel table is an input
like ``receptors.csv``: a ``receptor`` column, and one row per kernel
level with its ``level`` and ``value``. Adding it again leaves the kernels
already there as they were.

Step 4: Weight the particles
----------------------------

Two transforms go in ``config.yaml`` (:doc:`../guides/transforms`). The
averaging kernel differs per sounding, so it reads the kernel table by
name. Pressure weighting is worked out from the particles, so every
receptor uses the same setting:

.. code-block:: yaml

   transforms:
     - kind: averaging_kernel
       table: kernels
       coordinate: pres
     - kind: pressure_weighting

TROPOMI gives its kernel on pressures, so ``coordinate: pres`` matches it
to each particle's pressure at its first output step, with nothing to
convert to height. TROPOMI's kernels are layer averages, and the reader's
``ak_pressure`` holds the layer midpoints. OCO-2 kernels are values at
levels, and GGG kernels are on a fixed pressure grid per site. A PROFFAST
EM27/SUN kernel is on a fixed altitude grid: give the altitudes in the
receptor's vertical reference and leave ``coordinate`` at its default,
the release height.

Step 5: Run
-----------

.. code-block:: bash

   stilt run ./xch4          # or stilt submit ./xch4 on a cluster

Every way of running, Slurm included, applies each receptor's own kernel,
because the table is part of the project.

Step 6: Compare enhancements
----------------------------

The footprint, weighted by the kernel and the pressure weights, gives the
modeled enhancement for a flux field. The observed enhancement is the
retrieved value minus a background. Here the background is the median of
the soundings outside the plume of a forward run
(:doc:`../guides/plume_background`); a mole-fraction field also works
(:doc:`../guides/background`).

``plume`` is the outline of the forward run's particles at the overpass
(:func:`~stilt.observations.plume_polygon`, as in that guide).

.. code-block:: python

   import xarray as xr
   from stilt.observations import plume_background

   flux = xr.open_dataarray("ch4_flux.nc")     # µmol m⁻² s⁻¹, on the footprint grid or coarser
   bg = plume_background(df["longitude"], df["latitude"], df["value"], plume)
   background = bg.value                       # ppb

   soundings = df.set_index("sounding_id")
   rows = []
   for rec in project.receptors.itertuples():
       foot = project.simulation(rec.receptor, "hrrr").footprint
       if foot is None:      # no particle reached the grid
           continue
       rows.append({
           "sounding_id": rec.sounding_id,
           "modeled": 1000 * float(foot.stilt.enhancement(flux).sum()),   # ppm to ppb
           "observed": soundings.loc[rec.sounding_id, "value"] - background,
       })
   compare = pd.DataFrame(rows).set_index("sounding_id")

A flux in µmol m⁻² s⁻¹ times a footprint gives ppm, and XCH4 is in ppb.
The enhancement adds no prior term: the background was retrieved too, so
the prior's share cancels in the difference.

Comparing absolute columns
~~~~~~~~~~~~~~~~~~~~~~~~~~

To compare the retrieved value itself, with a background from a model
field, add the prior the retrieval carries where it is not sensitive.
:func:`~stilt.observations.modelled_column` does that:

.. code-block:: python

   import numpy as np
   from stilt.observations import modelled_column

   cams = xr.open_dataarray("cams_ch4.nc").rename(level="pres")   # (time, pres, lat, lon), ppb

   sounding_id = compare.index[0]              # any sounding with a footprint
   rec = project.receptors.set_index("sounding_id").loc[sounding_id]
   sim = project.simulation(rec.receptor, "hrrr")
   row = soundings.loc[sounding_id]

   p = row.pressure_levels                     # layer boundaries, hPa, surface first
   w = -np.diff(p) / (p[0] - p[-1])            # each layer's share of the column
   up = row.altitude_levels[:-1] >= row.surface_altitude + 3000   # layers above the receptor's top
   profile = cams.sel(time=row.time, lat=row.latitude, lon=row.longitude, method="nearest")
   above = float(np.sum((w * row.ak * profile.interp(pres=row.ak_pressure).values)[up]))

   column = modelled_column(
       1000 * float(sim.footprint.stilt.enhancement(flux).sum()),
       sim.background(cams).value + above,
       ak=row.ak, prior=row.apriori, pressure_weight=w,
   )

``sim.background(cams)`` covers the receptor's levels only, up to 3 km.
``above`` is the rest of the column: the field on the retrieval's layers
above the receptor's top, weighted by the kernel and the pressure weights.
That air saw no fluxes in the domain, so the field alone gives it
(:doc:`../guides/background`).

Your own instrument
-------------------

A new instrument needs only a reader: a function that returns a table
with the columns of
:data:`~stilt.observations.readers.schema.SOUNDING_SCHEMA`. The reader
handles the product's conventions,
such as unit conversions, quality flags, which variable holds the kernel,
and how to rebuild the pressure grid. Every step after reading is the same.
:doc:`../guides/readers` says which module to copy.
