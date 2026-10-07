Satellite And Column Observations
=================================

A tower measurement becomes a receptor directly. Satellite soundings and
column retrievals (TCCON, EM27/SUN) take more steps. You choose which
soundings to run, turn each one into a receptor, and weight each one's
particles with its own averaging kernel. ``stilt.observations`` and
``stilt.transforms`` have small functions for each step, ported from
X-STILT. They work on the table your product reader returns.

There is no observation class to fill in. A Level 2 file is already a table
with one row per sounding. Keep it as a :class:`pandas.DataFrame` and pass
PYSTILT the columns it needs.

The four steps
--------------

1. **Read** the product into a DataFrame with one row per sounding. It
   needs the time, longitude, latitude, and averaging kernel, plus whatever
   else you use (surface altitude, viewing angles, pixel corners, quality
   flags). :doc:`../guides/readers` has readers for TROPOMI, OCO-2, TCCON,
   and GGG files, and lists the columns a reader should return.
2. **Select** the soundings to run. Filter on quality flags and time with
   pandas. :func:`~stilt.observations.select_observations_spatial` does
   X-STILT's near-field plus background sampling.
   :func:`~stilt.observations.group_by_overpass` labels each row with its
   overpass.
3. **Build receptors**, one per row. Use a :class:`~stilt.ColumnReceptor`
   for a nadir column, or :meth:`Receptor.from_points
   <stilt.Receptor.from_points>` with
   :func:`~stilt.observations.slant_points` for a slant path
   (:doc:`../guides/slant_columns`).
   :func:`~stilt.observations.jitter_points` spreads several receptors
   across one large pixel.
4. **Weight** the particles with transforms (:doc:`transforms`). Pressure
   weighting is worked out from the particles, so every receptor uses the
   same setting. The averaging kernel differs per sounding. Write the
   kernels to a table in the project with
   :func:`~stilt.transforms.averaging_kernel_table` and point the
   ``averaging_kernel`` transform at it. Every way of running the model,
   Slurm included, then applies the right kernel to each receptor.

A worked example
----------------

.. code-block:: python

   import pandas as pd
   import stilt
   from stilt import ColumnReceptor
   from stilt.observations import group_by_overpass, read_tropomi_ch4, select_observations_spatial
   from stilt.transforms import averaging_kernel_table

   model = stilt.Model(project="./xch4")

   # 1. read: one row per sounding, kernels as arrays in two columns
   df = read_tropomi_ch4(path)          # or read_oco2, read_tccon, or your own
   df = df[df.good]                     # quality flags are a pandas filter

   # 2. select: label overpasses, then thin each one to X-STILT's grid
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

   # 3. build: one receptor per row
   receptors = [ColumnReceptor(r.time, r.longitude, r.latitude, 0, 3000) for r in df.itertuples()]
   model.register(receptors=receptors)

   # 4. weight: each sounding's kernel, keyed by its receptor, in the project
   table = averaging_kernel_table(receptors, levels=df.ak_pressure, values=df.ak)
   table.to_parquet(model.project.directory / "kernels.parquet")

   model.run()

The footprint transforms go in ``config.yaml``:

.. code-block:: yaml

   transforms:
     - kind: averaging_kernel
       table: kernels.parquet
       coordinate: pres
     - kind: pressure_weighting

The kernel table is an input file, like ``receptors.csv``. It has a
``receptor`` column with the receptor id, and one row per kernel point with
its ``level`` and ``value``. It can be Parquet or CSV. The ``table`` path is
relative to the project root, so a Slurm or Kubernetes worker finds it too.
A receptor with no rows in the table raises an error.

Kernels on pressure levels
--------------------------

Most satellite products give the kernel on the retrieval's own pressure
grid. The grid differs per sounding because it starts at the surface
pressure. Pass those pressures as ``levels`` and set ``coordinate: pres``.
The transform then uses each particle's pressure at its first output step,
so nothing has to be converted to height.

OCO-2 kernels are values at levels. TROPOMI kernels are layer averages, so
pass the layer midpoints (the TROPOMI reader's ``ak_pressure`` already
holds them). GGG kernels are on a fixed pressure grid per site, which
:func:`~stilt.observations.read_ggg_netcdf` returns as ``ak_pressure``.

A PROFFAST EM27/SUN kernel is tabulated by solar zenith angle on a fixed
altitude grid. Pick the kernel for each window's zenith angle and pass the
shared grid once as ``levels``. Give the altitudes in the receptor's
vertical reference, because the default coordinate is the release height
``xhgt``.

Adding your own instrument
--------------------------

A new instrument needs only a reader, a function that returns a DataFrame
with the columns the steps above use. Nothing is registered or subclassed.
Add any product-specific columns your analysis needs. The reader handles
the product's conventions, such as unit conversions, quality flags, which
variable holds the kernel, and how to rebuild the pressure grid. The steps
after reading are the same for every instrument. :doc:`../guides/readers`
lists the columns and says which module to copy.

What this layer does not do
---------------------------

It has no readers for background or flux fields. A background mole-fraction
field comes in as an xarray array (:doc:`../guides/background`). The
background can also come from the swath itself, beside a plume from a
forward run (:doc:`../guides/plume_background`). A flux field also comes in
as an xarray array (:doc:`../guides/transport_error`).

The prior term of a column observation operator is left to the inversion.
That term is the sum over levels of pressure weight × (1 − kernel) × prior
profile.
