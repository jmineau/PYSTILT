Satellite And Column Observations
=================================

A single tower measurement becomes a receptor directly. Satellite soundings
and column retrievals (TCCON, EM27/SUN) need more steps: choosing which
soundings to run, turning each into the right receptor, and weighting each
one's particles with its own averaging kernel afterwards.
``stilt.observations`` and ``stilt.transforms`` provide the pieces for that,
following X-STILT, as small functions that work on the table your product
reader produces.

There is no observation object to fill in. A Level 2 file is already a
table with one row per sounding, so keep it as a :class:`pandas.DataFrame`
and hand PYSTILT the columns it needs.

The four steps
--------------

1. **Read** your product into a DataFrame with one row per sounding: time,
   longitude, latitude, the retrieval's averaging kernel, and whatever else
   you need (surface altitude, viewing angles, pixel corners, quality
   flags). Product readers live outside PYSTILT.
2. **Select** which soundings to run with pandas for quality flags and time
   windows, :func:`~stilt.observations.select_observations_spatial` for
   X-STILT's near-field plus background sampling, and
   :func:`~stilt.observations.group_by_overpass` to label each row with its
   overpass.
3. **Build receptors**, one per row, with the :class:`~stilt.Receptor`
   classes: :class:`~stilt.ColumnReceptor` for a nadir column,
   :meth:`Receptor.from_points <stilt.Receptor.from_points>` over
   :func:`~stilt.observations.slant_points` for a slant path
   (:doc:`../guides/slant_columns`), and
   :func:`~stilt.observations.jitter_points` for several receptors across
   one large pixel.
4. **Weight** with particle transforms (:doc:`transforms`). Pressure
   weighting is derived from the particles and is the same for every
   receptor. The averaging kernel differs per sounding, so write the kernels
   to a table in the project with
   :func:`~stilt.transforms.averaging_kernel_table` and point the footprint's
   ``averaging_kernel`` transform at it. Every runner, including Slurm
   arrays, then applies the right kernel to the right receptor.

A worked example
----------------

.. code-block:: python

   import pandas as pd
   import stilt
   from stilt import ColumnReceptor
   from stilt.observations import group_by_overpass, select_observations_spatial
   from stilt.transforms import averaging_kernel_table

   model = stilt.Model(project="./xch4")

   # 1. read: one row per sounding, kernels as arrays in two columns
   df = read_my_product(path)          # your reader
   df = df[df.qa_value > 0.5]           # quality flags are a pandas filter

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

with the footprint declared once in ``config.yaml``:

.. code-block:: yaml

   footprints:
     column:
       grid: slv
       transforms:
         - kind: averaging_kernel
           table: kernels.parquet
           coordinate: pres
         - kind: pressure_weighting

The kernel table is an input file beside ``receptors.csv``: one ``receptor``
id per sounding and one ``level`` / ``value`` row per kernel point. It can be
Parquet or CSV, and the ``table`` path is relative to the project root, so a
Slurm or Kubernetes worker that rebuilds the model from the project finds
it. A receptor with no row in the table is an error, never silently
unweighted.

Kernels on pressure levels
--------------------------

Most satellite products give the kernel on the retrieval's own pressure
grid, which differs per sounding because it hangs off the surface pressure.
Pass those pressures as ``levels`` and set ``coordinate: pres``; the
transform reads each particle's release pressure, so nothing has to be
converted to height. OCO-2 kernels are values at levels, TROPOMI kernels are
layer means: for a layer product pass the layer midpoints. A ground-based
EM27 kernel is tabulated by solar zenith angle on a fixed altitude grid; pick
the column for each window's zenith angle and pass the shared grid once as
``levels``.

Adding your own instrument
--------------------------

Nothing is registered or subclassed. A new instrument is a reader that
returns a DataFrame with the columns the steps above use, plus whatever
product-specific columns your analysis wants. The reader is where the
product's conventions live: unit conversions, quality flags, which variable
holds the kernel, rebuilding the pressure grid from surface pressure and
layer thickness. Everything after that is the same for every instrument.

What this layer does not do
---------------------------

It ships no product readers and no readers for background fields: a
mole-fraction field comes in as an xarray array
(:doc:`../guides/background`), a flux field the same way
(:doc:`../guides/transport_error`). The prior term of a column observation
operator (pressure weight times one minus kernel times the a priori
profile) belongs to the inversion, not the footprint.
