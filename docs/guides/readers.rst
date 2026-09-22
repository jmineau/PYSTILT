Reading Retrieval Products
==========================

A column retrieval arrives as a product file: a TROPOMI orbit, an OCO-2
Lite file, a TCCON site file. :mod:`stilt.observations.readers` reads
those into a table of soundings with one row per sounding and the same
columns whichever instrument they came from, so the rest of the workflow
(:doc:`../advanced/observations`) does not know or care which product it
started from. Each instrument has its own module, and a product not covered
here is one more module of the same shape.

.. code-block:: python

   from stilt.observations import read_tropomi_ch4

   df = read_tropomi_ch4(
       "S5P_OFFL_L2__CH4____20231019T192749_20231019T210918_31176_03_020500_20231021T113213.nc",
       lon_range=(-113.5, -110.5), lat_range=(39.5, 42.0),
   )
   df = df[df.good]

The readers
-----------

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Reader
     - Files
     - Checked against
   * - :func:`~stilt.observations.read_tropomi_ch4`
     - Operational Sentinel-5P L2 CH4 orbits (``S5P_*_L2__CH4___``, v2.x)
       and the TROPOMI+GOSAT blended files (``S5P_BLND_L2__CH4___``)
     - a 2023 operational orbit (processor 2.5) and a 2018 blended file
   * - :func:`~stilt.observations.read_oco2`
     - OCO-2 and OCO-3 L2 Lite XCO2 files (``oco2_LtCO2_*.nc4``,
       ``oco3_LtCO2_*.nc4``, Lite FP v10/v11)
     - the documented v11 layout only; Lite files sit behind an Earthdata
       login, so no real granule has been run yet
   * - :func:`~stilt.observations.read_tccon`
     - TCCON GGG2020 public site files from CaltechDATA
       (``*.public.nc``, ``*.public.qc.nc``)
     - the Indianapolis GGG2020.R0 file

The satellite readers take ``lon_range`` and ``lat_range`` to keep only
the pixels in a box, which matters on a whole orbit; the TCCON reader takes
a ``species`` (``xco2``, ``xch4``, ``xco``, ...) and a ``time_range``, which
matters on a multi-year site file. Nothing else is filtered: quality flags
become the ``good`` column and the choice is yours.

The columns
-----------

Every reader fills these. Vertical arrays run from the surface upward,
pressures are hPa, altitudes are metres above mean sea level, and azimuths
are degrees clockwise from north giving the bearing toward the instrument
or the sun, which is what :func:`~stilt.observations.slant_points` takes.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Column
     - Meaning
   * - ``sounding_id``
     - A string that identifies the sounding within the product.
   * - ``time``
     - UTC time of the sounding, timezone-naive.
   * - ``longitude``, ``latitude``
     - Pixel centre or station location, degrees.
   * - ``surface_altitude``
     - Surface (or station) altitude, m MSL.
   * - ``surface_pressure``
     - Surface pressure the retrieval used, hPa.
   * - ``value``, ``uncertainty``
     - The column-average dry-air mole fraction and its one-sigma error, in
       ``units`` (``ppb`` or ``ppm``), for the gas named in ``species``.
   * - ``good``
     - The product's own recommended quality screen as a boolean.
   * - ``ak_pressure``, ``ak``
     - The averaging kernel and the pressures it is defined at, one array per
       row. Layer products (TROPOMI) give the layer midpoints. Pass them to
       :func:`~stilt.transforms.averaging_kernel_table` with
       ``coordinate: pres``.
   * - ``pressure_levels``
     - The retrieval's own pressure grid, for
       :func:`~stilt.observations.pressure_altitudes`.

and these where the product has them:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Column
     - Meaning
   * - ``zenith``, ``azimuth``
     - The viewing geometry for a slant path: sensor angles toward the
       satellite, solar angles for a ground-based spectrometer. The blended
       TROPOMI files carry none.
   * - ``solar_zenith``, ``solar_azimuth``
     - Solar angles alongside, for satellites.
   * - ``altitude_levels``
     - The retrieval's heights of ``pressure_levels``, m MSL, where the
       product provides them (TROPOMI, TCCON). Use these for a slant path
       rather than converting pressures.
   * - ``apriori``
     - The prior profile as a mole fraction in ``units`` on the kernel's
       levels; ``apriori_column`` the prior column value (OCO-2).
   * - ``longitude_bounds``, ``latitude_bounds``
     - Pixel corners, one array per row, for
       :func:`~stilt.observations.jitter_points`.

Product-specific columns keep product names (``qa_value``,
``quality_flag``, ``operation_mode``, ``xch4_uncorrected``, ...).

From a file to receptors
------------------------

The recipe in :doc:`../advanced/observations` becomes:

.. code-block:: python

   import stilt
   from stilt.observations import group_by_overpass, read_tropomi_ch4, slant_points
   from stilt.transforms import averaging_kernel_table

   model = stilt.Model(project="./xch4")

   df = read_tropomi_ch4(path, lon_range=(-113.5, -110.5), lat_range=(39.5, 42.0))
   df = df[df.good]
   df["overpass"] = group_by_overpass(df["time"])

   receptors = [
       stilt.Receptor.from_points(
           r.time,
           slant_points(
               r.longitude, r.latitude,
               r.altitude_levels[r.altitude_levels < r.surface_altitude + 3000],
               zenith=r.zenith, azimuth=r.azimuth,
           ),
           altitude_ref="msl",
       )
       for r in df.itertuples()
   ]
   model.register(receptors=receptors)

   averaging_kernel_table(receptors, levels=df.ak_pressure, values=df.ak).to_parquet(
       model.project.directory / "kernels.parquet"
   )
   model.run()

with ``averaging_kernel`` (``table: kernels.parquet``, ``coordinate: pres``)
and ``pressure_weighting`` in the footprint config. For a nadir column,
:class:`~stilt.ColumnReceptor` from ``surface_altitude`` replaces the slant
points. For OCO-2, whose file gives pressures but not heights,
:func:`~stilt.observations.pressure_altitudes` turns ``pressure_levels``
into the altitudes. A TCCON prior grid does not include the station itself
(its lowest level sits below ground and is dropped), so prepend
``surface_altitude`` to anchor the path at the instrument.

Adding an instrument
--------------------

Copy the module closest to your product under
``stilt/observations/readers/`` (``tropomi.py`` for a swath product,
``tccon.py`` for a ground station) and change what it reads: the reader is
where the product's conventions live, and the columns above are the
contract. Keep it a function that returns the table. Give it a small slice
of a real file under ``tests/data/products`` and a test that checks the
columns, the units, and that the vertical arrays start at the surface;
``tests/data/products/make_samples.py`` shows how the existing slices were
cut. EM27/SUN (PROFFAST), MethaneAIR and MethaneSAT readers are welcome
this way; the maintainers have no files for them.
