Reading Retrieval Products
==========================

A column retrieval comes as a product file, such as a TROPOMI orbit, an
OCO-2 Lite file, a TCCON site file, or a day of EM27/SUN spectra from GGG.
The readers in :mod:`stilt.observations` turn each of these into a table
with one row per sounding. The columns are the same for every instrument
(:data:`~stilt.observations.readers.schema.SOUNDING_SCHEMA`), so the rest of the workflow
(:doc:`../tutorials/satellite_column`) is too.

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
     - Tested on
   * - :func:`~stilt.observations.read_tropomi_ch4`
     - Operational Sentinel-5P L2 CH4 orbits (``S5P_*_L2__CH4___``, v2.x)
       and the TROPOMI+GOSAT blended files (``S5P_BLND_L2__CH4___``)
     - A 2023 operational orbit (processor 2.5) and a 2018 blended file.
   * - :func:`~stilt.observations.read_oco2`
     - OCO-2 and OCO-3 L2 Lite XCO2 files (``oco2_LtCO2_*.nc4``,
       ``oco3_LtCO2_*.nc4``, Lite FP v10/v11)
     - A synthetic file with the documented v11 layout. No real granule
       has been read yet.
   * - :func:`~stilt.observations.read_tccon`
     - TCCON GGG2020 public site files from CaltechDATA
       (``*.public.nc``, ``*.public.qc.nc``)
     - The Indianapolis GGG2020.R0 file.
   * - :func:`~stilt.observations.read_ggg_oof`
     - GGG2020 ``.oof`` files (``*.vav.ada.aia.oof``), one instrument-day
       each. This is how EGI delivers EM27/SUN retrievals.
     - Real EM27/SUN days. The committed test file is synthetic.
   * - :func:`~stilt.observations.read_ggg_netcdf`
     - GGG2020 netCDF files: the ``*.private.nc`` a GGG run writes, and the
       public files (``read_tccon`` calls this reader)
     - The public layout on a real TCCON file. The private layout only on
       a synthetic file.

The satellite readers take ``lon_range`` and ``lat_range`` to keep only the
pixels inside a box. Use them on a whole orbit. The GGG readers take a
``species`` (``xco2``, ``xch4``, ``xco``, ...), and the netCDF readers also
take a ``time_range`` for multi-year site files. The readers do not filter
anything else. Quality flags become the ``good`` column, and you decide
what to keep.

EM27/SUN
~~~~~~~~

GGG (through EGI) writes one ``.oof`` file per instrument-day and, from the
same run, one ``*.private.nc``. The ``.oof`` has the soundings but no
averaging kernel or prior, so :func:`~stilt.observations.read_ggg_oof`
leaves out the ``ak``, ``ak_pressure`` and ``pressure_levels`` columns. Take
the kernels from the private file with
:func:`~stilt.observations.read_ggg_netcdf`, or from a site kernel table
keyed by solar zenith angle. The :doc:`slant_columns` guide shows both.

To read a campaign as one table:

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   from stilt.observations import read_ggg_oof

   days = sorted(Path("EM27_oof/ha").glob("ha*.vav.ada.aia.oof"))
   df = pd.concat([read_ggg_oof(p, "xch4") for p in days], ignore_index=True)
   df = df[df.good]

There is no reader yet for EM27/SUN data processed with PROFFAST (the
COCCON pipeline).

The columns
-----------

Vertical arrays run from the surface upward. Pressures are in hPa and
altitudes in metres above mean sea level. Azimuths are degrees clockwise
from north, pointing from the ground toward the instrument or the sun. This
is the convention :func:`~stilt.observations.slant_points` uses.

Every reader returns these columns:

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
     - The column-average dry-air mole fraction and its one-sigma error.
   * - ``species``, ``units``
     - The gas (``xch4``, ``xco2``, ...) and the units of ``value``
       (``ppb`` or ``ppm``).
   * - ``good``
     - The product's recommended quality screen, as a boolean.

All readers except :func:`~stilt.observations.read_ggg_oof` also return:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Column
     - Meaning
   * - ``ak_pressure``, ``ak``
     - The averaging kernel and the pressures it is given at, one array per
       row. For a layer product (TROPOMI) the pressures are the layer
       midpoints. Pass both to
       :func:`~stilt.transforms.averaging_kernel_table` and set
       ``coordinate: pres`` on the ``averaging_kernel`` transform.
   * - ``pressure_levels``
     - The retrieval's own pressure grid, for
       :func:`~stilt.observations.pressure_altitudes`.

Some columns appear only when the product has them:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Column
     - Meaning
   * - ``zenith``, ``azimuth``
     - The viewing geometry for a slant path. For a satellite these are
       the sensor angles toward the satellite. For a ground-based
       spectrometer they are the solar angles. The blended TROPOMI files
       have neither.
   * - ``solar_zenith``, ``solar_azimuth``
     - The solar angles. In GGG files they equal ``zenith`` and
       ``azimuth``.
   * - ``altitude_levels``
     - The heights of ``pressure_levels``, m MSL, in operational TROPOMI
       and GGG netCDF files. Use these for a slant path instead of
       converting pressures.
   * - ``apriori``
     - The prior profile as a mole fraction in ``units``. TROPOMI and OCO-2
       give it on the kernel's levels, GGG on ``pressure_levels``.
   * - ``apriori_column``
     - The prior column value (OCO-2).
   * - ``longitude_bounds``, ``latitude_bounds``
     - Pixel corners, one array per row, for
       :func:`~stilt.observations.jitter_points` (TROPOMI, OCO-2).

Other columns keep the product's own names, for example ``qa_value``,
``quality_flag``, ``operation_mode``, ``xch4_uncorrected``, ``flag``,
``zmin``, and the other ``x<gas>`` columns of a GGG file. A private GGG file
adds ``ak_extrapolated``. It is true where the spectrum's slant xgas fell
outside GGG's kernel table.

From a file to receptors
------------------------

:func:`~stilt.observations.receptors_from_soundings` makes a receptor for
each sounding and the table of their averaging kernels. Here each TROPOMI
sounding becomes a slant receptor on the retrieval's own levels, up to
3 km above the surface:

.. code-block:: python

   import stilt
   from stilt.observations import read_tropomi_ch4, receptors_from_soundings

   project = stilt.Project("./xch4")      # made with stilt init or Project.init

   df = read_tropomi_ch4(path, lon_range=(-113.5, -110.5), lat_range=(39.5, 42.0))
   receptors, kernels = receptors_from_soundings(df[df.good], "slant", top=3000)
   project.add_receptors(receptors)
   project.add_table("kernels", kernels)    # tables/kernels.parquet
   project.run()

Each receptor keeps its ``sounding_id`` as a column of ``receptors.csv``,
so the results join back to the soundings. The footprint config lists two
transforms, the first reading the kernels by the table's name:

.. code-block:: yaml

   transforms:
     - kind: averaging_kernel
       table: kernels
       coordinate: pres
     - kind: pressure_weighting

A few products need a change to this recipe:

- For a nadir column, pass ``"column"``: a :class:`~stilt.ColumnReceptor`
  from the ground up to ``top``.
- OCO-2 files give pressures but no heights. Add ``altitude_levels`` from
  :func:`~stilt.observations.pressure_altitudes` before a slant receptor.
- A TCCON prior grid starts at sea level, below the station. Drop the
  levels below ``surface_altitude`` and put ``surface_altitude`` first, so
  the path starts at the instrument.

Adding an instrument
--------------------

Copy the reader module closest to your product from
``stilt/observations/readers/``. Use ``tropomi.py`` for a swath product and
``ggg.py`` for a ground station. The reader holds all of the product's
conventions, and its output must have the columns above, which
:data:`~stilt.observations.readers.schema.SOUNDING_SCHEMA` lists in code. Keep it a plain
function that returns the table.

Add a small slice of a real file under ``tests/observations/data/products``,
and a test that calls :func:`~stilt.observations.check_soundings` on the
table and checks the units and that the vertical arrays start at the
surface. ``tests/observations/data/products/make_samples.py`` shows how the existing
slices were cut. Readers for PROFFAST EM27/SUN output, MethaneAIR, and
MethaneSAT are welcome. The maintainers have no files to write them
against.
