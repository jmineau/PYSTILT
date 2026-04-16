.. _meteorology:

Meteorology
===========

STILT requires gridded meteorological fields in the NOAA ARL binary format.
PYSTILT does not download met files. Obtain ARL files separately and point
:class:`~stilt.MetConfig` at the directory where they live. To inspect or
validate ARL files, see the `arl-met <https://github.com/jmineau/arl-met>`_
package.


Obtaining ARL Meteorology
--------------------------

The NOAA Air Resources Laboratory archives ARL-format files for several
operational models:

* **HRRR** (3 km, CONUS) — best for regional US studies. Available via
  `NOAA ARL archives <https://www.ready.noaa.gov/archives.php>`_.
* **NAM** (12 km, North America) — coarser but broader coverage.
* **GFS** (0.25°, global) — use when simulations leave CONUS, or for
  longer back-trajectories.
* **ERA5 / WRF** — can be converted to ARL format (not included in PYSTILT).

NOAA ARL archive: https://www.ready.noaa.gov/archives.php


Configuring a Meteorology Source
----------------------------------

Each meteorology source is described by a :class:`~stilt.MetConfig`:

.. code-block:: python

   import stilt

   hrrr_met = stilt.MetConfig(
       directory="/data/met/hrrr",      # directory containing ARL files
       file_format="hrrr_%Y%m%d.arl",  # strftime format for file names
       file_tres="1h",                  # time resolution of each file
       n_min=2,                         # minimum ARL files required to run
   )

The ``file_format`` field uses Python's :func:`datetime.strftime` tokens.
PYSTILT uses it to discover which files cover the time range of each
simulation.

Multiple meteorology sources can be registered with a :class:`~stilt.Model`:

.. code-block:: python

   config = stilt.ModelConfig(
       mets={
           "hrrr": hrrr_met,
           "gfs": stilt.MetConfig(
               directory="/data/met/gfs",
               file_format="gfs_%Y%m%d_%Hz.arl",
               file_tres="3h",
           ),
       },
       ...
   )


MetStream at Runtime
--------------------

At run time, PYSTILT creates a :class:`~stilt.MetStream` from each
:class:`~stilt.MetConfig`. When ``STILT_MET_ARCHIVE`` is configured, relative
meteorology directories are resolved from that archive root and the required
ARL files are staged into each simulation's local compute directory before
HYSPLIT runs:

.. code-block:: python

   import pandas as pd

   source = stilt.MetStream(
       name="hrrr",
       **hrrr_met.model_dump(),
   )
   file_list = source.get_files(
       r_time=pd.Timestamp("2023-07-15 18:00", tz="UTC"),
       n_hours=-24,
   )
   print(file_list)

If fewer than ``n_min`` files are found for a simulation, the simulation is
marked as failed with :attr:`~stilt.errors.FailureReason.MISSING_MET_FILES`.


Checking Met Availability
--------------------------

Before running, you can check which simulations have not yet been run:

.. code-block:: python

   pending = model.get_pending(mets=["hrrr"])
   print(f"{len(pending)} simulations have not been run yet")

The :meth:`~stilt.Model.get_pending` method returns simulation IDs for
receptors that have no completed trajectory in the repository. Simulations can
fail at runtime if the required met files are not found (see
:attr:`~stilt.errors.FailureReason.MISSING_MET_FILES`).


Reading ARL Files Directly
---------------------------

`arl-met <https://github.com/jmineau/arl-met>`_ is a companion package that
can read NOAA ARL binary met files into xarray Datasets. It is not required to
run PYSTILT but is useful for inspecting met data, checking domain coverage,
or validating wind fields. Future ``arl-met`` cropping and ARL-writing support
is expected to be the mechanism PYSTILT leverages for subgrid meteorology
workflows:

.. code-block:: python

   # pip install arl-met
   import arl_met

   ds = arl_met.open("hrrr_20230715.arl")
   ds["UWND"].isel(time=0).plot()   # u-wind at first time step

.. warning::

   ``arl-met`` is still evolving. Today it is mainly useful for inspection and
   validation; future cropping and ARL-writing capabilities are the pieces
   PYSTILT plans to leverage for subgrid met workflows.
