.. _what_is_stilt:

What Is STILT?
==============

Suppose a tower in Salt Lake City measures a methane spike at 2 pm. Where did
that methane come from? A landfill to the south, a refinery to the north, or
the valley's gas pipes? STILT answers that by working backward from the
measurement: it follows the air that arrived at the tower back in time, and
records where that air was close enough to the ground to pick up emissions.

STILT (Stochastic Time-Inverted Lagrangian Transport) is widely used in
atmospheric science for exactly this: linking trace-gas measurements from
towers, aircraft, vehicles, and satellites to the surface areas that
influenced them. PYSTILT is a Python version of STILT.

How STILT works, in three steps
-------------------------------

1. **Say where and when you measured.** STILT calls this location and time a
   :term:`receptor`. It can be a single point (a tower inlet), a vertical
   column (a ground-based spectrometer), or a set of points along a slanted
   line of sight (a satellite sounding).
2. **Release particles and run them backward.** STILT releases a cloud of
   imaginary air parcels, called :term:`particles <particle>`, at the receptor
   and moves them backward in time using gridded wind fields from a weather
   model (the :term:`meteorology`). Each particle also gets random turbulent
   motion, so the cloud spreads out the way real air mixes. The particle
   paths are the :term:`trajectories <trajectory>`.
3. **Turn the particle paths into a footprint.** Wherever particles spend time
   near the ground, the surface there can influence the measurement. STILT
   counts this up on a map grid. The result is a :term:`footprint`, and
   each grid cell's value says how much a unit of emissions there would
   change the measurement.

.. figure:: https://uataq.github.io/stilt/static/img/footprint.png
   :alt: Example footprint showing upwind surface influence
   :align: center

   A footprint for a receptor over Salt Lake City. Warm colors mark the
   areas that most strongly influenced the measurement.

What you can do with a footprint
--------------------------------

Multiply a footprint by an emissions map (a flux inventory) and add it up,
and you get the concentration increase your receptor should have seen. Do
that for many measurements and compare with what was actually observed, and
you can test an inventory or estimate emissions in an inversion.
:doc:`../tutorials/flux_inversion` shows the first half of that.

What you need
-------------

- **Python 3.10 or newer** and PYSTILT (:doc:`installation`).
- **Meteorology in ARL format.** STILT needs gridded winds, temperature, and
  turbulence fields from a weather model such as HRRR, NAM, or GDAS, in the
  format used by NOAA's Air Resources Laboratory (:term:`ARL`). Many research
  groups keep an archive of these files. If you don't have one, PYSTILT can
  download them from NOAA for you (see :doc:`../guides/meteorology`).
- **Your measurement times and locations.**

You do not need to install HYSPLIT, the Fortran program STILT uses to move
particles. PYSTILT includes it for Linux and macOS (Intel).

Why PYSTILT?
------------

PYSTILT reproduces STILT-R's footprints (checked cell by cell against STILT-R)
and adds:

* **Python all the way through.** No R installation needed; results load
  straight into pandas and xarray.
* **Settings checked before you run.** A typo in ``config.yaml`` is reported
  right away, not an hour into a cluster job.
* **Reruns that pick up where they left off.** PYSTILT checks which
  simulations already have their outputs and runs only the rest.
* **Laptop to cluster with the same project.** Switching from your computer to
  a Slurm cluster is a few lines of configuration.
* **Column and satellite support.** Column and slanted receptors, averaging
  kernels, and pressure weighting are built in.
* **Standard file formats.** Trajectories are written as Parquet and
  footprints as NetCDF.

Where PYSTILT comes from
------------------------

PYSTILT builds on a line of open-source transport models:

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Model
     - What it is
     - Reference / link
   * - **HYSPLIT**
     - NOAA ARL's particle transport model. PYSTILT runs HYSPLIT to move the
       particles.
     - `NOAA READY <https://www.ready.noaa.gov/HYSPLIT.php>`_
   * - **STILT**
     - Adds improved boundary-layer mixing, near-field dilution corrections,
       and footprint calculation on top of HYSPLIT.
     - `Lin et al. 2003 <https://doi.org/10.1029/2002JD003161>`_
   * - **STILT-R v2**
     - The R version of STILT. PYSTILT's project layout and settings follow
       it closely.
     - `uataq/stilt <https://github.com/uataq/stilt>`_ ·
       `Fasoli et al. 2018 <https://doi.org/10.5194/gmd-11-2813-2018>`_
   * - **X-STILT**
     - STILT-R extensions for column and satellite measurements, including
       averaging kernels and transport-error analysis.
     - `uataq/X-STILT <https://github.com/uataq/X-STILT>`_ ·
       `Wu et al. 2018 <https://doi.org/10.5194/gmd-11-4843-2018>`_
   * - **stiltctl**
     - Tools for running large STILT workloads in the cloud.
     - `uataq/stiltctl <https://github.com/uataq/stiltctl>`_

If you use STILT in published work, please cite Lin et al. (2003) and
Fasoli et al. (2018), plus Wu et al. (2018) for column work, and PYSTILT
itself through its `Zenodo DOI <https://doi.org/10.5281/zenodo.22211796>`_.

Next: :doc:`installation`.
