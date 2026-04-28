.. _what_is_stilt:

What Is STILT?
==============

STILT (Stochastic Time-Inverted Lagrangian Transport) is a Lagrangian particle
dispersion model for simulating atmospheric transport backward in time. It is
used widely in atmospheric science to relate measurements of trace gases at a
receptor location, for example a tower, aircraft, or satellite sounding, to
the upstream surface areas that influenced that measurement.

PYSTILT is a pure-Python implementation of STILT built around the
`HYSPLIT <https://www.ready.noaa.gov/HYSPLIT.php>`_ Fortran binary. It brings
the core transport model, the execution runtime, and a first observation layer
into one modern, pip-installable Python package while working toward broader
parity with the older R-based ecosystem.

The Science In Three Steps
--------------------------

1. **Define a receptor**. This is a point in space and time representing where
   and when you made a measurement, or want to simulate one. In PYSTILT this is
   a :class:`~stilt.Receptor`.
2. **Release and backward-track particles**. An ensemble of theoretical air
   parcels is released from the receptor and followed backward through time
   using gridded meteorological wind fields. Random turbulent velocities are
   added to each particle. The result is a :class:`~stilt.Trajectories` object.
3. **Calculate the footprint**. Particle positions and heights through time are
   converted into a gridded surface-influence function quantifying how strongly
   each upwind surface area influences the receptor. Footprints can then be
   convolved with a surface flux inventory to estimate concentration
   enhancements.

.. figure:: https://uataq.github.io/stilt/static/img/footprint.png
   :alt: Example footprint showing upwind surface influence
   :align: center

   Surface-influence footprint for a receptor over Salt Lake City. Warm colors
   indicate strong upwind influence.

The STILT / HYSPLIT Ecosystem
-----------------------------

PYSTILT sits at the top of a lineage of open-source atmospheric transport
models:

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Model
     - Description
     - Reference / Link
   * - **HYSPLIT**
     - NOAA ARL's hybrid single-particle Lagrangian integrated trajectory
       model. PYSTILT uses HYSPLIT to compute particle trajectories.
     - `NOAA READY <https://www.ready.noaa.gov/HYSPLIT.php>`_
   * - **STILT**
     - The core framework extending HYSPLIT with improved boundary-layer
       mixing, near-field dilution corrections, and footprint calculations.
     - `Lin et al. 2003 <https://doi.org/10.1029/2002JD003161>`_
   * - **R-STILT v2**
     - The redesigned R interface that shaped much of the modern STILT project
       layout and user workflow.
     - `uataq/stilt <https://github.com/uataq/stilt>`_ ·
       `Fasoli et al. 2018 <https://doi.org/10.5194/gmd-11-2813-2018>`_
   * - **X-STILT**
     - R-STILT extensions for column and satellite workflows, including
       averaging kernels and transport-error analyses.
     - `uataq/X-STILT <https://github.com/uataq/X-STILT>`_ ·
       `Wu et al. 2018 <https://doi.org/10.5194/gmd-11-4843-2018>`_
   * - **stiltctl**
     - Operational Python tooling for large STILT workloads on Kubernetes and
       cloud infrastructure.
     - `uataq/stiltctl <https://github.com/uataq/stiltctl>`_
   * - **PYSTILT**
     - The current Python package aiming to unify transport, output runtime,
       and observation-facing workflows.
     - `jmineau/PYSTILT <https://github.com/jmineau/PYSTILT>`_

Key Improvements Over R-STILT
-----------------------------

* **Pure Python**. No R runtime is required.
* **Strong typing**. Configuration is validated before runs start.
* **Flexible receptors**. Point, column, slant, and multipoint workflows share
  the same receptor foundation.
* **Multiple execution backends**. Local, Slurm, and Kubernetes are part of one
  execution model.
* **Persistent index**. Simulation status is stored and resumable.
* **Standard outputs**. Trajectories are stored as Parquet and footprints as
  NetCDF.

Next Steps
----------

* :doc:`installation`
* :doc:`quickstart`
* :doc:`../guides/meteorology`
* :doc:`../guides/index`
