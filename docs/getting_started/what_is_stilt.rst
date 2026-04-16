.. _what_is_stilt:

What Is STILT?
==============

STILT (Stochastic Time-Inverted Lagrangian Transport) is a Lagrangian particle
dispersion model for simulating atmospheric transport backward in time. It is
used widely in atmospheric science to relate measurements of trace gases at a
receptor location (e.g., a tower, aircraft, or satellite sounding) to the
upstream surface areas that influenced that measurement.

PYSTILT is a pure-Python implementation of STILT built around the `HYSPLIT
<https://www.ready.noaa.gov/HYSPLIT.php>`_ Fortran binary. It brings the core
transport model, queue/service runtime, and a first observation layer into one
modern, pip-installable Python package while working toward broader parity with
the older R-based ecosystem.


The Science in Three Steps
---------------------------

1. **Define a receptor** — a point in space and time representing where and
   when you made a measurement (or want to simulate one). In PYSTILT this is a
   :class:`~stilt.Receptor`.

2. **Release and backward-track particles** — an ensemble of theoretical air
   parcels is released from the receptor and followed backward through time
   using gridded meteorological wind fields. Random turbulent velocities are
   added to each particle via a Markov chain process. The result is a
   :class:`~stilt.Trajectories` object.

3. **Calculate the footprint** — the positions and heights of particles at each
   time step are converted into a gridded surface-influence function (the
   *footprint*) quantifying how strongly each upwind surface area influences
   the receptor. Footprints can be convolved with any surface flux inventory to
   estimate the contribution of emissions to the observed concentration.


.. figure:: https://uataq.github.io/stilt/static/img/footprint.png
   :alt: Example footprint showing upwind surface influence
   :align: center

   Surface-influence footprint for a receptor over Salt Lake City.
   Warm colours indicate areas with strong upwind influence.


The STILT / HYSPLIT Ecosystem
------------------------------

PYSTILT sits at the top of a lineage of open-source atmospheric transport
models. To give credit where it is due and to help you navigate the literature:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Model
     - Description
     - Reference / Link
   * - **HYSPLIT**
     - NOAA ARL's hybrid single-particle Lagrangian integrated trajectory
       model. PYSTILT uses the bundled HYSPLIT binary to compute particle
       trajectories.
     - `NOAA READY <https://www.ready.noaa.gov/HYSPLIT.php>`_
   * - **STILT** (original)
     - Core Lagrangian framework extending HYSPLIT with improved boundary layer
       mixing, near-field dilution corrections, and Gaussian-kernel footprint
       calculations (Lin et al. 2003).
     - `Lin et al. 2003 <https://doi.org/10.1029/2002JD003161>`_
   * - **R-STILT v2**
     - Redesigned R interface by Fasoli et al. 2018. Introduced kernel-density
       gridding, HPC parallelism, CLI, and structured project layouts. The
       direct predecessor to PYSTILT.
     - `uataq/stilt <https://github.com/uataq/stilt>`_ ·
       `Fasoli et al. 2018 <https://doi.org/10.5194/gmd-11-2813-2018>`_
   * - **X-STILT**
     - Extension of R-STILT for vertically integrated column concentrations
       (OCO-2, TROPOMI, TCCON). Supports satellite averaging kernels and
       transport error analyses (Wu et al. 2018, 2023).
     - `uataq/X-STILT <https://github.com/uataq/X-STILT>`_ ·
       `Wu et al. 2018 <https://doi.org/10.5194/gmd-11-4843-2018>`_
   * - **stiltctl**
     - Operational Python toolkit for running large STILT workloads on
       Kubernetes / cloud infrastructure.
     - `uataq/stiltctl <https://github.com/uataq/stiltctl>`_
   * - **PYSTILT**
     - This package. Pure-Python, pip-installable reimplementation covering the
       capabilities of R-STILT, stiltctl, and X-STILT in a unified API.
     - `jmineau/PYSTILT <https://github.com/jmineau/PYSTILT>`_


Key Improvements Over R-STILT
-------------------------------

* **Pure Python** — no R dependency; install with ``pip install pystilt``.
* **Strong typing** — all configuration is validated by Pydantic models;
  configuration errors are caught before any simulations run.
* **Flexible receptors** — point, column, slant-path, and multi-point receptors
  all share the same :class:`~stilt.Receptor` interface, unifying the
  surface-station and column-retrieval workflows.
* **Multiple execution backends** — local, Slurm, and Kubernetes executors are
  interchangeable via the ``execution`` dict passed to :class:`~stilt.ModelConfig`.
* **Persistent state** — simulation status is tracked in a SQLite database;
  runs are resumable and incremental.
* **Standard outputs** — trajectories as Parquet, footprints as NetCDF
  following the CF convention.


Next Steps
----------

* :doc:`installation`: install PYSTILT in your environment.
* :doc:`quickstart`: run your first simulation in five minutes.
* :doc:`../migration`: coming from R-STILT, stiltctl, or X-STILT? See the
  concept-mapping table.
