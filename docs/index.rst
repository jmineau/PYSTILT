PYSTILT
=======

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.22211796.svg
   :target: https://doi.org/10.5281/zenodo.22211796
   :alt: DOI

.. rst-class:: hero-copy

PYSTILT is a Python version of STILT, an atmospheric transport model. It
tells you where the air in a trace-gas measurement came from. Give PYSTILT the
place and time of your measurement and some meteorology, and it computes a
footprint. The footprint is a map of the upwind surface areas that influenced
the measurement, and by how much.

PYSTILT works for towers, aircraft, mobile platforms, and satellite or
ground-based column measurements. The same project can run one simulation on a
laptop or tens of thousands on an HPC cluster.

.. note::

   PYSTILT is alpha software. Names and options may still change between
   releases. The :doc:`roadmap` shows what is settled.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :fas:`wind` New to STILT?
      :link: getting_started/what_is_stilt
      :link-type: doc

      What a footprint is, how STILT makes one, and what you need to get
      started.

   .. grid-item-card:: :fas:`play` Your first footprint
      :link: getting_started/quickstart
      :link-type: doc

      Install PYSTILT and go from one measurement to a footprint map in a
      few steps.

   .. grid-item-card:: :fas:`book` User guide
      :link: guides/index
      :link-type: doc

      How to describe your measurements, set up meteorology, run on a
      cluster, and load and plot results.

   .. grid-item-card:: :fas:`graduation-cap` Tutorials
      :link: tutorials/index
      :link-type: doc

      Complete examples. Run a week of tower footprints, turn footprints
      into modelled concentrations, and scale up on a Slurm cluster.

   .. grid-item-card:: :fas:`right-left` Coming from STILT-R?
      :link: migration/stilt_r
      :link-type: doc

      Where each ``run_stilt.r`` setting lives in PYSTILT.

   .. grid-item-card:: :fas:`code` API reference
      :link: reference/index
      :link-type: doc

      Every class, function, and configuration option.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   tutorials/index
   guides/index
   migration/index
   reference/index
   advanced/index
   roadmap
   development
