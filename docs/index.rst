PYSTILT
=======

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.22211796.svg
   :target: https://doi.org/10.5281/zenodo.22211796
   :alt: DOI

.. rst-class:: hero-copy

PYSTILT is a Python version of STILT, the atmospheric transport model. It
answers a question every trace-gas measurement raises: **where did this air
come from?** Tell PYSTILT where and when you measured, point it at
meteorology, and it gives you a footprint: a map of which upwind surface areas
influenced your measurement, and by how much.

Use it for towers, aircraft, mobile platforms, and satellite or ground-based
column measurements. Run one simulation on a laptop, or tens of thousands on
an HPC cluster, with the same project.

.. note::

   PYSTILT is in **alpha**. Names and options may still change between
   releases. See the :doc:`roadmap` for what is settled and what is not.

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

      Task-by-task help: describe your measurements, set up meteorology,
      run on a cluster, and load and plot results.

   .. grid-item-card:: :fas:`graduation-cap` Tutorials
      :link: tutorials/index
      :link-type: doc

      Worked examples: a week of footprints at a tower, and turning
      footprints into modeled concentrations.

   .. grid-item-card:: :fas:`right-left` Coming from STILT-R?
      :link: migration/r_stilt
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
   reference/index
   roadmap
   development
