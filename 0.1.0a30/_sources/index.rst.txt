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

.. raw:: html

   <figure class="home-animation">
     <video autoplay loop muted playsinline poster="_static/home-animation-poster.jpg"
            aria-label="Particles travel backward in time from a tower in northern Wisconsin, north into Canada, while a footprint map fills in behind them.">
       <source src="_static/home-animation.mp4" type="video/mp4">
     </video>
     <figcaption>
       The WLEF tall tower in Park Falls, Wisconsin, at 2 pm on 8 February 2019.
       Left: PYSTILT's particles traveling back in time from the 396 m inlet.
       Right: the footprint they make. Particles that reach the northern edge of
       the HRRR meteorology stop there.
     </figcaption>
   </figure>

PYSTILT works for towers, aircraft, mobile platforms, and satellite or
ground-based column measurements. The same project can run one simulation on a
laptop or tens of thousands on an HPC cluster.

.. note::

   PYSTILT is alpha software. Names and options may still change between
   releases. The :doc:`development/roadmap` shows what is settled.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :fas:`play` Getting Started
      :link: getting_started/index
      :link-type: doc

      What a footprint is, how to install PYSTILT, and your first footprint
      map in a few steps.

   .. grid-item-card:: :fas:`graduation-cap` Tutorials
      :link: tutorials/index
      :link-type: doc

      Complete studies. A day of tower footprints, modeled concentrations
      from an inventory, and footprints of a satellite column.

   .. grid-item-card:: :fas:`book` User Guide
      :link: guides/index
      :link-type: doc

      How to describe your measurements, set up meteorology, run on a
      cluster, and load and plot results.

   .. grid-item-card:: :fas:`code` API Reference
      :link: reference/index
      :link-type: doc

      Every class, function, and configuration option.

Used STILT-R, X-STILT, stiltctl or Jena STILT before?
:ref:`Coming from another STILT <coming-from-another-stilt>` says where
their settings and steps are in PYSTILT. Run HYSPLIT on its own?
:doc:`guides/hysplit_users` maps ``CONTROL`` and ``SETUP.CFG`` to PYSTILT's
settings.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   tutorials/index
   guides/index
   reference/index
   development/index
