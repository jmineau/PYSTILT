PYSTILT
=======

.. rst-class:: hero-copy

PYSTILT is an alpha Python implementation of the STILT transport workflow. It
combines receptor and footprint science APIs with a modern execution layer for
local runs, Slurm arrays, and Kubernetes workers.

.. note::

   PYSTILT is an **alpha development release**. APIs, configuration fields,
   executor semantics, and project layout details may change while the package
   settles.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :fas:`play` Getting Started
      :link: getting_started/index
      :link-type: doc

      Start with STILT background, installation, and a first trajectory or
      footprint from Python or the CLI.

   .. grid-item-card:: :fas:`book` User Guides
      :link: guides/index
      :link-type: doc

      Learn meteorology, outputs, inline Python use, executors, migration
      paths, and deeper operational topics.

   .. grid-item-card:: :fas:`graduation-cap` Tutorials
      :link: tutorials/index
      :link-type: doc

      Work through cleaned-up examples for a stationary tower workflow, Slurm
      execution, and footprint-to-flux calculations.

   .. grid-item-card:: :fas:`code` Reference
      :link: reference/index
      :link-type: doc

      Browse grouped API and configuration reference pages for core objects,
      meteorology, execution, observations, storage, and HYSPLIT internals.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   guides/index
   tutorials/index
   reference/index
