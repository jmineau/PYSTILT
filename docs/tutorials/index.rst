Tutorials
=========

Step-by-step worked examples covering common PYSTILT workflows.

The tutorials use sample ARL-format meteorology files from the
`stilt-tutorials <https://github.com/uataq/stilt-tutorials>`_ repository.
Clone it once and point your ``met_directory`` at the appropriate subdirectory:

.. code-block:: bash

   git clone https://github.com/uataq/stilt-tutorials sample-data

Then set, for example::

   met_directory = "sample-data/01-wbb/met"

.. note::

   PYSTILT does not fetch meteorology for you yet. ``arl-met`` is the
   companion package for inspecting ARL files today, and future ``arl-met``
   cropping / ARL-writing support is expected to be what PYSTILT leverages for
   more automated meteorology workflows.

.. toctree::
   :maxdepth: 2

   wbb_stationary
   column_satellite
   hpc_slurm
   flux_inversion
