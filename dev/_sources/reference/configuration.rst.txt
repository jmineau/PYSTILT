Configuration
=============

.. currentmodule:: stilt

``config.yaml`` is blocks, each a field of :class:`ProjectConfig`:

.. code-block:: yaml

   mets:
     hrrr: /path/to/hrrr_arl
   models:
     hysplit: {n_hours: -24, numpar: 200}
   stages:
     footprint:
       class: stilt.footprint.FootprintStage
       grid: {xmin: -113.0, xmax: -110.5, ymin: 40.0, ymax: 42.0, xres: 0.01, yres: 0.01}
   variants:
     hrrr: {}
     hrrr-long: {n_hours: -72}
     hrrr-coarse: {stages: {footprint: {grid: {xres: 0.05, yres: 0.05}}}}
     hrrr-particles: {stages: {footprint: null}}
   execution:
     backend: local
   output: output

``mets`` names the meteorology (for HYSPLIT,
:class:`~runtraj.models.hysplit.HysplitMetConfig`). ``models`` names the transport
models, each block with that model's settings
(:class:`~runtraj.models.hysplit.HysplitConfig` for ``hysplit``).
``stages`` names what is made from each simulation's particles: the
footprint, with the footprint settings (:class:`FootprintConfig`). A stage's
block is named for its stage, so the footprint's is ``footprint``, the name
of its folder in the output directory and of its column in
``project.status()``.
``variants`` says what runs, and ``execution`` where
(:class:`ExecutionConfig`). Each class lives next to the code that uses it
(``stilt.footprint.config``, ``runtraj.execution.config``,
``runtraj.models.hysplit``), and all but the transport model's are
importable from ``stilt``.

No setting sits at the top level. One written there is refused with a
message that names its block, such as ``'n_hours' is a hysplit setting.
Write it under models: hysplit:.`` A key may be written once in each
block: YAML would keep the last of two, so PYSTILT refuses the file and
names the line of the second.
Each page below lists its fields with their defaults and is generated from
the code.

.. tip::
  New to PYSTILT? The :doc:`Projects and variants guide <../guides/projects>`
  covers the few settings most projects need. This page lists all of them.

Every key
---------

These tables are generated from the code, so they always match the
installed version.

Top level
~~~~~~~~~

The blocks, and where the results go.

.. config-model:: runtraj.config.ProjectConfig
   :declared-only:

HYSPLIT
~~~~~~~

The keys of the ``hysplit`` block under ``models:``, the transport model's
settings (:doc:`hysplit`). They are the defaults of every variant that runs
HYSPLIT, and a variant may override any of them by writing the key itself.
Left out, the block is HYSPLIT with the defaults below, which are HYSPLIT's
own. ``n_hours`` has no default, and ``capemin``, ``kblt``, ``kmixd``,
``krand`` and ``vscales`` are unset unless given: HYSPLIT chooses them when
it runs. ``stilt init`` and :meth:`stilt.Project.init` write STILT-R's values
where they differ (:data:`stilt.project.STILT_R_HYSPLIT`; the list is in
:doc:`../guides/migration/stilt_r`).

.. config-model:: runtraj.models.hysplit.HysplitConfig

A model in another package gets a block of any name that gives the model
class's import path as ``class``, then that model's own settings:

.. code-block:: yaml

   models:
     hysplit: {n_hours: -24}
     mine: {class: mypkg.models.MyModel, n_hours: -24}
   variants:
     hrrr: {model: hysplit}
     hrrr-mine: {model: mine, met: hrrr}

With two or more models, each variant names its own with ``model``. A model
has one block: a second set of settings for it is a second variant.

Footprint
~~~~~~~~~

The keys of the footprint's block under ``stages:``, after ``class:
stilt.footprint.FootprintStage`` (:class:`~stilt.footprint.FootprintStage`).
They are the defaults of every variant. A variant changes some under its
own ``stages`` key, under ``footprint``, and sets the stage to null to make
particles only:

.. code-block:: yaml

   variants:
     hrrr-coarse: {stages: {footprint: {grid: {xres: 0.05, yres: 0.05}}}}
     hrrr-particles: {stages: {footprint: null}}

A ``grid`` given by a variant changes the block's grid field by field, so
the first variant above keeps the block's bounds. Without a ``stages``
block, a project makes particles only.

.. config-model:: stilt.footprint.config.FootprintConfig

Footprint grids
^^^^^^^^^^^^^^^

Longitudes count modulo 360, so a grid across 180° can be written
``xmin: 170``, ``xmax: 190`` or ``xmin: 170``, ``xmax: -170``, and the
footprint's longitudes are then 170 to 190. An ``xmax`` west of ``xmin``
is read as crossing 180° when the two are on either side of 0° or one is
written past 180°: with both in one hemisphere (``xmin: -110.5``,
``xmax: -113``) they are taken to be swapped and the grid is refused. A grid spans at most 360°. A
particle cloud on both sides of 180° is measured as one cloud for the
smoothing. A grid that goes all the way around (``xmin: -180``,
``xmax: 180``, with cells that fit 360°) is a circle, so smoothing that
passes the east edge returns at the west edge. On a grid that ends at or
near 180°, particles just across 180° smooth onto the grid, as particles
just past any other edge do.

Projected grids (a ``crs`` such as ``EPSG:32612``) keep their bounds in
degrees and take ``xres`` and ``yres`` in the projection's units. The
grid covers the whole box in the projection, so on a wide or polar box it
reaches past the box's projected corners, where the curved edges bow out.
A grid that :meth:`stilt.Mesh.to_grid` picks for a projected
``geometry`` covers every one of its areas. A grid
of more than a billion cells is refused, which catches a resolution in
degrees on a grid in meters. The two
steps that depend on the cell size work in the grid's units. The smoothing
kernel's width is worked out in degrees and converted, so a footprint is
smoothed about as much on 1 km UTM cells as on 0.01° cells. The early-time
interpolation, which adds particle positions between output steps in the
first 100 minutes when the particles move more than a cell per step, runs
on 100 m UTM cells as it does on 0.001° cells. :ref:`stilt-r-parity` says
where STILT-R differs on these grids.

Time layers (``tres``) are recorded one way, so ``60min`` is recorded as
``1h`` and two spellings of one length share a settings folder. Each layer
holds the influence of the particles in its minutes, so the layers sum to
the same total footprint whatever ``tres`` is.

Loading ``config.yaml`` does not read a ``geometry`` file.
PYSTILT reads it once, when it works out the variants' settings, and picks
the grid then (:meth:`stilt.Mesh.to_grid`), so a variant with a geometry
cannot change only part of the grid (``grid: {xres: 0.1}`` under its
``stages``). A ``grid``
given with a ``geometry`` is used as given. Each footprint file stores a
code for the geometry it was made for, which is how
``foot.stilt.aggregate`` knows when a mesh no longer matches.

Footprint maps
~~~~~~~~~~~~~~

To show each footprint on a web map, add a block for the map stage after
the footprint's (:class:`~stilt.footprint.MapStage`, which needs the
``visualization`` extra). Its keys are the color scale, the same for every
image so that two hours compare by eye:

.. code-block:: yaml

   stages:
     footprint:
       class: stilt.footprint.FootprintStage
       grid: {xmin: -113.0, xmax: -110.5, ymin: 40.0, ymax: 42.0, xres: 0.01, yres: 0.01}
     map:
       class: stilt.footprint.MapStage
       vmin: 1.0e-5
       vmax: 0.1

Each release gets ``<release id>.json`` in the map's folder of the output
directory, written after its footprint. It holds the footprint summed over
time as a PNG data URL (``image``), the image's corners as
``[[south, west], [north, east]]`` (``bounds``), the release, its time in
UTC, the color scale and the footprint's units. The image's rows are spaced
as on a web map, so Leaflet places it with
``L.imageOverlay(entry.image, entry.bounds)``. An empty footprint gets a
file with ``image`` and ``bounds`` null. A footprint grid across 180°
cannot be drawn, and its map fails.

.. config-model:: stilt.footprint.maps.MapConfig

A met
~~~~~

The keys of one entry under ``mets:``, for HYSPLIT
(:doc:`../guides/meteorology`).

.. config-model:: runtraj.models.hysplit.HysplitMetConfig

Execution
~~~~~~~~~

The keys under ``execution:`` (:doc:`../guides/running`). ``backend`` names
where the project runs, and every setting of that place is in the block of
its name. Only that block is read, so a config can hold both.

.. config-model:: runtraj.execution.config.ExecutionConfig

The keys under ``execution: local:``, a run on this machine.

.. config-model:: runtraj.execution.config.LocalConfig

The keys under ``execution: slurm:``, a Slurm job array
(:doc:`../guides/slurm`). Any other key is an ``sbatch`` option by its name.

.. config-model:: runtraj.execution.config.SlurmConfig

The keys under ``execution: slurm: fetch:``, the Slurm resources of the job
that gets the met files before a job array (:ref:`slurm-fetch`).

.. config-model:: runtraj.execution.config.FetchConfig

Config objects
--------------

A config is what you write. ``project.variants`` turns each declared
variant into a :class:`runtraj.config.Variant`: its transport settings
checked by the model's config class, and what the file alone does not
say, such as the grid of a footprint given by a geometry and the transport
model build. A variant with ``realizations: N`` stays one variant.

The ``RUNTRAJ_COMPUTE_ROOT`` environment variable sets the compute root,
where each run's workdir is made. It never changes a result, and projects
can share one.

.. autosummary::
   :toctree: _api
   :nosignatures:

   ProjectConfig
   FootprintConfig
   footprint.FootprintStage
   footprint.MapConfig
   footprint.MapStage
   ExecutionConfig
   Bounds
   Grid

Parameters
----------

The settings that shape a run's particles belong to the transport model,
and are written in its block under ``models:`` (``hysplit`` when the config
has no ``models``). HYSPLIT's are in
:class:`runtraj.models.hysplit.HysplitConfig`, under HYSPLIT's names (see
:doc:`hysplit`). Each model's block holds all of its settings, the ones
PYSTILT reads itself among them (``n_hours`` and ``seed``): nothing passes
from one model's block to another's.

What is done with the particles afterwards is the footprint stage's, under
``stages: footprint:``, and does not change the particles or their settings
folder: ``hnf_plume``, the near-field correction, applied when the
footprint is made, and ``footprint_depth``, ``pbl_height`` and
``pbl_height_min``, which say how PYSTILT computes ``foot`` when the
footprint needs one (:doc:`particles`). HYSPLIT computes its own ``foot``
with ``veght``, a HYSPLIT setting. A footprint made from that ``foot`` reads
``veght`` for the near-field correction, so a ``footprint_depth`` written
for it must equal ``veght``; a different one is refused, saying to set
``veght``, or ``pbl_height`` so that PYSTILT computes ``foot``.


Geometry specifications
-----------------------

The kinds of area you can name in the ``geometry`` footprint setting.
:meth:`stilt.Mesh.from_spec` reads one into a :class:`stilt.Mesh`.

.. autosummary::
   :toctree: _api
   :nosignatures:

   footprint.config.FileGeometrySpec
   footprint.config.H3GeometrySpec
   footprint.config.WindowsGeometrySpec
