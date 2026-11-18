Architecture
============

How the code is organized, how a run goes from a project to result files,
how PYSTILT knows which work is finished, and the implementation details that
are easy to break. This page is for contributors, for anyone building on
PYSTILT, and for anyone running at scale or sharing an output directory
between projects. ``AGENTS.md`` at the repository root has the short version
and the rules a change must keep.

The design in brief
-------------------

There is no index, manifest, or registry. A project is a directory of inputs
(:class:`stilt.Project`: ``config.yaml`` and ``receptors.csv``) and an output
directory (``stilt.output.Output``) that ``config.yaml`` names and that
several projects can share. The simulations a project defines are its
receptors crossed with the named variants in ``config.yaml``, which must
declare at least one.

Each variant resolves into a ``Variant`` (``stilt.config``) with two records
of settings (``stilt.identity``), and their hashes name its folders.
Variants with equal run settings share one run per receptor and differ only
in the footprint made from it. Changing a setting never overwrites a result:
it hashes to a new folder. :doc:`../reference/layout` describes the files.

Whether a simulation is complete is decided by the files in the output
directory, by ``stilt.output.completed``: the particles exist, and the
footprint when the variant has a grid. ``Output.complete``,
``Simulation.is_complete``, and ``Project.status`` all apply it.
:ref:`finished-work` explains the consequences.

``Project`` reads, and the workers write results. ``Project(path)`` opens a
project directory: its config, its receptors, the simulations they define,
and a view of their results. Its only writes are to its inputs.
``Project.init`` (or ``stilt init``) writes ``config.yaml`` once and PYSTILT
never rewrites it, ``add_receptors`` appends to ``receptors.csv``, and
``add_table`` writes ``tables/<name>.parquet``. It knows no scheduler or
compute root. ``stilt.execution.run`` and ``submit``, which ``Project.run()``
and ``Project.submit()`` call, find the receptors with missing results,
resolve the compute root, and start the workers. The workers are the only
code that writes results.

Tables are plain DataFrames, and the verbs are on ``Project``.
``project.simulations`` is a pandas DataFrame, one row per simulation. Select
it as in pandas (``sims[sims.variant == "hrrr"]``) and pass the selection to
the project: ``status``, ``incomplete``, ``particles`` (one long table),
``footprints`` (one lazy dataset stacked on the hour), ``jacobian``, or
``transport_error`` (the selection with the estimate's columns). A
selection is any table with ``receptor`` and ``variant`` columns (pandas,
polars, pyarrow) or a boolean mask over ``project.simulations``, and the
methods read only those two columns. ``project.receptors`` is a plain
DataFrame too. ``project.receptor(id)`` and ``project.simulation(id,
variant)`` return values, and ``sim.particles`` and ``sim.footprint`` are
data. ``Output`` has no folder classes: its methods (``path``,
``make_path``, ``read``, ``present``, ``complete``, ``table``, and the
failure records) take the kind of result (``"particles"``, or the name of a
stage such as ``"footprint"``), the variant, and receptor ids.

``stilt.__all__``, plus the ``__all__`` of each subpackage, is the public
surface. Everything else is internal and can change.

Module map
----------

.. code-block:: text

   src/stilt/
     cli.py             Typer CLI; a thin adapter over Project and execution
     project.py         Project: the project directory, its receptors and
                        simulations (DataFrames), run/submit, and a
                        selection's status, particles, footprints, Jacobian
     check.py           check_project: what would stop a project's runs,
                        found from its inputs before any run (Project.check,
                        stilt check); a transport model's check_met finds the
                        met's part
     fetch.py           fetch_project: gets the met files the unfinished
                        simulations need (Project.fetch, stilt fetch); a
                        transport model's fetch_met does the model's part
     output.py          Output: the output directory, a tree per kind of
                        result (particles, and each stage by its name) and a
                        folder per settings hash; finding a variant's folder,
                        which receptors have results (present, complete),
                        where a file is and where a stage writes it (path,
                        make_path), failure records, writing particles
     stages.py          the Stage protocol: what a class named under stages:
                        provides; get_stage, the order stages run in, and
                        check_stage for the tests of a stage of your own
     migration.py       read_stilt_r: STILT-R and X-STILT by-id output as
                        receptors, particle tables, and footprints;
                        import_stilt_r writes it into an output directory
                        with the settings its files record
     simulation.py      Simulation: a frozen value (receptor, variant, output)
                        that knows where its results are and whether they exist
     background.py      background: the mole fraction at a receptor from a
                        field sampled where the particles end (sim.background)
     transport_error.py transport_error: the transport error of the
                        enhancement, one receptor or a selection in threads
     config.py          ProjectConfig: reads config.yaml's blocks (mets, models,
                        stages, variants), refuses a setting outside its
                        block and an unknown key naming the nearest setting,
                        checks each declared variant; resolve() makes each a
                        Variant (its configs, the model build, its stages'
                        settings, and the hash of its particles and of each
                        stage)
     identity.py        settings records and their hashes: what a run was
                        made with, in _settings.yaml, and how a stage's
                        record is hashed with what it reads
     receptors/
       models.py        receptor types (frozen pydantic models: point, column,
                        multipoint), their times, and their ids
       table.py         the receptor table behind the CSV reader, writer, and
                        appender: receptor_rows checks it once, and
                        receptors_from_rows builds receptors from checked rows
       validation.py    the checks a receptor must pass, written once for many
                        points and shared by the models and the table
     particles/         the particle table (a DataFrame)
       table.py         its columns, particle files, release heights, foot for
                        a model that writes none, the near-field plume correction
                        (which calc_footprint applies)
       accessor.py      the .stilt pandas accessor
     footprint/         the footprint (a DataArray)
       config.py        FootprintConfig and the geometry specs; the record
                        of a footprint's settings
       stage.py         FootprintStage, the stage a stages: block names:
                        reads the geometry, grids the particles, and writes
                        the footprint file
       gridding.py      calc_footprint, as STILT-R's (fidelity-guarded)
       aggregation.py   aggregate and jacobian: footprints summed onto other
                        geometries, with time binning and target weights
       io.py            footprint files and CF-1.8 NetCDF; open_footprints
       targets.py       the geometries footprints are summed onto (Mesh,
                        Zones), the overlap weights, and to_grid
       accessor.py      the .stilt xarray accessor
     sampling.py        sampling a gridded field (a flux, a mole fraction) at
                        points; sample_met, ARL met at particle positions
     spatial.py         rasters and CRS without shapely: Bounds, Grid,
                        horizontal_dims, is_longlat, same_crs, haversine_km
     meteorology.py     run_window, and the wind-error statistics
                        (variogram, fit_variogram)
     transforms/        pre-footprint particle transforms, one module each
                        (averaging_kernel, pressure_weighting, lifetime); the
                        loader, YAML I/O, and apply_transforms in __init__.py
     exceptions.py      every exception class, all under StiltError
     visualization.py   matplotlib helpers (optional dependency)
     _paths.py          local paths and object-store paths behind one interface
     _plugins.py        import_class: a model or a stage class named by its
                        import path in config.yaml
     execution/         ExecutionConfig (config.py); the runner (plans what is
                        missing, runs it here or submits a Slurm job array);
                        the worker (runs the transport model in a workdir,
                        then each stage, for one or many simulations)
     observations/      the X-STILT port: product readers and SOUNDING_SCHEMA,
                        overpass grouping and sounding selection, slant
                        geometry and receptors_from_soundings, modelled_column,
                        plume backgrounds; receptors_from_track for the
                        track of a moving platform; arrays in, plain
                        values out
     transport/         the TransportModel and TransportConfig protocols,
                        ModelRun, ModelInfo, get_model and its MODELS table,
                        run_model and run_trajectories (__init__.py); one
                        subpackage per transport model, which owns its config;
                        stilt_r.py, the stilt-r model of imported STILT-R
                        runs, which runs nothing
       hysplit/         HYSPLIT: HysplitConfig (config.py); MetConfig and Met,
                        finding, downloading, and cropping ARL files through
                        arlmet (met.py); Catalog, what a folder's files hold,
                        read from their headers (catalog.py); HysplitModel
                        (model.py); the driver
                        (driver.py: write_inputs, read_particle_dat); the
                        CONTROL file (control.py) and SETUP.CFG namelist
                        (namelist.py) writers and readers; failure reasons read from its
                        log (failures.py). No HYSPLIT file: find_hycs_std
                        and data_tables take directory, else PATH

   tests/               pytest; folders follow src/stilt (execution/,
                        footprint/, observations/, particles/,
                        transport/hysplit/); integration/ holds the end-to-end
                        runs, r_stilt/ the STILT-R comparisons, and fixtures/
                        the factories and helpers tests share

How a run goes
--------------

A run starts from ``Project.run()``, ``Project.submit()``, or ``stilt run``.
``stilt.execution.run`` finds the receptors with missing results and either
runs them in this process (``backend: local``) or submits them as one Slurm
job array (``backend: slurm``).

For Slurm, ``submit`` writes
``_slurm/<stamp>/{receptors.txt,receptors.parquet,execution.yaml,job.sh}`` in
the project and calls ``sbatch``. Each task runs ``stilt run --receptors
receptors.parquet --task $SLURM_ARRAY_TASK_ID/N --execution execution.yaml
--cpus <cpus>``: a task runs its share in its own process, as a local run
does, so the script hands it the array's ``cpus``.
The parquet file holds the submitted receptors' checked rows, which the
task's project takes in place of ``receptors.csv``. ``submit`` returns the job
id at once, and ``run`` waits by polling ``sacct``.

When a simulation whose particles are missing uses a met that downloads
(the transport model's ``downloads_met``), ``submit`` first submits
``fetch.sh`` (``fetch_script``), one task of ``stilt fetch --receptors
receptors.parquet`` with the resources of ``execution.slurm.fetch``, and gives the
array ``--dependency=afterany:<fetch job>``. ``afterany`` because Slurm
cannot tell ``stilt fetch``'s exit 1 (some files missing) from 3 (stopped,
archive unreachable), and in both cases the runs whose met is there should
go. A local run with ``cpus`` above 1 calls ``Project.fetch`` before the pool
starts, for the same reason: workers that start the same file download it
twice. A task (``--task``) never fetches.

A task that stops with work left requeues itself (``scontrol requeue``) when
it got SIGUSR1, which the script asks for two minutes before the time limit
(``--signal=B:USR1@120``), or when it was preempted (Slurm set the job's
``PreemptTime``). Slurm checks ``--signal`` times only about once a minute,
so a preempted task may never get SIGUSR1. A task stopped by ``scancel`` is
not requeued, and a task stops for good after ``MAX_REQUEUES`` requeues.

``run(task=(i, n))`` (``stilt run --task i/n``) runs one share here, whatever
the backend. The share is taken from all the receptors before the complete
ones are dropped (``task_share``), so tasks that start at different times
never overlap. Any scheduler that can start a command line with an index can
run a project this way.

The unit of work is a receptor. ``run_receptor`` runs the transport model
once per distinct run-settings hash, then runs the stages of every variant
that shares those particles (``run_stages``). A worker handles the variants
of its receptor in config order. A variant whose particles are missing runs
the model, and the variants that share those particles reuse them.

A stage is what is made from a simulation's results after the model runs:
one file per receptor, from the particles, from the files of other stages,
or both. The footprint is the one stage PYSTILT has
(``stilt.footprint.FootprintStage``). A ``stages`` block of ``config.yaml``
names its stage's class by import path, and the class provides the members
of ``stilt.stages.Stage``: a ``name``, a ``config_class``, the ``suffix`` of
its files, the stages it ``reads``, and ``resolve``, ``settings``, ``check``
and ``run``. So the config, the settings records, the check before a run,
the output directory and the worker handle any stage and import none.

For each variant the worker runs the stage files that are missing, a stage
after the ones it reads. A stage's file is made once for the variants whose
settings of it, and of what it reads, are equal, since they share the file.
It is made again whenever what it reads was made again in the same call:
new particles remake every stage file, and a stage that reads another's
file is remade with it. ``Stage.run(config, sim, path)`` writes the file
itself, under a temporary name the worker moves into place, and the worker
never looks inside. The stage's earlier file is removed before it runs, so
a stage that fails leaves none, old or half-written. A stage whose input
failed is left for the next run. A batched model (``batched = True``,
with ``run_many``) is the exception: ``run_receptors`` first runs each of its
variant groups once for all the receptors that need particles, then the
per-receptor loop does the rest. HYSPLIT is not batched.

HYSPLIT has to run in a local folder that holds its input files, the run's
workdir. Each run's workdir is a new path in the compute root
(``_new_workdir`` in the worker: the receptor id, the variant, the
realization in an ensemble, and a random part, as in
``202301011200_-111.85_40.77_5_hrrr_3f9a1c2e``), so projects can share a
compute root and two runs never clear each other's files. A simulation that
runs twice at once keeps its own files in each. The compute root comes from
``--compute-root``, the ``PYSTILT_COMPUTE_ROOT`` environment variable, or
``$TMPDIR/pystilt/<project>`` (``resolve_compute_root`` in the runner;
``Project`` does not read it). It changes where PYSTILT runs and never what it
calculates, so it is not in ``config.yaml``.

After a successful run the particle file and the log are written to the output
directory and the workdir is removed. A failed run's workdir, or every one
with ``keep_workdir: true`` under ``execution:`` in ``config.yaml``, is first
copied to the output's ``scratch/`` folder, replacing an earlier copy for the
same simulation, so CONTROL, SETUP.CFG, and MESSAGE survive the job. A run
killed outright (``SIGKILL``, or a node that fails) cannot clean up, and its
workdir stays in the compute root until that folder is cleaned.

Files are written through a temporary name and renamed into place, so a reader
never sees a partial file, and two workers that run the same receptor by
accident produce equivalent files with the last rename winning. Each of them
runs in its own workdir. No lock is needed.

Slurm tasks rebuild the model from the project in another process on another
node, so anything a worker needs must be in the project or the output
directory, never only in memory.

The observation-driven path sits above this. A reader yields a DataFrame of
soundings, ``stilt.observations`` helpers thin and group it, and each row
becomes a receptor (``receptors_from_soundings``, which also gives the kernel
table, kept with ``project.add_table``). Then ``add_receptors`` and a run go
as usual. Observation logic stays out of ``project.py``.

.. _finished-work:

How PYSTILT tracks finished work
--------------------------------

This section explains what happens behind ``stilt status`` and reruns. You
don't need it to use PYSTILT, but it helps when running at scale, across
machines, or with several projects sharing one output directory.

The result files are the record
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PYSTILT keeps no separate list of which simulations have run. A simulation
is complete when the model's output file exists, and so does the file of
every stage its variant uses (:attr:`stilt.Simulation.is_complete`;
:func:`stilt.output.completed` is the one rule, for one simulation and for
many). For a variant with a footprint that is:

- the particle file,
  ``particles/settings=<variant>-<hash>/date=<day>/<receptor>.parquet``
- the footprint file in the variant's ``footprint/`` folder. A footprint
  that no particle reached is a file with no rows, marked
  ``stilt:empty: true`` in its metadata, and it counts.

A stage does not decide completion: it writes its file or it fails. A
variant that sets a stage to null needs no file of it.

Each result has one expected path, so checking one simulation is a file
lookup. For many simulations (``stilt status``, or planning a run) PYSTILT
lists the date folders the selection falls in, once each, and reads the
answer for every receptor from that. That keeps a project of a hundred
thousand receptors quick to check, and a few simulations of it cheap.
A notebook, a Slurm task, and another project sharing the directory all see
the same files, so they agree on what's finished without talking to each
other. Deleting a result file makes that simulation unfinished again.

A project's simulations are its receptors (``receptors.csv``) run under each
of its variants (``config.yaml``), and ``project.simulations`` is exactly that
set, one row per simulation. Both files are in the project folder before
anything runs, so any worker on any machine can open the project from the
folder alone.

Settings name the folders
^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`../reference/layout` lists what each ``_settings.yaml`` and result
file holds.

Each variant resolves into a record of settings for its particles and one
for each stage it uses (:mod:`stilt.identity`). Its run settings hold
everything that changes the particles: the HYSPLIT parameters, the content
of the meteorology, and the transport model's build. Their hash names the
``particles/`` folder. A stage writes the record of its own settings
(``Stage.settings``): the footprint's holds the footprint config, its grid,
and the hash of the geometry the grid was derived for. Hashed together with
the particles' hash, and the hash of each stage it reads, it names the
folder in the stage's tree (``footprint/``). Each folder holds a
``_settings.yaml`` with the settings written out in full.

That is how PYSTILT knows a setting changed: the variant now hashes to a
folder that does not exist yet, so its receptors run again into it. The old
folder is untouched. It is also how two variants come to share particles:
if only their footprint fields differ, their transport settings hash the
same and they resolve to one ``particles/`` folder.

A folder is found by reading its ``_settings.yaml`` back through the current
config classes and hashing that, rather than by comparing stored digests. A
stage's ``config_class`` reads back what its ``settings`` wrote
(``stilt.stages.check_stage`` holds a stage to that in its tests); a record
whose stage is not installed here is kept as written.
So a setting added in a later version, with a default, still matches folders
written before it existed; a default whose meaning changed does not, which
is right, because the particles would differ.

``stilt status`` lists the folders that no variant of the project uses.
PYSTILT never deletes a folder, because another project may share the
directory; deletion is by hand.

Import layers
-------------

The import contracts in ``pyproject.toml``, checked by ``lint-imports`` in CI
(``just imports``), hold this shape. One ``layers`` contract lists every
top-level module from the CLI down to ``_paths`` and ``exceptions``. A module
imports only the layers below it, and modules that import each other share a
layer. The contract is exhaustive, so a new top-level module fails until it
is placed. ``stilt.migration`` sits above the project config, because
``import_stilt_r`` writes into an output directory through variants. Two
short contracts hold what a layer cannot: only ``stilt.transport`` (its
``get_model``) imports HYSPLIT's package, and the project config reads no
results and runs nothing. STILT-R ran HYSPLIT, so the STILT-R reader and the
``stilt-r`` model are the named exceptions: they read HYSPLIT's ``CONTROL``
and ``SETUP.CFG`` files and record HYSPLIT's settings.

A third short contract holds the line between the core that makes particle
runs and what STILT adds on top of the particles: footprints, transforms,
observations, sampling fields at particles, background, transport error and
the STILT-R model. Outside those modules nothing imports them, except where
the contract's ``ignore_imports`` lists it, each with its reason:
``Simulation``'s footprint, background and transport-error reads,
``Project``'s bulk reads, and ``add_foot``'s use of ``sample_met``. The
config, the settings records, the check, the output directory and the
worker cross nowhere: a stage reaches them by the import path
``config.yaml`` names. The list is the work a code
boundary would leave; a new crossing fails ``just imports``. Two crossings
are not imports between modules, so the contract cannot see them: the
near-field correction (which only the footprint side calls) and
``add_foot`` are functions in ``stilt.particles.table``, and a package's ``__init__`` importing its own
module (the transport registry and the STILT-R model) is not checked.

Configuration
-------------

``ProjectConfig`` is the root, and ``config.yaml`` is its blocks: ``mets``,
``models``, ``stages`` and ``variants`` each map a name to a block, beside
``execution`` and ``output``. No setting sits at the top level; one written
there is refused with a message naming its block.

- A ``models`` block holds one transport model's parameters, the defaults
  of every variant that runs it. The block's name is the handle a variant
  uses (``model:``); the model is the block's ``class`` (an import path) or,
  without one, the built-in model of the block's name. A class has one
  block. The parameters are checked by the model's own config class
  (``stilt.transport.hysplit.HysplitConfig``), reached through
  ``stilt.transport.get_model``: by name when the file loads, and by value
  as a variant runs them (at load for a variant that changes none, else in
  ``resolve``), since a block need not be complete (the ``stilt-r`` model
  has no defaults, and ``n_hours`` has none in any model). Without
  ``models``, the project runs ``hysplit``.
- ``HysplitConfig`` has HYSPLIT's own defaults, in the STILT mode that
  ``FIXED_SETUP`` sets. The settings HYSPLIT decides after reading
  ``SETUP.CFG`` (``capemin``, ``kblt``, ``kmixd``, ``krand``, ``vscales``)
  default to unset and are then not written. STILT-R's values are PYSTILT's
  side: ``stilt.config.STILT_R_HYSPLIT``, which ``stilt init`` (the starter
  config) and ``Project.init`` write into the ``hysplit`` block where it
  gives none, so what a project runs is in its file. ``run_trajectories``
  and a hand-written config get HYSPLIT's. A run records the resolved
  values, so a config that gives STILT-R's values names the same settings
  folder as the old defaults did, while an unset setting is recorded as
  ``null``.
- A ``stages`` block names what is made from each simulation's particles
  by its ``class``. The footprint, ``stilt.footprint.FootprintStage``, is
  the one stage, and its block holds the ``FootprintConfig`` fields. A stage
  runs for every variant; a variant changes its settings under its own
  ``stages`` key (``grid`` merges field by field) or sets it to null for
  particles only. A footprint left with neither a grid nor a geometry is an
  error.
- A model is a variant axis, like a met. A variant's own keys are ``met``,
  ``model`` (needed with several models), ``realizations`` and ``stages``;
  every other key is a setting of its model. Each model's block holds all of
  its parameters, the ones PYSTILT's own code reads among them (the base
  ``TransportConfig``: ``n_hours``, ``seed``). A parameter goes on the base
  only when the core reads it; ``numpar`` and every ``SETUP.CFG`` setting,
  ``veght`` among them, are HYSPLIT's.
- A stored particle file holds what the model wrote. The footprint's own
  fields say what is done with it afterwards: ``hnf_plume`` (the near-field
  correction, applied when a footprint is made), and ``footprint_depth``,
  ``pbl_height`` and ``pbl_height_min``, with which the footprint stage
  computes ``foot`` when it needs one (``pbl_height`` set, or particles
  without ``foot``) and stores it nowhere. On the model's own ``foot`` the
  stage's ``resolve`` takes the run's ``veght`` as ``footprint_depth``, for
  the near-field correction and the record, and refuses a different one
  written in the config. So variants that differ only in how ``foot`` is
  computed share one run, and no stage sets a model setting.
- When it loads, ``ProjectConfig`` checks each block and each declared
  variant (names, met, model, ``realizations``, stage overrides) and rejects
  an unknown key with the nearest known one (``difflib``), reading no other
  file. ``from:`` is rejected.
- ``ProjectConfig.resolve(directory)`` (behind ``project.variants``) turns
  each variant into ``Variant`` objects. It validates the variant's transport
  settings with its model's config class once, keeps ``realizations: N`` as
  one ensemble variant (realization ``k`` runs with ``seed + k``), reads each
  geometry once (a relative file from the project directory) to derive the
  grid. The transport model's build is read on first use
  (``Variant.build``). ``Project.init`` resolves before it writes
  ``config.yaml``.
- Each part owns its config. ``FootprintConfig`` is in
  ``stilt.footprint.config``, ``ExecutionConfig`` in
  ``stilt.execution.config``, and a model's config and met config in its
  package (``met_config_class``; HYSPLIT's ``MetConfig`` is in
  ``stilt.transport.hysplit``). ``ProjectConfig.mets`` holds each met as
  written, and ``ProjectConfig.met_config(name, model)`` checks it with the
  met config of the model that reads it (``model`` a ``models`` block's
  name). With several models, a met no variant uses is not checked. ``Bounds`` and ``Grid`` are in
  ``stilt.spatial``, which needs no shapely, since more than footprints use
  rasters. ``stilt.config`` composes them, so nothing below the project's
  layer imports ``stilt.config``.
- Config classes do no I/O when they validate, so a ``config.yaml`` loads
  offline. Reading a geometry happens in ``ProjectConfig.resolve``; reading
  which weather a met holds (``Variant.met_settings``) and asking the model
  its build (``Variant.build``: for HYSPLIT, the SHA-256 of ``hycs_std`` and
  the version it prints) happen when a variant's hash is first needed. So a
  project opens on a machine without the met's files or ``hycs_std``, and
  ``Output`` compares stored folders on what can be read
  (``Variant.readable_run_settings``).
- A transport config's fields that change no result are listed in its
  ``UNRECORDED`` class variable (HYSPLIT's: ``directory``) and
  left out of the settings records. A met's record is built, not dumped: its
  config's ``settings()``, which for HYSPLIT is the source id in the ARL
  headers (from the first file under ``directory``, or the archive's
  ``source`` for ``download``) and the crop (``MetConfig.crop()``, from
  ``subgrid_bounds`` and ``subgrid_levels``). Where the files are and how
  they are named are never recorded, so moving them keeps every result, and
  two products named alike hash apart. The grid and levels are not in the
  record either, since a product's header changes over its archive's life;
  the met files a run read are recorded in its particle file.
- Lookup reads each stored ``_settings.yaml`` back through the current
  config classes (``stilt.identity``) and re-hashes, so a field added later
  with a default still matches. Stored records ignore keys this version does
  not know.
- Every field is a plain pydantic ``Field(default, description=...)``, and
  the settings inside a block stay flat (``models={"hysplit": {"numpar": ...,
  "seed": ...}}``).
  ``CONTRIBUTING.md`` explains how a HYSPLIT field is routed to
  ``SETUP.CFG``, ``CONTROL``, ``WINDERR``, or ``ZIERR``.
- ``ExecutionConfig`` (``execution:`` in ``config.yaml``) says where
  receptors run. ``backend`` names the place, and each backend's settings
  are in the block of its name (``LocalConfig``, ``SlurmConfig``), ``cpus``
  included; only ``timeout`` and ``keep_workdir`` sit beside ``backend``.
  ``ExecutionConfig.cpus`` and ``.fetch`` read the block ``backend`` names.
  A backend's setting written outside its block is refused. The ``slurm``
  block takes any other key as an ``sbatch`` option, and refuses one that
  reads as a misspelled setting.
- Particle transforms are declared as a default or per variant
  (``transforms: [{kind: ...}]``); ``kind`` may also be the import path of a
  user class.

Implementation notes
--------------------

Results are plain data
^^^^^^^^^^^^^^^^^^^^^^

``sim.particles`` is a pandas DataFrame and ``sim.footprint`` an xarray
DataArray. PYSTILT's methods on them live in ``.stilt`` accessors. A
footprint carries its receptor id as a scalar coordinate (kept through
arithmetic) and the full receptor and settings as attributes
(``stilt_receptor``, ``stilt_footprint``), which the accessor reads. Every
result file records what it needs to be read alone (``read_particles``,
``read_footprint``).

``sim.particles`` and ``sim.footprint`` are ``cached_property`` attributes of
a frozen value. A missing file raises ``FileNotFoundError``, which is never
cached, so the next read tries again, rather than caching a ``None`` that
would hide the result when it lands. ``None`` is only for final states (no
grid, an empty footprint). Use ``has_particles`` and ``has_footprint`` to
test presence; never write to a simulation's ``__dict__``.

``project.simulations`` is built once, and each access returns a copy, so a
user's new column stays theirs. ``add_receptors`` drops the cache. A
``config.yaml`` edited by hand needs a new ``Project(path)``.

Bulk methods and listings
^^^^^^^^^^^^^^^^^^^^^^^^^

A method of ``Project`` that takes a selection works from folder listings and
receptor ids. It never builds a ``Simulation`` per row, which costs
milliseconds each and adds up to many minutes on tens of thousands of
footprints. It opens a file per row only when the file's contents are the
answer (``footprints``, ``jacobian``, ``transport_error``), and builds receptors together
(``Project._receptors``) when it needs them.

``Output.directory`` is a ``Path`` on a filesystem and a universal path
(``UPath``) for a URL (``stilt._paths.location``). Code that touches the
output checks ``isinstance(path, Path)`` for the local fast path
(``os.scandir``, a temporary file and rename, pyarrow on the path) and
otherwise uses the store's fsspec filesystem (``path.fs``), putting a file
directly. Parquet is read through ``stilt._paths.readable``.

Empty footprints
^^^^^^^^^^^^^^^^

When no particle reaches the grid, ``calc_footprint`` raises
``EmptyFootprint`` and the footprint stage writes a footprint file
with no rows, marked ``stilt:empty`` in its metadata. ``sim.is_complete`` and
``sim.has_footprint`` are true, ``sim.footprint`` is ``None``, and
``project.footprints()`` gives it no row and lists it in ``attrs["empty"]``.
PYSTILT never synthesizes a zero-valued footprint for it: a zero enhancement
would flow into a comparison or an inversion unnoticed.

Failure records
^^^^^^^^^^^^^^^

When the model run or a stage fails, the worker writes
``<receptor id>.failure.yaml`` in the
logs of the folder whose result failed (``logs/settings=<key>/date=.../``:
the particles' folder for the run, the stage's own for a stage), and
removes it when that result is written. Its ``stage`` is ``particles`` or
the stage's name. Completion never reads it: a
simulation is done by its result files alone. ``sim.failure``, the
``state``, ``stage``, ``reason``, and ``message`` columns of ``status()``, and
``stilt status`` read it, only for simulations missing a result.

Met that ran out
^^^^^^^^^^^^^^^^

Met that ran out is a failure, and a domain exit is not. The transport model
tells them apart, not the core. HYSPLIT's driver reads its ``WARNING`` file
after each run and fails ``no more meteorology`` (and the other ``metpos``
time warnings in ``FAILURE_PHRASES``) as ``MET_COVERAGE``. Particles that all
left the met's domain or its crop are a complete run, so a particle file may
stop before ``n_hours``.

HYSPLIT's warning is not reliable. It writes only its first ``metpos``
warning per run, so a domain exit can hide met that ran out later. With
``krand: 4``, about one run in five that reaches a met file cut short drops
every particle there with no warning at all and reports ``Complete Hysplit``.
So ``Met.files_for`` fails before running when no file holds a time the run
needs, naming the hours, and ``Met.check`` fails when a file is damaged or the files lack a time
step the run needs (arlmet's ``File.check()`` and ``File.times``, read once
per file in a process).

A local met's files are found from a table of what their headers say
(``catalog.py``): the first index record gives the first time and the size of a
time step, the second the spacing, and the file's size the number of steps. One
table per folder is kept in ``subgrid_dir/catalogs/`` or the user's cache. A run
trusts the table until it has a gap or a chosen file's size or modified time has
changed, and then lists the folder once. ``stilt check`` always brings it up to
date. The first build takes a lock, so a job array does not read an archive once
per task. Stray names (``.partial``, ``.bad``, backups, and ``atomic_path``'s
``.name.<id>.tmp``) stay out by name, because a crop being written can read as
whole time steps.

When local files are cropped, ``crop_dir`` has its own table (``Met.crops``).
``Met._cover`` asks the crops first and the archive only for the stretches they
do not hold, then drops any file wholly inside another, a crop before an
archive file. So a crop serves a run by what it holds, whatever its name and
whether or not the archive still has its source, and the archive's table is not
read at all when the crops hold the whole run. ``Met.readable`` keeps a file
already in ``crop_dir`` as it is, and makes a crop of an archive file again
when the crop under its name does not hold the source's steps (``_holds_as``,
two probes).

After the run, the driver's ``check_met_stop`` catches a silent stop that the
check before it missed, whatever its cause. A run that stopped before
``n_hours`` fails as ``MET_COVERAGE`` when any particle at its last time step
is more than ``MET_STOP_CELLS`` (5) grid cells from every side of the first
met file's grid. The top is not an edge: HYSPLIT's particle mode holds a
particle that rises above it at the top (``advpnt.f``), so the top cannot end
a run. HYSPLIT limits a step to ``tratio`` (0.75) of a cell, so the limit is in
cells, not distance. On HRRR (3 km), NAM (12 km) and NARR (32 km) grids and
their crops, particles that left ended at most 2.0 cells in; on domains a
hundred cells across or more, the furthest-inside particle of runs the met
stopped was 34 cells in or more. On crops 45 to 120 cells across, 3 to 9 % of
the met stops tested passed as domain exits, none in a run's first 6 hours,
with at most one particle in six still in the air and under 3 % of the
footprint lost. HYSPLIT writes ``off spatial domain of all grids`` after a
domain exit and after a silent met stop, so its log cannot tell them apart. The check runs only when ``outdt`` is 0: with fewer rows, a
particle's last row can be far inside the domain though it left. It takes a
particle table and a grid, nothing of HYSPLIT's, so the core
can call it for another transport model that knows its domain.

Realizations
^^^^^^^^^^^^

``realizations: N`` makes one variant whose simulations are ``(receptor,
variant, k)``, with ``k`` in ``0..N-1``, the ``realization`` column of
``project.simulations``. They share one settings folder, with a
``realization=k/`` partition each. The run record says ``ensemble: true``
with the base seed, and realization ``k`` runs with ``seed + k``
(``Variant.transport_for(k)``). An ensemble of one is still an ensemble, so
raising ``N`` only adds partitions. A single run records neither key.

The HYSPLIT seed
^^^^^^^^^^^^^^^^

``SETUP.CFG`` gets ``SEED = -(|seed| + 1)`` (``setup_seed`` in the HYSPLIT
driver), not the user's value. HYSPLIT sets its generator state to
``-1 + SEED``, and under ``krand=2`` the generator (``ran1``) re-initializes
only from a negative value and collapses every state ``>= -1`` onto one
stream, so a positive ``SEED`` would be inert. ``krand=4`` discards the seed
(a clock draw with about 5000 distinct values), and ``krand=1`` uses it only
for the initial turbulent velocity, so ``seed`` requires ``krand=2``.
``krand`` is restricted to HYSPLIT's documented modes because any other value
silently degenerates the turbulence draws.

Realization 0, and a single wind-error variant, use the default seed, as
STILT-R's error run does. The ``winderr`` fidelity scenario (an ``hrrr-err``
variant) relies on this, and the R fidelity fixture applies the same seed
mapping.

Line sources
^^^^^^^^^^^^

In HYSPLIT's ``emspnt.f``, consecutive ``CONTROL`` starting locations at the
same latitude and longitude become one vertical line source, and only the
last pair is released. That is how ``ColumnReceptor`` works (two lines:
bottom and top), and why a ``MultiPointReceptor`` may not repeat a horizontal
location (the constructor raises). HYSPLIT 5.1.0 releases column
particles bottom to top in ``indx`` order (the table's ``particle``), which
``add_release_heights`` falls back to when a model writes no ``age = 0``
release row.

Pressure weighting
^^^^^^^^^^^^^^^^^^

``PressureWeighting`` fits ``ln p = b + a·z`` to the particles' first-step
``(zagl, pres)`` and gives each distinct release height the pressure slab
centered on it (``particle_pwf``), split evenly among the particles released
there (a multipoint receptor releases many per point). The fit is made in the
receptor's own datum: for ``altitude_ref="msl"`` it uses ``zagl + zsfc`` (so
``zsfc`` must be in ``varsiwant``), and the ground closing the bottom slab is
the terrain under the lowest point. ``PressureWeighting.apply`` reads
``altitude_ref`` from the receptor it is given; with none it assumes AGL.
``AveragingKernel`` holds only the kernel. ``calc_footprint`` divides by the
particle count, so weights are scaled by ``N``. Weights sum to the column's
mass fraction (less than 1) by design: the rest of the atmosphere is above
the column top.

The hypsometric fit is load-bearing. HYSPLIT's first output step is already
one time step of turbulence past release: on a real 1000-particle column, 394
of 999 adjacent particles are non-monotone in pressure against release
height. Raw first-step pressures give neighboring particles weights spanning
a factor of 145; the fit gives 1.49, the true hydrostatic ratio across 3 km.
Never "simplify" ``particle_pwf`` to use ``pres`` directly
(``tests/integration/test_pressure_weighting.py`` guards this).

Exact release positions would not help either. ``PARTICLE.DAT`` starts at
``t = -DELT``, never ``t = 0``, but the scatter is not only transport:
``emspnt.f`` places each particle uniformly at random inside its own
``1/numpar`` slab. The air a particle represents is its slab, known from
``numpar`` and the column, which is what ``release_height`` is. Particle data
is needed only for the two-parameter ``p(z)`` fit.

Release heights
^^^^^^^^^^^^^^^

The particle table's contract asks a model to write each particle's release
at ``age = 0``. From it ``add_release_heights`` (``particles/table.py``, a
core step the worker applies to any model's particles, as is
``correct_near_field``) finds which point of a multipoint receptor a particle
left from, or which slab of a column. ``release_height`` is that point's
altitude or that slab's center, never the random height inside the slab.

For a multipoint or slant receptor, ``_multipoint_release_heights`` uses, in
order: ``age = 0`` rows when present (exact); a match on height when every
release altitude differs; then horizontal position, with a warning when
points are closer than 1 km. HYSPLIT 5.1.0, STILT-R's build, writes no ``age = 0``
row. A build that does can be used through ``directory``, and nothing in
PYSTILT changes. ``tests/transport/hysplit/test_release_assignment.py``
covers closely spaced slant points.
