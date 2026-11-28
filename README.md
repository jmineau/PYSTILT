# PYSTILT

[![Tests](https://github.com/jmineau/PYSTILT/actions/workflows/tests.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/tests.yml)
[![Documentation](https://github.com/jmineau/PYSTILT/actions/workflows/docs.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/docs.yml)
[![Code Quality](https://github.com/jmineau/PYSTILT/actions/workflows/quality.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/jmineau/PYSTILT/branch/main/graph/badge.svg)](https://codecov.io/gh/jmineau/PYSTILT)
[![STILT-R parity](https://github.com/jmineau/PYSTILT/actions/workflows/stilt-r.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/stilt-r.yml)
[![PyPI version](https://badge.fury.io/py/pystilt.svg)](https://badge.fury.io/py/pystilt)
[![Python Version](https://img.shields.io/pypi/pyversions/pystilt.svg)](https://pypi.org/project/pystilt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22211796.svg)](https://doi.org/10.5281/zenodo.22211796)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyrefly](https://img.shields.io/badge/types-pyrefly-blue.svg)](https://pyrefly.org)

PYSTILT is a Python version of the [STILT](https://uataq.github.io/stilt/) atmospheric transport model.
It follows air backward in time from a measurement with [HYSPLIT](https://www.ready.noaa.gov/HYSPLIT.php)
and computes a footprint, a map of which upwind surface areas influenced the measurement and by how much.
PYSTILT is built on [runtraj](https://github.com/jmineau/runtraj), which runs HYSPLIT and makes the
particles; PYSTILT adds the footprint and what is computed from it. Installing PYSTILT installs
runtraj, and some errors and imports below name it.
If you run HYSPLIT on its own, PYSTILT also runs it from Python for particles alone; see
[PYSTILT for HYSPLIT users](https://jmineau.github.io/PYSTILT/stable/guides/hysplit_users.html).

## Status

PYSTILT is alpha software. There are no backward compatibility guarantees before v1.0, and the
public API may still change.

The test suite covers HYSPLIT runs, trajectories and footprints,
agreement with STILT-R, and runs on a local machine and on Slurm.

## Installation

```bash
pip install "pystilt[visualization]"
```

This installs PYSTILT and matplotlib and cartopy for plotting maps. For projected grids,
aggregation over shapefiles, downloading meteorology from NOAA, and reading STILT-R output,
install everything:

```bash
pip install "pystilt[complete]"
```

PYSTILT runs NOAA's HYSPLIT program, `hycs_std`, but does not include it or its data tables.
Download HYSPLIT from [NOAA's Air Resources Laboratory](https://www.ready.noaa.gov/HYSPLIT.php),
which asks you to accept the
[HYSPLIT terms of use](https://www.ready.noaa.gov/HYSPLIT_agreement.php) first. Unpack it, then
set `directory` under `models: hysplit:` in `config.yaml` to the folder it makes, or put that
folder's `exec/` on `PATH`:

```yaml
models:
  hysplit:
    directory: /path/to/hysplit
```

[Installation](https://jmineau.github.io/PYSTILT/stable/getting_started/installation.html)
has the details and other builds.

## Quickstart

Describe a measurement as a release (STILT's receptor: a release placed at a
measurement and followed backward in time), follow particles back from it
through the meteorology, and calculate its footprint:

```python
import stilt

release = stilt.PointRelease(
    time="2023-07-15 18:00",   # UTC
    longitude=-111.848,
    latitude=40.766,
    altitude=10,               # meters above ground
)
met = "/data/hrrr"             # a folder of ARL files; change it to yours (see below)
grid = stilt.Grid(xmin=-113.0, xmax=-110.5, ymin=40.0, ymax=42.0, xres=0.01, yres=0.01)

particles = stilt.run_trajectories(release, met, n_hours=-24, numpar=100)
foot = stilt.calc_footprint(particles, release, grid)
```

The met folder must hold ARL-format files that cover the release and the 24 hours before it;
without them the run stops with `MeteorologyError: No met file in ...`. If you have none,
PYSTILT can download them from NOAA's archive
(`pip install "pystilt[download]"`); the
[Meteorology guide](https://jmineau.github.io/PYSTILT/stable/guides/meteorology.html)
says how.

`particles` is a pandas DataFrame, one row per particle per step, and
`foot` an xarray DataArray, one map per hour.

For many releases, a project runs each one and keeps the results:

```python
project = stilt.Project.init(
    "./my_project",
    releases=[release],
    mets={"hrrr": met},
    models={"hysplit": {"n_hours": -24, "numpar": 100}},
    stages={"footprint": {"class": "stilt.footprint.FootprintStage", "grid": grid}},
    variants={"hrrr": {}},
)
project.run()   # returns when the run is done

sim = project.simulation(release.id, "hrrr")
sim.particles, sim.footprint
```

Everything is saved in `./my_project`. Later, open it again with
`stilt.Project("./my_project")`. A second `run()` only runs what is missing. To change a
setting, edit `config.yaml`.

## Command line

```bash
stilt init ./my_project          # write a starter config.yaml and releases.csv
stilt check ./my_project         # list what would stop a run, before any run
stilt run ./my_project           # run every release that is not done yet
stilt run ./my_project --cpus 8  # the same, eight releases at a time
stilt status ./my_project        # what has finished, and the settings folders
```

`stilt run` exits with 0 when every simulation it ran is complete, 1 when some failed, and 3
when it was stopped, or a connection error stopped some, before they finished. An error it did not expect exits with 3 when it is a
connection or I/O error (a bucket that cannot be reached), which a retry may get past, and with 4
for any other, which a retry would repeat. `stilt check` exits with 0 when it finds nothing and 1
when it lists anything. Every command exits with 2 when its command line is wrong or the project's
files are invalid (`config.yaml` that is not valid YAML, a bad row in `releases.csv`); `stilt run`
and `stilt submit` also when a met directory does not exist or holds no met files, and `stilt status`
also then, unless the output holds no result the project could own (it reports every simulation pending).

## Variants

A variant is a named set of settings. Every release runs once under each variant. `config.yaml`
is blocks: `models` holds the transport model's settings and `stages` the footprint's, which are
the defaults, and a variant lists only what it changes. The `variants` block declares them all;
`hrrr: {}` runs the defaults.

```yaml
mets:
  hrrr: /data/hrrr
models:
  hysplit: {n_hours: -24, numpar: 1000}
stages:
  footprint:
    class: stilt.footprint.FootprintStage
    grid: {xmin: -113.0, xmax: -110.5, ymin: 40.0, ymax: 42.0, xres: 0.01, yres: 0.01}
variants:
  hrrr: {}                                  # the defaults
  hrrr-zi08: {ziscale: 0.8}                 # mixed-layer height scaled by 0.8
  hrrr-err:                                 # wind-error ensemble for transport_error
    siguverr: 2.6
    tluverr: 260
    zcoruverr: 450
    horcoruverr: 14
    realizations: 4                         # realizations 0 to 3 of hrrr-err
    stages: {footprint: null}               # particles only, no footprint
  hrrr-ak:                                  # another footprint from the hrrr particles
    stages:
      footprint:
        transforms: [{kind: averaging_kernel, table: kernels}]
```

Each release under each variant is one simulation. Results go to the output directory
(`./output` by default), in a folder named for the settings they were made with. Variants whose
transport settings match share one HYSPLIT run and differ only in the footprint, as `hrrr-ak`
does here. Adding a variant runs only what is new, and changing a setting writes to a new folder
beside the old one.

## On a cluster

Add an `execution` section to `config.yaml` and run the same command. PYSTILT splits the
unfinished releases among the tasks of one Slurm job array and submits it:

```yaml
execution:
  backend: slurm
  slurm:
    n_workers: 200        # array tasks running at once
    account: my-account
    partition: my-partition
    time: "02:00:00"
    mem: 4G
```

```bash
stilt submit ./my_project       # submits and returns; stilt run waits
stilt status ./my_project
```

Each task runs `stilt run --task I/N`, so a task is a command line you can read and rerun. A
task that is preempted or nears its time limit is requeued and skips the releases it
finished. Whether a simulation is finished is always decided by its files in the output
directory. The same `--task` runs a project as a Kubernetes indexed Job.

## Column and satellite soundings

Read the soundings into a table with one row per sounding, then make a release for each.
Each sounding has its own averaging kernel, which goes into the project as a table:

```python
import stilt
from stilt.observations import read_tropomi_ch4, releases_from_soundings

df = read_tropomi_ch4(path)                      # or read_oco2, read_tccon, or your own reader
releases, kernels = releases_from_soundings(df[df.good], "slant", top=3000)

project = stilt.Project("./my_project")         # a project with a config.yaml
project.add_releases(releases)
project.add_table("kernels", kernels)
project.run()
```

Then name the table under `stages: footprint:` in `config.yaml`:

```yaml
stages:
  footprint:
    class: stilt.footprint.FootprintStage
    grid: {xmin: -114.0, xmax: -111.0, ymin: 39.0, ymax: 42.0, xres: 0.01, yres: 0.01}
    transforms:
      - kind: averaging_kernel
        table: kernels
        coordinate: pres
      - kind: pressure_weighting
```

`"column"` makes vertical columns in place of slant lines of sight. The
[satellite tutorial](https://jmineau.github.io/PYSTILT/stable/tutorials/satellite_column.html) goes from
an orbit to modeled and observed enhancements.

## Particle transforms

A transform changes how much each particle counts before the footprint is calculated. List them
under `stages: footprint:` in `config.yaml`, as a default or for one variant:

```yaml
stages:
  footprint:
    class: stilt.footprint.FootprintStage
    grid: {xmin: -114.0, xmax: -111.0, ymin: 39.0, ymax: 42.0, xres: 0.01, yres: 0.01}
    transforms:
      - kind: averaging_kernel
        levels: [0.0, 1000.0, 2000.0]
        values: [1.0, 0.8, 0.5]
      - kind: pressure_weighting
      - kind: first_order_lifetime
        lifetime_hours: 4.0
      - kind: mypkg.transforms.BoundaryLayerOnly   # your own pydantic class with an apply() method
        max_height: 1200
```

A transform is any object with an `apply(particles, release=None, directory=None)` method. The
[transforms guide](https://jmineau.github.io/PYSTILT/stable/guides/transforms.html)
covers the column-weighting science and how to write your own.

## Loading results

```python
import pandas as pd

import stilt

sims = project.simulations                       # one row per release and variant
january = sims[(sims.variant == "hrrr") & sims.time.between("2023-01-01", "2023-01-31")]

project.status(january)                          # which results exist
footprints = project.footprints(january)         # one dataset: release, hour, lat, lon
mean = footprints.foot.sum("hour").mean("release")

windows = stilt.Mesh.from_windows([(-111.9, 40.7), (-111.8, 40.8)], size=0.05)
time_bins = pd.interval_range(
    start=pd.Timestamp("2022-12-31 00:00"),  # backward footprints reach a day back
    end=pd.Timestamp("2023-02-01 00:00"),
    freq="1h",
    closed="left",
)
H = project.jacobian(january, windows, time_bins)  # sparse: releases by (hour, window)
```

`runtraj.sampling.sample_met` adds met columns (`mlht`, `dens`, `pres`, `zsfc`) to a particle
table, sampled from the met files at each particle's position and time:

```python
from runtraj.particles import particles_metadata
from runtraj.sampling import sample_met

met_files = particles_metadata(sim.particles_path).met_files
sampled = sample_met(sim.particles.drop(columns="dens"), met_files, columns=["dens"])
```

If a simulation's particles never reach the grid, PYSTILT writes a footprint file with no cells
and the reason. The simulation counts as finished, and `project.footprints()` lists it in
`attrs["empty"]`.

Runs made with STILT-R or X-STILT are read from their `out/by-id` folder, one simulation at a
time, as a release, a particle table, and a footprint (`pip install "pystilt[stilt-r]"`):

```python
from stilt.migration import read_stilt_r

for release, particles, foot in read_stilt_r("my_project/out/by-id"):
    print(release.id, len(particles), foot is not None)
```

`stilt.migration.import_stilt_r` writes them into a project's output directory instead, and prints
the `stilt-r` variant that reads them, so `project.status()` and `project.jacobian()` work on runs
STILT-R made.

## STILT-R parity

PYSTILT footprints match the [STILT-R](https://github.com/uataq/stilt) footprints to a relative
tolerance of 1e-7 in every grid cell, on longitude/latitude grids. The test
suite checks this on 21 scenarios against a pinned STILT-R commit. A few cases differ on
purpose: smoothing on a projected grid, grids and particles near 180°, and the near-field
correction for releases above sea level. The NetCDF files are laid out differently from STILT-R's, so read them as standard
CF-1.8 NetCDF. The [STILT-R parity page](https://jmineau.github.io/PYSTILT/stable/development/stilt_r_parity.html)
of the development docs lists every difference.

## Roadmap

What is implemented and what is planned is on the
[roadmap](https://jmineau.github.io/PYSTILT/stable/development/roadmap.html) page of the documentation.

## Use of AI coding agents

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science.

## Documentation

The full documentation is at [https://jmineau.github.io/PYSTILT/](https://jmineau.github.io/PYSTILT/).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up and submit changes.

## License

PYSTILT is released under the MIT License. See the LICENSE file.

PYSTILT runs HYSPLIT's `hycs_std`, which it does not include. HYSPLIT is NOAA Air
Resources Laboratory (ARL) software, not covered by the MIT License, and is subject to the
[HYSPLIT terms of use](https://www.ready.noaa.gov/HYSPLIT_agreement.php), which ask that
publications using it cite NOAA ARL (Stein et al., 2015,
[doi:10.1175/BAMS-D-14-00110.1](https://doi.org/10.1175/BAMS-D-14-00110.1)) and do not
present results as official NOAA material.

## Author

**James Mineau** - [jmineau](https://github.com/jmineau)
