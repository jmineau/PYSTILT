# PYSTILT

[![Tests](https://github.com/jmineau/PYSTILT/actions/workflows/tests.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/tests.yml)
[![Documentation](https://github.com/jmineau/PYSTILT/actions/workflows/docs.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/docs.yml)
[![Code Quality](https://github.com/jmineau/PYSTILT/actions/workflows/quality.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/jmineau/PYSTILT/branch/main/graph/badge.svg)](https://codecov.io/gh/jmineau/PYSTILT)
[![PyPI version](https://badge.fury.io/py/pystilt.svg)](https://badge.fury.io/py/pystilt)
[![Python Version](https://img.shields.io/pypi/pyversions/pystilt.svg)](https://pypi.org/project/pystilt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22211796.svg)](https://doi.org/10.5281/zenodo.22211796)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/pyright-checked-brightgreen.svg)](https://github.com/microsoft/pyright)

PYSTILT is a Python version of the [STILT](https://uataq.github.io/stilt/) atmospheric transport model.
It follows air backward in time from a measurement with [HYSPLIT](https://www.ready.noaa.gov/HYSPLIT.php)
and computes a footprint, a map of which upwind surface areas influenced the measurement and by how much.

## Status

PYSTILT is alpha software. There are no backward compatibility guarantees before v1.0, and the
public API may still change.

The transport core is stable. The test suite covers HYSPLIT runs, trajectories and footprints,
agreement with STILT-R, and runs on a local machine and on Slurm.

## Installation

```bash
pip install pystilt
```

For plotting, projected grids, aggregation over shapefiles, and downloading meteorology from
NOAA, install everything:

```bash
pip install "pystilt[complete]"
```

## Quickstart

Describe a measurement, follow particles back from it through the
meteorology, and calculate its footprint:

```python
import stilt

receptor = stilt.PointReceptor(
    time="2023-07-15 18:00",   # UTC
    longitude=-111.848,
    latitude=40.766,
    altitude=10,               # metres above ground
)
met = {"directory": "/data/hrrr", "file_format": "%Y%m%d_%H", "file_tres": "6h"}
grid = stilt.Grid(xmin=-113.0, xmax=-110.5, ymin=40.0, ymax=42.0, xres=0.01, yres=0.01)

particles = stilt.run_trajectories(receptor, met, n_hours=-24, numpar=100)
foot = stilt.calc_footprint(particles, receptor, grid)
```

`particles` is a pandas DataFrame, one row per particle per step, and
`foot` an xarray DataArray, one map per hour.

For many receptors, a project runs each one and keeps the results:

```python
project = stilt.Project.init(
    "./my_project",
    receptors=[receptor],
    mets={"hrrr": met},
    variants={"hrrr": {}},
    n_hours=-24,
    numpar=100,
    grid=grid,
)
project.run()   # returns when the run is done

sim = project.simulation(receptor.id, "hrrr")
sim.particles, sim.footprint
```

Everything is saved in `./my_project`. Later, open it again with
`stilt.Project("./my_project")`. A second `run()` only runs what is missing. To change a
setting, edit `config.yaml`.

## Command line

```bash
stilt init ./my_project          # write a starter config.yaml and receptors.csv
stilt run ./my_project           # run every receptor that is not done yet
stilt run ./my_project --cpus 8  # the same, eight receptors at a time
stilt status ./my_project        # what has finished, and the settings folders
```

`stilt run` exits with 0 when every simulation it ran is complete, 1 when some failed, and 3
when it was stopped before they finished. Every command exits with 2 when its command line is wrong.

## Variants

A variant is a named set of settings. Every receptor runs once under each variant. The top-level
settings in `config.yaml` are the defaults, and a variant lists only what it changes. The
`variants` section declares them all; `hrrr: {}` runs the defaults.

```yaml
variants:
  hrrr: {}                                  # the defaults
  hrrr-zi08: {ziscale: 0.8}                 # mixed-layer height scaled by 0.8
  hrrr-err:                                 # wind-error ensemble for transport_error
    siguverr: 2.6
    tluverr: 260
    zcoruverr: 450
    horcoruverr: 14
    realizations: 4                         # realizations 0 to 3 of hrrr-err
    grid: null                              # particles only, no footprint
  hrrr-ak:                                  # another footprint from the hrrr particles
    transforms: [{kind: averaging_kernel, table: kernels}]
```

Each receptor under each variant is one simulation. Results go to the output directory
(`./output` by default), in a folder named for the settings they were made with. Variants whose
transport settings match share one HYSPLIT run and differ only in the footprint, as `hrrr-ak`
does here. Adding a variant runs only what is new, and changing a setting writes to a new folder
beside the old one.

## On a cluster

Add an `execution` section to `config.yaml` and run the same command. PYSTILT splits the
unfinished receptors among the tasks of one Slurm job array and submits it:

```yaml
execution:
  backend: slurm
  n_workers: 200          # array tasks
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
task that is preempted or nears its time limit requeues itself and skips the receptors it
finished. Whether a simulation is finished is always decided by its files in the output
directory. The same `--task` runs a project as a Kubernetes indexed Job.

## Column and satellite soundings

Read the soundings into a table with one row per sounding, then make a receptor for each.
Each sounding has its own averaging kernel, which goes into the project as a table:

```python
import stilt
from stilt.observations import read_tropomi_ch4, receptors_from_soundings

df = read_tropomi_ch4(path)                      # or read_oco2, read_tccon, or your own reader
receptors, kernels = receptors_from_soundings(df[df.good], "slant", top=3000)

project = stilt.Project("./my_project")         # a project with a config.yaml
project.add_receptors(receptors)
project.add_table("kernels", kernels)
project.run()
```

Then name the table in `config.yaml`:

```yaml
grid: {xmin: -114.0, xmax: -111.0, ymin: 39.0, ymax: 42.0, xres: 0.01, yres: 0.01}
transforms:
  - kind: averaging_kernel
    table: kernels
    coordinate: pres
  - kind: pressure_weighting
```

`"column"` makes vertical columns in place of slant lines of sight. The
[satellite tutorial](https://jmineau.github.io/PYSTILT/tutorials/satellite_column.html) goes from
an orbit to modelled and observed enhancements.

## Particle transforms

A transform changes how much each particle counts before the footprint is calculated. List them
in `config.yaml`, as a default or for one variant:

```yaml
transforms:
  - kind: averaging_kernel
    levels: [0.0, 1000.0, 2000.0]
    values: [1.0, 0.8, 0.5]
  - kind: pressure_weighting
  - kind: first_order_lifetime
    lifetime_hours: 4.0
  - kind: mypkg.transforms.MyWeighting   # your own pydantic class with an apply() method
    some_field: 3
```

A transform is any object with an `apply(particles, receptor=None, directory=None)` method. The
[transforms guide](https://jmineau.github.io/PYSTILT/guides/transforms.html)
covers the column-weighting science and how to write your own.

## Loading results

```python
import pandas as pd

import stilt

sims = project.simulations                       # one row per receptor and variant
january = sims[(sims.variant == "hrrr") & sims.time.between("2023-01-01", "2023-01-31")]

project.status(january)                          # which results exist
footprints = project.footprints(january)         # one dataset: receptor, hour, lat, lon
mean = footprints.foot.sum("hour").mean("receptor")

windows = stilt.Mesh.from_windows([(-111.9, 40.7), (-111.8, 40.8)], size=0.05)
time_bins = pd.interval_range(
    start=pd.Timestamp("2022-12-31 00:00"),  # backward footprints reach a day back
    end=pd.Timestamp("2023-02-01 00:00"),
    freq="1h",
    closed="left",
)
H = project.jacobian(january, windows, time_bins)  # sparse: receptors by (hour, window)
```

If a simulation's particles never reach the grid, PYSTILT writes a footprint file with no cells
and the reason. The simulation counts as finished, and `project.footprints()` lists it in
`attrs["empty"]`.

## STILT-R parity

PYSTILT footprints match the [STILT-R](https://github.com/uataq/stilt) footprints to a relative
tolerance of 1e-7 in every grid cell. The test suite checks this against a pinned STILT-R
commit. The NetCDF files are laid out differently from STILT-R's, so read them as standard
CF-1.8 NetCDF. The [STILT-R parity section](https://jmineau.github.io/PYSTILT/development.html#stilt-r-parity)
of the development docs has the details.

## Roadmap

What is implemented and what is planned is on the
[roadmap](https://jmineau.github.io/PYSTILT/roadmap.html) page of the documentation.

## Use of AI coding agents

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science.

## Documentation

The full documentation is at [https://jmineau.github.io/PYSTILT/](https://jmineau.github.io/PYSTILT/).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up and submit changes.

## License

PYSTILT is released under the MIT License. See the LICENSE file.

## Author

**James Mineau** - [jmineau](https://github.com/jmineau)
