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

Describe a measurement, point PYSTILT at meteorology and a footprint grid, and run:

```python
import stilt

receptor = stilt.PointReceptor(
    time="2023-07-15 18:00",   # UTC
    longitude=-111.848,
    latitude=40.766,
    altitude=10,               # metres above ground
)

project = stilt.Project.init(
    "./my_project",
    receptors=[receptor],
    n_hours=-24,
    numpar=100,
    mets={
        "hrrr": {
            "directory": "/data/hrrr",
            "file_format": "%Y%m%d_%H",
            "file_tres": "6h",
        }
    },
    grid={
        "xmin": -113.0, "xmax": -110.5,
        "ymin": 40.0, "ymax": 42.0,
        "xres": 0.01, "yres": 0.01,
    },
)

project.run()   # returns when the run is done

sim = project.simulation(receptor.id, "hrrr")
particles = sim.particles   # a pandas DataFrame, one row per particle per step
foot = sim.footprint        # an xarray DataArray, one map per hour
```

Everything is saved in `./my_project`. Later, open it again with
`stilt.Project("./my_project")`. A second `run()` only runs what is missing. To change a
setting, edit `config.yaml`.

## Command line

```bash
stilt init ./my_project          # write a starter config.yaml and receptors.csv
stilt run ./my_project           # run every receptor that is not done yet
stilt run ./my_project --n-workers 8   # the same, in 8 parallel processes
stilt status ./my_project        # what has finished
```

To run on a Slurm cluster, add an `execution` section to `config.yaml`. `stilt run` then submits
a job array and returns. No database is needed.

```yaml
execution:
  backend: slurm
  n_workers: 200          # array tasks
  account: my-account     # other keys become #SBATCH options
  partition: my-partition
  time: "02:00:00"
```

## Variants

A variant is a named set of settings. Every receptor runs once under each variant. The top-level
settings in `config.yaml` are the defaults, and a variant lists only what it changes. With no
`variants` section there is one variant per met, named after it.

```yaml
variants:
  hrrr: {}                                  # the defaults
  hrrr-zi08: {ziscale: 0.8}                 # mixed-layer height scaled by 0.8
  hrrr-err:                                 # wind-error ensemble for transport_error
    siguverr: 2.6
    tluverr: 260
    zcoruverr: 450
    horcoruverr: 14
    realizations: 4                         # hrrr-err-0 .. hrrr-err-3
    grid: null                              # particles only, no footprint
  hrrr-ak:                                  # another footprint from the hrrr particles
    transforms: [{kind: averaging_kernel, table: kernels.parquet}]
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
stilt run ./my_project          # submits and returns; add --wait to wait
stilt status ./my_project
```

A task that is preempted or runs out of time is submitted again and skips the receptors it
finished. Whether a simulation is finished is always decided by its files in the output
directory.

## Column and satellite soundings

Read the soundings into a table with one row per sounding. Group and thin the rows with
`stilt.observations`, then make one receptor per row. Each sounding has its own averaging kernel,
so write the kernels to a table in the project:

```python
import stilt
from stilt.observations import group_by_overpass, read_tropomi_ch4
from stilt.transforms import averaging_kernel_table

df = read_tropomi_ch4(path)                      # or read_oco2, read_tccon, or your own reader
df["overpass"] = group_by_overpass(df["time"])   # label rows by overpass

project = stilt.Project("./my_project")         # a project with a config.yaml
receptors = [
    stilt.ColumnReceptor(time=r.time, longitude=r.longitude, latitude=r.latitude, bottom=0, top=3000)
    for r in df.itertuples()
]
project.add_receptors(receptors)

averaging_kernel_table(receptors, levels=df.ak_pressure, values=df.ak).to_parquet(
    project.directory / "kernels.parquet"
)
project.run()
```

Then name the table in `config.yaml`:

```yaml
grid: {xmin: -114.0, xmax: -111.0, ymin: 39.0, ymax: 42.0, xres: 0.01, yres: 0.01}
transforms:
  - kind: averaging_kernel
    table: kernels.parquet
    coordinate: pres
  - kind: pressure_weighting
```

For slanted lines of sight, use `slant_points` and `Receptor.from_points`.
`pressure_altitudes` converts a retrieval's pressure levels to altitudes. See the
[observations guide](https://jmineau.github.io/PYSTILT/advanced/observations.html)
and the [slant columns guide](https://jmineau.github.io/PYSTILT/guides/slant_columns.html).

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

A transform is any object with an `apply(particles, context)` method. The
[transforms guide](https://jmineau.github.io/PYSTILT/advanced/transforms.html)
covers the column-weighting science and how to write your own.

## Loading results

```python
import pandas as pd

sims = project.simulations                       # one row per receptor and variant
january = sims[(sims.variant == "hrrr") & sims.time.between("2023-01-01", "2023-01-31")]

project.status(january)                          # which results exist
footprints = project.load_footprints(january)    # keyed by (receptor, variant)

coords = [(-111.9, 40.7), (-111.8, 40.8)]
time_bins = pd.interval_range(
    start=pd.Timestamp("2023-01-01 00:00"),
    end=pd.Timestamp("2023-01-02 00:00"),
    freq="1h",
    closed="left",
)

for sim_id, footprint in footprints.items():       # keyed by (receptor, variant)
    hourly = footprint.stilt.aggregate(target=coords, time_bins=time_bins)
```

If a simulation's particles never reach the grid, PYSTILT writes a footprint file with no cells
and the reason. The simulation counts as finished, and `load_footprints()` leaves it out.

## STILT-R parity

PYSTILT footprints match the [STILT-R](https://github.com/uataq/stilt) footprints to a relative
tolerance of 1e-7 in every grid cell. The test suite checks this against a pinned STILT-R
commit. The NetCDF files are laid out differently from STILT-R's, so read them as standard
CF-1.8 NetCDF. The [STILT-R parity section](https://jmineau.github.io/PYSTILT/development.html#stilt-r-parity)
of the development docs has the details.

## Roadmap

PYSTILT borrows from two sister projects. [X-STILT](https://github.com/uataq/X-STILT) is the
source of its column and satellite science. [stiltctl](https://github.com/uataq/stiltctl)
showed the thin call path from the CLI through the project to workers handed receptors. Its
queue-backed and Kubernetes execution was tried and removed in favour of batches of receptors
submitted to Slurm through [submitit](https://github.com/facebookincubator/submitit). The full
[roadmap](https://jmineau.github.io/PYSTILT/roadmap.html) has more detail.

### Execution

| Feature | Status |
|---|---|
| Thin CLI → Project → worker call path | Implemented |
| Local runs, in one process or a process pool | Implemented |
| Slurm job arrays, with preempted tasks resubmitted | Implemented |
| Queue-backed workers (PostgreSQL), Kubernetes deployment | Removed |
| Cloud object store outputs (GCS, S3) | Not planned |

### Column and satellite science (from X-STILT)

PYSTILT does not try to match every X-STILT feature. It takes over X-STILT's approach to
observations and column weighting, and leaves the rest.

| Feature | Status |
|---|---|
| `stilt.observations` helpers (overpass grouping, sounding selection, jitter, slant geometry) | Implemented |
| Column receptor support | Implemented |
| Averaging-kernel and pressure-weighting particle transforms | Implemented |
| First-order lifetime decay transform | Implemented |
| Declarative transforms in config (default or per variant) | Implemented |
| Slant-column receptor support | Implemented |
| Slant altitudes from a retrieval's pressure levels (`pressure_altitudes`) | Implemented |
| User-defined transforms (`kind: my.module.Class`) | Implemented |
| Per-sounding averaging kernels in batch runs (`averaging_kernel` with `table:`) | Implemented |
| Product readers (OCO-2/3, TROPOMI, TCCON) | Implemented: `read_tropomi_ch4`, `read_oco2`, `read_tccon`; other instruments as one module each |
| Transport error on the modelled enhancement (`transport_error`) | Implemented |
| Modelled enhancement from a flux field (`foot.stilt.enhancement`) | Implemented |
| Background from a mole-fraction field at the trajectory endpoints (`background`) | Implemented |
| Satellite-derived plume background (forward trajectories) | Implemented |
| Forward runs (positive `n_hours`) for plume and dispersion studies | Implemented; see the plume-background guide |
| Emission-error propagation to the modelled enhancement | Recipe on `foot.stilt.enhancement`; correlated case in fips |
| Inventory readers | Out of scope: a flux field is an xarray array |

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
