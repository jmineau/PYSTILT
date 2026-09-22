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

PYSTILT is a Python implementation of the [STILT](https://uataq.github.io/stilt/) Lagrangian atmospheric transport model.
It runs backward trajectories with [HYSPLIT](https://www.ready.noaa.gov/HYSPLIT.php) and computes receptor footprints
that map where upwind surface fluxes influence a measurement.

The project is in alpha and focused on a unified execution model that works for one-off runs,
large batch runs, and streaming queue workers.

## Status

PYSTILT is in alpha development. No backward compatibility guarantees before v1.0.

The core transport is stable: HYSPLIT execution, trajectory and footprint generation,
numerical STILT-R parity, and the local and SLURM execution paths are all exercised by the
test suite. The public API may change while the package settles.

## Choose a workflow

- **One-off transport runs** for local analysis and notebooks:
  use `Model.run()` or `stilt run`.
- **Queue-backed batch or service runs** for cloud execution:
  use `Model.register()`, `stilt register`, `stilt pull-worker`, and
  `stilt serve` with a PostgreSQL work queue configured via `PYSTILT_DB_URL`.
  Slurm needs no database: `stilt run --backend slurm`.
- **Column and satellite workflows** for science-facing code:
  use `stilt.observations` to group, select, and lay out soundings as
  `Receptor` objects, and a per-receptor averaging-kernel table to weight
  them inside the same runtime.

## Roadmap

PYSTILT draws design and science inspiration from two sister projects:
[X-STILT](https://github.com/uataq/X-STILT) for column and satellite science workflows, and
[stiltctl](https://github.com/jmineau/air-tracker-stiltctl) for cloud-native execution
patterns. The tables below track what has been absorbed and what remains in scope.
See the full [roadmap](https://jmineau.github.io/PYSTILT/roadmap.html) for more details.

### Execution and orchestration (from stiltctl)

| Feature | Status |
|---|---|
| Pull-mode queue workers (`stilt pull-worker`) | Implemented |
| Long-lived streaming mode (`stilt serve`) | Implemented |
| PostgreSQL-backed work queue for distributed coordination | Implemented |
| Thin CLI → Model → worker call path | Implemented |
| Kubernetes worker deployment | Partial |
| Cloud object store outputs (GCS, S3) | In scope |

### Column and satellite science (from X-STILT)

Full X-STILT feature parity is not a goal. PYSTILT absorbs X-STILT's observation-layer
design and column-weighting concepts without trying to replicate every script.

| Feature | Status |
|---|---|
| `stilt.observations` helpers (overpass grouping, sounding selection, jitter, slant geometry) | Implemented |
| Column receptor support | Implemented |
| Averaging-kernel and pressure-weighting particle transforms | Implemented |
| First-order lifetime decay transform | Implemented |
| Declarative per-footprint transforms in config | Implemented |
| Slant-column receptor support | Implemented |
| User-defined transforms (`kind: my.module.Class`) | Implemented |
| Per-sounding averaging kernels in batch runs (`averaging_kernel` with `table:`) | Implemented |
| Product readers (OCO-2/3, TROPOMI, TCCON) | Out of scope: your reader produces a table of soundings |
| Transport error on the modelled enhancement (`transport_error`) | Implemented |
| Modelled enhancement from a flux field (`Footprint.enhancement`) | Implemented |
| Inventory coupling and background estimation | Deferred |

## Installation

```bash
pip install pystilt
```

For Slurm, Kubernetes, projections, plotting, and cloud object stores:

```bash
pip install "pystilt[complete]"
```

## Quickstart: one-off run

Define a receptor, configure meteorology and footprint grid, then run:

```python
import pandas as pd
import stilt

receptor = stilt.Receptor(
    time=pd.Timestamp("2023-07-15 18:00", tz="UTC"),
    latitude=40.766,
    longitude=-111.848,
    altitude=10,
)

model = stilt.Model(
    project="./my_project",
    receptors=[receptor],
    config=stilt.ModelConfig(
        n_hours=-24,
        numpar=100,
        mets={
            "hrrr": stilt.MetConfig(
                directory="/data/hrrr",
                file_format="hrrr_%Y%m%d.arl",
                file_tres="1h",
            )
        },
        footprints={
            "default": stilt.FootprintConfig(
                grid=stilt.Grid(
                    xmin=-113.0,
                    xmax=-110.5,
                    ymin=40.0,
                    ymax=42.0,
                    xres=0.01,
                    yres=0.01,
                )
            )
        },
    ),
)

handle = model.run()
handle.wait()

sim = list(model.simulations.values())[0]
traj = sim.trajectories
foot = sim.get_footprint("default")
```

## Quickstart: queue/service runtime

```bash
# Queue workers require a PostgreSQL work queue.
export PYSTILT_DB_URL=postgresql://user:pass@host:5432/pystilt

# Initialize project files (config.yaml and receptors.csv)
stilt init ./my_project

# Run with local workers (blocks until complete)
stilt run ./my_project --backend local --n-workers 8

# Persist inputs and enqueue every simulation (receptors x mets)
stilt register ./my_project

# Drain queue from worker processes (batch mode)
stilt pull-worker ./my_project

# Long-lived queue workers (streaming mode)
stilt serve ./my_project

# Check project status
stilt status ./my_project
```

The same queue model is available in Python:

```python
import stilt
from stilt.execution import pull_simulations

model = stilt.Model(project="./my_project")
model.register()
pull_simulations(model, follow=False)  # batch mode
print(model.status())
```

Workers claim simulations from the queue and record done/failed there;
whether outputs exist is always read from the project itself.

## Quickstart: column and satellite soundings

Your reader produces a table with one row per sounding. `stilt.observations`
groups and selects rows, each row becomes a `Receptor`, and each sounding's
averaging kernel goes into a table in the project so every runner applies
the right kernel to the right receptor:

```python
import stilt
from stilt.observations import group_by_overpass
from stilt.transforms import averaging_kernel_table

df = read_my_product(path)                       # your reader: time, longitude, latitude, ak_pressure, ak, ...
df["overpass"] = group_by_overpass(df["time"])   # label rows by overpass; thin with pandas or select_observations_spatial

model = stilt.Model(project="./my_project")      # existing project config on disk
receptors = [stilt.ColumnReceptor(r.time, r.longitude, r.latitude, 0, 3000) for r in df.itertuples()]
model.register(receptors=receptors)

averaging_kernel_table(receptors, levels=df.ak_pressure, values=df.ak).to_parquet(
    model.project.directory / "kernels.parquet"
)
model.run()
```

with the footprint declared once in `config.yaml`:

```yaml
footprints:
  column:
    grid: slv
    transforms:
      - kind: averaging_kernel
        table: kernels.parquet
        coordinate: pres
      - kind: pressure_weighting
```

Slant paths come from `slant_points` and `Receptor.from_points`. See the
[observations guide](https://jmineau.github.io/PYSTILT/advanced/observations.html)
and the [slant columns guide](https://jmineau.github.io/PYSTILT/guides/slant_columns.html).

## Particle transforms

Per-footprint transforms rescale each particle's influence before the
footprint is rasterized. Declare them in config, or pass them in Python:

```yaml
footprints:
  column:
    grid: slv
    transforms:
      - kind: averaging_kernel
        levels: [0.0, 1000.0, 2000.0]
        values: [1.0, 0.8, 0.5]
      - kind: pressure_weighting      # derived from the particles, X-STILT style
      - kind: first_order_lifetime
        lifetime_hours: 4.0
      - kind: mypkg.transforms.MyWeighting   # your own pydantic class with apply()
        some_field: 3
```

A transform is any object with `apply(particles, context)`. See the
[transforms guide](https://jmineau.github.io/PYSTILT/advanced/transforms.html)
for the column-weighting science and for writing your own.

## Accessing results

```python
import pandas as pd

for sim in model.simulations.values():
    traj = sim.trajectories
    foot = sim.get_footprint("default")

# Load footprints across all matching simulations
footprints = model.footprints["default"].load(
    time_range=("2023-01-01", "2023-01-31")
)

coords = [(-111.9, 40.7), (-111.8, 40.8)]
time_bins = pd.interval_range(
    start=pd.Timestamp("2023-01-01 00:00", tz="UTC"),
    end=pd.Timestamp("2023-01-02 00:00", tz="UTC"),
    freq="1h",
)

for footprint in footprints:
    hourly = footprint.aggregate(target=coords, time_bins=time_bins)
```

If a footprint is tracked as `complete-empty`, no NetCDF file is expected for that footprint.
The model APIs treat it as a successful terminal outcome while skipping missing file loads.

## STILT-R parity

PYSTILT footprints match the [uataq/stilt](https://github.com/uataq/stilt) R
implementation on **numerical values** at `rtol=1e-7` per cell, validated by
end-to-end fidelity scenarios against a **pinned upstream commit**.
NetCDF output is not byte-compatible with STILT-R;
it should be read as generic CF-1.8 NetCDF.
See [STILT-R.md](STILT-R.md) for more details.

## Documentation

Full documentation is available at [https://jmineau.github.io/PYSTILT/](https://jmineau.github.io/PYSTILT/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**James Mineau** - [jmineau](https://github.com/jmineau)
