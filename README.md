# PYSTILT

[![Tests](https://github.com/jmineau/PYSTILT/actions/workflows/tests.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/tests.yml)
[![Documentation](https://github.com/jmineau/PYSTILT/actions/workflows/docs.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/docs.yml)
[![Code Quality](https://github.com/jmineau/PYSTILT/actions/workflows/quality.yml/badge.svg)](https://github.com/jmineau/PYSTILT/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/jmineau/PYSTILT/branch/main/graph/badge.svg)](https://codecov.io/gh/jmineau/PYSTILT)
[![PyPI version](https://badge.fury.io/py/pystilt.svg)](https://badge.fury.io/py/pystilt)
[![Python Version](https://img.shields.io/pypi/pyversions/pystilt.svg)](https://pypi.org/project/pystilt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/pyright-checked-brightgreen.svg)](https://github.com/microsoft/pyright)

PYSTILT is a Python implementation of the [STILT](https://uataq.github.io/stilt/) Lagrangian atmospheric transport model.
It runs backward trajectories with [HYSPLIT](https://www.ready.noaa.gov/HYSPLIT.php) and computes receptor footprints
that map where upwind surface fluxes influence a measurement.

The project is in alpha and focused on a unified execution model that works for one-off runs,
large batch runs, and streaming queue workers.

## Choose a workflow

- **One-off transport runs** for local analysis and notebooks:
  use `Model.run()` or `stilt run`.
- **Queue-backed batch or service runs** for HPC/cloud execution:
  use `Model.submit()`, `stilt.Service`, `stilt worker`, and `stilt serve`.
- **Observation-driven workflows** for science-facing code:
  use `stilt.observations` to turn normalized observations into `Receptor`
  objects before feeding them into the same runtime.

## Alpha execution semantics

| Area | Current behavior |
|---|---|
| Delivery guarantee | At-least-once processing. A simulation can be retried after interruption or failure. |
| Trajectory status | `pending -> running -> complete` or `failed`. |
| Footprint status | `complete`, `complete-empty`, or `failed` per footprint name. |
| Empty footprint | Treated as terminal success (`complete-empty`), not failure. |
| Reruns | `skip_existing=True` avoids rework for already complete outputs; `skip_existing=False` forces rerun. |

For details see the user guide pages on running and execution.

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
# Initialize project files (config.yaml and receptors.csv)
stilt init ./my_project

# Run with local workers (blocks until complete)
stilt run ./my_project --backend local --n-workers 8

# Register work without starting workers
stilt submit ./my_project --batch-id daily_2026_04_13

# Drain queue from worker processes (batch mode)
stilt worker ./my_project --cpus 4

# Long-lived queue workers (streaming mode)
stilt serve ./my_project --cpus 4

# Check project status
stilt status ./my_project
```

The same queue model is available in Python:

```python
import stilt

service = stilt.Service(project="./my_project")
service.submit(batch_id="daily_2026_04_14")
service.drain(cpus=4)   # batch mode
# or: service.serve(cpus=4)  # long-lived worker
```

In all modes, workers claim simulations from the repository and write terminal
state directly back to the same queue model.

## Quickstart: observation layer

PYSTILT also includes a narrow science-facing layer in `stilt.observations`.
It is designed to sit above `Receptor`, not replace the transport/runtime core.

```python
import stilt
from stilt.observations import PointSensor

sensor = PointSensor(name="tower", supported_species=("co2",))
observations = [
    sensor.make_observation(
        time="2023-01-01 12:00:00",
        latitude=40.77,
        longitude=-111.85,
        altitude=30.0,
        observation_id="tower-001",
    )
]

[scene] = sensor.group_scenes(observations)
receptors = [sensor.build_receptor(obs) for obs in scene.observations]

service = stilt.Service(project="./my_project")  # existing project config on disk
service.submit(receptors=receptors, batch_id=scene.batch_id)
```

Direct `Observation(...)` construction is still available when you already
have a separate product-specific normalization layer. The sensor helper just
keeps the common path less repetitive.

This layer currently focuses on:

- normalized `Observation` and `Scene` objects
- geometry/operator metadata
- generic point/column sensor families
- observation-to-receptor conversion

See `docs/user_guide/observations.rst` for the intended workflow boundary.

## Declarative transforms

Per-footprint transforms can be declared in config instead of embedded as
ad hoc callbacks.

```yaml
footprints:
  column:
    grid: slv
    transforms:
      - kind: vertical_operator
        mode: ak_pwf
        levels: [0.0, 1000.0, 2000.0]
        values: [0.2, 0.5, 0.3]
        coordinate: xhgt
      - kind: first_order_lifetime
        lifetime_hours: 4.0
        time_column: time
        time_unit: min
```

The built-in transform interface is intentionally small:

- vertical operator weighting
- first-order lifetime decay
- runtime typed transforms for more advanced Python workflows

## Accessing results

```python
import pandas as pd

for sim in model.simulations.values():
    traj = sim.trajectories
    foot = sim.get_footprint("default")

# Load footprints across all matching simulations
footprints = model.get_footprints(
    "default",
    time_range=("2023-01-01", "2023-01-31"),
)

coords = [(-111.9, 40.7), (-111.8, 40.8)]
time_bins = pd.interval_range(
    start=pd.Timestamp("2023-01-01 00:00", tz="UTC"),
    end=pd.Timestamp("2023-01-02 00:00", tz="UTC"),
    freq="1h",
)

for footprint in footprints:
    hourly = footprint.aggregate(coords=coords, time_bins=time_bins)
```

If a footprint is tracked as `complete-empty`, no NetCDF file is expected for that footprint.
The model APIs treat it as a successful terminal outcome while skipping missing file loads.

## Documentation map

- `docs/getting_started/quickstart.rst`: first local run
- `docs/user_guide/running.rst`: run semantics and queue workflows
- `docs/user_guide/service.rst`: `stilt.Service` and Kubernetes helpers
- `docs/user_guide/observations.rst`: observation/scenes/sensors/transforms
- `examples/cloud/`: minimal Kubernetes deployment templates

## Documentation

Full documentation is available at [https://jmineau.github.io/PYSTILT/](https://jmineau.github.io/PYSTILT/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**James Mineau** - [jmineau](https://github.com/jmineau)
