"""Release-critical integration coverage for queue service and transforms."""

from __future__ import annotations

import numpy as np

from stilt.config import (
    FirstOrderLifetimeTransformSpec,
    FootprintConfig,
    ModelConfig,
)
from stilt.model import Model
from stilt.service import Service

from .conftest import integration


def _footprint_total(footprint) -> float:
    """Return the total scalar sensitivity in one footprint field."""
    return float(np.asarray(footprint.data.sum()))


@integration
def test_service_submit_and_drain_runs_batch_end_to_end(
    tmp_path,
    wbb_receptor,
    wbb_config,
):
    """Service.submit + Service.drain should execute one queued batch end to end."""
    service = Service(
        project=tmp_path / "service_queue",
        config=wbb_config,
        receptors=[wbb_receptor],
    )

    sim_ids = service.submit(batch_id="service_batch")
    assert len(sim_ids) == 1

    pending = service.status()
    assert pending.total == 1
    assert pending.pending == 1
    assert pending.running == 0
    assert pending.completed == 0
    assert pending.failed == 0

    service.drain(cpus=1, poll_interval=0.1, lease_ttl=30.0)

    complete = service.status()
    assert complete.total == 1
    assert complete.pending == 0
    assert complete.running == 0
    assert complete.completed == 1
    assert complete.failed == 0

    batch = service.batch_status("service_batch")
    assert batch.total == 1
    assert batch.completed == 1
    assert batch.is_complete is True

    assert service.active_claims() == []
    attempts = service.attempts(sim_ids[0])
    assert len(attempts) == 1
    assert attempts[0].outcome == "complete"

    sim_dir = service.model.directory / "simulations" / "by-id" / sim_ids[0]
    assert list(sim_dir.glob("*.parquet")), "No trajectory parquet from service drain"
    assert list(sim_dir.glob("*_foot.nc")), "No footprint artifact from service drain"


@integration
def test_declarative_transform_config_changes_real_footprint(
    tmp_path,
    met_dir,
    wbb_receptor,
    wbb_grid,
):
    """Per-footprint transform specs should affect real footprint output."""
    config = ModelConfig(
        mets={
            "hrrr": {
                "directory": met_dir,
                "file_format": "%Y%m%d.%Hz.hrrra",
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
        footprints={
            "baseline": FootprintConfig(grid=wbb_grid),
            "lifetime": FootprintConfig(
                grid=wbb_grid,
                transforms=[
                    FirstOrderLifetimeTransformSpec(
                        kind="first_order_lifetime",
                        lifetime_hours=1.0,
                        time_column="time",
                        time_unit="min",
                    )
                ],
            ),
        },
    )

    model = Model(
        project=tmp_path / "configured_transforms",
        config=config,
        receptors=[wbb_receptor],
    )
    model.run()

    [baseline] = model.get_footprints("baseline")
    [lifetime] = model.get_footprints("lifetime")

    assert len(lifetime.config.transforms) == 1

    baseline_values = baseline.data.to_numpy()
    lifetime_values = lifetime.data.to_numpy()

    assert baseline_values.shape == lifetime_values.shape
    assert np.all(lifetime_values <= baseline_values + 1e-12)
    assert np.any(lifetime_values < baseline_values - 1e-12)
    assert _footprint_total(lifetime) < _footprint_total(baseline)
