"""Release-critical integration coverage for queue runtime and transforms."""

from __future__ import annotations

import numpy as np
import pytest

from stilt.config import (
    FirstOrderLifetimeTransformSpec,
    FootprintConfig,
    ModelConfig,
)
from stilt.errors import ConfigValidationError
from stilt.execution import pull_simulations
from stilt.model import Model

from .conftest import integration


def _footprint_total(footprint) -> float:
    """Return the total scalar sensitivity in one footprint field."""
    return float(np.asarray(footprint.data.sum()))


@integration
def test_pull_simulations_requires_runtime_queue_backend(
    tmp_path,
    wbb_receptor,
    wbb_config,
):
    """pull_simulations should fail clearly when only the local SQLite index exists."""
    model = Model(
        project=tmp_path / "service_queue",
        config=wbb_config,
        receptors=[wbb_receptor],
    )

    sim_ids = model.register_pending()
    assert len(sim_ids) == 1

    pending = model.status()
    assert pending.total == 1
    assert pending.pending == 1
    assert pending.running == 0
    assert pending.completed == 0
    assert pending.failed == 0

    with pytest.raises(ConfigValidationError, match="claim-capable index backend"):
        pull_simulations(model, poll_interval=0.1)


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

    [baseline] = model.footprints["baseline"].load()
    [lifetime] = model.footprints["lifetime"].load()

    assert len(lifetime.config.transforms) == 1

    baseline_values = baseline.data.to_numpy()
    lifetime_values = lifetime.data.to_numpy()

    assert baseline_values.shape == lifetime_values.shape
    assert np.all(lifetime_values <= baseline_values + 1e-12)
    assert np.any(lifetime_values < baseline_values - 1e-12)
    assert _footprint_total(lifetime) < _footprint_total(baseline)
