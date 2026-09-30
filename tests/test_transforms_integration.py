"""Integration coverage for particle transforms declared in the config."""

from __future__ import annotations

import numpy as np

from stilt.config import ModelConfig
from stilt.model import Model
from stilt.transforms import FirstOrderLifetime

from .conftest import integration


def _footprint_total(footprint) -> float:
    """Return the total scalar sensitivity in one footprint field."""
    return float(np.asarray(footprint.data.sum()))


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
                "file_format": "%Y%m%d_%H",
                "file_tres": "6h",
            }
        },
        n_hours=-6,
        numpar=100,
        grid=wbb_grid,
        variants={
            "hrrr": {},
            "lifetime": {
                "transforms": [FirstOrderLifetime(lifetime_hours=1.0)],
            },
        },
    )

    model = Model(
        project=tmp_path / "configured_transforms",
        config=config,
        receptors=[wbb_receptor],
    )
    model.run()

    [baseline] = model.simulations.sel(variant="hrrr").footprint.load().values()
    [lifetime] = model.simulations.sel(variant="lifetime").footprint.load().values()

    assert len(lifetime.config.transforms) == 1

    baseline_values = baseline.data.to_numpy()
    lifetime_values = lifetime.data.to_numpy()

    assert baseline_values.shape == lifetime_values.shape
    assert np.all(lifetime_values <= baseline_values + 1e-12)
    assert np.any(lifetime_values < baseline_values - 1e-12)
    assert _footprint_total(lifetime) < _footprint_total(baseline)
