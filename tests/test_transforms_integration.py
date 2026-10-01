"""Integration coverage for particle transforms declared in the config."""

from __future__ import annotations

import numpy as np

from stilt.config import ProjectConfig
from stilt.project import Project
from stilt.transforms import FirstOrderLifetime

from .conftest import integration


def _footprint_total(footprint) -> float:
    """Return the total scalar sensitivity in one footprint field."""
    return float(np.asarray(footprint.sum()))


@integration
def test_declarative_transform_config_changes_real_footprint(
    tmp_path,
    met_dir,
    wbb_receptor,
    wbb_grid,
):
    """Per-footprint transform specs should affect real footprint output."""
    config = ProjectConfig(
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

    project = Project.init(
        tmp_path / "configured_transforms", config=config, receptors=[wbb_receptor]
    )
    project.run()

    sims = project.simulations
    [baseline] = project.load_footprints(sims[sims.variant == "hrrr"]).values()
    [lifetime] = project.load_footprints(sims[sims.variant == "lifetime"]).values()

    assert len(lifetime.stilt.config.transforms) == 1

    baseline_values = baseline.to_numpy()
    lifetime_values = lifetime.to_numpy()

    assert baseline_values.shape == lifetime_values.shape
    assert np.all(lifetime_values <= baseline_values + 1e-12)
    assert np.any(lifetime_values < baseline_values - 1e-12)
    assert _footprint_total(lifetime) < _footprint_total(baseline)
