"""
Every name ``varsiwant`` accepts is one the bundled hycs_std reads (#324).

``hycs_std`` stops at the first ``varsiwant`` name it does not know, before it
writes a particle, so a name the config allowed but HYSPLIT did not read gave a
run with no particles.
"""

from __future__ import annotations

import datetime as dt
from typing import get_args

from stilt.receptors import PointReceptor
from stilt.transport import run_model
from stilt.transport.hysplit import HysplitConfig
from stilt.transport.hysplit.driver import PARTICLE_COLUMNS

from ..conftest import integration
from .conftest import reference_met

NAMES = list(get_args(get_args(HysplitConfig.model_fields["varsiwant"].annotation)[0]))


@integration
def test_hycs_std_writes_a_column_for_every_name_varsiwant_accepts(tmp_path, met_dir):
    receptor = PointReceptor(
        time=dt.datetime(2021, 1, 15, 3, 0),
        longitude=-111.85,
        latitude=40.77,
        altitude=30.0,
    )
    config = HysplitConfig(n_hours=-2, numpar=20, hnf_plume=False, varsiwant=NAMES)

    run = run_model("hysplit", receptor, config, reference_met(met_dir), tmp_path, 600)

    assert not run.particles.empty
    assert list(run.particles.columns) == [PARTICLE_COLUMNS.get(n, n) for n in NAMES]
    assert "zcfx" in run.particles.columns
