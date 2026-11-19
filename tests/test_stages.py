"""The stage interface, held with two toy stages that import nothing from the footprint."""

from __future__ import annotations

import pytest
from runtraj.config import ProjectConfig
from runtraj.stages import get_stage

from .fixtures.factories import make_met_config


def _config(tmp_path, **blocks) -> ProjectConfig:
    """Return a config with one met, HYSPLIT, and the blocks given."""
    blocks.setdefault("models", {"hysplit": {"n_hours": -24}})
    blocks.setdefault("variants", {"hrrr": {}})
    return ProjectConfig(mets={"hrrr": make_met_config(tmp_path / "met")}, **blocks)


# -- a stage class ------------------------------------------------------------


def test_a_stage_must_be_named_by_an_import_path():
    with pytest.raises(ValueError, match="import path of its class.*not 'footprint'"):
        get_stage("footprint")


# -- the order stages run in --------------------------------------------------


# -- the promises check_stage holds ------------------------------------------


# -- stages in a config -------------------------------------------------------
