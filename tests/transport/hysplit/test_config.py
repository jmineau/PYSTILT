"""Tests for HysplitConfig: the HYSPLIT parameters and how they are checked."""

import pytest
from pydantic import ValidationError

from stilt.transport.hysplit import HysplitConfig


def test_winderrtf_partial_xy_raises():
    with pytest.raises(ValidationError):
        HysplitConfig(siguverr=1.0)  # only one of four XY params set


def test_winderrtf_partial_zi_raises():
    with pytest.raises(ValidationError):
        HysplitConfig(sigzierr=0.6, tlzierr=60.0)  # missing horcorzierr


def test_stilt_params_flat_construction():
    """HysplitConfig accepts all fields flat (no met fields)."""
    p = HysplitConfig(
        n_hours=-24,
        numpar=500,
    )
    assert p.n_hours == -24
    assert p.numpar == 500


def test_stilt_params_ziscale_defaults_to_scalar_one():
    p = HysplitConfig()
    assert p.ziscale == 1.0


def test_saved_config_no_longer_carries_zicontroltf():
    assert "zicontroltf" not in HysplitConfig(ziscale=0.8).model_dump()


def test_zicontroltf_is_not_a_setting():
    with pytest.raises(ValidationError, match="zicontroltf"):
        HysplitConfig(zicontroltf=1)


@pytest.mark.parametrize("ziscale", [0.0, [1.0, 0.0]])
def test_ziscale_zero_is_rejected(ziscale):
    with pytest.raises(ValueError, match="ziscale of 0"):
        HysplitConfig(ziscale=ziscale)


def test_ziscale_empty_list_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        HysplitConfig(ziscale=[])


def test_ziscale_rejects_multiple_per_simulation_lists():
    with pytest.raises(ValueError, match="Per-simulation ziscale lists"):
        HysplitConfig(ziscale=[[0.8, 0.8], [0.9, 0.9]])


@pytest.mark.parametrize("krand", [0, 1, 3, 4, 10, 13])
def test_seed_requires_krand_2(krand):
    with pytest.raises(ValueError, match="requires krand=2"):
        HysplitConfig(seed=1, krand=krand)


@pytest.mark.parametrize("krand", [0, 1, 2, 3, 4, 10, 11, 12, 13])
def test_krand_accepts_hysplit_modes(krand):
    assert HysplitConfig(krand=krand).krand == krand


@pytest.mark.parametrize("krand", [-1, 5, 9, 14, 20])
def test_krand_rejects_undocumented_values(krand):
    with pytest.raises(ValueError, match="krand"):
        HysplitConfig(krand=krand)
