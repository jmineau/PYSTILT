"""Tests for stilt.observations.modelled_column."""

import numpy as np
import pytest

from stilt.observations import modelled_column


def test_without_a_kernel_the_column_is_enhancement_plus_background():
    assert modelled_column(12.0, 1900.0) == 1912.0


def test_the_prior_fills_where_the_retrieval_is_not_sensitive():
    w = np.array([0.5, 0.3, 0.2])
    prior = np.array([1900.0, 1880.0, 1800.0])
    # Fully sensitive everywhere: no prior term.
    assert (
        modelled_column(10.0, 1890.0, ak=np.ones(3), pressure_weight=w, prior=prior)
        == 1900.0
    )
    # Blind everywhere: the column is the prior's.
    blind = modelled_column(0.0, 0.0, ak=np.zeros(3), pressure_weight=w, prior=prior)
    assert blind == pytest.approx(float(w @ prior))
    half = modelled_column(
        10.0, 1890.0, ak=np.full(3, 0.5), pressure_weight=w, prior=prior
    )
    assert half == pytest.approx(1900.0 + 0.5 * float(w @ prior))


def test_the_kernel_terms_come_together_and_match_in_length():
    with pytest.raises(ValueError, match="together"):
        modelled_column(1.0, 2.0, ak=[1.0])
    with pytest.raises(ValueError, match="differ in length"):
        modelled_column(1.0, 2.0, ak=[1.0, 1.0], pressure_weight=[1.0], prior=[1.0])
