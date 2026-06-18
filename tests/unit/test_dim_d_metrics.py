"""Unit tests for vcbench.dimensions.dim_d_cross_modal.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import pearsonr

from vcbench.dimensions.dim_d_cross_modal.metrics import (
    mean_pearson_per_protein,
    median_pearson_per_protein,
    pearson_per_protein,
    rmse,
)


def test_pearson_per_protein_identity_returns_ones():
    rng = np.random.default_rng(0)
    Y = rng.normal(0, 1, size=(100, 5))
    out = pearson_per_protein(Y, Y)
    np.testing.assert_allclose(out, 1.0, atol=1e-9)


def test_pearson_per_protein_negation_returns_minus_ones():
    rng = np.random.default_rng(1)
    Y = rng.normal(0, 1, size=(100, 5))
    out = pearson_per_protein(Y, -Y)
    np.testing.assert_allclose(out, -1.0, atol=1e-9)


def test_pearson_per_protein_independent_columns_match_scipy():
    rng = np.random.default_rng(2)
    n_cells, n_proteins = 200, 7
    obs = rng.normal(0, 1, size=(n_cells, n_proteins))
    pred = rng.normal(0, 1, size=(n_cells, n_proteins))
    rs = pearson_per_protein(pred, obs)
    expected = [pearsonr(pred[:, j], obs[:, j])[0] for j in range(n_proteins)]
    np.testing.assert_allclose(rs, expected, atol=1e-12)


def test_pearson_per_protein_zero_variance_protein_returns_nan():
    n_cells = 50
    pred = np.zeros((n_cells, 3))   # all proteins zero-variance
    obs = np.random.default_rng(0).normal(0, 1, size=(n_cells, 3))
    rs = pearson_per_protein(pred, obs)
    assert np.all(np.isnan(rs))


def test_mean_pearson_skips_nan_proteins():
    n_cells = 50
    rng = np.random.default_rng(3)
    obs = rng.normal(0, 1, size=(n_cells, 3))
    pred = obs.copy()
    pred[:, 1] = 0.0   # make protein 1 zero-variance → NaN
    val = mean_pearson_per_protein(pred, obs)
    # The two valid proteins both have ρ=1, so mean should be 1.0
    assert val == pytest.approx(1.0, abs=1e-9)


def test_mean_pearson_all_nan_returns_nan():
    n_cells = 50
    pred = np.zeros((n_cells, 3))
    obs = np.zeros((n_cells, 3))
    assert math.isnan(mean_pearson_per_protein(pred, obs))


def test_median_pearson():
    n_cells = 100
    rng = np.random.default_rng(4)
    obs = rng.normal(0, 1, size=(n_cells, 3))
    pred = obs.copy()
    val = median_pearson_per_protein(pred, obs)
    assert val == pytest.approx(1.0, abs=1e-9)


def test_rmse_perfect_prediction_is_zero():
    Y = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert rmse(Y, Y) == 0.0


def test_rmse_known_value():
    pred = np.array([[1.0, 2.0]])
    obs = np.array([[2.0, 4.0]])
    # squared diffs: 1, 4 → mean 2.5 → sqrt = 1.581...
    assert rmse(pred, obs) == pytest.approx(math.sqrt(2.5), abs=1e-12)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pearson_per_protein(np.zeros((10, 3)), np.zeros((10, 4)))
    with pytest.raises(ValueError):
        rmse(np.zeros((5, 3)), np.zeros((6, 3)))


def test_one_dim_input_raises():
    with pytest.raises(ValueError):
        pearson_per_protein(np.zeros(10), np.zeros(10))


def test_rmse_empty_input_raises():
    with pytest.raises(ValueError):
        rmse(np.zeros((0, 3)), np.zeros((0, 3)))
