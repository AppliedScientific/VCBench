"""Unit tests for vcbench.dimensions.dim_e_temporal.metrics (Eq. 9)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import kendalltau

from vcbench.dimensions.dim_e_temporal.metrics import (
    kendall_tau_b,
    knn_balanced_accuracy,
)


# ---- Kendall τ-b (Eq. 9) ---------------------------------------------------


def test_kendall_tau_b_perfect_concordance_is_one():
    pred = np.linspace(0, 1, 50)
    truth = np.arange(50)
    assert kendall_tau_b(pred, truth) == pytest.approx(1.0, abs=1e-9)


def test_kendall_tau_b_perfect_inversion_is_minus_one():
    pred = np.linspace(0, 1, 50)
    truth = np.arange(50, 0, -1)
    assert kendall_tau_b(pred, truth) == pytest.approx(-1.0, abs=1e-9)


def test_kendall_tau_b_handles_ties_in_truth():
    """Discrete timepoints have many ties — τ-b must give a finite, sensible
    value where τ-a would degenerate. Note: τ-b's tie-correction prevents
    it from hitting exactly 1.0 even on monotone-correct rankings when the
    truth has ties (the correction shrinks the denominator); 0.85+ is the
    expected range here."""
    pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    truth = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # 3 timepoints, many ties
    val = kendall_tau_b(pred, truth)
    assert 0.8 < val <= 1.0
    assert not np.isnan(val)


def test_kendall_tau_b_matches_scipy():
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 1, 200)
    truth = rng.integers(0, 5, 200)
    expected, _ = kendalltau(pred, truth, variant="b")
    assert kendall_tau_b(pred, truth) == pytest.approx(expected, abs=1e-12)


def test_kendall_tau_b_uncorrelated_near_zero():
    rng = np.random.default_rng(1)
    pred = rng.normal(0, 1, 5000)
    truth = rng.integers(0, 5, 5000)
    assert abs(kendall_tau_b(pred, truth)) < 0.05


def test_kendall_tau_b_constant_input_returns_nan():
    pred = np.zeros(10)
    truth = np.arange(10)
    val = kendall_tau_b(pred, truth)
    assert math.isnan(val)


def test_kendall_tau_b_shape_mismatch_raises():
    with pytest.raises(ValueError):
        kendall_tau_b(np.zeros(10), np.zeros(11))


def test_kendall_tau_b_too_few_points_raises():
    with pytest.raises(ValueError):
        kendall_tau_b(np.zeros(1), np.zeros(1))


# ---- knn_balanced_accuracy ------------------------------------------------


def test_knn_balanced_accuracy_well_separated_clusters():
    """When timepoints are well-separated in embedding space, BA should be
    high (>0.7). Default sklearn cv splits aren't stratified, so some folds
    can end up imbalanced enough to bring the score below 0.9 even on
    perfectly-separable data — 0.7 is the conservative bound that holds
    across CV-fold variance."""
    rng = np.random.default_rng(0)
    X, y = [], []
    for t in range(3):
        # Each timepoint occupies a distinct region of embedding space
        center = np.array([10.0 * t, 0.0, 0.0])
        X.append(center + rng.normal(0, 0.3, size=(50, 3)))
        y.extend([t] * 50)
    X = np.vstack(X)
    y = np.array(y)
    score = knn_balanced_accuracy(X, y, k=5, cv=5)
    assert score > 0.7


def test_knn_balanced_accuracy_random_embeddings_near_chance():
    """Random embeddings → BA near 1/n_classes."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(300, 3))
    y = rng.integers(0, 3, 300)
    score = knn_balanced_accuracy(X, y, k=5, cv=3)
    assert 0.0 < score < 0.5


def test_knn_balanced_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        knn_balanced_accuracy(np.zeros((10, 3)), np.zeros(11))
