"""Unit tests for vcbench.dimensions.dim_c_grn.metrics (Eq. 6)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from vcbench.dimensions.dim_c_grn.metrics import auprc, auroc, epr


def test_auroc_perfect_ranker():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(y_true, y_score) == pytest.approx(1.0)


def test_auroc_inverse_ranker():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert auroc(y_true, y_score) == pytest.approx(0.0)


def test_auroc_random_ranker_near_half():
    rng = np.random.default_rng(0)
    n = 5000
    y_true = rng.integers(0, 2, n)
    y_score = rng.normal(0, 1, n)
    assert abs(auroc(y_true, y_score) - 0.5) < 0.05


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(1)
    n = 1000
    y_true = rng.integers(0, 2, n)
    y_score = rng.normal(0, 1, n)
    assert auroc(y_true, y_score) == pytest.approx(
        roc_auc_score(y_true, y_score), abs=1e-12
    )


def test_auroc_single_class_returns_nan():
    assert math.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert math.isnan(auroc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_auprc_matches_sklearn():
    rng = np.random.default_rng(2)
    n = 1000
    y_true = rng.integers(0, 2, n)
    y_score = rng.normal(0, 1, n)
    assert auprc(y_true, y_score) == pytest.approx(
        average_precision_score(y_true, y_score), abs=1e-12
    )


def test_auprc_single_class_returns_nan():
    assert math.isnan(auprc([1, 1, 1], [0.1, 0.2, 0.3]))


# ---- EPR (Eq. 6) ----------------------------------------------------------


def test_epr_perfect_ranker_equals_inverse_base_rate():
    """If model puts all K true edges in top-K, EPR = 1 / base_rate."""
    n_total = 1000
    n_true = 10
    y_true = np.zeros(n_total, dtype=int)
    y_true[:n_true] = 1
    # Score puts the 10 true edges first
    y_score = np.zeros(n_total)
    y_score[:n_true] = 1.0
    base_rate = n_true / n_total
    assert epr(y_true, y_score) == pytest.approx(1.0 / base_rate, abs=1e-9)


def test_epr_random_ranker_near_one():
    """A random ranker should put ~K * base_rate true edges in top-K → EPR ≈ 1."""
    rng = np.random.default_rng(0)
    n_total = 5000
    n_true = 100
    y_true = np.zeros(n_total, dtype=int)
    y_true[rng.choice(n_total, size=n_true, replace=False)] = 1
    y_score = rng.normal(0, 1, n_total)
    val = epr(y_true, y_score)
    assert 0.0 < val < 3.0   # generous bound for sampling


def test_epr_inverse_ranker_zero_or_low():
    """If model anti-ranks (true edges all at bottom), top-K contains zero
    true edges → EPR = 0."""
    n_total = 100
    y_true = np.zeros(n_total, dtype=int)
    y_true[:10] = 1   # first 10 are true
    y_score = np.arange(n_total, dtype=float)  # ascending → true edges have lowest scores
    assert epr(y_true, y_score) == pytest.approx(0.0)


def test_epr_single_class_returns_nan():
    assert math.isnan(epr([1, 1, 1], [0.1, 0.2, 0.3]))


def test_epr_matches_manual_formula_on_known_input():
    """K=3 true edges in N=10 candidates; predictions place 2/3 true in top-3.
    Precision@3 = 2/3, base rate = 3/10, EPR = (2/3) / (3/10) = 20/9."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    y_score = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    # Top 3 by score: indices 0, 1, 2 — true labels [1, 1, 0]
    # Precision@3 = 2/3, base = 3/10, EPR = 20/9 ≈ 2.222
    assert epr(y_true, y_score) == pytest.approx(20 / 9, abs=1e-9)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        auroc([0, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        auprc([0, 1, 0], [0.1, 0.2])
    with pytest.raises(ValueError):
        epr([0], [0.1, 0.2])
