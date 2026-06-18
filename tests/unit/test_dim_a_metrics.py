"""Unit tests for vcbench.dimensions.dim_a_perturbation.metrics (Eq. 1, Eq. 3)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from vcbench.dimensions.dim_a_perturbation.metrics import (
    DES_TOP_K_DEFAULT,
    des,
    prr,
)


# ---- PRR (Eq. 1) ----------------------------------------------------------


def test_prr_identity_is_one():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 500)
    assert prr(v, v) == pytest.approx(1.0, abs=1e-9)


def test_prr_negation_is_minus_one():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 500)
    assert prr(v, -v) == pytest.approx(-1.0, abs=1e-9)


def test_prr_uncorrelated_near_zero():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert abs(prr(a, b)) < 0.05


def test_prr_matches_scipy_pearsonr():
    """PRR is exactly scipy.stats.pearsonr applied to the deltas — no clever
    preprocessing, no centring assumption beyond what pearsonr already does."""
    rng = np.random.default_rng(3)
    a = rng.normal(2.0, 1.5, 250)
    b = a * 0.7 + rng.normal(0, 0.4, 250)
    expected, _ = pearsonr(a, b)
    assert prr(a, b) == pytest.approx(expected, abs=1e-12)


def test_prr_zero_variance_returns_zero():
    """When predicted (or observed) Δ is flat, PRR is 0.0 by convention.
    Matches the existing _evaluate_perturbation_fallback divide-by-zero guard."""
    real = np.random.default_rng(4).normal(0, 1, 100)
    flat = np.zeros(100)
    assert prr(flat, real) == 0.0
    assert prr(real, flat) == 0.0


def test_prr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        prr(np.zeros(10), np.zeros(11))


# ---- DES (Eq. 3) ----------------------------------------------------------


def test_des_all_signs_match_is_one():
    real = np.array([3.0, -2.0, 5.0, -1.5, 4.0, -0.5] * 5)  # 30 genes
    pred = np.sign(real) * np.abs(real) * 0.5
    assert des(pred, real) == pytest.approx(1.0, abs=1e-12)


def test_des_all_signs_opposite_is_zero():
    real = np.array([3.0, -2.0, 5.0, -1.5, 4.0, -0.5] * 5)
    pred = -np.sign(real) * np.abs(real) * 0.5
    assert des(pred, real) == pytest.approx(0.0, abs=1e-12)


def test_des_top_k_selects_largest_absolute_real_delta():
    """Mismatched-sign genes hidden in the bottom-of-rank should NOT count."""
    G = 100
    real = np.zeros(G)
    pred = np.zeros(G)
    # Top-20 genes (largest |real|): all signs match
    real[:20] = np.linspace(10, 1, 20)
    pred[:20] = real[:20]
    # Bottom-80 genes (small |real|): all signs OPPOSITE — must not enter DES
    real[20:] = np.linspace(0.1, 0.001, 80)
    pred[20:] = -real[20:]
    assert des(pred, real, top_k=20) == pytest.approx(1.0, abs=1e-12)


def test_des_uses_g_when_g_lt_top_k():
    """min(20, G) cap behaviour: G=10 → DES averaged over 10 genes."""
    rng = np.random.default_rng(7)
    real = rng.normal(0, 1, 10)
    pred = real.copy()
    assert des(pred, real, top_k=20) == 1.0
    # And K=5 should also work without issue
    assert des(pred, real, top_k=5) == 1.0


def test_des_default_top_k_is_20():
    assert DES_TOP_K_DEFAULT == 20


def test_des_invalid_top_k_raises():
    with pytest.raises(ValueError):
        des(np.zeros(10), np.zeros(10), top_k=0)
    with pytest.raises(ValueError):
        des(np.zeros(10), np.zeros(10), top_k=-1)


def test_des_shape_mismatch_raises():
    with pytest.raises(ValueError):
        des(np.zeros(10), np.zeros(11))


def test_des_zero_in_pred_does_not_match_nonzero_real():
    """sign(0)==0, so a zero predicted Δ does not match a positive real Δ.
    This is the documented behaviour of the existing fallback path."""
    real = np.array([1.0] * 20)  # all positive
    pred = np.zeros(20)
    assert des(pred, real) == 0.0
