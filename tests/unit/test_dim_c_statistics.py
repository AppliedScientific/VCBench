"""Unit tests for vcbench.dimensions.dim_c_grn.statistics (Eq. 7)."""

from __future__ import annotations

import numpy as np
import pytest

from vcbench.dimensions.dim_c_grn.metrics import auprc, epr
from vcbench.dimensions.dim_c_grn.statistics import (
    BootstrapCI,
    PairedBootstrapResult,
    benjamini_hochberg,
    bootstrap_ci,
    paired_bootstrap_test,
)


# ---- bootstrap_ci ----------------------------------------------------------


def test_bootstrap_ci_returns_dataclass_with_finite_fields():
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, n)
    y_score = rng.normal(0, 1, n)
    ci = bootstrap_ci(y_true, y_score, auprc, n_bootstrap=200, seed=1)
    assert isinstance(ci, BootstrapCI)
    assert ci.ci_lower <= ci.mean <= ci.ci_upper
    assert ci.se >= 0.0
    assert ci.n_bootstrap > 0


def test_bootstrap_ci_seed_is_deterministic():
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.integers(0, 2, n)
    y_score = rng.normal(0, 1, n)
    a = bootstrap_ci(y_true, y_score, auprc, n_bootstrap=200, seed=42)
    b = bootstrap_ci(y_true, y_score, auprc, n_bootstrap=200, seed=42)
    assert a.mean == b.mean
    assert a.ci_lower == b.ci_lower
    assert a.ci_upper == b.ci_upper


def test_bootstrap_ci_invalid_confidence_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([0, 1], [0.1, 0.2], auprc, confidence=0.0)
    with pytest.raises(ValueError):
        bootstrap_ci([0, 1], [0.1, 0.2], auprc, confidence=1.0)


def test_bootstrap_ci_too_few_iterations_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([0, 1], [0.1, 0.2], auprc, n_bootstrap=1)


# ---- paired_bootstrap_test ------------------------------------------------


def test_paired_bootstrap_strong_model_beats_weak_baseline():
    """When model puts true edges in top-K and baseline doesn't, Δ > 0 with low p."""
    rng = np.random.default_rng(0)
    n = 1000
    n_true = 50
    y_true = np.zeros(n, dtype=int)
    y_true[rng.choice(n, n_true, replace=False)] = 1
    # Strong model: rank correlated with true labels + small noise
    y_model = y_true + rng.normal(0, 0.3, n)
    # Random baseline
    y_baseline = rng.normal(0, 1, n)
    result = paired_bootstrap_test(
        y_true, y_model, y_baseline, auprc, n_bootstrap=500, seed=0
    )
    assert isinstance(result, PairedBootstrapResult)
    assert result.delta_mean > 0
    assert result.p_value_one_sided < 0.05


def test_paired_bootstrap_two_random_predictors_yields_ci_bracketing_zero():
    """Two random predictors → no consistent advantage → 95% Δ CI must contain zero."""
    rng = np.random.default_rng(1)
    n = 1000
    y_true = rng.integers(0, 2, n)
    y_a = rng.normal(0, 1, n)
    y_b = rng.normal(0, 1, n)
    result = paired_bootstrap_test(y_true, y_a, y_b, auprc, n_bootstrap=500, seed=2)
    # CI brackets zero — neither random predictor consistently beats the other.
    assert result.delta_ci_lower <= 0.0 <= result.delta_ci_upper


def test_paired_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_test([0, 1], [0.1], [0.2, 0.3], auprc)


# ---- benjamini_hochberg ----------------------------------------------------


def test_bh_empty_input():
    assert len(benjamini_hochberg([])) == 0


def test_bh_returns_in_input_order():
    """The output q-values must be in the SAME order as the input p-values."""
    p = [0.001, 0.5, 0.04, 0.9]
    q = benjamini_hochberg(p)
    assert len(q) == 4
    # The 4th element (p=0.9) is the largest p; in BH it should also be the
    # largest q. Verify the per-input-position semantics, not by sorting.
    assert q[3] == max(q)
    assert q[0] == min(q)


def test_bh_monotone_after_sort():
    """After sorting q by input p, q must be non-decreasing."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50)
    q = benjamini_hochberg(p)
    order = np.argsort(p)
    q_in_p_order = q[order]
    diffs = np.diff(q_in_p_order)
    assert np.all(diffs >= -1e-12), "BH q-values must be non-decreasing in p-rank"


def test_bh_clipped_to_unit_interval():
    p = [0.5] * 10
    q = benjamini_hochberg(p)
    assert np.all((q >= 0.0) & (q <= 1.0))


def test_bh_all_significant_input_yields_small_q():
    """If every p-value is tiny, every q-value should also be tiny (well below 0.05)."""
    p = [1e-6] * 10
    q = benjamini_hochberg(p)
    assert np.all(q < 1e-3)


def test_bh_invalid_p_raises():
    with pytest.raises(ValueError):
        benjamini_hochberg([1.5, 0.2])
    with pytest.raises(ValueError):
        benjamini_hochberg([-0.1, 0.2])


def test_bh_known_two_test_case():
    """Verifies the m=2 BH correction value: scgpt p≈0.000 + geneformer p≈0.692
    → q values [0.000, 0.692] (the larger p-value is unchanged because BH at
    rank 2 multiplies by 2/2 = 1)."""
    p = [0.0, 0.692]   # scgpt, geneformer
    q = benjamini_hochberg(p)
    assert q[0] == pytest.approx(0.0, abs=1e-12)
    assert q[1] == pytest.approx(0.692, abs=1e-12)
