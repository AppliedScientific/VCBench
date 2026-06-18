"""Unit tests for vcbench.probes.spread_error_correlation (Eq. 10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcbench.probes import SpreadErrorResult, spread_error_correlation


def _make_synthetic(n_perts: int, n_genes: int, *, seed: int):
    """Build matched (predictions, ground_truth) frames for n_perts perturbations."""
    rng = np.random.default_rng(seed)
    rows_pred, rows_gt = [], []
    expr_cols = [f"g{i}" for i in range(n_genes)]
    # Control rows in ground-truth (anchors the Δ-expression computation).
    for _ in range(20):
        rows_gt.append({"perturbation": "ctrl",
                        **dict(zip(expr_cols, rng.normal(0, 1, n_genes)))})
    for k in range(n_perts):
        for _ in range(5):
            rows_pred.append({"perturbation": f"p{k}",
                              **dict(zip(expr_cols, rng.normal(0, 1, n_genes)))})
            rows_gt.append({"perturbation": f"p{k}",
                            **dict(zip(expr_cols, rng.normal(0, 1, n_genes)))})
    return pd.DataFrame(rows_pred), pd.DataFrame(rows_gt), expr_cols


def test_returns_result_dataclass_with_expected_n():
    pred, gt, _ = _make_synthetic(n_perts=20, n_genes=50, seed=0)
    result = spread_error_correlation(pred, gt)
    assert isinstance(result, SpreadErrorResult)
    assert result.n_perturbations == 20
    assert -1.0 <= result.rho <= 1.0
    assert 0.0 <= result.pvalue <= 1.0


def test_perfect_positive_correlation():
    """If predicted spread is rigged to align with predicted error, ρ → +1."""
    rng = np.random.default_rng(42)
    n_genes = 30
    expr_cols = [f"g{i}" for i in range(n_genes)]
    pred_rows, gt_rows = [], []
    for _ in range(10):
        gt_rows.append({"perturbation": "ctrl", **dict(zip(expr_cols, np.zeros(n_genes)))})
    for k in range(15):
        # spread grows monotonically with k; ground-truth deltas stay near zero.
        scale = float(k + 1)
        pred_vec = rng.normal(0, scale, n_genes)
        pred_rows.append({"perturbation": f"p{k}", **dict(zip(expr_cols, pred_vec))})
        gt_rows.append({"perturbation": f"p{k}", **dict(zip(expr_cols, np.zeros(n_genes)))})
    pred = pd.DataFrame(pred_rows)
    gt = pd.DataFrame(gt_rows)
    result = spread_error_correlation(pred, gt)
    # spread = var(pred_delta) ↑ with k; error = mean|pred-gt| also ↑ with scale.
    assert result.rho > 0.7


def test_missing_control_raises():
    rng = np.random.default_rng(0)
    expr_cols = [f"g{i}" for i in range(10)]
    pred = pd.DataFrame([
        {"perturbation": "p0", **dict(zip(expr_cols, rng.normal(0, 1, 10)))}
        for _ in range(5)
    ])
    gt = pd.DataFrame([
        {"perturbation": "p0", **dict(zip(expr_cols, rng.normal(0, 1, 10)))}
        for _ in range(5)
    ])
    with pytest.raises(ValueError, match="control"):
        spread_error_correlation(pred, gt)


def test_missing_perturbation_column_raises():
    pred = pd.DataFrame({"g0": [0.1, 0.2]})
    gt = pd.DataFrame({"g0": [0.1, 0.2]})
    with pytest.raises(ValueError, match="perturbation"):
        spread_error_correlation(pred, gt)


def test_too_few_perturbations_raises():
    rng = np.random.default_rng(0)
    pred, gt, _ = _make_synthetic(n_perts=2, n_genes=20, seed=1)
    with pytest.raises(ValueError, match="at least 3"):
        spread_error_correlation(pred, gt)
