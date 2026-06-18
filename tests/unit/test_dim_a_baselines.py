"""Unit tests for vcbench.dimensions.dim_a_perturbation.baselines (Eq. 2)."""

from __future__ import annotations

import numpy as np
import pytest

from vcbench.dimensions.dim_a_perturbation.baselines import (
    additive_baseline,
    mean_baseline,
    no_change_baseline,
)
from vcbench.dimensions.dim_a_perturbation.metrics import des, prr


# ---- additive_baseline (Eq. 2) ---------------------------------------------


def test_additive_baseline_known_construction():
    """Numerical sanity: y_hat = ctrl + delta_a + delta_b, element-wise."""
    ctrl = np.array([10.0, 20.0, 30.0, 40.0])
    da = np.array([1.0, 0.0, -2.0, 5.0])
    db = np.array([0.0, 3.0, -1.0, -4.0])
    expected = np.array([11.0, 23.0, 27.0, 41.0])
    np.testing.assert_array_equal(additive_baseline(ctrl, da, db), expected)


def test_additive_baseline_perfect_recovery():
    """If real_delta_AB == delta_a + delta_b exactly, PRR=1.0 and DES=1.0."""
    rng = np.random.default_rng(11)
    G = 200
    ctrl = rng.normal(5, 2, G)
    da = rng.normal(0, 1, G)
    db = rng.normal(0, 1, G)
    real_AB_mean = ctrl + da + db                # ground-truth additive case
    pred_AB_mean = additive_baseline(ctrl, da, db)
    real_delta = real_AB_mean - ctrl
    pred_delta = pred_AB_mean - ctrl
    assert prr(pred_delta, real_delta) == pytest.approx(1.0, abs=1e-12)
    assert des(pred_delta, real_delta) == pytest.approx(1.0, abs=1e-12)


def test_additive_baseline_shape_mismatch_raises():
    with pytest.raises(ValueError):
        additive_baseline(np.zeros(10), np.zeros(11), np.zeros(10))


# ---- mean_baseline ---------------------------------------------------------


def test_mean_baseline_dict_input():
    profiles = {
        "p1": np.array([1.0, 2.0, 3.0]),
        "p2": np.array([3.0, 4.0, 5.0]),
    }
    np.testing.assert_array_equal(
        mean_baseline(profiles), np.array([2.0, 3.0, 4.0])
    )


def test_mean_baseline_array_input():
    arr = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    np.testing.assert_array_equal(
        mean_baseline(arr), np.array([3.0, 4.0, 5.0])
    )


def test_mean_baseline_empty_dict_raises():
    with pytest.raises(ValueError):
        mean_baseline({})


def test_mean_baseline_wrong_array_dim_raises():
    with pytest.raises(ValueError):
        mean_baseline(np.zeros(10))   # 1-D


# ---- no_change_baseline ----------------------------------------------------


def test_no_change_baseline_returns_copy_of_ctrl():
    ctrl = np.array([1.0, 2.0, 3.0])
    out = no_change_baseline(ctrl)
    np.testing.assert_array_equal(out, ctrl)
    out[0] = 999.0
    assert ctrl[0] == 1.0   # original untouched (copy semantics)


def test_no_change_baseline_yields_zero_PRR_and_DES():
    """Critical reference: no-change predicts ctrl, so pred_delta is 0
    everywhere → PRR=0 (zero variance branch), DES=0 (sign(0)=0)."""
    rng = np.random.default_rng(13)
    ctrl = rng.normal(5, 2, 100)
    real_pert = ctrl + rng.normal(0, 1, 100)   # nontrivial real perturbation
    pred_pert = no_change_baseline(ctrl)
    real_delta = real_pert - ctrl
    pred_delta = pred_pert - ctrl
    assert prr(pred_delta, real_delta) == 0.0
    assert des(pred_delta, real_delta) == 0.0
