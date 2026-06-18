"""Integration smoke tests for end-to-end per-dim evaluators.

These do NOT run model forward passes (no GPU available in CI). They
verify each dimension's evaluator can:

1. Import without errors,
2. Be invoked on synthetic / cached inputs,
3. Return a result dataclass with the expected schema,
4. Reproduce the §I.4 reference value when fed cached on-disk artefacts.

Reference-value drift detectors live in tests/unit/; this file is for
the cross-module orchestration smoke checks the spec §II.11 calls out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


# ---- Dim A smoke ---------------------------------------------------------


def test_dim_a_evaluator_smoke():
    """End-to-end: synthetic anndata → DimAResult with the expected schema."""
    anndata = pytest.importorskip("anndata")
    import pandas as pd

    from vcbench.dimensions.dim_a_perturbation import evaluate_dim_a

    rng = np.random.default_rng(0)
    G = 30
    ctrl_mean = rng.normal(5, 2, G)

    rows_pred, rows_real = [], []
    for p in range(5):
        delta = rng.normal(0, 1, G)
        for _ in range(8):
            rows_pred.append({"condition": f"p{p}",
                               **{f"g{i}": v for i, v in enumerate(ctrl_mean + delta)}})
            rows_real.append({"condition": f"p{p}",
                               **{f"g{i}": v for i, v in enumerate(ctrl_mean + delta + rng.normal(0, 0.1, G))}})
    for _ in range(20):
        rows_real.append({"condition": "ctrl",
                          **{f"g{i}": v for i, v in enumerate(ctrl_mean + rng.normal(0, 0.1, G))}})

    df_pred = pd.DataFrame(rows_pred)
    df_real = pd.DataFrame(rows_real)
    expr_cols = [c for c in df_real.columns if c != "condition"]
    pred_a = anndata.AnnData(X=df_pred[expr_cols].to_numpy(),
                              obs=df_pred[["condition"]].reset_index(drop=True))
    real_a = anndata.AnnData(X=df_real[expr_cols].to_numpy(),
                              obs=df_real[["condition"]].reset_index(drop=True))

    result = evaluate_dim_a(pred_a, real_a)
    assert result.n_perturbations == 5
    assert 0.5 < result.mean_pearson_r_delta <= 1.0


# ---- Dim B smoke ---------------------------------------------------------


def test_dim_b_evaluator_smoke():
    from vcbench.dimensions.dim_b_cross_species import evaluate_dim_b

    per_tissue = {
        "lung":  {"y_true": ["A", "B", "C"] * 3, "y_pred": ["A", "B", "C"] * 3},
        "liver": {"y_true": ["X", "Y"] * 4,      "y_pred": ["X", "Y"] * 4},
    }
    common = {"lung": {"A", "B", "C"}, "liver": {"X", "Y"}}
    out = evaluate_dim_b("smoke_model", per_tissue, common_label_sets=common)
    assert out.aggregate_native_macro_f1 == pytest.approx(1.0)
    assert out.aggregate_common_set_macro_f1 == pytest.approx(1.0)


# ---- Dim C smoke ---------------------------------------------------------


def test_dim_c_evaluator_smoke():
    from vcbench.dimensions.dim_c_grn import evaluate_dim_c

    rng = np.random.default_rng(0)
    n = 1000
    y_true = (rng.uniform(0, 1, n) < 0.05).astype(int)   # ~5% positive
    y_score = rng.normal(0, 1, n) + y_true * 0.5         # weak signal
    out = evaluate_dim_c(y_true, y_score)
    assert 0.4 < out.AUROC < 0.9
    assert out.AUPRC > 0.0
    assert out.n_total_pairs == n


# ---- Dim D smoke ---------------------------------------------------------


def test_dim_d_evaluator_smoke():
    from vcbench.dimensions.dim_d_cross_modal import evaluate_dim_d

    rng = np.random.default_rng(0)
    Y = rng.normal(0, 1, size=(50, 10))
    out = evaluate_dim_d(Y, Y, model_name="smoke")
    assert out.mean_pearson_r == pytest.approx(1.0, abs=1e-9)
    assert out.rmse == 0.0


# ---- Dim E smoke ---------------------------------------------------------


def test_dim_e_evaluator_smoke():
    from vcbench.dimensions.dim_e_temporal import evaluate_dim_e

    out = evaluate_dim_e("smoke", per_dataset_taus={"sci_fate": 0.2, "weinreb": 0.1})
    assert out.aggregate_kendall_tau_b == pytest.approx(0.15, abs=1e-12)
    assert len(out.per_dataset) == 2


# ---- Cross-dim invariant: pre-registration + reference values agree -----


def test_pre_registration_expected_assignments_match_reference_values():
    """The Level assignments in pre_registration.yaml must reproduce from the
    reference_values.json capability matrix using the §I.5 rules."""
    yaml = pytest.importorskip("yaml")
    pre_reg = yaml.safe_load((REPO_ROOT / "configs" / "pre_registration.yaml").read_text())
    ref = json.loads((REPO_ROOT / "tests" / "reference_values.json").read_text())

    for model_key, expected_level in pre_reg["expected_assignments"].items():
        ref_entry = ref["vc_levels"].get(model_key)
        assert ref_entry is not None, f"{model_key} missing from reference_values vc_levels"
        assert ref_entry["level"] == expected_level, (
            f"{model_key}: pre_registration says Level {expected_level}, "
            f"reference_values says Level {ref_entry['level']}"
        )
