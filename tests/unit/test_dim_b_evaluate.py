"""Unit tests for vcbench.dimensions.dim_b_cross_species.evaluate +
reference-value drift detectors against results/dim_b/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vcbench.dimensions.dim_b_cross_species import (
    DimBPerTissueResult,
    DimBResult,
    evaluate_dim_b,
)


def test_evaluate_dim_b_perfect_predictions_per_tissue():
    """Perfect predictions in every tissue → both protocol aggregates = 1.0."""
    per_tissue = {
        "lung":  {"y_true": ["A", "B", "C"] * 5, "y_pred": ["A", "B", "C"] * 5},
        "liver": {"y_true": ["X", "Y"] * 5,      "y_pred": ["X", "Y"] * 5},
    }
    common_sets = {"lung": {"A", "B", "C"}, "liver": {"X", "Y"}}
    result = evaluate_dim_b("test_model", per_tissue, common_label_sets=common_sets)
    assert isinstance(result, DimBResult)
    assert result.aggregate_native_macro_f1 == pytest.approx(1.0)
    assert result.aggregate_common_set_macro_f1 == pytest.approx(1.0)
    assert len(result.per_tissue) == 2


def test_evaluate_dim_b_derives_common_set_from_method_vocabularies():
    """If common_label_sets isn't provided, it's computed from method vocabularies."""
    per_tissue = {"lung": {"y_true": ["A", "B"], "y_pred": ["A", "B"]}}
    method_vocabs = {
        "m1": {"lung": {"A", "B", "C"}},
        "m2": {"lung": {"A", "B"}},
    }
    result = evaluate_dim_b("test_model", per_tissue, method_label_vocabularies=method_vocabs)
    # Common-set is {"A", "B"} — both are in y_true, so n_eval_cells = 2
    assert result.per_tissue[0].common_set is not None
    assert result.per_tissue[0].common_set.n_classes == 2


def test_evaluate_dim_b_tissue_with_no_common_set_yields_none():
    """If a tissue's intersection is empty, common_set score is None for that tissue."""
    per_tissue = {"lung": {"y_true": ["A"], "y_pred": ["A"]}}
    common_sets = {"lung": set()}  # empty
    result = evaluate_dim_b("test_model", per_tissue, common_label_sets=common_sets)
    assert result.per_tissue[0].common_set is None
    assert result.aggregate_common_set_macro_f1 is None


def test_evaluate_dim_b_requires_one_of_two_kwargs():
    per_tissue = {"lung": {"y_true": ["A"], "y_pred": ["A"]}}
    with pytest.raises(ValueError, match="must supply"):
        evaluate_dim_b("test_model", per_tissue)


def test_evaluate_dim_b_missing_keys_raises():
    per_tissue = {"lung": {"y_true": ["A"]}}  # no y_pred
    with pytest.raises(ValueError, match="y_true"):
        evaluate_dim_b("test_model", per_tissue, common_label_sets={"lung": {"A"}})


# ---- Reference-value drift detectors against on-disk artefacts -------------

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_PATH = REPO_ROOT / "tests" / "reference_values.json"
DIM_B_DIR = REPO_ROOT / "results" / "dim_b"


def _ref():
    return json.loads(REF_PATH.read_text())


def test_pca_knn_per_tissue_common_set_matches_reference():
    """Locks results/dim_b/baselines_pca_knn_common.json per-tissue common-set
    macroF1 to the §I.4 / §I.3 Eq. 5 reference table within ±0.001."""
    p = DIM_B_DIR / "baselines_pca_knn_common.json"
    if not p.exists():
        pytest.skip("baselines_pca_knn_common.json not present locally")
    on_disk = json.loads(p.read_text())
    ref_pca = _ref()["dim_b"]["per_tissue_common_set_macroF1"]["pca_knn"]
    for i, tissue in enumerate(on_disk["tissues"]):
        on_disk_v = on_disk["per_tissue_common_set"][i]["macro_f1"]
        assert on_disk_v == pytest.approx(ref_pca[tissue], abs=1e-3), tissue


def test_pca_knn_aggregate_common_set_matches_reference():
    p = DIM_B_DIR / "baselines_pca_knn_common.json"
    if not p.exists():
        pytest.skip("baselines_pca_knn_common.json not present locally")
    on_disk = json.loads(p.read_text())
    ref = _ref()["dim_b"]["common_set_aggregate_macroF1"]["pca_knn"]["value"]
    assert on_disk["aggregate_common_set_macroF1"] == pytest.approx(ref, abs=1e-3)


def test_fm_per_tissue_common_set_matches_reference():
    """Locks the FM common-set per-tissue values that feed the manuscript's
    'PCA+kNN beats every FM in every tissue (except TF on liver)' claim."""
    csv_path = DIM_B_DIR / "common_label_macroF1.csv"
    if not csv_path.exists():
        pytest.skip("common_label_macroF1.csv not present locally")
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(csv_path)
    ref = _ref()["dim_b"]["per_tissue_common_set_macroF1"]
    for model_key in ("geneformer", "scgpt", "uce"):
        for tissue, ref_v in ref[model_key].items():
            row = df[(df["model"] == model_key) & (df["tissue"] == tissue)]
            if row.empty:
                continue
            actual = float(row.iloc[0]["macroF1_common_other"])
            assert actual == pytest.approx(ref_v, abs=1e-3), f"{model_key}/{tissue}"


def test_tf_liver_beats_pca_knn_invariant():
    """The single foundation-model-beats-baseline cell in the matrix:
    TF common-set liver (0.495) > PCA+kNN common-set liver (0.446).

    Anchored test — if either value drifts enough to invert the ordering
    we want CI to scream because the body claim depends on this exception."""
    p_pca = DIM_B_DIR / "baselines_pca_knn_common.json"
    p_tf = DIM_B_DIR / "transcriptformer_common_lung_liver.json"
    if not (p_pca.exists() and p_tf.exists()):
        pytest.skip("Dim B common-set artefacts not present locally")
    pca = json.loads(p_pca.read_text())
    tf = json.loads(p_tf.read_text())
    liver_idx = pca["tissues"].index("liver")
    pca_liver = pca["per_tissue_common_set"][liver_idx]["macro_f1"]
    tf_liver = tf["tf_per_tissue"]["liver"]["common_macroF1"]
    assert tf_liver > pca_liver, (
        f"TF liver ({tf_liver}) no longer beats PCA+kNN liver ({pca_liver}) — "
        "body claim 'except (TF, liver)' may be invalid"
    )
