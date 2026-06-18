"""Unit tests for vcbench.dimensions.dim_d_cross_modal.evaluate +
reference-value drift detectors against results/dim_d/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vcbench.dimensions.dim_d_cross_modal import DimDResult, evaluate_dim_d


def test_evaluate_dim_d_perfect_prediction():
    rng = np.random.default_rng(0)
    Y = rng.normal(0, 1, size=(60, 10))
    out = evaluate_dim_d(Y, Y, model_name="test")
    assert isinstance(out, DimDResult)
    assert out.mean_pearson_r == pytest.approx(1.0, abs=1e-9)
    assert out.median_pearson_r == pytest.approx(1.0, abs=1e-9)
    assert out.rmse == 0.0
    assert out.n_proteins == 10
    assert out.n_valid_proteins == 10
    assert out.model == "test"


def test_evaluate_dim_d_to_dict_schema_matches_legacy():
    """Output dict must match the field schema of
    results/dim_d/<model>/crossmodal_results.json."""
    rng = np.random.default_rng(0)
    Y = rng.normal(0, 1, size=(60, 10))
    d = evaluate_dim_d(Y, Y, model_name="test").to_dict()
    assert set(d.keys()) == {
        "mean_pearson_r",
        "median_pearson_r",
        "rmse",
        "n_proteins",
        "n_valid_proteins",
        "model",
    }


# ---- Reference-value drift detectors against on-disk artefacts -------------

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_PATH = REPO_ROOT / "tests" / "reference_values.json"
DIM_D_DIR = REPO_ROOT / "results" / "dim_d"


def _ref():
    return json.loads(REF_PATH.read_text())


@pytest.mark.parametrize(
    "subdir,ref_key",
    [
        ("transcriptformer", "transcriptformer"),
        ("uce",              "uce"),
        ("scgpt",            "scgpt"),
        ("geneformer",       "geneformer"),
        ("mean_celltype",    "mean_celltype"),
        ("sclinear",         "scLinear"),
    ],
)
def test_dim_d_pearson_matches_reference(subdir, ref_key):
    """Locks results/dim_d/<model>/crossmodal_results.json mean Pearson R values
    against §I.4 Table 2 within ±0.005."""
    p = DIM_D_DIR / subdir / "crossmodal_results.json"
    if not p.exists():
        pytest.skip(f"{subdir} not present locally")
    on_disk = json.loads(p.read_text())
    ref = _ref()["dim_d"]["table2_pearson"][ref_key]
    assert on_disk["mean_pearson_r"] == pytest.approx(ref, abs=0.005), \
        f"{subdir} drifted from §I.4 ref {ref}"


def test_tf_beats_mean_celltype_invariant():
    """Locks the load-bearing §I.4 invariant: TF beats mean-celltype on Dim D
    (the only Level-2 cell in the entire capability matrix). If this flips,
    the manuscript's Level mapping for TF flips too."""
    p_tf = DIM_D_DIR / "transcriptformer" / "crossmodal_results.json"
    p_mc = DIM_D_DIR / "mean_celltype" / "crossmodal_results.json"
    if not (p_tf.exists() and p_mc.exists()):
        pytest.skip("Dim D artefacts not present locally")
    tf = json.loads(p_tf.read_text())["mean_pearson_r"]
    mc = json.loads(p_mc.read_text())["mean_pearson_r"]
    assert tf > mc, (
        f"TranscriptFormer mean Pearson ({tf}) no longer exceeds mean-celltype "
        f"baseline ({mc}) — VC Level 2 assignment for TF would change"
    )
