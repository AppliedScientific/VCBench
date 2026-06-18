"""Unit tests for vcbench.dimensions.dim_e_temporal.evaluate +
reference-value drift detectors against results/dim_e/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcbench.dimensions.dim_e_temporal import DimEResult, evaluate_dim_e


def test_evaluate_dim_e_basic():
    out = evaluate_dim_e(
        "test_model",
        per_dataset_taus={"sci_fate": 0.225, "weinreb": 0.153},
        per_dataset_n_cells={"sci_fate": 6567, "weinreb": 49008},
    )
    assert isinstance(out, DimEResult)
    assert out.model == "test_model"
    assert len(out.per_dataset) == 2
    assert out.aggregate_kendall_tau_b == pytest.approx((0.225 + 0.153) / 2, abs=1e-12)


def test_evaluate_dim_e_carries_notes():
    out = evaluate_dim_e(
        "transcriptformer",
        per_dataset_taus={"sci_fate": 0.04, "weinreb": 0.04},
        notes={"weinreb": "10x5K bootstrap; ARPACK non-convergent on full graph"},
    )
    weinreb_row = next(r for r in out.per_dataset if r.dataset == "weinreb")
    assert "ARPACK" in weinreb_row.note


# ---- Reference-value drift detectors against on-disk artefacts ------------

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_PATH = REPO_ROOT / "tests" / "reference_values.json"
DIM_E_DIR = REPO_ROOT / "results" / "dim_e"


def _ref():
    return json.loads(REF_PATH.read_text())


@pytest.mark.parametrize("subdir,ref_key", [
    ("uce",              "uce"),
    ("transcriptformer", "transcriptformer"),
    ("geneformer",       "geneformer"),
    ("scgpt",            "scgpt"),
    ("pca_dpt",          "pca_dpt"),
])
def test_dim_e_aggregate_matches_reference(subdir, ref_key):
    """Locks the unweighted-mean across-dataset τ-b in
    results/dim_e/<model>/{sci_fate,weinreb}/temporal_results.json
    against §I.4 Table 2 within ±0.005."""
    sci_fate_p = DIM_E_DIR / subdir / "sci_fate" / "temporal_results.json"
    weinreb_p = DIM_E_DIR / subdir / "weinreb" / "temporal_results.json"
    if not (sci_fate_p.exists() and weinreb_p.exists()):
        pytest.skip(f"{subdir} per-dataset artefacts not present locally")
    sf = json.loads(sci_fate_p.read_text())
    wb = json.loads(weinreb_p.read_text())
    sf_tau = sf.get("kendall_tau_b") or sf.get("kendall_tau")
    wb_tau = wb.get("kendall_tau_b") or wb.get("kendall_tau")
    aggregate = (sf_tau + wb_tau) / 2.0
    ref = _ref()["dim_e"]["table2_kendall_tau_b"][ref_key]
    # TF gets a wider tolerance: Weinreb is bootstrapped (10 subsamples)
    # so its mean carries non-trivial sampling noise (std ≈ 0.078).
    tol = 0.01 if subdir == "transcriptformer" else 0.005
    assert aggregate == pytest.approx(ref, abs=tol), \
        f"{subdir} aggregate {aggregate:.4f} drifted from §I.4 ref {ref}"


def test_scgpt_weinreb_temporal_inversion_locked():
    """Locks scGPT Weinreb-specific τ-b ≈ -0.103 (the inversion result that
    aggregation-rule choice protects). Drift here would change the inversion
    narrative in the manuscript Results section."""
    p = DIM_E_DIR / "scgpt" / "weinreb" / "temporal_results.json"
    if not p.exists():
        pytest.skip("scgpt/weinreb artefact not present locally")
    on_disk = json.loads(p.read_text())
    tau = on_disk.get("kendall_tau_b") or on_disk.get("kendall_tau")
    ref = _ref()["dim_e"]["scgpt_weinreb_temporal_inversion"]
    assert tau == pytest.approx(ref, abs=0.005)


def test_tf_weinreb_bootstrap_std_locked():
    """Locks the TF Weinreb bootstrap std ≈ 0.078 (the ±value cited in §I.4
    footnote and on Fig 5 Dim E TF error bar)."""
    p = DIM_E_DIR / "transcriptformer" / "weinreb" / "temporal_results.json"
    if not p.exists():
        pytest.skip("transcriptformer/weinreb bootstrap artefact not present")
    on_disk = json.loads(p.read_text())
    std = on_disk.get("kendall_tau_std")
    if std is None:
        pytest.skip("kendall_tau_std field absent (non-bootstrap run)")
    ref = _ref()["dim_e"]["transcriptformer_weinreb_bootstrap_std"]
    assert std == pytest.approx(ref, abs=0.005)


def test_pca_dpt_baseline_aggregate_matches_reference():
    """The PCA+DPT baseline aggregate τ-b is the binding Dim E threshold (0.190).
    If this drifts the entire Dim E Level decision changes."""
    p = REPO_ROOT / "results" / "baselines" / "temporal_pca_dpt.json"
    if not p.exists():
        pytest.skip("baselines/temporal_pca_dpt.json not present locally")
    on_disk = json.loads(p.read_text())
    aggregate = (on_disk["sci_fate"]["kendall_tau_b"]
                 + on_disk["weinreb"]["kendall_tau_b"]) / 2.0
    ref = _ref()["dim_e"]["table2_kendall_tau_b"]["pca_dpt"]
    assert aggregate == pytest.approx(ref, abs=0.005)
