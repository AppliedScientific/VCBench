"""Unit tests for vcbench.dimensions.dim_a_perturbation.evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from vcbench.dimensions.dim_a_perturbation import evaluate_dim_a
from vcbench.dimensions.dim_a_perturbation.evaluate import DimAResult


def _build_anndata(
    per_pert_means: dict[str, np.ndarray],
    rows_per_pert: int = 5,
    noise_std: float = 1e-6,
):
    """Create a small AnnData where each perturbation has ``rows_per_pert``
    cells whose expression vectors equal ``per_pert_means[p]`` plus optional
    Gaussian noise.

    Pass ``noise_std=0`` to make the per-perturbation mean *exactly* equal to
    the input target — required for tests that assert PRR == 0 exactly (any
    residual noise lets the PRR's zero-variance guard flip on, producing a
    small but nonzero correlation).
    """
    rng = np.random.default_rng(99)
    rows = []
    obs = []
    for p, mean_vec in per_pert_means.items():
        for _ in range(rows_per_pert):
            if noise_std > 0:
                rows.append(mean_vec + rng.normal(0, noise_std, size=mean_vec.shape))
            else:
                rows.append(mean_vec.copy())
            obs.append({"condition": p})
    X = np.vstack(rows)
    return anndata.AnnData(X=X, obs=pd.DataFrame(obs))


def test_evaluate_dim_a_perfect_prediction():
    """If predicted == observed exactly for every perturbation, PRR=1 and DES=1."""
    rng = np.random.default_rng(0)
    G = 50
    ctrl_mean = rng.normal(5, 2, G)
    real_means = {
        "ctrl": ctrl_mean,
        "p1": ctrl_mean + rng.normal(0, 1, G),
        "p2": ctrl_mean + rng.normal(0, 1, G),
        "p3": ctrl_mean + rng.normal(0, 1, G),
    }
    pred_means = {p: v for p, v in real_means.items() if p != "ctrl"}
    real = _build_anndata(real_means, rows_per_pert=10)
    pred = _build_anndata(pred_means, rows_per_pert=10)
    result = evaluate_dim_a(pred, real)
    assert isinstance(result, DimAResult)
    assert result.n_perturbations == 3
    assert result.mean_pearson_r_delta == pytest.approx(1.0, abs=1e-3)
    assert result.mean_direction_score == pytest.approx(1.0, abs=1e-12)


def test_evaluate_dim_a_no_change_predictor_yields_zero_PRR():
    """Predicted = ctrl mean for every perturbation → pred_delta is 0 → PRR=0, DES=0."""
    rng = np.random.default_rng(1)
    G = 50
    ctrl_mean = rng.normal(5, 2, G)
    real_means = {
        "ctrl": ctrl_mean,
        "p1": ctrl_mean + rng.normal(0, 1, G),
        "p2": ctrl_mean + rng.normal(0, 1, G),
        "p3": ctrl_mean + rng.normal(0, 1, G),
    }
    pred_means = {"p1": ctrl_mean, "p2": ctrl_mean, "p3": ctrl_mean}
    # noise_std=0 throughout: ensures inferred ctrl_mean is exact, so
    # pred_delta is identically zero and PRR's zero-variance guard fires.
    real = _build_anndata(real_means, rows_per_pert=10, noise_std=0)
    pred = _build_anndata(pred_means, rows_per_pert=10, noise_std=0)
    result = evaluate_dim_a(pred, real)
    assert result.mean_pearson_r_delta == pytest.approx(0.0, abs=1e-12)
    assert result.mean_direction_score == pytest.approx(0.0, abs=1e-12)


def test_evaluate_dim_a_missing_control_raises():
    rng = np.random.default_rng(2)
    means = {"p1": rng.normal(0, 1, 20), "p2": rng.normal(0, 1, 20)}
    real = _build_anndata(means)
    pred = _build_anndata(means)
    with pytest.raises(ValueError, match="control"):
        evaluate_dim_a(pred, real)


def test_evaluate_dim_a_no_overlap_raises():
    rng = np.random.default_rng(3)
    real = _build_anndata({"ctrl": rng.normal(0, 1, 20), "p1": rng.normal(0, 1, 20)})
    pred = _build_anndata({"q1": rng.normal(0, 1, 20)})
    with pytest.raises(ValueError, match="no perturbations"):
        evaluate_dim_a(pred, real)


# ----------------------------------------------------------------------
# Gene-vocabulary alignment regression tests.
# These guard against a class of bug where adata_pred uses
# integer-positional gene indices ("0", "1", "2", ...) and adata_real
# uses Ensembl IDs ("ENSG00000...") — same shape but zero gene-name
# overlap, producing noise-floor PRR (~0.1) and DES (~0.5).


def _build_anndata_with_var_names(per_pert_means, var_names):
    """Like _build_anndata but lets the caller pin var.index explicitly."""
    a = _build_anndata(per_pert_means, rows_per_pert=5, noise_std=0)
    a.var.index = pd.Index([str(v) for v in var_names])
    return a


def test_evaluate_dim_a_gene_count_mismatch_raises():
    """Different gene COUNT → fail fast with helpful message."""
    rng = np.random.default_rng(0)
    G = 10
    ctrl = rng.normal(5, 2, G)
    real = _build_anndata({"ctrl": ctrl, "p1": ctrl + rng.normal(0, 1, G)}, noise_std=0)
    pred_means_short = {"p1": (ctrl[:5] + rng.normal(0, 1, 5))}
    pred = _build_anndata(pred_means_short, rows_per_pert=5, noise_std=0)
    with pytest.raises(ValueError, match="gene count mismatch"):
        evaluate_dim_a(pred, real)


def test_evaluate_dim_a_zero_gene_overlap_raises():
    """Same shape but DIFFERENT vocabulary (e.g. integer indices vs Ensembl)
    must raise — this is the exact gene-vocabulary mismatch bug class."""
    rng = np.random.default_rng(0)
    G = 10
    ctrl = rng.normal(5, 2, G)
    means = {"ctrl": ctrl, "p1": ctrl + rng.normal(0, 1, G)}
    real = _build_anndata_with_var_names(
        means, var_names=[f"ENSG{i:011d}" for i in range(G)]
    )
    pred = _build_anndata_with_var_names(
        {"p1": ctrl + rng.normal(0, 1, G)},
        var_names=[str(i) for i in range(G)],
    )
    with pytest.raises(ValueError, match="ZERO\\s+overlap"):
        evaluate_dim_a(pred, real)


def test_evaluate_dim_a_partial_gene_overlap_with_wrong_order_raises():
    """Same vocabulary but shuffled order also raises (positional
    aggregation would compare wrong genes against each other)."""
    rng = np.random.default_rng(0)
    G = 10
    ctrl = rng.normal(5, 2, G)
    names = [f"GENE{i}" for i in range(G)]
    means = {"ctrl": ctrl, "p1": ctrl + rng.normal(0, 1, G)}
    real = _build_anndata_with_var_names(means, var_names=names)
    pred = _build_anndata_with_var_names(
        {"p1": ctrl + rng.normal(0, 1, G)},
        var_names=names[::-1],   # same set, reversed order
    )
    with pytest.raises(ValueError, match="positional"):
        evaluate_dim_a(pred, real)


def test_evaluate_dim_a_aligned_var_index_passes():
    """Sanity: identical var.index in same order works fine (no false positive)."""
    rng = np.random.default_rng(0)
    G = 10
    ctrl = rng.normal(5, 2, G)
    names = [f"GENE{i}" for i in range(G)]
    real = _build_anndata_with_var_names(
        {"ctrl": ctrl, "p1": ctrl + rng.normal(0, 1, G)}, var_names=names,
    )
    pred = _build_anndata_with_var_names(
        {"p1": ctrl + rng.normal(0, 1, G)}, var_names=names,
    )
    # Should not raise; should produce a sensible result.
    result = evaluate_dim_a(pred, real)
    assert result.n_perturbations == 1


# ----------------------------------------------------------------------
# Anchor-convention reconciliation tests.
# Document and lock in the difference between vcbench's principled
# shared-real-control anchor (default, §I.3 convention) and the per-model
# anchor used by upstream `cell-eval pearson_delta` (Arc Institute). Under
# matched anchor conventions the two evaluators agree to within numerical
# precision (1e-6); under mismatched conventions they can diverge by ~0.02
# PRR or more, with the cell-eval (per-model) number generally inflated
# because the predicted control absorbs systematic baseline offsets.


def _scipy_pearsonr_per_pert_celleval_style(pred, real, control_label="ctrl"):
    """Pure-numpy reference reproduction of cell-eval's per-perturbation
    Pearson-on-Δ algorithm, exactly as implemented in
    ``cell_eval/_types/_anndata.py::BulkArrays.perturbation_effect`` and
    ``cell_eval/metrics/_anndata.py::pearson_delta``:

        pred_delta = pert_pred_mean - ctrl_pred_mean       # PER-MODEL anchor
        real_delta = pert_real_mean - ctrl_real_mean       # real anchor

    Returns a dict ``{condition: r}``.
    """
    from scipy.stats import pearsonr  # type: ignore

    real_conds = np.asarray(real.obs["condition"])
    pred_conds = np.asarray(pred.obs["condition"])
    ctrl_real_mean = np.asarray(real[real_conds == control_label].X).mean(axis=0).ravel()
    ctrl_pred_mean = np.asarray(pred[pred_conds == control_label].X).mean(axis=0).ravel()
    out: dict[str, float] = {}
    for c in sorted(set(pred_conds) - {control_label}):
        rm = real_conds == c
        pm = pred_conds == c
        if not rm.any() or not pm.any():
            continue
        pert_pred = np.asarray(pred[pm].X).mean(axis=0).ravel()
        pert_real = np.asarray(real[rm].X).mean(axis=0).ravel()
        pd_delta = pert_pred - ctrl_pred_mean
        rd_delta = pert_real - ctrl_real_mean
        out[c] = float(pearsonr(pd_delta, rd_delta)[0])
    return out


def test_control_anchor_pred_reproduces_cell_eval_algorithm():
    """``control_anchor='pred'`` must reproduce upstream cell-eval's
    per-perturbation Pearson-on-Δ to numerical precision (≤ 1e-9 absolute).

    This is the locked equivalence — if it ever breaks, vcbench's claim
    that 'cell-eval and vcbench numbers cross-validate to numerical
    precision under matched anchor conventions' becomes false.
    Cross-checked on Arc State predictions at max abs diff = 2e-6.
    """
    rng = np.random.default_rng(7)
    G = 60
    # Construct a non-trivial scenario where the predicted control is
    # systematically OFFSET from the real control. This is the regime
    # where the two anchor conventions diverge most strongly.
    ctrl_real = rng.normal(5, 2, G)
    ctrl_pred = ctrl_real + rng.normal(0.3, 0.5, G)  # baseline shift
    real_means = {
        "ctrl": ctrl_real,
        "p1": ctrl_real + rng.normal(0, 1, G),
        "p2": ctrl_real + rng.normal(0, 1, G),
        "p3": ctrl_real + rng.normal(0, 1, G),
    }
    pred_means = {
        "ctrl": ctrl_pred,
        "p1": ctrl_pred + rng.normal(0, 1, G),
        "p2": ctrl_pred + rng.normal(0, 1, G),
        "p3": ctrl_pred + rng.normal(0, 1, G),
    }
    real = _build_anndata(real_means, rows_per_pert=20, noise_std=0.05)
    pred = _build_anndata(pred_means, rows_per_pert=20, noise_std=0.05)

    # Reference (cell-eval style — per-model anchor)
    ref = _scipy_pearsonr_per_pert_celleval_style(pred, real)

    # vcbench under per-model anchor
    res = evaluate_dim_a(pred, real, control_anchor="pred")
    vc_per_pert = dict(zip(res.per_perturbation["condition"], res.per_perturbation["pearson_r_delta"]))

    assert set(ref.keys()) == set(vc_per_pert.keys())
    for c in ref:
        assert abs(ref[c] - vc_per_pert[c]) < 1e-9, (
            f"vcbench(pred-anchor) and cell-eval-style algorithm disagree on {c!r}: "
            f"{vc_per_pert[c]} vs {ref[c]}"
        )


def test_real_anchor_is_more_conservative_than_pred_anchor_under_baseline_drift():
    """The two conventions are ASYMMETRIC, not interchangeable.

    Under per-gene baseline drift between ctrl_pred and ctrl_real, the
    real-anchor PRR is **strictly lower** than the pred-anchor PRR — i.e.
    vcbench's principled real-anchor default produces a more conservative
    (lower) number than cell-eval's per-model pred-anchor under the same
    inputs. Mechanism: pred-anchor strips per-gene ctrl_pred − ctrl_real
    drift out of pred_delta (the drift is a property of ctrl_pred itself);
    real-anchor leaves the drift in pred_delta, which adds noise that
    decorrelates pred_delta from real_delta and lowers R. This asymmetry
    is why VCBench uses the real anchor canonically for VC Level
    decisions: it prevents systematically-biased models from
    hiding baseline offsets in their predicted control. The
    pred anchor exists only so VCBench can numerically reproduce
    upstream cell-eval reports; it is NOT a competing canonical
    convention.

    Empirical evidence from real Arc State predictions:
        0.4021 (real) < 0.4076 (pred) — Δ 0.0055

    This test fails first if someone accidentally collapses the two
    anchor branches into one, OR if someone accidentally swaps the
    direction of the asymmetry (e.g. by anchoring real_delta on
    ctrl_pred). Pearson R is invariant under uniform additive shifts,
    so the realistic regime is per-gene non-uniform drift — exactly
    what the Arc State predictions exhibit (max abs per-gene
    ctrl_pred − ctrl_real was 0.066 across the 5,045 genes).
    """
    rng = np.random.default_rng(8)
    G = 40
    ctrl_real = rng.normal(5, 2, G)
    # Per-gene non-uniform shift — varies in [−0.8, 0.8] across genes.
    per_gene_shift = rng.normal(0, 0.5, G)
    ctrl_pred = ctrl_real + per_gene_shift
    real_means = {
        "ctrl": ctrl_real,
        "p1": ctrl_real + rng.normal(0, 1, G),
        "p2": ctrl_real + rng.normal(0, 1, G),
    }
    pred_means = {
        "ctrl": ctrl_pred,
        "p1": ctrl_pred + rng.normal(0, 1, G),
        "p2": ctrl_pred + rng.normal(0, 1, G),
    }
    real = _build_anndata(real_means, rows_per_pert=20, noise_std=0.05)
    pred = _build_anndata(pred_means, rows_per_pert=20, noise_std=0.05)

    res_real = evaluate_dim_a(pred, real, control_anchor="real")
    res_pred = evaluate_dim_a(pred, real, control_anchor="pred")

    # Branches must produce different numbers (catches accidental collapse).
    assert abs(res_real.mean_pearson_r_delta - res_pred.mean_pearson_r_delta) > 1e-3, (
        "Anchor conventions collapsed to identical numbers under a deliberate "
        "per-gene baseline shift — the two branches must produce different "
        f"deltas. real_anchor PRR={res_real.mean_pearson_r_delta:.6f}, "
        f"pred_anchor PRR={res_pred.mean_pearson_r_delta:.6f}."
    )

    # ASYMMETRY: real-anchor is STRICTLY MORE CONSERVATIVE than pred-anchor.
    # This is the load-bearing property — it's what makes the real anchor
    # the right default for cross-model benchmarking.
    assert res_real.mean_pearson_r_delta < res_pred.mean_pearson_r_delta, (
        "Real-anchor PRR must be ≤ pred-anchor PRR under per-gene baseline "
        f"drift (real_anchor={res_real.mean_pearson_r_delta:.6f}, "
        f"pred_anchor={res_pred.mean_pearson_r_delta:.6f}). If this assertion "
        "fails, the asymmetry direction has been corrupted — the canonical "
        "convention claim no longer holds."
    )


def test_control_anchor_uniform_shift_does_not_change_R():
    """Documentation test: Pearson R is invariant under a UNIFORM additive
    shift to one of the variables, so a global ``ctrl_pred = ctrl_real + c``
    shift produces identical PRR under both anchor conventions. This is the
    mathematical reason the convention only matters for non-uniform per-gene
    baseline differences (which is the realistic regime for foundation-model
    predictions)."""
    rng = np.random.default_rng(11)
    G = 40
    ctrl_real = rng.normal(5, 2, G)
    ctrl_pred = ctrl_real + 1.5  # uniform +1.5
    real_means = {
        "ctrl": ctrl_real,
        "p1": ctrl_real + rng.normal(0, 1, G),
    }
    pred_means = {
        "ctrl": ctrl_pred,
        "p1": ctrl_pred + rng.normal(0, 1, G),
    }
    real = _build_anndata(real_means, rows_per_pert=20, noise_std=0.05)
    pred = _build_anndata(pred_means, rows_per_pert=20, noise_std=0.05)
    res_real = evaluate_dim_a(pred, real, control_anchor="real")
    res_pred = evaluate_dim_a(pred, real, control_anchor="pred")
    assert res_real.mean_pearson_r_delta == pytest.approx(
        res_pred.mean_pearson_r_delta, abs=1e-12
    )


def test_control_anchor_pred_falls_back_with_warning_when_pred_lacks_ctrl():
    """If the caller asks for ``control_anchor='pred'`` but adata_pred has
    no control cells (some reduced-output predict pipelines), vcbench
    should fall back to the real anchor and emit a warning rather than
    crash."""
    import warnings

    rng = np.random.default_rng(9)
    G = 30
    ctrl_real = rng.normal(5, 2, G)
    real = _build_anndata({"ctrl": ctrl_real, "p1": ctrl_real + rng.normal(0, 1, G)},
                          rows_per_pert=10, noise_std=0.05)
    # No "ctrl" rows in pred
    pred = _build_anndata({"p1": ctrl_real + rng.normal(0, 1, G)},
                          rows_per_pert=10, noise_std=0.05)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_pred = evaluate_dim_a(pred, real, control_anchor="pred")
        msgs = [str(w.message) for w in caught]
    assert any("falling back to the real-control anchor" in m for m in msgs), (
        f"expected fallback warning, got: {msgs}"
    )
    res_real = evaluate_dim_a(pred, real, control_anchor="real")
    # With the fallback, pred-anchor should produce the same number as real-anchor
    assert res_pred.mean_pearson_r_delta == pytest.approx(
        res_real.mean_pearson_r_delta, abs=1e-12
    )


def test_control_anchor_invalid_value_raises():
    rng = np.random.default_rng(10)
    G = 20
    ctrl = rng.normal(5, 2, G)
    real = _build_anndata({"ctrl": ctrl, "p1": ctrl + rng.normal(0, 1, G)})
    pred = _build_anndata({"p1": ctrl + rng.normal(0, 1, G)})
    with pytest.raises(ValueError, match="control_anchor"):
        evaluate_dim_a(pred, real, control_anchor="bogus")  # type: ignore[arg-type]


def test_to_aggregate_dict_schema_matches_legacy_cell_eval_results_json():
    """The aggregate dict must use the exact field names that
    results/dim_a/<model>/cell_eval_results.json has on disk, so the
    schema-equivalence to the legacy fallback path is preserved."""
    rng = np.random.default_rng(4)
    G = 30
    ctrl_mean = rng.normal(5, 2, G)
    real = _build_anndata({"ctrl": ctrl_mean, "p1": ctrl_mean + rng.normal(0, 1, G)})
    pred = _build_anndata({"p1": ctrl_mean + rng.normal(0, 1, G)})
    out = evaluate_dim_a(pred, real).to_aggregate_dict()
    assert set(out.keys()) == {
        "mean_pearson_r_delta",
        "median_pearson_r_delta",
        "mean_mse_delta",
        "mean_direction_score",
        "n_perturbations",
    }


# ---- Reference-value drift detection --------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_PATH = REPO_ROOT / "tests" / "reference_values.json"
ON_DISK_DIR = REPO_ROOT / "results" / "dim_a"


def _load_ref():
    return json.loads(REF_PATH.read_text())


def _load_on_disk_aggregate(rel_path: str) -> dict:
    """Helper: read one of the existing cell_eval_results.json files and unwrap
    the {"0": value} index that pandas to_dict adds when persisting a 1-row df."""
    p = ON_DISK_DIR / rel_path
    raw = json.loads(p.read_text())
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict) and set(v.keys()) == {"0"}:
            out[k] = v["0"]
        else:
            out[k] = v
    return out


@pytest.mark.parametrize(
    "manifest_path,model_key,subset_key",
    [
        ("baselines/additive_results.json",  "additive_baseline",  "additive_evaluable_71"),
        ("baselines/mean_results.json",      "mean_baseline",      "full_107"),
        ("baselines/no_change_results.json", "no_change_baseline", "full_107"),
    ],
)
def test_baseline_on_disk_values_match_reference_fixture(
    manifest_path, model_key, subset_key
):
    """If results/dim_a/baselines/* drifts away from §I.4 numbers, this fails.
    Locks the manuscript-cited baseline values to the on-disk artefacts."""
    if not (ON_DISK_DIR / manifest_path).exists():
        pytest.skip(f"{manifest_path} not present locally")
    ref = _load_ref()["dim_a"][model_key][subset_key]
    actual = _load_on_disk_aggregate(manifest_path)
    assert actual["mean_pearson_r_delta"] == pytest.approx(ref["PRR"], abs=1e-3)
    if "DES" in ref:
        assert actual["mean_direction_score"] == pytest.approx(ref["DES"], abs=1e-3)


def test_scgpt_partition_decomposition_matches_reference():
    """Locks the scGPT shared/novel/full PRR breakdown — the load-bearing
    partition values for the §3.1 'compositional generalization' Results claim.

    Schema: results/dim_a/scgpt/partition_pds_v2.json wraps each subset under
    ``shared_doubles`` / ``novel_singles`` / ``full`` with the cell-eval
    fallback field names (mean_pearson_r_delta etc).
    """
    p = ON_DISK_DIR / "scgpt" / "partition_pds_v2.json"
    if not p.exists():
        pytest.skip("partition_pds_v2.json not present locally")
    on_disk = json.loads(p.read_text())
    ref = _load_ref()["dim_a"]["scgpt_ft"]
    assert on_disk["shared_doubles"]["n_perturbations"] == 71
    assert on_disk["novel_singles"]["n_perturbations"] == 36
    assert on_disk["full"]["n_perturbations"] == 107
    assert on_disk["shared_doubles"]["mean_pearson_r_delta"] == pytest.approx(
        ref["additive_evaluable_71"]["PRR"], abs=1e-3
    )
    assert on_disk["novel_singles"]["mean_pearson_r_delta"] == pytest.approx(
        ref["novel_36"]["PRR"], abs=1e-3
    )
    assert on_disk["full"]["mean_pearson_r_delta"] == pytest.approx(
        ref["full_107"]["PRR"], abs=1e-3
    )
