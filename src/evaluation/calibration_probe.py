"""
Calibration probe for in silico experimentation claims.

Tests whether a model's prediction uncertainty (variance of predicted LFC
across genes) correlates with its actual prediction error (MAE).

If well-calibrated, a model should be more uncertain on perturbations where
it performs poorly. Expected result: near-zero correlation (no calibration
signal), converting an "untestable" claim into an empirical negative.

Runs on CPU in seconds. No new data needed — uses existing Dim A outputs.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

RESULTS_DIM_A = os.path.join(PROJECT_DIR, "results", "dim_a")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "calibration")

MODELS = ["geneformer", "scgpt", "state", "transcriptformer"]


def calibration_probe(pred_lfc, true_lfc):
    """
    Test whether prediction variance correlates with prediction error.

    Args:
        pred_lfc: (n_perturbations, n_genes) predicted log-fold changes
        true_lfc: (n_perturbations, n_genes) true log-fold changes

    Returns:
        dict with calibration Spearman rho, p-value, and interpretation
    """
    per_pert_entropy = pred_lfc.var(axis=1)  # Variance across genes
    per_pert_mae = np.abs(pred_lfc - true_lfc).mean(axis=1)  # MAE per perturbation

    # Filter out zero-variance perturbations
    valid = per_pert_entropy > 1e-10
    if valid.sum() < 5:
        return {
            "calibration_rho": float("nan"),
            "calibration_p": float("nan"),
            "n_perturbations": int(valid.sum()),
            "interpretation": "insufficient_data",
        }

    rho, p = spearmanr(per_pert_entropy[valid], per_pert_mae[valid])

    # Interpret: |rho| > 0.3 is meaningful, > 0.5 is strong
    if abs(rho) > 0.5:
        interpretation = "calibrated"
    elif abs(rho) > 0.3:
        interpretation = "weakly_calibrated"
    else:
        interpretation = "uncalibrated"

    return {
        "calibration_rho": float(rho),
        "calibration_p": float(p),
        "n_perturbations": int(valid.sum()),
        "mean_entropy": float(per_pert_entropy[valid].mean()),
        "mean_mae": float(per_pert_mae[valid].mean()),
        "interpretation": interpretation,
    }


def load_perturbation_predictions(model_name):
    """Load predicted and real perturbation data for a model."""
    import anndata as ad
    from scipy.sparse import issparse

    model_dir = os.path.join(RESULTS_DIM_A, model_name)
    pred_path = os.path.join(model_dir, "predictions.h5ad")

    if not os.path.exists(pred_path):
        return None, None

    adata_pred = ad.read_h5ad(pred_path)
    adata_real = ad.read_h5ad(os.path.join(PROCESSED_DIR, "norman.h5ad"))

    # Compute LFC against control mean
    ctrl_mask = adata_real.obs["condition"] == "ctrl"
    X_ctrl = adata_real[ctrl_mask].X
    if issparse(X_ctrl):
        X_ctrl = X_ctrl.toarray()
    ctrl_mean = np.array(X_ctrl).mean(axis=0)

    X_pred = adata_pred.X
    if issparse(X_pred):
        X_pred = X_pred.toarray()
    pred_lfc = np.array(X_pred) - ctrl_mean

    # Get real LFC for matching perturbations
    pred_perts = adata_pred.obs["condition"].values if "condition" in adata_pred.obs else None
    if pred_perts is None:
        return None, None

    real_lfcs = []
    for pert in pred_perts:
        mask = adata_real.obs["condition"] == pert
        if mask.sum() > 0:
            X_pert = adata_real[mask].X
            if issparse(X_pert):
                X_pert = X_pert.toarray()
            real_lfcs.append(np.array(X_pert).mean(axis=0) - ctrl_mean)
        else:
            real_lfcs.append(np.zeros_like(ctrl_mean))

    true_lfc = np.stack(real_lfcs)

    # Align gene dimensions
    n_genes = min(pred_lfc.shape[1], true_lfc.shape[1])
    return pred_lfc[:, :n_genes], true_lfc[:, :n_genes]


def main():
    print("=== Calibration Probe ===\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for model_name in MODELS:
        print(f"{model_name}:")
        pred_lfc, true_lfc = load_perturbation_predictions(model_name)

        if pred_lfc is None:
            print(f"  SKIP: no predictions found")
            continue

        result = calibration_probe(pred_lfc, true_lfc)
        result["model"] = model_name
        all_results[model_name] = result

        print(f"  rho={result['calibration_rho']:.4f}, "
              f"p={result['calibration_p']:.4e}, "
              f"interpretation={result['interpretation']}")

    out_path = os.path.join(OUTPUT_DIR, "calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
