"""
Evaluation metrics for all five VCBench dimensions.

Dimension A (Perturbation): cell-eval wrapper (PDS, DES, MAE, composite)
Dimension B (Cross-Species): kNN transfer F1 scores
Dimension C (GRN): AUROC, AUPRC, Early Precision Ratio (EPR)
Dimension D (Cross-Modal): Per-protein Pearson R and RMSE
Dimension E (Temporal): Kendall tau-b and kNN balanced accuracy
"""

import os

import numpy as np
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier


def evaluate_perturbation(adata_pred, adata_real, pert_col="condition", control="ctrl"):
    """
    Wrapper around cell-eval for perturbation prediction evaluation.

    Returns per-perturbation and aggregate results (PDS, DES, MAE, composite).
    Falls back to built-in metrics if cell_eval is not installed (e.g. Python 3.9).
    """
    try:
        from cell_eval import MetricsEvaluator

        evaluator = MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=control,
            pert_col=pert_col,
            num_threads=os.cpu_count(),
        )
        results, agg_results = evaluator.compute()
        return results, agg_results
    except ImportError:
        print("WARNING: cell_eval not installed. Using built-in perturbation metrics.")
        return _evaluate_perturbation_fallback(adata_pred, adata_real, pert_col, control)
    except (ValueError, RuntimeError) as e:
        print(f"WARNING: cell_eval failed ({e}). Using built-in perturbation metrics.")
        return _evaluate_perturbation_fallback(adata_pred, adata_real, pert_col, control)


def _evaluate_perturbation_fallback(adata_pred, adata_real, pert_col="condition", control="ctrl"):
    """
    Built-in perturbation evaluation: per-perturbation Pearson R, MSE, and
    direction score (fraction of top-20 DEGs with correct sign of change).
    """
    import pandas as pd
    from scipy.sparse import issparse

    conditions = adata_pred.obs[pert_col].values
    real_conditions = adata_real.obs[pert_col].values

    # Get control mean expression
    ctrl_mask = real_conditions == control
    X_ctrl = adata_real[ctrl_mask].X
    X_ctrl = X_ctrl.toarray() if issparse(X_ctrl) else np.array(X_ctrl)
    ctrl_mean = X_ctrl.mean(axis=0)

    per_pert = []
    for cond in sorted(set(conditions)):
        if cond == control:
            continue

        # Predicted
        pred_mask = conditions == cond
        X_pred = adata_pred[pred_mask].X
        X_pred = X_pred.toarray() if issparse(X_pred) else np.array(X_pred)
        pred_mean = X_pred.mean(axis=0).flatten()

        # Real
        real_mask = real_conditions == cond
        if real_mask.sum() == 0:
            continue
        X_real = adata_real[real_mask].X
        X_real = X_real.toarray() if issparse(X_real) else np.array(X_real)
        real_mean = X_real.mean(axis=0).flatten()

        # Delta from control
        pred_delta = pred_mean - ctrl_mean.flatten()
        real_delta = real_mean - ctrl_mean.flatten()

        # Pearson R on deltas
        if np.std(pred_delta) > 1e-10 and np.std(real_delta) > 1e-10:
            r, _ = pearsonr(pred_delta, real_delta)
        else:
            r = 0.0

        # MSE
        mse = float(np.mean((pred_delta - real_delta) ** 2))

        # Direction score: top-20 DEGs by absolute real delta
        top_k = min(20, len(real_delta))
        top_idx = np.argsort(-np.abs(real_delta))[:top_k]
        direction_correct = np.sign(pred_delta[top_idx]) == np.sign(real_delta[top_idx])
        direction_score = float(direction_correct.mean())

        per_pert.append({
            "condition": cond,
            "pearson_r_delta": float(r),
            "mse_delta": mse,
            "direction_score_top20": direction_score,
        })

    results_df = pd.DataFrame(per_pert)
    agg = pd.DataFrame([{
        "mean_pearson_r_delta": results_df["pearson_r_delta"].mean(),
        "median_pearson_r_delta": results_df["pearson_r_delta"].median(),
        "mean_mse_delta": results_df["mse_delta"].mean(),
        "mean_direction_score": results_df["direction_score_top20"].mean(),
        "n_perturbations": len(results_df),
    }])
    return results_df, agg


def evaluate_cross_species(human_emb, human_labels, mouse_emb, mouse_labels, k=5):
    """
    kNN transfer from human to mouse embeddings.

    Fits kNN on human embeddings, predicts mouse cell types.
    Only evaluates on shared cell types between species.
    """
    # Handle embedding/label shape mismatches (e.g. UCE filters some cells)
    if human_emb.shape[0] != human_labels.shape[0]:
        n = min(human_emb.shape[0], human_labels.shape[0])
        human_emb = human_emb[:n]
        human_labels = human_labels[:n]
    if mouse_emb.shape[0] != mouse_labels.shape[0]:
        n = min(mouse_emb.shape[0], mouse_labels.shape[0])
        mouse_emb = mouse_emb[:n]
        mouse_labels = mouse_labels[:n]

    shared_types = set(human_labels) & set(mouse_labels)
    if not shared_types:
        return {"macro_f1": 0.0, "weighted_f1": 0.0, "n_shared_types": 0}

    h_mask = np.isin(human_labels, list(shared_types))
    m_mask = np.isin(mouse_labels, list(shared_types))

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(human_emb[h_mask], human_labels[h_mask])
    preds = knn.predict(mouse_emb[m_mask])

    return {
        "macro_f1": f1_score(mouse_labels[m_mask], preds, average="macro"),
        "weighted_f1": f1_score(mouse_labels[m_mask], preds, average="weighted"),
        "n_shared_types": len(shared_types),
    }


def evaluate_grn(predicted_df, ground_truth_edges, all_tf_gene_pairs):
    """
    AUROC, AUPRC, and Early Precision Ratio (EPR) for predicted GRN edges.

    Args:
        predicted_df: DataFrame with columns ['TF', 'target', 'score']
        ground_truth_edges: Set of (TF, target) tuples
        all_tf_gene_pairs: List of all possible (TF, target) pairs to evaluate
    """
    pair_to_score = {
        (r["TF"], r["target"]): r["score"] for _, r in predicted_df.iterrows()
    }

    y_true, y_score = [], []
    for pair in all_tf_gene_pairs:
        y_true.append(1 if pair in ground_truth_edges else 0)
        y_score.append(pair_to_score.get(pair, 0.0))

    y_true, y_score = np.array(y_true), np.array(y_score)

    # Guard: metrics fail on single-class input
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print(
            f"WARNING: Ground truth has {int(y_true.sum())}/{len(y_true)} positive "
            "edges. Cannot compute AUROC/AUPRC. Check vocabulary intersection."
        )
        return {
            "AUROC": float("nan"),
            "AUPRC": float("nan"),
            "AUPRC_ratio": float("nan"),
            "EPR": float("nan"),
            "edge_density": y_true.sum() / len(y_true) if len(y_true) > 0 else 0,
            "n_true_edges": int(y_true.sum()),
            "n_total_pairs": len(y_true),
        }

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    edge_density = y_true.sum() / len(y_true)

    # Early Precision Ratio: precision in top-k predictions / random baseline
    k = int(y_true.sum())
    sorted_idx = np.argsort(-y_score)
    early_precision = y_true[sorted_idx[:k]].sum() / k
    epr = early_precision / edge_density

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "AUPRC_ratio": auprc / edge_density,
        "EPR": epr,
        "edge_density": float(edge_density),
        "n_true_edges": int(y_true.sum()),
        "n_total_pairs": len(y_true),
    }


def evaluate_crossmodal(y_pred, y_true):
    """
    Dimension D: Per-protein Pearson R and RMSE for RNA→protein prediction.

    Args:
        y_pred: (n_cells, n_proteins) predicted protein expression
        y_true: (n_cells, n_proteins) ground truth protein expression
    """
    n_proteins = y_true.shape[1]
    per_protein_r = []
    for i in range(n_proteins):
        pred_col = y_pred[:, i]
        true_col = y_true[:, i]
        # Skip constant columns (zero variance)
        if np.std(true_col) < 1e-10 or np.std(pred_col) < 1e-10:
            per_protein_r.append(float("nan"))
        else:
            r, _ = pearsonr(pred_col, true_col)
            per_protein_r.append(r)

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    return {
        "mean_pearson_r": float(np.nanmean(per_protein_r)),
        "median_pearson_r": float(np.nanmedian(per_protein_r)),
        "rmse": rmse,
        "n_proteins": n_proteins,
        "n_valid_proteins": int(np.sum(~np.isnan(per_protein_r))),
    }


def evaluate_temporal(embeddings, true_time, k=15):
    """
    Dimension E: Temporal ordering via DPT and kNN classification.

    Computes:
    - Kendall tau-b between diffusion pseudotime (DPT) and true collection time
    - kNN balanced accuracy for timepoint classification (5-fold CV)

    Args:
        embeddings: (n_cells, n_dims) cell embeddings
        true_time: (n_cells,) numeric collection times
        k: Number of neighbors for kNN graph and classifier
    """
    import scanpy as sc

    # Build temporary AnnData for scanpy DPT computation
    adata_tmp = sc.AnnData(X=embeddings)
    sc.pp.neighbors(adata_tmp, n_neighbors=k, use_rep="X")
    sc.tl.diffmap(adata_tmp)

    # Root cell = earliest time
    root_idx = int(np.argmin(true_time))
    adata_tmp.uns["iroot"] = root_idx
    sc.tl.dpt(adata_tmp)

    dpt = adata_tmp.obs["dpt_pseudotime"].values

    # Handle infinite DPT values (disconnected components)
    valid = np.isfinite(dpt)
    if valid.sum() < len(dpt) * 0.5:
        print(f"  WARNING: {(~valid).sum()}/{len(dpt)} cells have infinite DPT")

    tau, p_value = kendalltau(dpt[valid], true_time[valid])

    # kNN classification of discrete timepoints (5-fold CV)
    time_labels = np.array(true_time)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(embeddings) - 1))
    preds = cross_val_predict(knn, embeddings, time_labels, cv=5)
    bal_acc = balanced_accuracy_score(time_labels, preds)

    return {
        "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
        "kendall_p": float(p_value) if not np.isnan(p_value) else 1.0,
        "knn_balanced_accuracy": float(bal_acc),
        "n_valid_dpt": int(valid.sum()),
        "n_cells": len(true_time),
    }
