"""
Perturbation prediction baselines for Dimension A.

Baselines:
- Additive: Ahlmann-Eltze model, y(A+B) = x_ctrl + LFC(A) + LFC(B)
  This baseline OUTPERFORMED all deep learning models on Norman.
- Mean: Predict average expression across all training perturbation conditions
- No-change: Predict control expression (LFC = 0)
"""

import numpy as np
from scipy.sparse import issparse


def additive_baseline(adata, pert_col="condition"):
    """
    Ahlmann-Eltze additive model: y(A+B) = x_ctrl + LFC(A) + LFC(B)

    For each double perturbation A+B, predicts the sum of individual effects.
    """
    ctrl = adata[adata.obs[pert_col] == "ctrl"]
    X_ctrl = ctrl.X.toarray() if issparse(ctrl.X) else np.array(ctrl.X)
    ctrl_mean = X_ctrl.mean(axis=0)

    # Compute log fold changes for single perturbations
    singles = [
        c
        for c in adata.obs[pert_col].unique()
        if "+ctrl" in c or "ctrl+" in c
    ]
    lfc = {}
    for s in singles:
        gene = s.replace("+ctrl", "").replace("ctrl+", "")
        cells = adata[adata.obs[pert_col] == s]
        X_s = cells.X.toarray() if issparse(cells.X) else np.array(cells.X)
        lfc[gene] = X_s.mean(axis=0) - ctrl_mean

    # Predict double perturbations as sum of individual effects
    doubles = [
        c
        for c in adata.obs[pert_col].unique()
        if "+" in c and "ctrl" not in c
    ]
    predictions = {}
    for d in doubles:
        gene_a, gene_b = d.split("+")
        if gene_a in lfc and gene_b in lfc:
            predictions[d] = ctrl_mean + lfc[gene_a] + lfc[gene_b]

    return predictions


def mean_baseline(adata, pert_col="condition"):
    """Predict average expression across all training perturbation conditions."""
    train_perts = [c for c in adata.obs[pert_col].unique() if c != "ctrl"]
    all_means = []
    for p in train_perts:
        cells = adata[adata.obs[pert_col] == p]
        X = cells.X.toarray() if issparse(cells.X) else np.array(cells.X)
        all_means.append(X.mean(axis=0))
    return np.mean(all_means, axis=0)


def no_change_baseline(adata, pert_col="condition"):
    """Predict control expression (LFC = 0)."""
    ctrl = adata[adata.obs[pert_col] == "ctrl"]
    X_ctrl = ctrl.X.toarray() if issparse(ctrl.X) else np.array(ctrl.X)
    return X_ctrl.mean(axis=0)
