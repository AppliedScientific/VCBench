"""Dim D baselines.

* :func:`mean_celltype_baseline` — Eq. 8. The binding non-FM baseline:
  predict each test cell's surface-protein vector as the per-cell-type
  mean over the training set, where the cell-type label is itself
  predicted by a kNN classifier on the RNA modality.

Why this baseline matters: it carries no information beyond the predicted
cell-type identity. An FM that fails to exceed 0.152 mean Pearson on the
NeurIPS CITE-seq protocol has not learned anything beyond cell-type
conditioning of protein abundance — a sharp, interpretable failure mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


@dataclass(frozen=True)
class MeanCelltypeFit:
    """Fitted mean-celltype baseline ready to predict on a test RNA matrix."""

    knn: KNeighborsClassifier
    celltype_means: dict[str, np.ndarray]   # celltype -> per-protein mean vector
    fallback_mean: np.ndarray               # global protein mean for unknown celltypes


def fit_mean_celltype_baseline(
    train_rna: np.ndarray,
    train_celltype: Sequence,
    train_protein: np.ndarray,
    *,
    k: int = 5,
    metric: str = "cosine",
) -> MeanCelltypeFit:
    """Fit the Eq. 8 mean-celltype baseline on training cells.

    Parameters
    ----------
    train_rna : np.ndarray
        Training RNA embeddings (or expression matrix), shape (n_train, n_features).
    train_celltype : sequence
        Cell-type labels for each training cell, length n_train.
    train_protein : np.ndarray
        Surface-protein abundance for each training cell, shape (n_train, n_proteins).
    k : int, default 5
        kNN neighbour count for the cell-type classifier.
    metric : str, default ``"cosine"``
        kNN distance metric.

    Returns
    -------
    MeanCelltypeFit
    """
    train_rna = np.asarray(train_rna, dtype=float)
    train_celltype = np.asarray(train_celltype)
    train_protein = np.asarray(train_protein, dtype=float)
    if not (
        len(train_rna) == len(train_celltype) == len(train_protein)
    ):
        raise ValueError("train_rna / train_celltype / train_protein length mismatch")

    knn = KNeighborsClassifier(n_neighbors=min(k, len(train_celltype)), metric=metric)
    knn.fit(train_rna, train_celltype)

    celltype_means: dict[str, np.ndarray] = {}
    for ct in np.unique(train_celltype):
        mask = train_celltype == ct
        celltype_means[str(ct)] = train_protein[mask].mean(axis=0)
    fallback = train_protein.mean(axis=0)
    return MeanCelltypeFit(
        knn=knn, celltype_means=celltype_means, fallback_mean=fallback
    )


def mean_celltype_baseline(
    fit: MeanCelltypeFit,
    test_rna: np.ndarray,
) -> np.ndarray:
    """Apply Eq. 8 to a test-cell RNA matrix and return predicted protein matrix.

    For each test cell:

    1. ``c_hat = knn.predict(rna_i)`` — predicted cell type from training-set
       kNN classifier.
    2. ``y_hat^mc_pi(i) = celltype_means[c_hat]`` — predicted protein vector
       is the training-set mean for that cell type.

    Cells whose predicted cell type is absent from ``celltype_means``
    (only possible if the kNN somehow returned an unseen label, which it
    can't by construction — but kept for robustness) fall back to the
    global training-protein mean.

    Parameters
    ----------
    fit : MeanCelltypeFit
        Fitted baseline returned by :func:`fit_mean_celltype_baseline`.
    test_rna : np.ndarray
        Test-cell RNA embeddings, shape (n_test, n_features).

    Returns
    -------
    np.ndarray
        Predicted protein matrix, shape (n_test, n_proteins).
    """
    test_rna = np.asarray(test_rna, dtype=float)
    predicted_celltypes = fit.knn.predict(test_rna)
    n_proteins = len(fit.fallback_mean)
    out = np.empty((len(test_rna), n_proteins), dtype=float)
    for i, ct in enumerate(predicted_celltypes):
        out[i] = fit.celltype_means.get(str(ct), fit.fallback_mean)
    return out
