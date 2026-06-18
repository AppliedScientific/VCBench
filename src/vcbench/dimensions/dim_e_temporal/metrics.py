"""Dim E metrics — Kendall τ-b (Eq. 9) and balanced kNN accuracy.

Eq. 9::

    tau_b = (n_c - n_d) / sqrt((n_0 - n_1) * (n_0 - n_2))

where n_c / n_d are concordant / discordant pairs, n_0 = C(n, 2), and
n_1 / n_2 are tie corrections on predicted / ground-truth orderings.

For VCBench Dim E, the predicted ordering is a pseudotime estimate and
the ground-truth ordering is a discrete timepoint label (Weinreb LARRY:
3 timepoints; sci-fate: 6 timepoints). The discrete ground truth produces
many ties — τ-b handles ties; τ-a doesn't.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier


def kendall_tau_b(predicted_order: Sequence, ground_truth_order: Sequence) -> float:
    """Kendall τ-b correlation between predicted and ground-truth orderings.

    Tie-aware (τ-b, not τ-a) so that discrete ground-truth timepoints
    don't degrade the metric. Returns NaN if either sequence has zero
    rank variance (e.g. all elements equal).

    Parameters
    ----------
    predicted_order : sequence
        Continuous pseudotime estimate, one per cell.
    ground_truth_order : sequence
        Discrete timepoint label (or any orderable sequence), one per cell.

    Returns
    -------
    float
        Kendall τ-b in [-1, 1], or NaN.
    """
    p = np.asarray(predicted_order, dtype=float)
    g = np.asarray(ground_truth_order, dtype=float)
    if p.shape != g.shape:
        raise ValueError(
            f"predicted_order shape {p.shape} != ground_truth_order shape {g.shape}"
        )
    if p.size < 2:
        raise ValueError("need at least 2 points for Kendall correlation")
    tau, _ = kendalltau(p, g, variant="b")
    if tau is None or np.isnan(tau):
        return float("nan")
    return float(tau)


def knn_balanced_accuracy(
    embeddings: np.ndarray,
    timepoint_labels: Sequence,
    *,
    k: int = 5,
    metric: str = "cosine",
    cv: int = 5,
) -> float:
    """Cross-validated balanced accuracy of a kNN timepoint classifier on cell embeddings.

    Companion metric to Kendall τ-b: τ-b summarises ranking quality, balanced
    accuracy summarises whether discrete timepoints are linearly separable in
    embedding space.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings or PCA scores, shape (n_cells, n_dims).
    timepoint_labels : sequence
        Discrete timepoint label per cell.
    k : int, default 5
    metric : str, default ``"cosine"``
    cv : int, default 5
        Cross-validation fold count.

    Returns
    -------
    float
        Balanced accuracy in [0, 1].
    """
    embeddings = np.asarray(embeddings, dtype=float)
    timepoint_labels = np.asarray(timepoint_labels)
    if len(embeddings) != len(timepoint_labels):
        raise ValueError(
            f"embeddings ({len(embeddings)}) and labels ({len(timepoint_labels)}) length mismatch"
        )
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    pred = cross_val_predict(knn, embeddings, timepoint_labels, cv=cv)
    return float(balanced_accuracy_score(timepoint_labels, pred))
