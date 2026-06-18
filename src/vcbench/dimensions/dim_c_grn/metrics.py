"""Dim C metrics — AUROC, AUPRC, and EPR (Eq. 6 from VCBench (2026)).

All three are computed against a binary ground truth ``y_true`` (1 = true
edge in BEELINE / TRRUST, 0 = non-edge) and a continuous predicted ranking
``y_score`` (higher = more likely edge).

Eq. 6 (EPR)::

    EPR = Precision@K / (K / N)

    K = number of true positive edges in ground truth
    N = total number of candidate (TF, target) pairs evaluated
    Precision@K = fraction of true edges among the top-K predicted edges

EPR=1 corresponds to a random ranker; EPR>1 means true edges are
concentrated above chance among the top-K predictions. ``K`` matches the
TP count in the ground truth, so EPR is precision-at-recall-of-1
normalised by the base rate.

The TRRUST evaluation has ~2.7e-4 edge density, so AUPRC values are
naturally tiny (order 10⁻³) and easy to misread. The conjunctive
passing rule (AUPRC AND EPR exceed all three baselines) was adopted
to prevent ranking signal from being dismissed because AUPRC alone
sits near the noise floor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _validate(y_true: Sequence, y_score: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true shape {y_true.shape} != y_score shape {y_score.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true.shape}")
    return y_true, y_score


def auroc(y_true: Sequence, y_score: Sequence) -> float:
    """Area under the ROC curve. Returns NaN on single-class input."""
    y_true, y_score = _validate(y_true, y_score)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: Sequence, y_score: Sequence) -> float:
    """Area under the precision-recall curve (sklearn ``average_precision_score``).

    Returns NaN on single-class input. For TRRUST-style highly imbalanced
    labels (edge density ~2.7e-4), expect values in the 10⁻³ range — see
    the ``EPR`` companion metric for a baseline-normalised view.
    """
    y_true, y_score = _validate(y_true, y_score)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def epr(y_true: Sequence, y_score: Sequence) -> float:
    """Early Precision Ratio (Eq. 6).

    K is set to the number of true edges in ``y_true``. Precision is computed
    among the top-K predictions ranked by ``y_score`` (descending), then
    divided by the base rate K/N. Returns NaN on single-class input.

    Tied scores: ``np.argsort`` provides a stable but arbitrary tie-break;
    for highly tied predictors (e.g. degree-null) the resulting EPR depends
    on tie-break order. Note in §I.3 — this matches the existing pipeline
    behaviour and the on-disk values in ``results/dim_c/bootstrap_cis.csv``.
    """
    y_true, y_score = _validate(y_true, y_score)
    n = len(y_true)
    k = int(y_true.sum())
    if k == 0 or k == n:
        return float("nan")
    base_rate = k / n
    sorted_idx = np.argsort(-y_score)
    top_k_precision = y_true[sorted_idx[:k]].sum() / k
    return float(top_k_precision / base_rate)
