"""Dim C end-to-end evaluator: AUROC + AUPRC + EPR for one (model, GRN) pair."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vcbench.dimensions.dim_c_grn.metrics import auprc, auroc, epr


@dataclass(frozen=True)
class DimCResult:
    """Aggregate Dim C scores for one (model, evaluation-edge-set) pair.

    Attributes
    ----------
    AUROC : float
    AUPRC : float
    EPR : float
    n_true_edges : int
    n_total_pairs : int
    edge_density : float
    """

    AUROC: float
    AUPRC: float
    EPR: float
    n_true_edges: int
    n_total_pairs: int
    edge_density: float


def evaluate_dim_c(y_true: Sequence, y_score: Sequence) -> DimCResult:
    """Score one (model, evaluation-edge-set) pair on AUROC / AUPRC / EPR.

    Parameters
    ----------
    y_true : sequence of {0, 1}
        Binary ground truth (1 = true edge in BEELINE / TRRUST, 0 = non-edge).
    y_score : sequence of float
        Predicted ranking score (higher = more likely edge). Must be aligned
        to ``y_true`` by edge index.

    Returns
    -------
    DimCResult
    """
    import numpy as np

    yt = np.asarray(y_true)
    n = len(yt)
    n_true = int(yt.sum())
    return DimCResult(
        AUROC=auroc(yt, y_score),
        AUPRC=auprc(yt, y_score),
        EPR=epr(yt, y_score),
        n_true_edges=n_true,
        n_total_pairs=n,
        edge_density=(n_true / n) if n else 0.0,
    )
