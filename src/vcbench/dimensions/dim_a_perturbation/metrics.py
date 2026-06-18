"""Dim A metrics — PRR (Eq. 1) and DES (Eq. 3) from VCBench (2026).

Both metrics are computed on **pseudo-bulk** Δ-expression vectors:

    Δy^p_g     = mean_i x_{ig} | condition=p   - mean_i x_{ig} | condition=ctrl       (observed)
    Δy_hat^p_g = mean_i x_hat_{ig} | condition=p - mean_i x_{ig} | condition=ctrl     (predicted)

Both deltas use the *observed* control mean as anchor; the model is not
asked to predict control. This matches the canonical Cell-Eval Δ construction
but the summarisation differs (correlation vs ranking — see §I.3 PRR/PDS
distinction).

PRR is bit-identical to the ``mean_pearson_r_delta`` field that
``src/evaluation/metrics.py::_evaluate_perturbation_fallback`` writes into
``results/dim_a/<model>/cell_eval_results.json``.

DES is bit-identical to the ``mean_direction_score`` field there.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

DES_TOP_K_DEFAULT: int = 20


def prr(pred_delta: np.ndarray, real_delta: np.ndarray) -> float:
    """Per-perturbation Perturbation Response Recovery (Eq. 1).

    Pearson correlation between predicted and observed Δ-expression vectors.
    Bounded in [-1, 1].

    Parameters
    ----------
    pred_delta, real_delta : np.ndarray
        1-D vectors of length G (genes). Both must be the same length.

    Returns
    -------
    float
        Pearson correlation. Returns 0.0 if either vector has zero variance
        (the convention ``_evaluate_perturbation_fallback`` adopts to avoid
        divide-by-zero on perfectly-flat predictions).

    Raises
    ------
    ValueError
        If the two vectors have different lengths.
    """
    pred_delta = np.asarray(pred_delta, dtype=float).ravel()
    real_delta = np.asarray(real_delta, dtype=float).ravel()
    if pred_delta.shape != real_delta.shape:
        raise ValueError(
            f"pred_delta shape {pred_delta.shape} != real_delta shape {real_delta.shape}"
        )
    if np.std(pred_delta) <= 1e-10 or np.std(real_delta) <= 1e-10:
        return 0.0
    r, _ = pearsonr(pred_delta, real_delta)
    return float(r)


def des(
    pred_delta: np.ndarray,
    real_delta: np.ndarray,
    top_k: int = DES_TOP_K_DEFAULT,
) -> float:
    """Direction score on the top-K most-perturbed genes (Eq. 3).

    For the K genes with largest |real_delta|, the fraction whose predicted
    delta matches the observed delta in sign. Bounded in [0, 1].

    Parameters
    ----------
    pred_delta, real_delta : np.ndarray
        1-D vectors of length G. Both must be the same length.
    top_k : int, default ``DES_TOP_K_DEFAULT`` (20)
        Number of top-DEG genes to score on. Capped at G if G < top_k.
        K=20 is the manuscript value; the parameter is exposed only so
        tests / future revisions can override.

    Returns
    -------
    float
        Sign-agreement fraction.

    Raises
    ------
    ValueError
        If shapes mismatch or top_k <= 0.
    """
    pred_delta = np.asarray(pred_delta, dtype=float).ravel()
    real_delta = np.asarray(real_delta, dtype=float).ravel()
    if pred_delta.shape != real_delta.shape:
        raise ValueError(
            f"pred_delta shape {pred_delta.shape} != real_delta shape {real_delta.shape}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    k = min(top_k, len(real_delta))
    if k == 0:
        return 0.0
    top_idx = np.argsort(-np.abs(real_delta))[:k]
    correct = np.sign(pred_delta[top_idx]) == np.sign(real_delta[top_idx])
    return float(correct.mean())
