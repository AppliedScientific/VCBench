"""Dim D metrics — per-protein Pearson R and RMSE.

The Dim D primary metric is the **mean across proteins** of per-protein
Pearson correlation between predicted and observed surface-protein abundance
across cells in the test set::

    For each protein π in {1, ..., N_pi}:
        rho_pi = Pearson_i(y_hat_{i,pi}, y_{i,pi})

    PrimaryMetric = (1 / N_pi) * sum_pi rho_pi

This summarises across the protein axis (not the cell axis), so it captures
"do per-cell predictions for protein π track per-cell observations" rather
than "are the predicted profiles for each cell close to truth".

The schema matches the on-disk ``results/dim_d/<model>/crossmodal_results.json``
produced by ``src/models/run_crossmodal_probes.py``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr


def pearson_per_protein(predictions: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Per-protein Pearson R across cells.

    Parameters
    ----------
    predictions : np.ndarray
        Shape (n_cells, n_proteins).
    observed : np.ndarray
        Shape (n_cells, n_proteins).

    Returns
    -------
    np.ndarray
        Shape (n_proteins,). NaN for proteins with zero variance in either
        predictions or observations (no defined Pearson).
    """
    predictions = np.asarray(predictions, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if predictions.shape != observed.shape:
        raise ValueError(
            f"shape mismatch: predictions {predictions.shape} vs observed {observed.shape}"
        )
    if predictions.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got shape {predictions.shape}")

    n_proteins = predictions.shape[1]
    out = np.empty(n_proteins, dtype=float)
    for j in range(n_proteins):
        p = predictions[:, j]
        o = observed[:, j]
        if np.std(p) <= 1e-12 or np.std(o) <= 1e-12:
            out[j] = float("nan")
            continue
        r, _ = pearsonr(p, o)
        out[j] = r
    return out


def mean_pearson_per_protein(predictions: np.ndarray, observed: np.ndarray) -> float:
    """Mean per-protein Pearson R, ignoring NaN proteins.

    This is the Dim D primary metric (binding cell in §I.4 Table 2).
    """
    rs = pearson_per_protein(predictions, observed)
    valid = ~np.isnan(rs)
    if not valid.any():
        return float("nan")
    return float(np.nanmean(rs[valid]))


def median_pearson_per_protein(predictions: np.ndarray, observed: np.ndarray) -> float:
    """Median per-protein Pearson R, ignoring NaN proteins."""
    rs = pearson_per_protein(predictions, observed)
    valid = ~np.isnan(rs)
    if not valid.any():
        return float("nan")
    return float(np.nanmedian(rs[valid]))


def rmse(predictions: np.ndarray, observed: np.ndarray) -> float:
    """Root-mean-squared error across all (cell, protein) entries.

    Computes a single scalar over the full matrix (matches the on-disk
    ``rmse`` field schema). Per-protein RMSE is available by slicing
    columns first if needed.
    """
    predictions = np.asarray(predictions, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if predictions.shape != observed.shape:
        raise ValueError(
            f"shape mismatch: predictions {predictions.shape} vs observed {observed.shape}"
        )
    if predictions.size == 0:
        raise ValueError("predictions is empty; RMSE is undefined")
    return math.sqrt(float(np.mean((predictions - observed) ** 2)))
