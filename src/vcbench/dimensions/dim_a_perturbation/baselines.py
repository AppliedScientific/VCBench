"""Dim A baselines.

* :func:`additive_baseline` — Ahlmann-Eltze additive (Eq. 2).
* :func:`mean_baseline`     — predict the training-perturbation mean response.
* :func:`no_change_baseline` — predict zero Δ-expression for every perturbation.

Each baseline returns a *predicted* per-gene expression vector for one
perturbation. To get a Dim A aggregate score, pair the baseline with
:func:`vcbench.dimensions.dim_a_perturbation.metrics.prr` (or ``des``)
applied to (predicted - control_mean) vs. (observed - control_mean).

Reference values (VCBench (2026), §I.4, Norman test set):

* Additive baseline, additive-evaluable subset (n=71): PRR = 0.890, DES = 0.999
* Mean-prediction baseline, full test (n=107):         PRR = 0.579
* No-change baseline, full test (n=107):               PRR = 0.000
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def additive_baseline(
    ctrl_mean: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
) -> np.ndarray:
    """Ahlmann-Eltze additive baseline for a combination perturbation A+B (Eq. 2).

    ::

        y_hat^{A+B}_g = xbar^ctrl_g + Δybar^A_g + Δybar^B_g

    Evaluable iff both A and B were observed as singletons in training; that
    constraint defines VCBench's 71-perturbation additive-evaluable subset of
    the 107-perturbation Norman test split.

    Parameters
    ----------
    ctrl_mean : np.ndarray
        Per-gene mean expression in control cells. Shape (G,).
    delta_a, delta_b : np.ndarray
        Per-gene observed Δ-expression vectors for the singleton perturbations
        A and B. Shape (G,) each.

    Returns
    -------
    np.ndarray
        Predicted per-gene expression vector for the combination A+B. Shape (G,).
    """
    ctrl_mean = np.asarray(ctrl_mean, dtype=float).ravel()
    delta_a = np.asarray(delta_a, dtype=float).ravel()
    delta_b = np.asarray(delta_b, dtype=float).ravel()
    if not (ctrl_mean.shape == delta_a.shape == delta_b.shape):
        raise ValueError(
            f"shape mismatch: ctrl_mean {ctrl_mean.shape}, "
            f"delta_a {delta_a.shape}, delta_b {delta_b.shape}"
        )
    return ctrl_mean + delta_a + delta_b


def mean_baseline(
    train_perturbation_means: Mapping[str, np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Mean-prediction baseline: average of all training perturbation profiles.

    Captures the **systematic-variation floor** of the held-out test set:
    everything a model could have learned from "perturbations on average look
    like X" without any perturbation-specific signal. The PRR of this baseline
    is the binding "trivial" baseline that VCBench Dim A uses to award Level 1
    (a model must beat 0.579 PRR on Norman to clear it).

    Parameters
    ----------
    train_perturbation_means : mapping or array
        Either a dict ``{perturbation_id: per_gene_mean_vector}`` or a 2-D array
        of shape (n_train_perts, n_genes). Both are reduced to a single per-gene
        mean.

    Returns
    -------
    np.ndarray
        Per-gene mean prediction shared across all test perturbations. Shape (G,).
    """
    if isinstance(train_perturbation_means, Mapping):
        if not train_perturbation_means:
            raise ValueError("train_perturbation_means is empty")
        stacked = np.vstack(
            [np.asarray(v, dtype=float).ravel() for v in train_perturbation_means.values()]
        )
    else:
        stacked = np.asarray(train_perturbation_means, dtype=float)
        if stacked.ndim != 2:
            raise ValueError(
                f"expected 2-D array of (n_perts, n_genes), got shape {stacked.shape}"
            )
    return stacked.mean(axis=0)


def no_change_baseline(ctrl_mean: np.ndarray) -> np.ndarray:
    """No-change baseline: predict the control mean for every perturbation.

    The most trivial possible predictor — Δy_hat is identically zero. PRR is
    undefined (zero variance) and reported as 0.0 by convention; DES is also 0.0
    because ``sign(0) == 0`` never matches the non-zero signs in real_delta.

    Parameters
    ----------
    ctrl_mean : np.ndarray
        Per-gene mean expression in control cells. Shape (G,).

    Returns
    -------
    np.ndarray
        Same as ``ctrl_mean``, returned as a fresh array.
    """
    return np.asarray(ctrl_mean, dtype=float).ravel().copy()
