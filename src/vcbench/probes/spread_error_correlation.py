"""Spread-error correlation probe — Eq. 10 from VCBench (2026).

For each test perturbation:

* spread  s_p = Var_g( predicted Δ-expression across genes )
* error   e_p = mean_g | predicted - observed |  (per-perturbation MAE)

Reported as Spearman rank correlation across the perturbation set, with
analytical p-value. ρ > 0 indicates the model assigns higher spread to harder
predictions; ρ ≈ 0 indicates spread is uninformative; ρ < 0 indicates inverse
behaviour.

The probe is presented as **necessary but not sufficient** for calibration:
passing it requires distributional output / ensembles / MC dropout / conformal
wrappers, none of which the FMs evaluated in VCBench v1 provide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class SpreadErrorResult:
    """Result of the spread-error correlation probe.

    Attributes
    ----------
    rho : float
        Spearman rank correlation between per-perturbation spread and error.
    pvalue : float
        Two-sided analytical p-value from ``scipy.stats.spearmanr``.
    n_perturbations : int
        Number of perturbations entering the correlation (after dropping
        any rows with NaN spread or error).
    """

    rho: float
    pvalue: float
    n_perturbations: int


def spread_error_correlation(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    perturbation_col: str = "perturbation",
    expression_cols: list[str] | None = None,
    control_label: str = "ctrl",
) -> SpreadErrorResult:
    """Spread-error correlation probe (Eq. 10 from VCBench (2026)).

    Parameters
    ----------
    predictions : pd.DataFrame
        One row per (perturbation × cell or pseudo-cell). Must contain the
        ``perturbation_col`` column plus per-gene expression columns.
    ground_truth : pd.DataFrame
        Same schema as ``predictions``. Must include rows tagged with
        ``control_label`` so the per-gene control mean can be computed.
    perturbation_col : str, default ``"perturbation"``
        Column name carrying the perturbation identifier.
    expression_cols : list[str] | None
        Gene columns to use. If ``None``, every column other than
        ``perturbation_col`` is treated as an expression column.
    control_label : str, default ``"ctrl"``
        Value of ``perturbation_col`` that identifies control cells in
        ``ground_truth`` (used to compute the Δ-expression anchor).

    Returns
    -------
    SpreadErrorResult
        See :class:`SpreadErrorResult`.

    Raises
    ------
    ValueError
        If the control label is absent from ``ground_truth``, or if fewer
        than 3 perturbations remain after computing spread/error (Spearman
        is undefined below 3 points).

    Notes
    -----
    Reference values from VCBench (2026), Norman fine-tuning runs:

    * Geneformer (FT on Norman): ρ = -0.119, p = 0.225 (n = 106)
    * scGPT (FT on Norman):      ρ = +0.131, p = 0.177 (n = 107)

    These match the manuscript's Supplementary Table 8.
    """
    if perturbation_col not in predictions.columns:
        raise ValueError(
            f"predictions missing required column {perturbation_col!r}"
        )
    if perturbation_col not in ground_truth.columns:
        raise ValueError(
            f"ground_truth missing required column {perturbation_col!r}"
        )

    if expression_cols is None:
        expression_cols = [c for c in predictions.columns if c != perturbation_col]
    if not expression_cols:
        raise ValueError("no expression columns available for spread/error computation")

    ctrl_mask = ground_truth[perturbation_col] == control_label
    if not ctrl_mask.any():
        raise ValueError(
            f"ground_truth has no rows with {perturbation_col}={control_label!r}; "
            "control mean cannot be computed"
        )
    ctrl_mean = ground_truth.loc[ctrl_mask, expression_cols].to_numpy().mean(axis=0)

    perts = sorted(set(predictions[perturbation_col]) - {control_label})
    spreads = []
    errors = []
    for p in perts:
        pred_p = predictions.loc[
            predictions[perturbation_col] == p, expression_cols
        ].to_numpy()
        gt_p = ground_truth.loc[
            ground_truth[perturbation_col] == p, expression_cols
        ].to_numpy()
        if len(pred_p) == 0 or len(gt_p) == 0:
            continue
        pred_mean = pred_p.mean(axis=0)
        gt_mean = gt_p.mean(axis=0)
        pred_delta = pred_mean - ctrl_mean
        spreads.append(float(np.var(pred_delta)))
        errors.append(float(np.mean(np.abs(pred_mean - gt_mean))))

    s = np.array(spreads)
    e = np.array(errors)
    valid = ~(np.isnan(s) | np.isnan(e))
    s, e = s[valid], e[valid]
    n = len(s)
    if n < 3:
        raise ValueError(
            f"need at least 3 perturbations for Spearman correlation; got {n}"
        )
    rho, pval = spearmanr(s, e)
    return SpreadErrorResult(rho=float(rho), pvalue=float(pval), n_perturbations=n)
