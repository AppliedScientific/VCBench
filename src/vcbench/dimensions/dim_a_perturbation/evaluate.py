"""Dim A end-to-end evaluator.

Takes (predicted, observed) AnnData and produces per-perturbation PRR / DES
plus aggregate values. Output schema is bit-identical to
``src/evaluation/metrics.py::_evaluate_perturbation_fallback`` so the existing
``results/dim_a/<model>/cell_eval_results.{csv,json}`` files reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from vcbench.dimensions.dim_a_perturbation.metrics import DES_TOP_K_DEFAULT, des, prr

ControlAnchor = Literal["real", "pred"]
"""Which control mean to subtract when forming Δ-expression vectors.

- ``"real"`` (vcbench default, principled): both ``pred_delta`` and ``real_delta``
  are anchored on the **observed** control mean. The control is a property of
  the data, not the model — see §I.3 of VCBench (2026). This prevents a
  systematically-biased predicted control from inflating per-perturbation
  Pearson R, which is the right behaviour for cross-model benchmarking.

- ``"pred"`` (cell-eval / Arc Institute upstream convention): ``pred_delta``
  is anchored on the model's *predicted* control mean (``pert_pred - ctrl_pred``)
  while ``real_delta`` stays anchored on the real control. This lets the
  predicted control absorb systematic baseline offsets, which generally inflates
  R when the model has a non-trivial baseline shift. Provided so VCBench can
  cross-validate against numbers reported by cell-eval to numerical precision.
  See ``tests/unit/test_dim_a_evaluate.py::test_control_anchor_pred_reproduces_cell_eval``.
"""


@dataclass(frozen=True)
class DimAResult:
    """Aggregate Dim A scores across all evaluated perturbations.

    Attributes
    ----------
    mean_pearson_r_delta : float
        PRR aggregate (mean across perturbations).
    median_pearson_r_delta : float
        Per-perturbation PRR median (less sensitive to outliers).
    mean_mse_delta : float
        Mean squared error on Δ-expression, averaged across perturbations.
    mean_direction_score : float
        DES aggregate (mean across perturbations).
    n_perturbations : int
        Number of perturbations entering the aggregate.
    per_perturbation : pd.DataFrame
        One row per perturbation with columns
        ``condition / pearson_r_delta / mse_delta / direction_score_top20``.
    """

    mean_pearson_r_delta: float
    median_pearson_r_delta: float
    mean_mse_delta: float
    mean_direction_score: float
    n_perturbations: int
    per_perturbation: pd.DataFrame

    def to_aggregate_dict(self) -> dict[str, float | int]:
        """Return the aggregate-only dict (matches ``cell_eval_results.json`` schema)."""
        return {
            "mean_pearson_r_delta": self.mean_pearson_r_delta,
            "median_pearson_r_delta": self.median_pearson_r_delta,
            "mean_mse_delta": self.mean_mse_delta,
            "mean_direction_score": self.mean_direction_score,
            "n_perturbations": self.n_perturbations,
        }


def _to_dense(x):
    """Coerce ``x`` to a dense numpy array; handles scipy.sparse if installed."""
    try:
        from scipy.sparse import issparse

        if issparse(x):
            return np.asarray(x.toarray())
    except ImportError:
        pass
    return np.asarray(x)


def evaluate_dim_a(
    adata_pred,
    adata_real,
    perturbation_col: str = "condition",
    control_label: str = "ctrl",
    top_k: int = DES_TOP_K_DEFAULT,
    control_anchor: ControlAnchor = "real",
) -> DimAResult:
    """Evaluate Dim A on (predicted, observed) AnnData pairs.

    For each perturbation present in both ``adata_pred`` and ``adata_real``
    (excluding the control label):

    1. Compute the predicted and observed per-gene mean profiles.
    2. Form Δ vs. the control mean. The anchor convention is selected by
       ``control_anchor``:

       - ``"real"`` (default, vcbench / §I.3 principled convention): both
         ``pred_delta`` and ``real_delta`` use the **observed** real control
         mean. Right for cross-model benchmarking — the control is a property
         of the data, not of the model.
       - ``"pred"`` (cell-eval / Arc Institute upstream convention):
         ``pred_delta = pert_pred - ctrl_pred`` (model's own predicted control)
         while ``real_delta = pert_real - ctrl_real``. Reproduces upstream
         cell-eval ``pearson_delta`` numbers to numerical precision.

    3. Compute :func:`vcbench.dimensions.dim_a_perturbation.metrics.prr`,
       :func:`vcbench.dimensions.dim_a_perturbation.metrics.des`, and a
       per-perturbation MSE on the Δ vectors.

    Parameters
    ----------
    adata_pred, adata_real : anndata.AnnData
        Predicted and observed expression. ``adata_real`` must contain at least
        one row tagged with ``control_label`` so the control mean can be
        anchored. Both AnnData must use the same gene order in ``var``.
    perturbation_col : str, default ``"condition"``
        Column in ``.obs`` carrying the perturbation identifier.
    control_label : str, default ``"ctrl"``
        Value of ``perturbation_col`` identifying control cells in ``adata_real``.
        Under ``control_anchor="pred"`` the same label is used to find control
        cells in ``adata_pred``; if no such cells exist, falls back to the
        ``"real"`` anchor for that delta with a warning.
    top_k : int, default 20
        DES top-K cap.
    control_anchor : {"real", "pred"}, default ``"real"``
        Anchor convention for forming ``pred_delta``. See module-level
        :data:`ControlAnchor` docstring.

    Returns
    -------
    DimAResult

    Raises
    ------
    ValueError
        If ``adata_real`` has no control cells, or if no perturbations are
        common to both AnnData, or if ``control_anchor`` is not one of
        ``"real"`` / ``"pred"``.
    """
    if control_anchor not in ("real", "pred"):
        raise ValueError(
            f"control_anchor must be 'real' or 'pred', got {control_anchor!r}"
        )
    if perturbation_col not in adata_pred.obs.columns:
        raise ValueError(
            f"adata_pred.obs missing required column {perturbation_col!r}"
        )
    if perturbation_col not in adata_real.obs.columns:
        raise ValueError(
            f"adata_real.obs missing required column {perturbation_col!r}"
        )

    # GUARD: gene-vocabulary alignment.
    # Per-gene Δ-expression aggregation assumes adata_pred.var.index[i] is the
    # SAME GENE as adata_real.var.index[i]. If the two AnnData objects use
    # different gene namings (e.g. predictor uses integer-positional indices
    # while real uses Ensembl IDs), the per-gene correlation collapses to
    # noise floor (~0.1 PRR, ~0.5 DES); the guard below catches this.
    #
    # Required: var.index sets must be equal AND in the same order.
    # Callers with mismatched orders should reindex first; mismatched sets
    # are an upstream pipeline bug.
    if adata_pred.shape[1] != adata_real.shape[1]:
        raise ValueError(
            f"gene count mismatch: adata_pred has {adata_pred.shape[1]} genes, "
            f"adata_real has {adata_real.shape[1]}. Reindex one to match the "
            f"other on the union or intersection of their gene vocabularies "
            f"before calling evaluate_dim_a."
        )
    if not adata_pred.var.index.equals(adata_real.var.index):
        # Same length but different IDs / order. Surface a precise diagnostic.
        pred_set = set(adata_pred.var.index.astype(str))
        real_set = set(adata_real.var.index.astype(str))
        intersect = pred_set & real_set
        if not intersect:
            raise ValueError(
                f"adata_pred.var.index and adata_real.var.index have ZERO "
                f"overlap (pred[:3]={list(adata_pred.var.index[:3])}, "
                f"real[:3]={list(adata_real.var.index[:3])}). The two h5ads "
                f"use different gene-naming schemes (e.g. integer-positional "
                f"vs Ensembl IDs). Reindex one to match the other before "
                f"calling evaluate_dim_a — direct positional comparison would "
                f"produce noise-floor PRR / DES."
            )
        n_aligned = (adata_pred.var.index == adata_real.var.index).sum()
        raise ValueError(
            f"adata_pred.var.index and adata_real.var.index share "
            f"{len(intersect)} of {adata_pred.shape[1]} gene names but only "
            f"{n_aligned} are at the same positional index. Reindex one to "
            f"match the other's order (e.g. "
            f"`adata_pred = adata_pred[:, adata_real.var.index]`) before "
            f"calling evaluate_dim_a."
        )

    real_conds = np.asarray(adata_real.obs[perturbation_col])
    pred_conds = np.asarray(adata_pred.obs[perturbation_col])

    ctrl_mask = real_conds == control_label
    if not ctrl_mask.any():
        raise ValueError(
            f"adata_real has no rows with {perturbation_col}={control_label!r}; "
            "control mean cannot be computed"
        )
    ctrl_mean_real = _to_dense(adata_real[ctrl_mask].X).mean(axis=0).ravel()

    # Under the "pred" anchor we also need the predicted control mean. If the
    # caller gave us a pred AnnData without control cells, fall back to the
    # real-anchor convention rather than crash; most predict pipelines (incl.
    # Arc State's `state tx predict`) emit predicted control cells alongside
    # perturbed predictions, but a few reduced-output formats don't.
    if control_anchor == "pred":
        pred_ctrl_mask = pred_conds == control_label
        if pred_ctrl_mask.any():
            ctrl_mean_pred = _to_dense(
                adata_pred[pred_ctrl_mask].X
            ).mean(axis=0).ravel()
        else:
            import warnings
            warnings.warn(
                f"control_anchor='pred' requested but adata_pred has no "
                f"{perturbation_col}={control_label!r} cells; falling back to "
                f"the real-control anchor for pred_delta. To suppress this, "
                f"either pass control_anchor='real' or include control cells "
                f"in adata_pred.",
                stacklevel=2,
            )
            ctrl_mean_pred = ctrl_mean_real
    else:
        ctrl_mean_pred = ctrl_mean_real  # unused under "real" anchor

    candidate_conds = sorted(set(pred_conds) - {control_label})
    rows: list[dict] = []
    for cond in candidate_conds:
        real_rows = real_conds == cond
        if not real_rows.any():
            continue
        pred_rows = pred_conds == cond
        if not pred_rows.any():
            continue
        pred_mean = _to_dense(adata_pred[pred_rows].X).mean(axis=0).ravel()
        real_mean = _to_dense(adata_real[real_rows].X).mean(axis=0).ravel()
        # pred_delta uses ctrl_mean_pred under "pred" anchor (which equals
        # ctrl_mean_real under "real" anchor by construction above).
        pred_delta = pred_mean - ctrl_mean_pred
        real_delta = real_mean - ctrl_mean_real

        rows.append(
            {
                "condition": cond,
                "pearson_r_delta": prr(pred_delta, real_delta),
                "mse_delta": float(np.mean((pred_delta - real_delta) ** 2)),
                "direction_score_top20": des(pred_delta, real_delta, top_k=top_k),
            }
        )

    if not rows:
        raise ValueError(
            "no perturbations common to adata_pred and adata_real (after dropping ctrl)"
        )

    df = pd.DataFrame(rows)
    return DimAResult(
        mean_pearson_r_delta=float(df["pearson_r_delta"].mean()),
        median_pearson_r_delta=float(df["pearson_r_delta"].median()),
        mean_mse_delta=float(df["mse_delta"].mean()),
        mean_direction_score=float(df["direction_score_top20"].mean()),
        n_perturbations=len(df),
        per_perturbation=df,
    )
