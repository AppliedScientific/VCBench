"""Dim D — cross-modal RNA→protein prediction metrics, baseline, and evaluator.

Implements:

* Per-protein Pearson R + mean-across-proteins aggregate
* Per-protein RMSE
* Mean-celltype baseline (Eq. 8) — predict the per-cell-type mean protein
  abundance from a kNN cell-type prediction on RNA embeddings

Reference values (VCBench (2026), §I.4 Table 2):

================== ====================
Method             Mean Pearson R
================== ====================
TranscriptFormer   0.232  (binding cell)
UCE                0.132
scGPT              0.064
Geneformer         0.001
Arc State          DNR
Mean-celltype      0.152  (binding baseline)
scLinear           0.129
================== ====================

TranscriptFormer is the only Level-2 cell in the §I.4 capability matrix
(beats Mean-celltype 0.152 → exceeds the strongest non-FM baseline on
≥ 1 dimension).
"""

from vcbench.dimensions.dim_d_cross_modal.baselines import (
    MeanCelltypeFit,
    fit_mean_celltype_baseline,
    mean_celltype_baseline,
)
from vcbench.dimensions.dim_d_cross_modal.evaluate import DimDResult, evaluate_dim_d
from vcbench.dimensions.dim_d_cross_modal.metrics import (
    mean_pearson_per_protein,
    median_pearson_per_protein,
    pearson_per_protein,
    rmse,
)

__all__ = [
    "pearson_per_protein",
    "mean_pearson_per_protein",
    "median_pearson_per_protein",
    "rmse",
    "MeanCelltypeFit",
    "fit_mean_celltype_baseline",
    "mean_celltype_baseline",
    "evaluate_dim_d",
    "DimDResult",
]
