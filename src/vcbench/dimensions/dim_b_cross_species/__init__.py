"""Dim B — cross-species transfer metrics, baselines, and dual-protocol evaluator.

Implements:

* Eq. 4 — macro F1 and weighted F1
* Eq. 5 — common-label-set protocol (re-exported from ``vcbench.protocols``)
* PCA + kNN baseline (k=5, cosine), the binding non-FM baseline

Reference values (VCBench (2026), §I.4):

Aggregate macro F1 (common-label-set, all five tissues):
    PCA+kNN          0.497  (binding baseline)
    Geneformer       0.171
    scGPT            0.123
    UCE              0.379  (heart + brain only)
    TranscriptFormer 0.351  (lung + liver only)

Per-tissue head-to-head (common-set): PCA+kNN beats every FM on lung, heart,
kidney, brain; TranscriptFormer (0.495) marginally beats PCA+kNN (0.446) on
liver — the only foundation-model-beats-baseline cell in the per-tissue matrix.
"""

from vcbench.dimensions.dim_b_cross_species.baselines import pca_knn_classifier
from vcbench.dimensions.dim_b_cross_species.evaluate import (
    DimBPerTissueResult,
    DimBResult,
    evaluate_dim_b,
)
from vcbench.dimensions.dim_b_cross_species.metrics import macro_f1, weighted_f1
from vcbench.dimensions.dim_b_cross_species.protocols import (
    score_native,
    score_under_common_set,
)
from vcbench.protocols import common_label_set

__all__ = [
    "macro_f1",
    "weighted_f1",
    "pca_knn_classifier",
    "common_label_set",
    "score_native",
    "score_under_common_set",
    "evaluate_dim_b",
    "DimBPerTissueResult",
    "DimBResult",
]
