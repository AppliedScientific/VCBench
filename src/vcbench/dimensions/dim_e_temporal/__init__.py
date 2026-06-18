"""Dim E — temporal ordering metrics, baseline, and aggregator.

Implements:

* Eq. 9 — Kendall τ-b on (predicted pseudotime, observed timepoint) pairs
* Per-dataset balanced kNN accuracy on timepoint labels
* Unweighted across-dataset aggregation (preserves the scGPT temporal-inversion
  finding that cell-count-weighted averaging would dilute)
* Bootstrap aggregation for the TF Weinreb special case (ARPACK
  non-convergence on full 49K-cell graph → 10×5K subsamples)

Reference values (VCBench (2026), §I.4 Table 2 + §I.3 Eq. 9):

================== ===================================
Method             Kendall τ-b (sci-fate + Weinreb avg)
================== ===================================
PCA + DPT          0.190  (binding baseline)
UCE                0.136  (approached but did not reach)
TranscriptFormer   0.041 ± 0.078 (Weinreb bootstrap on 10×5K subsamples)
Geneformer        -0.017
scGPT             -0.057  (Weinreb specifically: -0.103, mild inversion)
================== ===================================

Why Kendall τ-b (not τ-a): time-course timepoints are discrete and produce
many ties; τ-b handles ties, τ-a doesn't.

Why unweighted mean across datasets: a cell-count-weighted average would let
the 49K-cell Weinreb dataset dominate the 6.5K-cell sci-fate dataset and
hide the dataset-level signal — in particular, scGPT's clean temporal
inversion on Weinreb (-0.103) would average to noise.
"""

from vcbench.dimensions.dim_e_temporal.aggregation import (
    aggregate_across_datasets,
    bootstrap_subsample,
)
from vcbench.dimensions.dim_e_temporal.evaluate import DimEResult, evaluate_dim_e
from vcbench.dimensions.dim_e_temporal.metrics import knn_balanced_accuracy, kendall_tau_b

__all__ = [
    "kendall_tau_b",
    "knn_balanced_accuracy",
    "aggregate_across_datasets",
    "bootstrap_subsample",
    "evaluate_dim_e",
    "DimEResult",
]
