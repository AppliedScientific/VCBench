"""Unit tests for vcbench.dimensions.dim_d_cross_modal.baselines (Eq. 8)."""

from __future__ import annotations

import numpy as np
import pytest

from vcbench.dimensions.dim_d_cross_modal.baselines import (
    MeanCelltypeFit,
    fit_mean_celltype_baseline,
    mean_celltype_baseline,
)
from vcbench.dimensions.dim_d_cross_modal.metrics import mean_pearson_per_protein


def _build_synthetic_cite_seq(
    n_per_class: int, n_features: int, n_proteins: int, *, seed: int,
    centers_rna: np.ndarray | None = None,
    centers_prot: np.ndarray | None = None,
):
    """Synthetic CITE-seq-like data: 3 cell types with distinct RNA + protein profiles.

    Pass shared ``centers_rna`` / ``centers_prot`` to make train and test draws
    share the same cluster geometry (the analogue of 'same cell types in the
    test set as in the training set' — required for the baseline to actually
    transfer cleanly).
    """
    rng = np.random.default_rng(seed)
    if centers_rna is None:
        centers_rna = rng.normal(0, 5, size=(3, n_features))
    if centers_prot is None:
        centers_prot = rng.normal(0, 3, size=(3, n_proteins))
    rna, ct, prot = [], [], []
    for c in range(3):
        rna.append(centers_rna[c] + rng.normal(0, 0.3, size=(n_per_class, n_features)))
        prot.append(centers_prot[c] + rng.normal(0, 0.3, size=(n_per_class, n_proteins)))
        ct.extend([f"ct_{c}"] * n_per_class)
    return np.vstack(rna), np.array(ct), np.vstack(prot)


def test_mean_celltype_returns_correct_shape():
    centers_rna = np.random.default_rng(100).normal(0, 5, size=(3, 30))
    centers_prot = np.random.default_rng(101).normal(0, 3, size=(3, 10))
    rna, ct, prot = _build_synthetic_cite_seq(
        50, 30, 10, seed=0, centers_rna=centers_rna, centers_prot=centers_prot
    )
    fit = fit_mean_celltype_baseline(rna, ct, prot)
    test_rna, _, _ = _build_synthetic_cite_seq(
        20, 30, 10, seed=1, centers_rna=centers_rna, centers_prot=centers_prot
    )
    pred = mean_celltype_baseline(fit, test_rna)
    assert pred.shape == (60, 10)


def test_mean_celltype_recovers_per_celltype_means():
    """When test RNA matches train RNA, predicted protein should equal training
    per-cell-type mean for the matching cell type."""
    rna, ct, prot = _build_synthetic_cite_seq(50, 30, 10, seed=0)
    fit = fit_mean_celltype_baseline(rna, ct, prot)
    pred = mean_celltype_baseline(fit, rna)
    # Each row of pred should be one of the three celltype means
    for ct_label, mean_vec in fit.celltype_means.items():
        # Find rows where the prediction matches this celltype's mean
        matches = np.all(pred == mean_vec, axis=1)
        # Most cells of this celltype should match (kNN may misclassify a few)
        true_count = (ct == ct_label).sum()
        match_count = matches.sum()
        # At least 90% of cells in each celltype should be correctly classified
        assert match_count >= 0.9 * true_count, (
            f"{ct_label}: matched {match_count}/{true_count}"
        )


def test_mean_celltype_beats_random_protein_prediction_in_pearson():
    """Mean-celltype baseline should clearly beat a random predictor on
    synthetic data where train and test share the same cluster geometry."""
    rng = np.random.default_rng(42)
    centers_rna = np.random.default_rng(110).normal(0, 5, size=(3, 30))
    centers_prot = np.random.default_rng(111).normal(0, 3, size=(3, 15))
    rna, ct, prot = _build_synthetic_cite_seq(
        80, 30, 15, seed=10, centers_rna=centers_rna, centers_prot=centers_prot
    )
    test_rna, _, test_prot = _build_synthetic_cite_seq(
        40, 30, 15, seed=11, centers_rna=centers_rna, centers_prot=centers_prot
    )
    fit = fit_mean_celltype_baseline(rna, ct, prot)
    mc_pred = mean_celltype_baseline(fit, test_rna)
    random_pred = rng.normal(0, 3, size=test_prot.shape)
    mc_score = mean_pearson_per_protein(mc_pred, test_prot)
    random_score = mean_pearson_per_protein(random_pred, test_prot)
    assert mc_score > random_score + 0.5


def test_fit_validates_shapes():
    with pytest.raises(ValueError, match="length mismatch"):
        fit_mean_celltype_baseline(
            np.zeros((10, 5)), np.array(["a"] * 9), np.zeros((10, 3))
        )


def test_fitted_object_carries_celltype_dictionary():
    rna, ct, prot = _build_synthetic_cite_seq(50, 30, 10, seed=0)
    fit = fit_mean_celltype_baseline(rna, ct, prot)
    assert isinstance(fit, MeanCelltypeFit)
    assert set(fit.celltype_means.keys()) == {"ct_0", "ct_1", "ct_2"}
    for v in fit.celltype_means.values():
        assert v.shape == (10,)
    assert fit.fallback_mean.shape == (10,)
