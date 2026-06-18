"""Unit tests for vcbench.dimensions.dim_b_cross_species.baselines (PCA+kNN)."""

from __future__ import annotations

import numpy as np
import pytest

from vcbench.dimensions.dim_b_cross_species.baselines import (
    KNN_K_DEFAULT,
    KNN_METRIC_DEFAULT,
    PCA_N_COMPONENTS_DEFAULT,
    pca_knn_classifier,
)
from vcbench.dimensions.dim_b_cross_species.metrics import macro_f1


def _clusters_at(centers: np.ndarray, n_per_class: int, *, seed: int, noise: float = 0.3):
    """Sample n_per_class cells around each row of `centers`."""
    rng = np.random.default_rng(seed)
    n_classes, n_genes = centers.shape
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.normal(0, noise, size=(n_per_class, n_genes)))
        y.extend([f"class_{c}"] * n_per_class)
    return np.vstack(X), np.array(y)


def test_pca_knn_recovers_well_separated_clusters():
    """When source and target share the same cluster geometry (same cell types
    in both species), PCA+kNN should recover labels at ≥ 0.9 macro F1.
    The 'cross-species' analogy is preserved by using independent noise draws
    for source and target around shared per-class centers."""
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 5, size=(4, 30))
    src_X, src_y = _clusters_at(centers, n_per_class=60, seed=1)
    tgt_X, tgt_y = _clusters_at(centers, n_per_class=30, seed=2)
    fit = pca_knn_classifier(src_X, src_y)
    preds = fit.predict(tgt_X)
    assert macro_f1(tgt_y, preds) > 0.9


def test_pca_knn_chance_on_uncorrelated_data():
    """Random gene-space alignment between species → kNN at chance (~1/n_classes)."""
    rng = np.random.default_rng(2)
    n_genes, n_classes = 30, 4
    src_X = rng.normal(0, 1, size=(160, n_genes))
    src_y = np.array([f"class_{i % n_classes}" for i in range(160)])
    tgt_X = rng.normal(0, 1, size=(80, n_genes))
    tgt_y = np.array([f"class_{i % n_classes}" for i in range(80)])
    fit = pca_knn_classifier(src_X, src_y)
    preds = fit.predict(tgt_X)
    score = macro_f1(tgt_y, preds)
    # 4-class chance ~0.25; allow generous margin for sampling noise
    assert 0.0 <= score < 0.45


def test_pca_knn_default_hyperparameters():
    assert PCA_N_COMPONENTS_DEFAULT == 50
    assert KNN_K_DEFAULT == 5
    assert KNN_METRIC_DEFAULT == "cosine"


def test_pca_components_capped_to_min_dim():
    """If n_components > min(n_cells, n_genes), it must be capped without error."""
    src_X = np.random.default_rng(3).normal(0, 1, size=(20, 5))
    src_y = np.array([f"c_{i % 3}" for i in range(20)])
    fit = pca_knn_classifier(src_X, src_y, n_components=100)
    assert fit.n_components <= 5  # capped


def test_pca_knn_shape_validation():
    with pytest.raises(ValueError):
        pca_knn_classifier(np.zeros((0, 10)), np.array([]))  # zero rows
    with pytest.raises(ValueError):
        pca_knn_classifier(np.zeros((10, 5)), np.array(["a"] * 9))  # length mismatch
    with pytest.raises(ValueError):
        pca_knn_classifier(np.zeros(10), np.array([]))  # 1-D


def test_predict_uses_fitted_pca_not_target_pca():
    """Calling predict on a target matrix must transform via the SOURCE-fitted
    PCA (i.e., PCA components are not re-fit on target). This is the core
    cross-species design: train on source, freeze, transfer to target."""
    centers = np.random.default_rng(10).normal(0, 5, size=(4, 30))
    src_X, src_y = _clusters_at(centers, n_per_class=60, seed=4)
    fit = pca_knn_classifier(src_X, src_y)
    # If the fit re-trained PCA on target we'd get different components every
    # call. Predict twice and assert the underlying PCA components are unchanged.
    components_before = fit.pca.components_.copy()
    fit.predict(src_X)
    fit.predict(src_X[:10])
    np.testing.assert_array_equal(fit.pca.components_, components_before)
