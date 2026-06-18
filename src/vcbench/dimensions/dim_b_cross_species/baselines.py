"""Dim B baselines.

* :func:`pca_knn_classifier` — 50-component PCA + k=5 cosine kNN, the binding
  non-FM baseline for Dim B. Fit on human cells, transferred to mouse cells
  via the ortholog-aligned gene matrix.

The implementation here is the sklearn-based reference reimplementation; it
matches ``src/baselines/cross_species.py::pca_knn_baseline`` (the legacy code
that produced ``results/dim_b/baselines_pca_knn_common.json``) on synthetic
data. The legacy code remains in-place for backward compatibility with
existing pipeline invocations.

Hyperparameters are pinned at the manuscript values (50 PCA components,
k=5, cosine metric); the function exposes them only so tests can override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

PCA_N_COMPONENTS_DEFAULT: int = 50
KNN_K_DEFAULT: int = 5
KNN_METRIC_DEFAULT: str = "cosine"


@dataclass(frozen=True)
class PCAKNNFit:
    """Fitted PCA+kNN classifier ready to predict on a target-species matrix.

    Attributes
    ----------
    pca : sklearn.decomposition.PCA
        PCA fitted on the source-species expression matrix.
    knn : sklearn.neighbors.KNeighborsClassifier
        kNN trained on PCA-projected source cells.
    n_components : int
        Effective number of PCA components used (capped at n_genes if smaller).
    n_train_cells : int
        Number of source-species cells the classifier was fit on.
    """

    pca: PCA
    knn: KNeighborsClassifier
    n_components: int
    n_train_cells: int

    def predict(self, target_X: np.ndarray) -> np.ndarray:
        """Predict labels for a target-species expression matrix."""
        target_X = np.asarray(target_X, dtype=float)
        return self.knn.predict(self.pca.transform(target_X))


def pca_knn_classifier(
    source_X: np.ndarray,
    source_labels: np.ndarray,
    *,
    n_components: int = PCA_N_COMPONENTS_DEFAULT,
    k: int = KNN_K_DEFAULT,
    metric: str = KNN_METRIC_DEFAULT,
) -> PCAKNNFit:
    """Fit the PCA + kNN cross-species baseline.

    Parameters
    ----------
    source_X : np.ndarray
        Source-species expression matrix, shape (n_source_cells, n_genes).
        Genes must be **ortholog-aligned** to the target matrix that
        :meth:`PCAKNNFit.predict` will be called on (column ``i`` is the
        same ortholog in both).
    source_labels : np.ndarray
        Cell-type labels for source cells, shape (n_source_cells,).
    n_components : int, default 50
        PCA component count, capped at n_genes if smaller.
    k : int, default 5
        kNN neighbour count.
    metric : str, default ``"cosine"``
        kNN distance metric.

    Returns
    -------
    PCAKNNFit
        Fitted classifier wrapping the PCA and kNN objects.

    Raises
    ------
    ValueError
        If source_X has zero rows or shape doesn't match labels.
    """
    source_X = np.asarray(source_X, dtype=float)
    source_labels = np.asarray(source_labels)
    if source_X.ndim != 2:
        raise ValueError(f"source_X must be 2-D, got shape {source_X.shape}")
    if source_X.shape[0] == 0:
        raise ValueError("source_X has zero rows")
    if source_X.shape[0] != len(source_labels):
        raise ValueError(
            f"source_X rows ({source_X.shape[0]}) != source_labels ({len(source_labels)})"
        )

    n_eff = min(n_components, source_X.shape[1], source_X.shape[0])
    pca = PCA(n_components=n_eff)
    src_proj = pca.fit_transform(source_X)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(source_labels)), metric=metric)
    knn.fit(src_proj, source_labels)
    return PCAKNNFit(
        pca=pca,
        knn=knn,
        n_components=n_eff,
        n_train_cells=source_X.shape[0],
    )
