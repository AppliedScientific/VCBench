"""
Cross-species transfer baseline for Dimension B.

PCA + kNN: Fit PCA on human data, project mouse orthologs into human space,
use kNN to transfer cell type labels. This mirrors the "naive transfer"
scenario where a model trained on one species is applied to another.
"""

import pickle

import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


def _ensure_gene_symbols(adata):
    """If var_names are numeric indices, remap to feature_name gene symbols."""
    if "feature_name" in adata.var.columns:
        try:
            # Check if var_names look like integers
            int(adata.var_names[0])
            import pandas as pd
            adata.var_names = pd.Index(adata.var["feature_name"].astype(str).values)
            adata.var_names_make_unique()
        except (ValueError, TypeError):
            pass  # Already gene symbols
    return adata


def pca_knn_baseline(human_adata, mouse_adata, ortholog_path, n_components=50, k=5):
    """
    PCA + kNN cross-species baseline.

    PCA is fit on human data only, then applied to mouse — matching what
    foundation models do (train on one species, apply to another).
    """
    # Ensure var_names are gene symbols, not numeric indices
    human_adata = _ensure_gene_symbols(human_adata)
    mouse_adata = _ensure_gene_symbols(mouse_adata)

    with open(ortholog_path, "rb") as f:
        maps = pickle.load(f)
    mouse_to_human = maps["m2h"]

    # Find shared genes via ortholog mapping, maintaining paired order
    human_var_set = set(human_adata.var_names)
    paired_mouse = []
    paired_human = []
    for mg in mouse_adata.var_names:
        if mg in mouse_to_human:
            hg = mouse_to_human[mg]
            if hg in human_var_set:
                paired_mouse.append(mg)
                paired_human.append(hg)

    if not paired_human:
        raise ValueError("No shared ortholog genes found between human and mouse data")

    # Extract ALIGNED expression matrices (column i is the same ortholog pair)
    h_X = human_adata[:, paired_human].X
    h_X = h_X.toarray() if issparse(h_X) else np.array(h_X)

    m_X = mouse_adata[:, paired_mouse].X
    m_X = m_X.toarray() if issparse(m_X) else np.array(m_X)

    # PCA: fit on human, transform mouse
    n_genes = len(paired_human)
    pca = PCA(n_components=min(n_components, n_genes))
    h_pca = pca.fit_transform(h_X)
    m_pca = pca.transform(m_X)

    # Filter to shared cell types
    shared_types = set(human_adata.obs["cell_type"]) & set(mouse_adata.obs["cell_type"])
    if not shared_types:
        return {
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "n_shared_types": 0,
            "n_shared_genes": n_genes,
        }

    h_mask = human_adata.obs["cell_type"].isin(shared_types).values
    m_mask = mouse_adata.obs["cell_type"].isin(shared_types).values

    # kNN classification
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(h_pca[h_mask], human_adata.obs["cell_type"].values[h_mask])
    preds = knn.predict(m_pca[m_mask])

    return {
        "macro_f1": f1_score(
            mouse_adata.obs["cell_type"].values[m_mask], preds, average="macro"
        ),
        "weighted_f1": f1_score(
            mouse_adata.obs["cell_type"].values[m_mask], preds, average="weighted"
        ),
        "n_shared_types": len(shared_types),
        "n_shared_genes": n_genes,
    }
