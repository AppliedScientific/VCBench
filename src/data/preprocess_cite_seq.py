"""
Preprocess CITE-seq data for Dimension D: cross-modal RNA→protein prediction.

Steps:
1. Log-normalize RNA, CLR-normalize protein (ADT)
2. Select 4,000 HVGs from RNA
3. Create site-based train/test split: sites 1-3 train, site 4 test
4. Save raw counts in .layers['counts'] for models needing them
"""

import os

import numpy as np
import scanpy as sc

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def clr_normalize(X):
    """Centered log-ratio normalization for protein (ADT) counts."""
    from scipy.sparse import issparse

    if issparse(X):
        X = X.toarray()
    X = np.array(X, dtype=np.float64)
    X = X + 1  # Pseudocount to avoid log(0)
    log_X = np.log(X)
    geometric_mean = np.exp(log_X.mean(axis=1, keepdims=True))
    return log_X - np.log(geometric_mean)


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_path = os.path.join(PROCESSED_DIR, "cite_train.h5ad")
    test_path = os.path.join(PROCESSED_DIR, "cite_test.h5ad")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("CITE-seq already preprocessed.")
        return

    raw_path = os.path.join(RAW_DIR, "neurips_cite_seq.h5ad")
    print(f"Loading {raw_path}...")
    adata = sc.read_h5ad(raw_path)

    # Extract protein expression from obsm
    protein_key = None
    for key in adata.obsm:
        if "protein" in key.lower() or "adt" in key.lower():
            protein_key = key
            break

    if protein_key is None:
        raise ValueError(
            "No protein expression found in .obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    protein_raw = adata.obsm[protein_key]
    print(f"  Protein data: {protein_raw.shape} from .obsm['{protein_key}']")

    # CLR-normalize protein
    protein_clr = clr_normalize(protein_raw)
    print(f"  Protein CLR-normalized: {protein_clr.shape}")

    # Save raw RNA counts before normalization
    adata.layers["counts"] = adata.X.copy()

    # Log-normalize RNA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select 4,000 HVGs using raw counts
    sc.pp.highly_variable_genes(
        adata, n_top_genes=4000, flavor="seurat_v3", layer="counts"
    )
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"  RNA after HVG selection: {adata.shape}")

    # Store protein in obsm
    adata.obsm["protein_clr"] = protein_clr

    # Identify site column for train/test split
    site_col = None
    for col in ["site", "Site", "batch", "donor_site"]:
        if col in adata.obs.columns:
            site_col = col
            break

    if site_col is None:
        # Fallback: use donor column or create random split
        print("WARNING: No 'site' column found. Using random 75/25 split.")
        rng = np.random.default_rng(42)
        n = adata.n_obs
        perm = rng.permutation(n)
        train_n = int(n * 0.75)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[perm[:train_n]] = True
        test_mask = ~train_mask
    else:
        sites = adata.obs[site_col].unique()
        print(f"  Sites: {sorted(sites)}")
        # Use last site as test, rest as train
        test_site = sorted(sites)[-1]
        train_mask = adata.obs[site_col] != test_site
        test_mask = adata.obs[site_col] == test_site
        print(f"  Train sites: {sorted(sites[:-1])}, Test site: {test_site}")

    adata_train = adata[train_mask].copy()
    adata_test = adata[test_mask].copy()

    adata_train.write_h5ad(train_path)
    adata_test.write_h5ad(test_path)
    print(f"  Train: {adata_train.shape} -> {train_path}")
    print(f"  Test:  {adata_test.shape} -> {test_path}")


if __name__ == "__main__":
    run()
