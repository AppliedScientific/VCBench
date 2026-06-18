"""
Preprocess temporal datasets for Dimension E: temporal ordering.

Steps:
1. Log-normalize, select HVGs
2. Store true_time in obs
3. No train/test split — temporal ordering is unsupervised
4. Save raw counts in .layers['counts'] for foundation models
"""

import os

import numpy as np
import scanpy as sc

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

DATASETS = {
    "sci_fate": {"min_cells": 5000, "min_timepoints": 3},
    "weinreb": {"min_cells": 40000, "min_timepoints": 3},
}


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for name, expected in DATASETS.items():
        in_path = os.path.join(RAW_DIR, f"{name}.h5ad")
        out_path = os.path.join(PROCESSED_DIR, f"{name}.h5ad")

        if os.path.exists(out_path):
            print(f"Already processed: {out_path}")
            continue

        if not os.path.exists(in_path):
            print(f"Not found, skipping: {in_path}")
            continue

        print(f"Processing {name}...")
        adata = sc.read_h5ad(in_path)

        # Verify time column
        assert "true_time" in adata.obs.columns, f"Missing true_time in {name}"
        n_timepoints = adata.obs["true_time"].nunique()
        print(f"  {adata.shape}, {n_timepoints} timepoints")

        # Drop cells with missing time
        valid_mask = adata.obs["true_time"].notna()
        if valid_mask.sum() < adata.n_obs:
            print(f"  Dropping {adata.n_obs - valid_mask.sum()} cells with missing time")
            adata = adata[valid_mask].copy()

        # Save raw counts before normalization
        adata.layers["counts"] = adata.X.copy()

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # HVG selection on raw counts
        sc.pp.highly_variable_genes(
            adata, n_top_genes=4000, flavor="seurat_v3", layer="counts"
        )
        adata = adata[:, adata.var.highly_variable].copy()

        # Ensure true_time is numeric
        adata.obs["true_time"] = adata.obs["true_time"].astype(float)

        adata.write_h5ad(out_path)
        print(f"  {name}: {adata.shape} -> {out_path}")
        print(f"  Time range: {adata.obs['true_time'].min()} - {adata.obs['true_time'].max()}")


if __name__ == "__main__":
    run()
