"""
Preprocess CELLxGENE Census files: normalize, select HVGs, preserve raw counts.

Foundation models (Geneformer, scGPT, UCE) require raw integer counts for
tokenization, so raw counts are saved in .layers['counts'] before normalization.
"""

import os

import scanpy as sc

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    census_files = [
        f for f in os.listdir(RAW_DIR)
        if f.startswith("census_") and f.endswith(".h5ad")
    ]

    if not census_files:
        print("No CELLxGENE Census files found in data/raw/")
        return

    for fname in sorted(census_files):
        in_path = os.path.join(RAW_DIR, fname)
        out_path = os.path.join(PROCESSED_DIR, fname)

        if os.path.exists(out_path):
            print(f"Already processed: {out_path}")
            continue

        print(f"Processing {fname}...")
        adata = sc.read_h5ad(in_path)

        # Save raw counts BEFORE normalization (models need these for tokenization)
        adata.layers["counts"] = adata.X.copy()

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Use seurat_v3 on raw counts for consistency with perturbation preprocessing
        sc.pp.highly_variable_genes(
            adata, n_top_genes=5000, flavor="seurat_v3", layer="counts"
        )
        adata = adata[:, adata.var.highly_variable].copy()

        # Set var_names to gene symbols for downstream ortholog matching
        if "feature_name" in adata.var.columns:
            adata.var_names = adata.var["feature_name"].astype(str).values
            adata.var_names_make_unique()

        # Verify ensembl_id (feature_id) survives HVG subset — needed by TranscriptFormer
        if "feature_id" in adata.var.columns:
            adata.var["ensembl_id"] = adata.var["feature_id"]
            assert adata.var["ensembl_id"].notna().all(), (
                f"ensembl_id has NaN values after HVG subset in {fname}"
            )

        adata.write_h5ad(out_path)
        print(f"  {fname}: {adata.shape}, raw counts in .layers['counts']")


if __name__ == "__main__":
    run()
