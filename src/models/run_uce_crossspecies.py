"""
UCE cross-species embedding extraction.

UCE was trained on a species-wide gene universe (~33,694 genes).
Ideally use RAW files with all genes, but HVG-subsetted data (data/processed/)
is acceptable — UCE gene embeddings handle missing genes via the protein
embedding lookup; it just uses fewer genes per cell.

UCE handles cross-species natively — no ortholog remapping needed.
"""

import os
import subprocess
import sys

import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_b", "uce")
UCE_DIR = os.path.join(PROJECT_DIR, "models", "uce", "UCE")
WEIGHTS_PATH = os.path.join(PROJECT_DIR, "models", "uce", "33l_8ep_1024t_1280.torch")

TISSUES = ["lung", "liver", "heart", "kidney", "brain"]


def _find_census_file(species_key, tissue):
    """Find census file — prefer raw (all genes), fall back to processed (HVG)."""
    raw_path = os.path.join(RAW_DIR, f"census_{species_key}_{tissue}.h5ad")
    if os.path.exists(raw_path):
        return raw_path
    processed_path = os.path.join(PROCESSED_DIR, f"census_{species_key}_{tissue}.h5ad")
    if os.path.exists(processed_path):
        return processed_path
    return None


def _prepare_uce_input(adata_path, species_key, tissue):
    """Ensure UCE input has gene symbols as var_names (not Ensembl IDs)."""
    import scanpy as sc

    uce_path = os.path.join(PROCESSED_DIR, f"census_{species_key}_{tissue}_uce.h5ad")
    if os.path.exists(uce_path):
        return uce_path

    adata = sc.read_h5ad(adata_path)

    # UCE expects gene symbols. Convert if needed.
    if adata.var_names[0].startswith("ENSG") or adata.var_names[0].startswith("ENSMUS"):
        if "feature_name" in adata.var.columns:
            adata.var_names = adata.var["feature_name"].astype(str).values
        elif "gene_name" in adata.var.columns:
            adata.var_names = adata.var["gene_name"].astype(str).values
        adata.var_names_make_unique()

    adata.write_h5ad(uce_path)
    return uce_path


def run_uce(adata_path, output_dir, species, batch_size=25):
    """Run UCE embedding extraction via its CLI script."""
    os.makedirs(output_dir, exist_ok=True)

    script = os.path.join(UCE_DIR, "eval_single_anndata.py")
    if not os.path.exists(script):
        print(f"ERROR: UCE script not found: {script}")
        return False

    cmd = [
        "python", "eval_single_anndata.py",
        "--adata_path", os.path.abspath(adata_path),
        "--dir", os.path.abspath(output_dir) + "/",
        "--species", species,
        "--model_loc", os.path.abspath(WEIGHTS_PATH),
        "--batch_size", str(batch_size),
        "--nlayers", "33",
    ]
    print(f"  Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, cwd=UCE_DIR)
        return True
    except subprocess.CalledProcessError as e:
        if "CUDA out of memory" in str(e):
            print(f"  OOM at batch_size={batch_size}, retrying with {batch_size // 2}")
            return run_uce(adata_path, output_dir, species, batch_size // 2)
        raise


def extract_and_save(tissue, species_key, species_label):
    """Extract UCE embeddings and save with labels."""
    # Checkpoint: skip if embeddings already exist
    emb_path = os.path.join(RESULTS_DIR, f"{species_label}_{tissue}_embeddings.npy")
    if os.path.exists(emb_path):
        print(f"  Already extracted: {emb_path}")
        return

    adata_path = _find_census_file(species_key, tissue)
    if adata_path is None:
        print(f"  Skipping {species_label} {tissue}: no census file found")
        return

    print(f"  Using: {adata_path}")

    # Prepare input (ensure gene symbols)
    uce_input = _prepare_uce_input(adata_path, species_key, tissue)

    output_dir = os.path.join(RESULTS_DIR, f"{species_label}_{tissue}")
    try:
        run_uce(uce_input, output_dir, species_label)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # UCE saves embeddings in the output AnnData's .obsm['X_uce']
    import scanpy as sc
    result_files = [f for f in os.listdir(output_dir) if f.endswith(".h5ad")]
    if result_files:
        result = sc.read_h5ad(os.path.join(output_dir, result_files[0]))
        if "X_uce" in result.obsm:
            emb = result.obsm["X_uce"]
            np.save(emb_path, emb)
            print(f"  UCE embeddings: {emb.shape}")

    # Save labels from original data
    orig = sc.read_h5ad(adata_path)
    np.save(
        os.path.join(RESULTS_DIR, f"{species_label}_{tissue}_labels.npy"),
        orig.obs["cell_type"].values,
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: UCE weights not found: {WEIGHTS_PATH}")
        print("Download from: https://figshare.com/articles/dataset/24320806")
        return

    print("=== UCE Cross-Species Embeddings ===")

    for tissue in TISSUES:
        print(f"\n--- {tissue} ---")
        extract_and_save(tissue, "homo_sapiens", "human")
        extract_and_save(tissue, "mus_musculus", "mouse")

    print("\n=== UCE cross-species embeddings complete ===")


if __name__ == "__main__":
    main()
