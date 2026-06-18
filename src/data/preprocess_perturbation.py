"""
Preprocess perturbation datasets: filter, normalize, select HVGs, save raw counts.

Processes:
- Replogle K562 essential -> 5000 HVGs, log-normalized, counts in .layers['counts']
- Norman -> already preprocessed by GEARS, just copy to processed dir
"""

import os
import shutil

import scanpy as sc

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def preprocess_perturbation(input_path, output_path, n_hvg=5000):
    """Filter, normalize, and select HVGs for a perturbation dataset."""
    adata = sc.read_h5ad(input_path)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Save raw counts before normalization
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_hvg, flavor="seurat_v3", layer="counts"
    )
    adata = adata[:, adata.var.highly_variable].copy()

    adata.write_h5ad(output_path)
    print(f"Saved: {adata.n_obs} cells x {adata.n_vars} genes -> {output_path}")
    return adata


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Replogle K562
    replogle_in = os.path.join(RAW_DIR, "K562_essential_raw_singlecell_01.h5ad")
    replogle_out = os.path.join(PROCESSED_DIR, "replogle_k562_essential.h5ad")
    if os.path.exists(replogle_in):
        print("Preprocessing Replogle K562 essential...")
        preprocess_perturbation(replogle_in, replogle_out)
    else:
        print(f"Skipping Replogle (not found: {replogle_in})")

    # Norman — already preprocessed to 5000 HVGs by GEARS
    norman_in = os.path.join(RAW_DIR, "norman", "perturb_processed.h5ad")
    norman_out = os.path.join(PROCESSED_DIR, "norman.h5ad")
    if os.path.exists(norman_in):
        print("Copying Norman dataset (already preprocessed by GEARS)...")
        shutil.copy(norman_in, norman_out)
        print(f"Copied: {norman_out}")
    else:
        print(f"Skipping Norman (not found: {norman_in})")


if __name__ == "__main__":
    run()
