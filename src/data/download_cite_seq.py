"""
Download NeurIPS 2021 Open Problems CITE-seq benchmark dataset.

66,175 bone marrow mononuclear cells, 12 donors, 4 sites.
13,431 genes (RNA) + 134 surface proteins (ADTs).
GEO accession: GSE194122.

Source: OpenProblems processed version on Figshare.
"""

import os
import subprocess

import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# OpenProblems processed CITE-seq data
FIGSHARE_URL = "https://figshare.com/ndownloader/articles/25958374/versions/1"
EXPECTED_CELLS = 66_000  # Approximate lower bound
EXPECTED_PROTEINS = 134


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, "neurips_cite_seq.h5ad")

    if os.path.exists(out_path):
        print(f"Already downloaded: {out_path}")
        return

    zip_path = os.path.join(RAW_DIR, "cite_seq_data.zip")
    print("Downloading CITE-seq data from Figshare...")
    subprocess.run(
        ["wget", "-O", zip_path, FIGSHARE_URL],
        check=True,
    )

    # Extract — look for .h5ad file
    print("Extracting...")
    subprocess.run(
        ["unzip", "-o", zip_path, "-d", os.path.join(RAW_DIR, "cite_seq_tmp")],
        check=True,
    )

    # Find the h5ad file in extracted contents
    tmp_dir = os.path.join(RAW_DIR, "cite_seq_tmp")
    h5ad_files = []
    for root, dirs, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith(".h5ad"):
                h5ad_files.append(os.path.join(root, f))

    if h5ad_files:
        os.rename(h5ad_files[0], out_path)
        print(f"Saved: {out_path}")
    else:
        # If no h5ad, try building from GEO directly
        print("No .h5ad found in Figshare download. Falling back to GEO...")
        download_from_geo(out_path)

    # Cleanup
    subprocess.run(["rm", "-rf", zip_path, tmp_dir])


def download_from_geo(out_path):
    """Fallback: download directly from GEO GSE194122."""
    import scanpy as sc
    import anndata as ad

    geo_url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/"
        "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
    )
    gz_path = out_path + ".gz"
    print(f"Downloading from GEO: {geo_url}")
    subprocess.run(["wget", "-O", gz_path, geo_url], check=True)
    subprocess.run(["gunzip", gz_path], check=True)
    print(f"Saved: {out_path}")


def verify():
    import scanpy as sc

    path = os.path.join(RAW_DIR, "neurips_cite_seq.h5ad")
    assert os.path.exists(path), f"Missing: {path}"

    adata = sc.read_h5ad(path)
    print(f"  Shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns[:10])}")

    assert adata.n_obs >= EXPECTED_CELLS, (
        f"Expected >= {EXPECTED_CELLS} cells, got {adata.n_obs}"
    )

    # Check for protein expression
    has_protein = (
        "protein_expression" in adata.obsm
        or any("ADT" in str(k) or "protein" in str(k).lower()
               for k in list(adata.obsm.keys()) + list(adata.obs.columns))
    )
    assert has_protein, "No protein expression data found in .obsm or .obs"

    print(f"CITE-seq data verified: {adata.n_obs} cells")
    return True


if __name__ == "__main__":
    download()
    verify()
