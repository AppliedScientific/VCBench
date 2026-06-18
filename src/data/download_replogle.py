"""
Download Replogle K562 Perturb-seq essential dataset from Figshare+.

Source: DOI 10.25452/figshare.plus.20029387
File: K562_essential_raw_singlecell_01.h5ad (~200K cells, ~2000 perturbations)
"""

import os
import subprocess
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DIR, "K562_essential_raw_singlecell_01.h5ad")

# Figshare+ download URL — must be obtained from the Figshare page
# (requires login/agreement for Figshare+ datasets)
FIGSHARE_URL = None  # Set manually after obtaining URL from Figshare+


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        print(f"Already downloaded: {OUTPUT_FILE}")
        return OUTPUT_FILE

    if FIGSHARE_URL is None:
        print(
            "ERROR: Figshare+ requires manual URL retrieval.\n"
            "1. Go to https://doi.org/10.25452/figshare.plus.20029387\n"
            "2. Accept terms and get download URL for K562_essential_raw_singlecell_01.h5ad\n"
            "3. Set FIGSHARE_URL in this script or run:\n"
            '   wget -O data/raw/K562_essential_raw_singlecell_01.h5ad "<URL>"'
        )
        sys.exit(1)

    print(f"Downloading Replogle K562 essential dataset...")
    subprocess.run(
        ["wget", "-O", OUTPUT_FILE, FIGSHARE_URL],
        check=True,
    )
    print(f"Downloaded: {OUTPUT_FILE}")
    return OUTPUT_FILE


def verify():
    import scanpy as sc

    adata = sc.read_h5ad(OUTPUT_FILE)
    assert adata.n_obs > 150_000, f"Expected >150K cells, got {adata.n_obs}"
    print(f"Replogle K562 verified: {adata.n_obs} cells x {adata.n_vars} genes")

    pert_cols = [
        c
        for c in adata.obs.columns
        if "gene" in c.lower() or "pert" in c.lower()
    ]
    print(f"Perturbation columns: {pert_cols}")
    return True


if __name__ == "__main__":
    download()
    verify()
