"""
Download temporal datasets for Dimension E: temporal ordering.

Datasets:
1. sci-fate (Cao 2020): 6,680 cells, 6 timepoints (0-10h), A549 + dexamethasone
   GEO: GSE131351
2. LARRY/Weinreb (Weinreb 2020): 49,116 cells, 3 timepoints (Day 2,4,6), mouse HSPCs
   GEO: GSE140802
"""

import os
import subprocess

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def download_sci_fate():
    """Download sci-fate dataset from GEO GSE131351."""
    out_path = os.path.join(RAW_DIR, "sci_fate.h5ad")
    if os.path.exists(out_path):
        print(f"Already downloaded: {out_path}")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    # sci-fate processed data — check GEO supplementary files
    geo_url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131351/suppl/"
        "GSE131351_RAW.tar"
    )
    tar_path = os.path.join(RAW_DIR, "sci_fate_raw.tar")
    tmp_dir = os.path.join(RAW_DIR, "sci_fate_tmp")

    print("Downloading sci-fate from GEO GSE131351...")
    subprocess.run(["wget", "-O", tar_path, geo_url], check=True)

    print("Extracting...")
    os.makedirs(tmp_dir, exist_ok=True)
    subprocess.run(["tar", "-xf", tar_path, "-C", tmp_dir], check=True)

    # Build AnnData from extracted files
    build_sci_fate_anndata(tmp_dir, out_path)

    # Cleanup
    subprocess.run(["rm", "-rf", tar_path, tmp_dir])


def build_sci_fate_anndata(tmp_dir, out_path):
    """Build AnnData from GEO extracted files."""
    import scanpy as sc
    import anndata as ad
    import pandas as pd
    import numpy as np
    from scipy.io import mmread

    # Look for matrix files
    mtx_files = []
    for root, dirs, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith((".mtx", ".mtx.gz", ".h5ad", ".csv.gz")):
                mtx_files.append(os.path.join(root, f))

    if any(f.endswith(".h5ad") for f in mtx_files):
        h5ad = [f for f in mtx_files if f.endswith(".h5ad")][0]
        adata = sc.read_h5ad(h5ad)
    else:
        # Try reading as 10x-style directory
        print(f"  Found files: {[os.path.basename(f) for f in mtx_files[:10]]}")
        print("  Attempting to read as 10x format...")
        adata = sc.read_10x_mtx(tmp_dir)

    # Ensure true_time exists
    if "true_time" not in adata.obs.columns:
        # Try to parse from other metadata
        for col in ["time", "timepoint", "Time", "hour", "hours"]:
            if col in adata.obs.columns:
                adata.obs["true_time"] = pd.to_numeric(
                    adata.obs[col], errors="coerce"
                )
                break
        else:
            print("WARNING: No time column found. Setting placeholder values.")
            adata.obs["true_time"] = 0.0

    adata.write_h5ad(out_path)
    print(f"  sci-fate: {adata.shape} -> {out_path}")


def download_weinreb():
    """Download Weinreb/LARRY dataset from GEO GSE140802."""
    out_path = os.path.join(RAW_DIR, "weinreb.h5ad")
    if os.path.exists(out_path):
        print(f"Already downloaded: {out_path}")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    # LARRY dataset processed data
    geo_url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE140nnn/GSE140802/suppl/"
        "GSE140802_RAW.tar"
    )
    tar_path = os.path.join(RAW_DIR, "weinreb_raw.tar")
    tmp_dir = os.path.join(RAW_DIR, "weinreb_tmp")

    print("Downloading Weinreb/LARRY from GEO GSE140802...")
    subprocess.run(["wget", "-O", tar_path, geo_url], check=True)

    print("Extracting...")
    os.makedirs(tmp_dir, exist_ok=True)
    subprocess.run(["tar", "-xf", tar_path, "-C", tmp_dir], check=True)

    build_weinreb_anndata(tmp_dir, out_path)

    # Cleanup
    subprocess.run(["rm", "-rf", tar_path, tmp_dir])


def build_weinreb_anndata(tmp_dir, out_path):
    """Build AnnData from Weinreb GEO files."""
    import scanpy as sc
    import pandas as pd
    import numpy as np

    # Look for data files
    all_files = []
    for root, dirs, files in os.walk(tmp_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    h5ad_files = [f for f in all_files if f.endswith(".h5ad")]
    if h5ad_files:
        adata = sc.read_h5ad(h5ad_files[0])
    else:
        print(f"  Found files: {[os.path.basename(f) for f in all_files[:10]]}")
        # Try standard 10x format
        adata = sc.read_10x_mtx(tmp_dir)

    # Map timepoints to numeric
    if "true_time" not in adata.obs.columns:
        time_map = {"Day 2": 2, "Day 4": 4, "Day 6": 6,
                    "d2": 2, "d4": 4, "d6": 6,
                    "2": 2, "4": 4, "6": 6}
        for col in ["Time_point", "time", "timepoint", "day", "Day"]:
            if col in adata.obs.columns:
                adata.obs["true_time"] = adata.obs[col].map(
                    lambda x: time_map.get(str(x), pd.to_numeric(x, errors="coerce"))
                )
                break
        else:
            print("WARNING: No time column found. Setting placeholder values.")
            adata.obs["true_time"] = 0.0

    adata.write_h5ad(out_path)
    print(f"  Weinreb: {adata.shape} -> {out_path}")


def verify():
    import scanpy as sc

    for name, min_cells in [("sci_fate", 5000), ("weinreb", 40000)]:
        path = os.path.join(RAW_DIR, f"{name}.h5ad")
        assert os.path.exists(path), f"Missing: {path}"
        adata = sc.read_h5ad(path)
        assert adata.n_obs >= min_cells, (
            f"{name}: expected >= {min_cells} cells, got {adata.n_obs}"
        )
        assert "true_time" in adata.obs.columns, (
            f"{name}: missing 'true_time' column"
        )
        n_timepoints = adata.obs["true_time"].nunique()
        print(f"  {name}: {adata.shape}, {n_timepoints} timepoints")

    print("Temporal datasets verified.")
    return True


if __name__ == "__main__":
    download_sci_fate()
    download_weinreb()
    verify()
