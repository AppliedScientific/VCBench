"""
Prepare GRN ground truth edge sets from BEELINE and TRRUST.

BEELINE ships expression data as CSV (genes x cells), not h5ad.
This script converts to AnnData and parses ground truth edge networks.
"""

import os
import pickle

import pandas as pd
import scanpy as sc

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def convert_beeline_to_h5ad():
    """Convert BEELINE ExpressionData.csv (genes x cells) -> AnnData (cells x genes)."""
    for dataset_name in ["hESC", "hHep"]:
        # BEELINE data may be under scRNA-Seq/ or Experimental/
        expr_path = os.path.join(
            RAW_DIR, "beeline", "BEELINE-data", "inputs", "scRNA-Seq",
            dataset_name, "ExpressionData.csv",
        )
        if not os.path.exists(expr_path):
            expr_path = os.path.join(
                RAW_DIR, "beeline", "inputs", "Experimental",
                dataset_name, "ExpressionData.csv",
            )
        out_path = os.path.join(PROCESSED_DIR, f"beeline_{dataset_name.lower()}.h5ad")

        if os.path.exists(out_path):
            print(f"Already converted: {out_path}")
            continue

        if not os.path.exists(expr_path):
            print(f"Skipping {dataset_name} (not found: {expr_path})")
            continue

        # BEELINE CSVs are genes x cells (rows = genes, columns = cells)
        expr = pd.read_csv(expr_path, index_col=0)
        adata = sc.AnnData(X=expr.T)  # Transpose to cells x genes
        adata.write_h5ad(out_path)
        print(f"BEELINE {dataset_name}: {adata.n_obs} cells x {adata.n_vars} genes -> {out_path}")


def parse_ground_truth():
    """Parse BEELINE and TRRUST ground truth edge sets."""
    gt = {}

    # BEELINE edge networks
    for dataset_name in ["hESC", "hHep"]:
        ref_path = os.path.join(
            RAW_DIR, "beeline", "BEELINE-data", "inputs", "scRNA-Seq",
            dataset_name, "refNetwork.csv",
        )
        if not os.path.exists(ref_path):
            ref_path = os.path.join(
                RAW_DIR, "beeline", "inputs", "Experimental",
                dataset_name, "refNetwork.csv",
            )
        if os.path.exists(ref_path):
            df = pd.read_csv(ref_path)
            edges = set(zip(df["Gene1"], df["Gene2"]))
            gt[f"beeline_{dataset_name.lower()}"] = edges
            print(f"BEELINE {dataset_name}: {len(edges)} edges")

    # TRRUST
    trrust_path = os.path.join(RAW_DIR, "trrust_rawdata.human.tsv")
    if os.path.exists(trrust_path):
        trrust = pd.read_csv(
            trrust_path, sep="\t", header=None,
            names=["TF", "Target", "Mode", "PMID"],
        )
        gt["trrust"] = set(zip(trrust["TF"], trrust["Target"]))
        gt["trrust_tfs"] = set(trrust["TF"].unique())
        print(f"TRRUST: {len(gt['trrust'])} edges, {len(gt['trrust_tfs'])} TFs")

    # Save
    out_path = os.path.join(PROCESSED_DIR, "grn_ground_truth.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(gt, f)
    print(f"Ground truth saved: {out_path}")


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    convert_beeline_to_h5ad()
    parse_ground_truth()


if __name__ == "__main__":
    run()
