"""
Data preparation utilities for model-specific input formats.

Handles Ensembl ID mapping, n_counts annotation, and ortholog remapping
needed by Geneformer, scGPT, and other foundation models.
"""

import json
import os
import pickle

import numpy as np
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "static")


def get_symbol_to_ensembl():
    """
    Load human gene symbol -> Ensembl ID mapping from static file.

    The mapping is generated once during benchmark setup via:
        python -m src.data.build_ensembl_map
    and committed to data/static/ensembl_symbol_map.json for reproducibility.
    """
    map_path = os.path.join(STATIC_DIR, "ensembl_symbol_map.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"Ensembl mapping not found: {map_path}\n"
            "Generate it with: python -m src.data.build_ensembl_map"
        )
    with open(map_path) as f:
        return json.load(f)


def prepare_for_geneformer(input_path, output_path):
    """
    Add ensembl_id and n_counts required by Geneformer tokenizer.

    Geneformer requires:
    - adata.var["ensembl_id"]: Ensembl gene IDs
    - adata.obs["n_counts"]: total raw counts per cell
    - adata.X or adata.layers["counts"]: raw integer counts
    """
    if os.path.exists(output_path):
        print(f"Already prepared: {output_path}")
        return sc.read_h5ad(output_path)

    adata = sc.read_h5ad(input_path)
    symbol_to_ensembl = get_symbol_to_ensembl()

    # Add Ensembl IDs — handle both symbol-indexed and ensembl-indexed data
    if adata.var_names[0].startswith("ENSG"):
        # var_names are already Ensembl IDs (e.g. Norman from GEARS)
        adata.var["ensembl_id"] = adata.var_names.tolist()
        print(f"Ensembl mapping: {adata.n_vars}/{adata.n_vars} genes (already Ensembl IDs)")
    elif "gene_name" in adata.var.columns:
        # Try gene_name column for symbol lookup
        adata.var["ensembl_id"] = [
            symbol_to_ensembl.get(g, "NA") for g in adata.var["gene_name"]
        ]
        n_before = adata.n_vars
        adata = adata[:, adata.var["ensembl_id"] != "NA"].copy()
        print(f"Ensembl mapping: {adata.n_vars}/{n_before} genes mapped (via gene_name)")
    else:
        adata.var["ensembl_id"] = [
            symbol_to_ensembl.get(g, "NA") for g in adata.var_names
        ]
        n_before = adata.n_vars
        adata = adata[:, adata.var["ensembl_id"] != "NA"].copy()
        print(f"Ensembl mapping: {adata.n_vars}/{n_before} genes mapped")

    # Add cell_type if missing (Geneformer tokenizer requires it)
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "unknown"
        print("WARNING: No 'cell_type' column — added placeholder 'unknown'")

    # Add n_counts from raw counts layer (or X as fallback)
    if "n_counts" in adata.obs.columns:
        print("n_counts already in obs")
    elif "counts" in adata.layers:
        counts = adata.layers["counts"]
        adata.obs["n_counts"] = np.array(counts.sum(axis=1)).flatten()
    else:
        from scipy.sparse import issparse
        X = adata.X
        totals = np.array(X.sum(axis=1)).flatten() if issparse(X) else X.sum(axis=1).flatten()
        adata.obs["n_counts"] = totals
        print(f"WARNING: No 'counts' layer found in {input_path}. "
              "Using X.sum() as n_counts — results may differ if X is normalized.")

    adata.write_h5ad(output_path)
    print(f"Geneformer-ready: {adata.shape} -> {output_path}")
    return adata


def remap_mouse_to_human(mouse_path, output_path, ortholog_path=None):
    """
    Remap mouse gene names to human orthologs for cross-species transfer.

    Used by models that only have human checkpoints (Geneformer, scGPT).
    """
    if os.path.exists(output_path):
        print(f"Already remapped: {output_path}")
        return sc.read_h5ad(output_path)

    if ortholog_path is None:
        ortholog_path = os.path.join(PROCESSED_DIR, "ortholog_maps.pkl")

    with open(ortholog_path, "rb") as f:
        maps = pickle.load(f)
    m2h = maps["m2h"]

    mouse = sc.read_h5ad(mouse_path)
    mappable = [g for g in mouse.var_names if g in m2h]
    mapped = mouse[:, mappable].copy()
    mapped.var_names = [m2h[g] for g in mappable]
    mapped.var_names_make_unique()

    mapped.write_h5ad(output_path)
    print(f"Remapped {len(mappable)}/{mouse.n_vars} mouse genes -> human orthologs")
    return mapped
