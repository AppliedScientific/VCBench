"""
Extract foundation model embeddings for Dim E temporal datasets.

Species-aware extraction:
  - sci-fate (human A549 cells): all 5 models participate directly
  - Weinreb LARRY (mouse hematopoietic): species handling varies:
      * UCE: native multi-species (--species mus_musculus)
      * TranscriptFormer: native multi-species (mouse gene embeddings)
      * Geneformer: remap mouse->human orthologs, then embed
      * scGPT: remap mouse->human orthologs, then embed
      * Arc State: N/A (human-only tokenizer, no ortholog support)

This is the same species logic used in Dim B (cross-species), now applied
to temporal datasets. Without ortholog mapping, human-vocabulary models
produce garbage on mouse gene symbols.

Output: {model}/{dataset}_embeddings.npy in results/dim_e/

Run in: respective model environments (GPU required for extraction)
"""

import os
import sys

import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_e")

# Dataset metadata
TEMPORAL_DATASETS = {
    "sci_fate": {
        "species": "human",
        "file": "sci_fate.h5ad",
    },
    "weinreb": {
        "species": "mouse",
        "file": "weinreb.h5ad",
    },
}

# Which models can process which species
MODEL_SPECIES_SUPPORT = {
    "geneformer": {"human": "direct", "mouse": "ortholog"},
    "scgpt": {"human": "direct", "mouse": "ortholog"},
    "uce": {"human": "direct", "mouse": "direct"},
    "transcriptformer": {"human": "direct", "mouse": "direct"},
    "state": {"human": "direct", "mouse": None},  # N/A on mouse
}


def _remap_for_human_model(adata, dataset_name):
    """Remap mouse genes to human orthologs for human-only models."""
    from src.models.data_prep import remap_mouse_to_human

    remapped_path = os.path.join(
        PROCESSED_DIR, f"{dataset_name}_human_orthologs.h5ad"
    )
    return remap_mouse_to_human(
        os.path.join(PROCESSED_DIR, f"{dataset_name}.h5ad"),
        remapped_path,
    )


def extract_geneformer(dataset_name, adata_path):
    """Extract Geneformer cell embeddings (mean-pool, pretrained)."""
    from geneformer import EmbExtractor, TranscriptomeTokenizer
    from src.models.data_prep import prepare_for_geneformer

    model_path = os.path.join(
        PROJECT_DIR, "models", "geneformer", "Geneformer", "Geneformer-V2-316M"
    )
    tokenized_dir = os.path.join(PROJECT_DIR, "data", "tokenized")
    os.makedirs(tokenized_dir, exist_ok=True)

    # Prepare (add ensembl_id, n_counts)
    gf_path = adata_path.replace(".h5ad", "_gf.h5ad")
    prepare_for_geneformer(adata_path, gf_path)

    # Tokenize — isolate target file to prevent tokenizer from globbing other h5ads
    # Check if cell_type exists (prepare_for_geneformer adds placeholder if missing)
    import tempfile
    import scanpy as _sc
    _tmp_adata = _sc.read_h5ad(gf_path)
    has_cell_type = "cell_type" in _tmp_adata.obs.columns
    del _tmp_adata

    basename = os.path.basename(gf_path).replace(".h5ad", "")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.symlink(os.path.abspath(gf_path), os.path.join(tmpdir, os.path.basename(gf_path)))
        custom_attrs = {"cell_type": "cell_type"} if has_cell_type else {}
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict=custom_attrs,
            use_h5ad_index=True,
        )
        tk.tokenize_data(
            tmpdir + "/",
            tokenized_dir + "/",
            basename,
            file_format="h5ad",
        )

    # Verify tokenized dataset is non-empty
    from datasets import load_from_disk
    tok_ds_path = os.path.join(tokenized_dir, f"{basename}.dataset")
    tok_ds = load_from_disk(tok_ds_path)
    if len(tok_ds) == 0:
        raise ValueError(f"Tokenized dataset is empty — no valid cells for {dataset_name}")
    print(f"  Tokenized {len(tok_ds)} cells for {dataset_name}")

    # Extract embeddings
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        emb_layer=-1,
        forward_batch_size=100,
        nproc=8,
    )
    out_dir = os.path.join(RESULTS_DIR, "geneformer")
    os.makedirs(out_dir, exist_ok=True)

    embs = embex.extract_embs(
        model_path,
        tok_ds_path,
        out_dir + "/",
        dataset_name,
    )
    return np.array(embs)


def extract_scgpt(dataset_name, adata_path):
    """Extract scGPT cell embeddings."""
    import scanpy as sc
    import torch
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab

    model_dir = os.path.join(PROJECT_DIR, "models", "scgpt", "scGPT_human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = GeneVocab.from_file(os.path.join(model_dir, "vocab.json"))
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=512, nhead=8, d_hid=512, nlayers=12,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
    )
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device),
        strict=False,
    )
    model = model.to(device)
    model.eval()

    adata = sc.read_h5ad(adata_path)
    gene_ids = np.array([
        vocab[g] if g in vocab else vocab["<pad>"] for g in adata.var_names
    ])

    from scipy.sparse import issparse
    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)

    # Batch embedding extraction
    all_embs = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch_x = torch.tensor(X[start:end], dtype=torch.float32).to(device)
            batch_ids = torch.tensor(gene_ids, dtype=torch.long).unsqueeze(0).expand(end - start, -1).to(device)

            output = model._encode(batch_ids, batch_x, src_key_padding_mask=None)
            # CLS token or mean pool
            embs = output.mean(dim=1)  # mean pool over genes
            all_embs.append(embs.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def _prepare_uce_input(adata_path, dataset_name):
    """Ensure UCE input has gene symbols as var_names (not Ensembl IDs)."""
    import scanpy as sc

    uce_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_uce.h5ad")
    if os.path.exists(uce_path):
        return uce_path

    adata = sc.read_h5ad(adata_path)

    # UCE expects gene symbols. If var_names are Ensembl IDs, convert.
    if adata.var_names[0].startswith("ENSG") or adata.var_names[0].startswith("ENSMUS"):
        if "gene_short_name" in adata.var.columns:
            adata.var_names = adata.var["gene_short_name"].astype(str).values
        elif "gene_name" in adata.var.columns:
            adata.var_names = adata.var["gene_name"].astype(str).values
        else:
            print(f"  WARNING: {dataset_name} has Ensembl IDs but no gene_name column for UCE")
            return adata_path
        adata.var_names_make_unique()

    adata.write_h5ad(uce_path)
    print(f"  UCE-ready: {adata.shape} -> {uce_path}")
    return uce_path


def extract_uce(dataset_name, adata_path, species):
    """Extract UCE cell embeddings (native multi-species)."""
    uce_dir = os.path.join(PROJECT_DIR, "models", "uce", "UCE")
    weights = os.path.join(PROJECT_DIR, "models", "uce", "33l_8ep_1024t_1280.torch")
    out_dir = os.path.join(RESULTS_DIR, "uce", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Ensure gene symbols (not Ensembl IDs) for UCE
    uce_input = _prepare_uce_input(adata_path, dataset_name)

    species_map = {"human": "human", "mouse": "mouse"}
    import subprocess
    # UCE uses relative paths (./model_files/), so must run from UCE dir
    cmd = [
        "python", "eval_single_anndata.py",
        "--adata_path", os.path.abspath(uce_input),
        "--dir", os.path.abspath(out_dir) + "/",
        "--species", species_map[species],
        "--model_loc", os.path.abspath(weights),
        "--batch_size", "25",
        "--nlayers", "33",
    ]
    print(f"  Running (cwd={uce_dir}): {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=uce_dir)

    # UCE saves embeddings in the output directory
    for f in os.listdir(out_dir):
        if f.endswith("_embeddings.npy") or f.endswith("_uce_adata.h5ad"):
            if f.endswith(".npy"):
                return np.load(os.path.join(out_dir, f))

    # Fallback: check for h5ad with X_uce
    import scanpy as sc
    for f in os.listdir(out_dir):
        if f.endswith(".h5ad"):
            adata = sc.read_h5ad(os.path.join(out_dir, f))
            if "X_uce" in adata.obsm:
                return np.array(adata.obsm["X_uce"])
    return None


def extract_transcriptformer(dataset_name, adata_path, species):
    """Extract TranscriptFormer embeddings (native multi-species)."""
    import scanpy as sc
    from src.models.run_transcriptformer_perturbation import (
        _build_inference_config,
        _get_embeddings,
    )

    adata = sc.read_h5ad(adata_path)

    # Ensure ensembl_id exists
    if "ensembl_id" not in adata.var.columns:
        from src.models.data_prep import get_symbol_to_ensembl
        symbol_map = get_symbol_to_ensembl()
        adata.var["ensembl_id"] = [symbol_map.get(g, "NA") for g in adata.var_names]
        adata = adata[:, adata.var["ensembl_id"] != "NA"].copy()

    # Ensure assay column exists (aux_vocab tokenizer expects it)
    if "assay" not in adata.obs.columns:
        adata.obs["assay"] = "single-cell RNA sequencing"

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    cfg = _build_inference_config()

    # For mouse data, TranscriptFormer may need species-specific config
    # tf-exemplar and tf-metazoa handle mouse natively
    return _get_embeddings(cfg, adata)


EXTRACTORS = {
    "geneformer": extract_geneformer,
    "scgpt": extract_scgpt,
    "uce": extract_uce,
    "transcriptformer": extract_transcriptformer,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract FM embeddings for temporal datasets (species-aware)"
    )
    parser.add_argument(
        "--model",
        choices=list(EXTRACTORS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--dataset",
        choices=list(TEMPORAL_DATASETS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    models = list(EXTRACTORS.keys()) if args.model == "all" else [args.model]
    datasets = list(TEMPORAL_DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    print("=== Temporal Embedding Extraction (Species-Aware) ===\n")

    for dataset_name in datasets:
        meta = TEMPORAL_DATASETS[dataset_name]
        species = meta["species"]
        adata_path = os.path.join(PROCESSED_DIR, meta["file"])

        if not os.path.exists(adata_path):
            print(f"SKIP {dataset_name}: not found at {adata_path}")
            continue

        print(f"\n--- {dataset_name} ({species}) ---")

        for model_name in models:
            support = MODEL_SPECIES_SUPPORT[model_name].get(species)

            if support is None:
                print(f"  {model_name}: N/A on {species} data (skipping)")
                continue

            out_dir = os.path.join(RESULTS_DIR, model_name, dataset_name)
            os.makedirs(out_dir, exist_ok=True)
            emb_path = os.path.join(out_dir, f"{dataset_name}_embeddings.npy")

            if os.path.exists(emb_path):
                print(f"  {model_name}: cached at {emb_path}")
                continue

            # Determine input data path
            if support == "ortholog" and species == "mouse":
                print(f"  {model_name}: remapping mouse->human orthologs...")
                remapped_adata = _remap_for_human_model(None, dataset_name)
                input_path = os.path.join(
                    PROCESSED_DIR, f"{dataset_name}_human_orthologs.h5ad"
                )
            else:
                input_path = adata_path

            print(f"  {model_name}: extracting embeddings ({support})...")

            try:
                if model_name in ["uce", "transcriptformer"]:
                    embs = EXTRACTORS[model_name](dataset_name, input_path, species)
                else:
                    embs = EXTRACTORS[model_name](dataset_name, input_path)

                if embs is not None:
                    np.save(emb_path, embs)
                    print(f"  {model_name}: {embs.shape} -> {emb_path}")
                else:
                    print(f"  {model_name}: extraction returned None")
            except Exception as e:
                print(f"  {model_name}: ERROR - {e}")

    print("\n=== Temporal embedding extraction complete ===")


if __name__ == "__main__":
    main()
