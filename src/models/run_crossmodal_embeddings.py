"""
Dimension D: Extract foundation model embeddings on CITE-seq data.

For each model, extract cell embeddings on cite_train and cite_test,
saving as cite_train_embeddings.npy and cite_test_embeddings.npy.
These are consumed by run_crossmodal_probes.py for Ridge regression.

Run with --model flag to select which model to extract.
Each model needs its own conda environment.
"""

import argparse
import os
import sys

import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_d")


def _check_done(model_name):
    """Check if embeddings already extracted."""
    model_dir = os.path.join(RESULTS_DIR, model_name)
    train_path = os.path.join(model_dir, "cite_train_embeddings.npy")
    test_path = os.path.join(model_dir, "cite_test_embeddings.npy")
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  {model_name}: already extracted")
        return True
    return False


def extract_geneformer():
    """Extract Geneformer cell embeddings on CITE-seq data."""
    if _check_done("geneformer"):
        return

    import scanpy as sc
    import tempfile
    from geneformer import EmbExtractor, TranscriptomeTokenizer
    from src.models.data_prep import prepare_for_geneformer

    model_path = os.path.join(
        PROJECT_DIR, "models", "geneformer", "Geneformer", "Geneformer-V2-316M"
    )
    tokenized_dir = os.path.join(PROJECT_DIR, "data", "tokenized")
    os.makedirs(tokenized_dir, exist_ok=True)
    model_dir = os.path.join(RESULTS_DIR, "geneformer")
    os.makedirs(model_dir, exist_ok=True)

    for split_name in ["cite_train", "cite_test"]:
        emb_path = os.path.join(model_dir, f"{split_name}_embeddings.npy")
        if os.path.exists(emb_path):
            print(f"  {split_name}: cached")
            continue

        adata_path = os.path.join(PROCESSED_DIR, f"{split_name}.h5ad")
        gf_path = os.path.join(PROCESSED_DIR, f"{split_name}_gf.h5ad")
        prepare_for_geneformer(adata_path, gf_path)

        # Check cell_type
        _tmp = sc.read_h5ad(gf_path)
        has_ct = "cell_type" in _tmp.obs.columns
        del _tmp

        basename = split_name + "_gf"
        tok_path = os.path.join(tokenized_dir, f"{basename}.dataset")

        if not os.path.exists(tok_path):
            with tempfile.TemporaryDirectory() as tmpdir:
                os.symlink(os.path.abspath(gf_path), os.path.join(tmpdir, f"{basename}.h5ad"))
                custom = {"cell_type": "cell_type"} if has_ct else {}
                tk = TranscriptomeTokenizer(
                    custom_attr_name_dict=custom,
                    use_h5ad_index=True,
                )
                tk.tokenize_data(
                    tmpdir + "/", tokenized_dir + "/", basename, file_format="h5ad"
                )

        embex = EmbExtractor(
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cell",
            cell_emb_style="mean_pool",
            emb_layer=-1,
            forward_batch_size=100,
            nproc=4,
        )
        embs = embex.extract_embs(
            model_path, tok_path, model_dir + "/", split_name,
        )
        np.save(emb_path, np.array(embs))
        print(f"  {split_name}: {np.array(embs).shape} -> {emb_path}")


def extract_scgpt():
    """Extract scGPT cell embeddings on CITE-seq data."""
    if _check_done("scgpt"):
        return

    import scanpy as sc
    import torch
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scipy.sparse import issparse

    model_dir_out = os.path.join(RESULTS_DIR, "scgpt")
    os.makedirs(model_dir_out, exist_ok=True)

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

    for split_name in ["cite_train", "cite_test"]:
        emb_path = os.path.join(model_dir_out, f"{split_name}_embeddings.npy")
        if os.path.exists(emb_path):
            print(f"  {split_name}: cached")
            continue

        adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, f"{split_name}.h5ad"))
        gene_ids = np.array([
            vocab[g] if g in vocab else vocab["<pad>"] for g in adata.var_names
        ])
        X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)

        all_embs = []
        batch_size = 64
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch_x = torch.tensor(X[start:end], dtype=torch.float32).to(device)
                batch_ids = (
                    torch.tensor(gene_ids, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(end - start, -1)
                    .to(device)
                )
                output = model._encode(batch_ids, batch_x, src_key_padding_mask=None)
                embs = output.mean(dim=1)
                all_embs.append(embs.cpu().numpy())

        embs = np.concatenate(all_embs, axis=0)
        np.save(emb_path, embs)
        print(f"  {split_name}: {embs.shape} -> {emb_path}")


def extract_uce():
    """Extract UCE cell embeddings on CITE-seq data."""
    if _check_done("uce"):
        return

    import subprocess
    import scanpy as sc

    uce_dir = os.path.join(PROJECT_DIR, "models", "uce", "UCE")
    weights = os.path.join(PROJECT_DIR, "models", "uce", "33l_8ep_1024t_1280.torch")
    model_dir = os.path.join(RESULTS_DIR, "uce")
    os.makedirs(model_dir, exist_ok=True)

    for split_name in ["cite_train", "cite_test"]:
        emb_path = os.path.join(model_dir, f"{split_name}_embeddings.npy")
        if os.path.exists(emb_path):
            print(f"  {split_name}: cached")
            continue

        adata_path = os.path.join(PROCESSED_DIR, f"{split_name}.h5ad")
        out_dir = os.path.join(model_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "python", "eval_single_anndata.py",
            "--adata_path", os.path.abspath(adata_path),
            "--dir", os.path.abspath(out_dir) + "/",
            "--species", "human",
            "--model_loc", os.path.abspath(weights),
            "--batch_size", "25",
            "--nlayers", "33",
        ]
        subprocess.run(cmd, check=True, cwd=uce_dir)

        # Find output
        for f in os.listdir(out_dir):
            if f.endswith(".h5ad"):
                result = sc.read_h5ad(os.path.join(out_dir, f))
                if "X_uce" in result.obsm:
                    embs = np.array(result.obsm["X_uce"])
                    np.save(emb_path, embs)
                    print(f"  {split_name}: {embs.shape} -> {emb_path}")
                    break


def extract_transcriptformer():
    """Extract TranscriptFormer cell embeddings on CITE-seq data."""
    if _check_done("transcriptformer"):
        return

    import scanpy as sc
    from src.models.run_transcriptformer_perturbation import (
        _build_inference_config,
        _get_embeddings,
    )

    model_dir = os.path.join(RESULTS_DIR, "transcriptformer")
    os.makedirs(model_dir, exist_ok=True)

    cfg = _build_inference_config()

    for split_name in ["cite_train", "cite_test"]:
        emb_path = os.path.join(model_dir, f"{split_name}_embeddings.npy")
        if os.path.exists(emb_path):
            print(f"  {split_name}: cached")
            continue

        adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, f"{split_name}.h5ad"))

        # Ensure ensembl_id
        if "ensembl_id" not in adata.var.columns:
            from src.models.data_prep import get_symbol_to_ensembl
            symbol_map = get_symbol_to_ensembl()
            adata.var["ensembl_id"] = [symbol_map.get(g, "NA") for g in adata.var_names]
            adata = adata[:, adata.var["ensembl_id"] != "NA"].copy()

        # Ensure assay column
        if "assay" not in adata.obs.columns:
            adata.obs["assay"] = "single-cell RNA sequencing"

        # Use raw counts if available
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"].copy()

        embs = _get_embeddings(cfg, adata)
        embs = np.array(embs)
        np.save(emb_path, embs)
        print(f"  {split_name}: {embs.shape} -> {emb_path}")


EXTRACTORS = {
    "geneformer": extract_geneformer,
    "scgpt": extract_scgpt,
    "uce": extract_uce,
    "transcriptformer": extract_transcriptformer,
}


def main():
    parser = argparse.ArgumentParser(
        description="Extract FM embeddings on CITE-seq data for Dim D"
    )
    parser.add_argument(
        "--model",
        choices=list(EXTRACTORS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    print("=== Dimension D: CITE-seq Embedding Extraction ===\n")

    models = list(EXTRACTORS.keys()) if args.model == "all" else [args.model]
    for model_name in models:
        print(f"\n{model_name}:")
        try:
            EXTRACTORS[model_name]()
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
