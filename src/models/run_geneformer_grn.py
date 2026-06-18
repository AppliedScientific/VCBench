"""
Geneformer GRN inference via attention weight extraction.

Uses V2-316M (18 layers, 18 heads). Extracts attention weights from
layer 13 (0-indexed), following Kendiukhov (2026) TRRUST recovery analysis.
DO NOT use flash attention — it never materializes the attention matrix.

Run in: vcbench-geneformer environment
GPU: ~6-10 GB VRAM (316M requires ~2-3 GB more than 104M)
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
TOKENIZED_DIR = os.path.join(PROJECT_DIR, "data", "tokenized")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_c", "geneformer")
MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "geneformer", "Geneformer", "Geneformer-V2-316M"
)

# Use layer 13 (0-indexed) for regulatory signal per Kendiukhov (2026)
# V2-316M has 18 layers; layer 13 gave AUROC 0.694 against TRRUST
ATTENTION_LAYER = 13


def load_vocab_mapping():
    """
    Load Geneformer token ID -> gene SYMBOL mapping.

    Geneformer V2 vocabulary maps Ensembl IDs to token IDs. We need to
    convert to gene symbols so we can match against TRRUST TF names.

    Pipeline: token_id -> Ensembl ID -> gene symbol
    Uses ensembl_mapping_dict (symbol -> Ensembl) inverted to (Ensembl -> symbol).
    """
    import pickle

    dict_path = os.path.join(MODEL_PATH, "gene_name_id_dict.pkl")
    if not os.path.exists(dict_path):
        # Fallback: try loading from geneformer package
        from geneformer import TranscriptomeTokenizer
        tk = TranscriptomeTokenizer({})
        gene_to_id = tk.gene_token_dict
    else:
        with open(dict_path, "rb") as f:
            gene_to_id = pickle.load(f)

    # gene_to_id maps Ensembl IDs -> token IDs
    id_to_ensembl = {v: k for k, v in gene_to_id.items()}
    print(f"Vocabulary: {len(id_to_ensembl)} gene tokens loaded (Ensembl IDs)")

    # Load Ensembl -> symbol mapping
    # ensembl_mapping_dict maps symbol -> Ensembl; we invert it
    import geneformer
    gf_dir = os.path.dirname(geneformer.__file__)
    ensembl_map_path = os.path.join(gf_dir, "ensembl_mapping_dict_gc104M.pkl")
    if not os.path.exists(ensembl_map_path):
        print("WARNING: ensembl_mapping_dict not found, returning Ensembl IDs")
        return id_to_ensembl

    with open(ensembl_map_path, "rb") as f:
        symbol_to_ensembl = pickle.load(f)

    # Invert: Ensembl -> symbol (take first match for duplicates)
    ensembl_to_symbol = {}
    for symbol, ensembl_id in symbol_to_ensembl.items():
        if ensembl_id not in ensembl_to_symbol:
            ensembl_to_symbol[ensembl_id] = symbol

    # Map token_id -> symbol
    id_to_gene = {}
    mapped = 0
    for token_id, ensembl_id in id_to_ensembl.items():
        symbol = ensembl_to_symbol.get(ensembl_id)
        if symbol:
            id_to_gene[token_id] = symbol
            mapped += 1
        # else: skip unmappable tokens

    print(f"Mapped {mapped}/{len(id_to_ensembl)} tokens to gene symbols")
    return id_to_gene


def _tokenize_beeline_hesc():
    """Tokenize BEELINE hESC data for Geneformer if not already done."""
    import tempfile

    import scanpy as sc

    dataset_path = os.path.join(TOKENIZED_DIR, "beeline_hesc_gf.dataset")
    if os.path.exists(dataset_path):
        return dataset_path

    os.makedirs(TOKENIZED_DIR, exist_ok=True)

    hesc_path = os.path.join(PROCESSED_DIR, "beeline_hesc.h5ad")
    if not os.path.exists(hesc_path):
        print(f"BEELINE hESC not found: {hesc_path}")
        return None

    adata = sc.read_h5ad(hesc_path)

    # Add ensembl_id if missing (Geneformer needs it)
    if "ensembl_id" not in adata.var.columns:
        if adata.var_names[0].startswith("ENSG"):
            adata.var["ensembl_id"] = adata.var_names.values
        else:
            from src.models.data_prep import get_symbol_to_ensembl
            symbol_map = get_symbol_to_ensembl()
            adata.var["ensembl_id"] = [symbol_map.get(g, "NA") for g in adata.var_names]
            adata = adata[:, adata.var["ensembl_id"] != "NA"].copy()

    # Ensure raw counts
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    # Geneformer needs n_counts in obs
    if "n_counts" not in adata.obs.columns:
        from scipy.sparse import issparse
        X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
        adata.obs["n_counts"] = X.sum(axis=1)

    # Save to temp and tokenize
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "beeline_hesc_gf.h5ad")
        adata.write_h5ad(tmp_path)

        from geneformer import TranscriptomeTokenizer
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"cell_type": "cell_type"}
            if "cell_type" in adata.obs.columns else {},
            nproc=4,
        )
        tk.tokenize_data(
            tmpdir + "/",
            TOKENIZED_DIR + "/",
            "beeline_hesc_gf",
            file_format="h5ad",
        )

    print(f"Tokenized BEELINE hESC: {dataset_path}")
    return dataset_path


def step1_extract_and_build_edges(id_to_gene, tf_set):
    """
    Extract layer-13 attention and build TF→target edge scores in one streaming pass.

    Memory-efficient design:
    - Uses a forward hook on layer 13 to capture ONLY that layer's attention,
      avoiding output_attentions=True which materializes all 18 layers (~5GB/batch).
    - Processes edges incrementally per batch — never accumulates attention matrices.
    - Peak memory: model (~1.2GB) + one layer's attention (~300MB) + edge dict.
      Total ~3-4GB, well within 16GB on CPU.

    Falls back to CPU if GPU is unavailable/OOM.
    """
    from transformers import BertForMaskedLM

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Tokenize if needed, then load (do this BEFORE loading model to save memory)
    dataset_path = _tokenize_beeline_hesc()
    if dataset_path is None:
        return None

    from datasets import load_from_disk
    dataset = load_from_disk(dataset_path)

    # Try GPU with float16 first, fall back to CPU
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            model = BertForMaskedLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
            model = model.to(device)
            batch_size = 4
            print(f"Model loaded on {device} (float16)")
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            print("GPU OOM, falling back to CPU")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            model = BertForMaskedLM.from_pretrained(MODEL_PATH)
            model = model.to(device)
            batch_size = 1
    else:
        print("No GPU available, using CPU (float32, batch_size=1)")
        device = torch.device("cpu")
        model = BertForMaskedLM.from_pretrained(MODEL_PATH)
        model = model.to(device)
        batch_size = 1

    model.eval()

    # Patch ONLY layer 13 to compute attention weights.
    # output_attentions=True propagates to ALL 18 layers, each materializing
    # (batch, heads, seq, seq) float32 attention matrices. With 18 layers that's
    # ~19GB at batch=2, seq=2048. Instead, we wrap layer 13's self-attention to
    # force output_attentions=True just for that layer, while the other 17 use
    # efficient SDPA (no attention matrix materialized).
    # Peak memory: model (1.2GB) + 1 layer attention (~300MB) ≈ 2GB.
    layer_13_self = model.bert.encoder.layer[ATTENTION_LAYER].attention.self
    _original_forward = layer_13_self.forward
    captured_attention = {}

    def _patched_forward(*args, **kwargs):
        kwargs["output_attentions"] = True
        result = _original_forward(*args, **kwargs)
        # result = (context_layer, attention_probs)
        if len(result) > 1 and result[1] is not None:
            captured_attention["attn"] = result[1].detach()
        return result

    layer_13_self.forward = _patched_forward

    # Accumulate edge scores incrementally (no attention matrix storage)
    # Resume from partial checkpoint if available
    partial_path = os.path.join(RESULTS_DIR, "edge_scores_partial.pkl")
    start_i = 0
    if os.path.exists(partial_path):
        import pickle as _pkl
        with open(partial_path, "rb") as _f:
            ckpt = _pkl.load(_f)
        gene_pair_scores = ckpt["scores"]
        gene_pair_counts = ckpt["counts"]
        start_i = ckpt["batch_idx"] + batch_size  # Resume from next batch
        print(f"  Resuming from checkpoint: {len(gene_pair_scores)} edges, starting at cell {start_i}")
    else:
        gene_pair_scores = {}
        gene_pair_counts = {}

    print(f"Extracting attention from {len(dataset)} cells (batch_size={batch_size}, start={start_i})...")
    for i in range(start_i, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        with torch.no_grad():
            # Run forward pass — only layer 13 computes attention (via patch)
            # output_attentions=False at model level so other 17 layers use SDPA
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
            )

        # Get layer 13 attention from patch, average across heads
        if "attn" not in captured_attention:
            print(f"  WARNING: No attention captured at batch {i}")
            continue

        # (batch, heads, seq, seq) -> (batch, seq, seq)
        avg_attn = captured_attention["attn"].mean(dim=1).cpu().float().numpy()
        batch_ids = input_ids.cpu().numpy()

        # Build edges incrementally for this batch
        for cell_idx in range(avg_attn.shape[0]):
            attn = avg_attn[cell_idx]
            genes = [id_to_gene.get(int(tid)) for tid in batch_ids[cell_idx]]

            for gi, gene_i in enumerate(genes):
                if gene_i is None or gene_i not in tf_set:
                    continue
                for gj, gene_j in enumerate(genes):
                    if gene_j is None or gi == gj:
                        continue
                    pair = (gene_i, gene_j)
                    gene_pair_scores[pair] = gene_pair_scores.get(pair, 0.0) + float(attn[gi, gj])
                    gene_pair_counts[pair] = gene_pair_counts.get(pair, 0) + 1

        # Free memory immediately
        del avg_attn, batch_ids
        captured_attention.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import gc; gc.collect()

        batch_num = i // batch_size
        if batch_num % 25 == 0:
            print(f"  Processed {min(i + batch_size, len(dataset))}/{len(dataset)} cells, "
                  f"{len(gene_pair_scores)} edges so far")
            sys.stdout.flush()
            # Incremental checkpoint every 25 batches
            if gene_pair_scores:
                partial_path = os.path.join(RESULTS_DIR, "edge_scores_partial.pkl")
                import pickle as _pkl
                with open(partial_path, "wb") as _f:
                    _pkl.dump({"scores": gene_pair_scores, "counts": gene_pair_counts, "batch_idx": i}, _f)
                print(f"  Checkpoint saved: {len(gene_pair_scores)} edges at cell {i}")
                sys.stdout.flush()

    layer_13_self.forward = _original_forward  # Restore original

    # Clean up partial checkpoint
    if os.path.exists(partial_path):
        os.remove(partial_path)

    # Build edge DataFrame: average attention per pair across all cells
    edges = []
    for (tf, target), total_score in gene_pair_scores.items():
        count = gene_pair_counts[(tf, target)]
        edges.append({
            "TF": tf,
            "target": target,
            "score": total_score / count,
        })

    if not edges:
        print("WARNING: No TF-target edges found. Check vocabulary mapping and TF list.")
        df = pd.DataFrame(columns=["TF", "target", "score"])
    else:
        df = pd.DataFrame(edges).sort_values("score", ascending=False)

    out_path = os.path.join(RESULTS_DIR, "predicted_edges.csv")
    df.to_csv(out_path, index=False)
    print(f"Edge list: {len(df)} edges -> {out_path}")
    return df


def step3_evaluate():
    """Evaluate GRN predictions against ground truth."""
    from src.models.grn_utils import evaluate_and_save_grn

    evaluate_and_save_grn(
        os.path.join(RESULTS_DIR, "predicted_edges.csv"),
        PROCESSED_DIR,
        RESULTS_DIR,
    )


def main():
    import pickle

    print("=== Geneformer GRN Inference (V2-316M, layer 13) ===")

    # Checkpoint: skip if final results exist
    final_results = os.path.join(RESULTS_DIR, "grn_results.json")
    if os.path.exists(final_results):
        print(f"Already complete: {final_results}")
        return

    # Load vocabulary and TF list upfront
    print("\nLoading vocabulary...")
    id_to_gene = load_vocab_mapping()

    print("Loading TF list from ground truth...")
    gt_path = os.path.join(PROCESSED_DIR, "grn_ground_truth.pkl")
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)
    tf_set = gt.get("trrust_tfs", set())
    print(f"  {len(tf_set)} TFs loaded from TRRUST")

    print("\n[1/2] Extracting attention + building edges (streaming)...")
    step1_extract_and_build_edges(id_to_gene, tf_set)

    print("\n[2/2] Evaluating...")
    step3_evaluate()

    print("\n=== Geneformer GRN complete ===")


if __name__ == "__main__":
    main()
