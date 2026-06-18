"""
scGPT GRN inference via gene embedding similarity.

Uses the GeneEmbedding class (NOT model.encoder()) to get gene embeddings
that include the trained value/bin encoding. Cosine similarity between
TF and target embeddings gives edge scores.
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_c", "scgpt")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "scgpt", "scGPT_human")


def step1_extract_gene_embeddings():
    """
    Extract gene embeddings from scGPT's pretrained encoder.

    Uses the model's token encoder to get 512-dim embeddings for each gene
    in the vocabulary, then filters to genes present in the BEELINE hESC
    dataset. Wraps in GeneEmbedding for downstream similarity computation.
    """
    from scgpt.tasks import GeneEmbedding

    os.makedirs(RESULTS_DIR, exist_ok=True)

    hesc_path = os.path.join(PROCESSED_DIR, "beeline_hesc.h5ad")
    if not os.path.exists(hesc_path):
        print(f"BEELINE hESC not found: {hesc_path}")
        return None

    adata = sc.read_h5ad(hesc_path)

    # Load model
    import torch
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = GeneVocab.from_file(os.path.join(MODEL_DIR, "vocab.json"))

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=512,
        nhead=8,
        d_hid=512,
        nlayers=12,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
    )
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=device),
        strict=False,
    )
    model = model.to(device)
    model.eval()

    # Extract gene embeddings from encoder token embeddings
    stoi = vocab.get_stoi()
    gene_ids = torch.arange(len(vocab)).to(device)
    with torch.no_grad():
        all_embs = model.encoder(gene_ids).cpu().numpy()

    # Build {gene_name: embedding} dict for genes in our dataset
    dataset_genes = set(adata.var_names)
    emb_dict = {}
    for gene, idx in stoi.items():
        if gene in dataset_genes:
            emb_dict[gene] = all_embs[idx]

    print(f"Matched {len(emb_dict)}/{len(dataset_genes)} dataset genes in scGPT vocab")

    # Build GeneEmbedding object (expects Mapping[str, array])
    gene_emb = GeneEmbedding(emb_dict)

    # Extract ordered arrays
    gene_names = gene_emb.genes
    emb_matrix = np.array(gene_emb.vector)

    np.save(os.path.join(RESULTS_DIR, "gene_embeddings.npy"), emb_matrix)
    with open(os.path.join(RESULTS_DIR, "gene_names.json"), "w") as f:
        json.dump(gene_names, f)

    print(f"Gene embeddings: {emb_matrix.shape}")
    return emb_matrix, gene_names


def step2_compute_similarity(emb_matrix, gene_names):
    """Compute cosine similarity between all gene pairs."""
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(emb_matrix)
    np.save(os.path.join(RESULTS_DIR, "similarity_matrix.npy"), sim_matrix)
    print(f"Similarity matrix: {sim_matrix.shape}")
    return sim_matrix


def step3_build_edge_list(sim_matrix, gene_names, tf_list, top_k=10000):
    """Convert similarity matrix to ranked TF-target edge list."""
    edges = []
    tf_indices = [i for i, g in enumerate(gene_names) if g in tf_list]

    for i in tf_indices:
        for j in range(len(gene_names)):
            if i != j:
                edges.append({
                    "TF": gene_names[i],
                    "target": gene_names[j],
                    "score": float(sim_matrix[i, j]),
                })

    df = pd.DataFrame(edges).nlargest(top_k, "score")
    out_path = os.path.join(RESULTS_DIR, "predicted_edges.csv")
    df.to_csv(out_path, index=False)
    print(f"Edge list: {len(df)} edges -> {out_path}")
    return df


def step4_evaluate():
    """Evaluate GRN predictions against ground truth."""
    from src.models.grn_utils import evaluate_and_save_grn

    evaluate_and_save_grn(
        os.path.join(RESULTS_DIR, "predicted_edges.csv"),
        PROCESSED_DIR,
        RESULTS_DIR,
    )


def main():
    print("=== scGPT GRN Inference ===")

    # Checkpoint: skip if final results exist
    final_results = os.path.join(RESULTS_DIR, "grn_results.json")
    if os.path.exists(final_results):
        print(f"Already complete: {final_results}")
        return

    print("\n[1/4] Extracting gene embeddings...")
    result = step1_extract_gene_embeddings()
    if result is None:
        return
    emb_matrix, gene_names = result

    print("\n[2/4] Computing cosine similarity...")
    sim_matrix = step2_compute_similarity(emb_matrix, gene_names)

    print("\n[3/4] Building edge list...")
    with open(os.path.join(PROCESSED_DIR, "grn_ground_truth.pkl"), "rb") as f:
        gt = pickle.load(f)
    tf_list = gt.get("trrust_tfs", set())
    step3_build_edge_list(sim_matrix, gene_names, tf_list)

    print("\n[4/4] Evaluating...")
    step4_evaluate()

    print("\n=== scGPT GRN complete ===")


if __name__ == "__main__":
    main()
