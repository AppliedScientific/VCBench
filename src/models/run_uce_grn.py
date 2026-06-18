"""
UCE GRN inference via ESM-2 protein embedding similarity.

UCE uses ESM-2 protein sequence embeddings (5120-dim) as gene tokens.
For GRN inference, we compute cosine similarity between TF and target
gene embeddings. Genes with similar protein structures/functions have
higher similarity scores.

This is analogous to scGPT's gene embedding cosine similarity approach,
but uses pre-trained protein language model features instead of
expression-learned gene tokens.

Run in: vcbench-pt118 environment (CPU-only, no GPU needed)
"""

import os, sys, pickle, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'dim_c', 'uce')
EMB_PATH = os.path.join(
    PROJECT_DIR, 'models', 'uce', 'UCE', 'model_files',
    'protein_embeddings', 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt'
)


def build_edges():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    out_path = os.path.join(RESULTS_DIR, 'predicted_edges.csv')
    if os.path.exists(out_path):
        print(f'Already done: {out_path}')
        return pd.read_csv(out_path)
    
    # Load ground truth TF list
    gt_path = os.path.join(PROCESSED_DIR, 'grn_ground_truth.pkl')
    with open(gt_path, 'rb') as f:
        gt = pickle.load(f)
    tf_set = gt.get('trrust_tfs', set())
    print(f'TFs from TRRUST: {len(tf_set)}')
    
    # Load BEELINE hESC gene list (vocabulary intersection)
    import scanpy as sc
    hesc = sc.read_h5ad(os.path.join(PROCESSED_DIR, 'beeline_hesc.h5ad'))
    dataset_genes = set(hesc.var_names)
    print(f'BEELINE hESC genes: {len(dataset_genes)}')
    
    # Load ESM-2 gene embeddings
    gene_embs = torch.load(EMB_PATH, map_location='cpu')
    print(f'ESM-2 embeddings: {len(gene_embs)} genes, {next(iter(gene_embs.values())).shape[0]}-dim')
    
    # Filter to genes in both ESM-2 vocabulary and dataset
    shared_genes = sorted(dataset_genes & set(gene_embs.keys()))
    shared_tfs = sorted(tf_set & set(shared_genes))
    print(f'Shared genes: {len(shared_genes)}, Shared TFs: {len(shared_tfs)}')
    
    if not shared_tfs:
        raise ValueError('No TFs found in shared vocabulary')
    
    # Build embedding matrix
    gene_list = shared_genes
    gene_idx = {g: i for i, g in enumerate(gene_list)}
    emb_matrix = np.stack([gene_embs[g].numpy() for g in gene_list])
    
    # Compute cosine similarity: TFs vs all genes
    tf_indices = [gene_idx[tf] for tf in shared_tfs]
    tf_embs = emb_matrix[tf_indices]
    
    sim_matrix = cosine_similarity(tf_embs, emb_matrix)  # (n_tfs, n_genes)
    
    # Build edge list
    edges = []
    for i, tf in enumerate(shared_tfs):
        for j, target in enumerate(gene_list):
            if tf != target:
                edges.append({
                    'TF': tf,
                    'target': target,
                    'score': float(abs(sim_matrix[i, j])),
                })
    
    df = pd.DataFrame(edges).sort_values('score', ascending=False)
    df.to_csv(out_path, index=False)
    print(f'Edge list: {len(df)} edges -> {out_path}')
    return df


def evaluate():
    from src.models.grn_utils import evaluate_and_save_grn
    evaluate_and_save_grn(
        os.path.join(RESULTS_DIR, 'predicted_edges.csv'),
        PROCESSED_DIR,
        RESULTS_DIR,
    )


def main():
    print('=== UCE GRN Inference (ESM-2 protein embedding similarity) ===')
    
    final = os.path.join(RESULTS_DIR, 'grn_eval_trrust.json')
    if os.path.exists(final):
        print(f'Already complete: {final}')
        return
    
    print('\n[1/2] Building edges from ESM-2 embeddings...')
    build_edges()
    
    print('\n[2/2] Evaluating...')
    evaluate()
    
    print('\n=== UCE GRN complete ===')


if __name__ == '__main__':
    main()
