"""
Dimension E: Temporal ordering via embedding-based pseudotime.

For each foundation model, for each dataset (sci-fate, Weinreb):
1. Extract cell embeddings
2. Build kNN graph in embedding space
3. Compute diffusion pseudotime (DPT) with root = earliest-time cell
4. Compute Kendall's tau-b between DPT and true_time
5. Compute kNN balanced accuracy for timepoint classification

Baselines: PCA + DPT on log-normalized HVG counts.

Critical context: Zhou et al. (2026, bioRxiv) found that zero-shot FM embeddings
underperform HVG baselines on temporal tasks. Expect poor FM performance.
That is the finding, not a bug.

Run in: vcbench-analysis environment (probes are CPU-only after embedding extraction)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_e")

DATASETS = ["weinreb", "sci_fate"]  # LARRY primary (mouse), sci-fate (human)

# Species-aware model participation.
# Weinreb LARRY is mouse data. Human-only models need ortholog mapping first
# (done in run_temporal_embeddings.py). Arc State cannot process mouse at all.
DATASET_MODELS = {
    "sci_fate": ["geneformer", "scgpt", "uce", "transcriptformer", "state"],
    "weinreb": ["geneformer", "scgpt", "uce", "transcriptformer"],  # State N/A on mouse
}
FM_MODELS = ["geneformer", "scgpt", "uce", "transcriptformer", "state"]  # full list for reference


def _parse_true_time(obs_series):
    """Parse true_time obs column to numeric, handling string suffixes like '4h', '6h'.

    Strips common time-unit suffixes (h, d, hr, hrs, day, days, min) before
    numeric conversion so that pd.to_numeric doesn't produce all-NaN.
    """
    values = obs_series.astype(str).str.strip()
    # Strip common time-unit suffixes
    values = values.str.replace(r'\s*(hours?|hrs?|h|days?|d|min)$', '', regex=True)
    return pd.to_numeric(values, errors="coerce").values.astype(float)


def _filter_nan_time(embeddings, true_time):
    """Remove cells with NaN true_time values."""
    valid = ~np.isnan(true_time)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped}/{len(true_time)} cells with NaN true_time")
    if valid.sum() == 0:
        raise ValueError("All true_time values are NaN after parsing. Check column format.")
    return embeddings[valid], true_time[valid]


def run_pca_dpt_baseline(dataset_name):
    """PCA (50 components) + DPT baseline."""
    print(f"\n[Baseline] PCA + DPT on {dataset_name}...")
    model_dir = os.path.join(RESULTS_DIR, "pca_dpt", dataset_name)
    os.makedirs(model_dir, exist_ok=True)

    adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, f"{dataset_name}.h5ad"))
    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    embeddings = pca.fit_transform(X)

    true_time = _parse_true_time(adata.obs["true_time"])
    embeddings, true_time = _filter_nan_time(embeddings, true_time)

    from src.evaluation.metrics import evaluate_temporal
    results = evaluate_temporal(embeddings, true_time, k=15)
    results["model"] = "PCA + DPT"
    results["dataset"] = dataset_name

    with open(os.path.join(model_dir, "temporal_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  KendallTau={results['kendall_tau']:.4f}, "
          f"kNN_BalAcc={results['knn_balanced_accuracy']:.4f}")
    return results


def run_fm_temporal_probe(model_name, dataset_name):
    """Evaluate temporal ordering from pre-extracted FM embeddings."""
    model_dir = os.path.join(RESULTS_DIR, model_name, dataset_name)

    emb_path = os.path.join(model_dir, f"{dataset_name}_embeddings.npy")
    if not os.path.exists(emb_path):
        # Also check parent dir
        emb_path = os.path.join(RESULTS_DIR, model_name, f"{dataset_name}_embeddings.npy")
    if not os.path.exists(emb_path):
        print(f"  SKIP {model_name}/{dataset_name}: embeddings not found")
        return None

    os.makedirs(model_dir, exist_ok=True)

    embeddings = np.load(emb_path)
    adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, f"{dataset_name}.h5ad"))
    true_time = _parse_true_time(adata.obs["true_time"])

    if embeddings.shape[0] != len(true_time):
        print(f"  SKIP {model_name}/{dataset_name}: embedding count mismatch "
              f"({embeddings.shape[0]} vs {len(true_time)} cells). "
              f"Delete {emb_path} and re-extract.")
        return None

    embeddings, true_time = _filter_nan_time(embeddings, true_time)

    from src.evaluation.metrics import evaluate_temporal
    results = evaluate_temporal(embeddings, true_time, k=15)
    results["model"] = model_name
    results["dataset"] = dataset_name

    with open(os.path.join(model_dir, "temporal_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  KendallTau={results['kendall_tau']:.4f}, "
          f"kNN_BalAcc={results['knn_balanced_accuracy']:.4f}")
    return results


def main():
    print("=== Dimension E: Temporal Ordering ===\n")

    # Checkpoint: skip entirely if final results exist
    final_results = os.path.join(RESULTS_DIR, "all_temporal_results.json")
    if os.path.exists(final_results):
        print(f"Already complete: {final_results}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for dataset_name in DATASETS:
        data_path = os.path.join(PROCESSED_DIR, f"{dataset_name}.h5ad")
        if not os.path.exists(data_path):
            print(f"SKIP {dataset_name}: not found at {data_path}")
            continue

        print(f"\n--- {dataset_name} ---")

        # PCA baseline
        result = run_pca_dpt_baseline(dataset_name)
        if result:
            all_results[f"PCA_DPT/{dataset_name}"] = result

        # Foundation models (species-aware participation)
        eligible_models = DATASET_MODELS.get(dataset_name, FM_MODELS)
        for model_name in eligible_models:
            print(f"\n{model_name} / {dataset_name}:")
            result = run_fm_temporal_probe(model_name, dataset_name)
            if result:
                all_results[f"{model_name}/{dataset_name}"] = result

    # Save combined
    out_path = os.path.join(RESULTS_DIR, "all_temporal_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results: {out_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Model/Dataset':<35} {'KendallTau':>12} {'kNN_BalAcc':>12}")
    print("-" * 62)
    for key, r in all_results.items():
        print(f"{key:<35} {r['kendall_tau']:>12.4f} {r['knn_balanced_accuracy']:>12.4f}")


if __name__ == "__main__":
    main()
