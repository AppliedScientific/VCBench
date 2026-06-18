"""
Dimension D: Cross-modal RNA→protein prediction via embedding probes.

For each foundation model:
1. Extract cell embeddings on train split (RNA only)
2. Fit ridge regression: embedding → 134 protein values
3. Extract embeddings on test split
4. Predict protein expression, evaluate (Pearson R, RMSE)

Baselines:
- PCA (50 components) + ridge
- scLinear (linear regression on full RNA features)
- Mean protein per cell type

All models participate — embedding probes are model-agnostic.
Run in: vcbench-analysis environment (probes are CPU-only after embedding extraction)
"""

import json
import os
import sys

import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_d")

FM_MODELS = ["geneformer", "scgpt", "uce", "transcriptformer", "state"]


def load_cite_data():
    """Load CITE-seq data, splitting into train/test if needed."""
    import scanpy as sc
    from sklearn.model_selection import train_test_split

    train_path = os.path.join(PROCESSED_DIR, "cite_train.h5ad")
    test_path = os.path.join(PROCESSED_DIR, "cite_test.h5ad")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return sc.read_h5ad(train_path), sc.read_h5ad(test_path)

    # Split from combined file
    adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, "neurips_cite_seq.h5ad"))
    train_idx, test_idx = train_test_split(
        np.arange(adata.n_obs), test_size=0.2, random_state=42
    )
    train = adata[train_idx].copy()
    test = adata[test_idx].copy()
    train.write_h5ad(train_path)
    test.write_h5ad(test_path)
    print(f"CITE-seq split: {train.n_obs} train, {test.n_obs} test")
    return train, test


def get_protein_matrices(train, test):
    """Extract CLR-normalized protein matrices.

    Handles different obsm key names across CITE-seq data versions:
    protein_clr, protein_expression, protein_counts, or protein.
    """
    protein_key = None
    for key in ["protein_clr", "protein_expression", "protein_counts", "protein"]:
        if key in train.obsm:
            protein_key = key
            break

    if protein_key is None:
        # Check all obsm keys for anything protein-related
        for key in train.obsm:
            if "protein" in key.lower() or "adt" in key.lower() or "cite" in key.lower():
                protein_key = key
                break

    if protein_key is None:
        raise KeyError(
            f"No protein matrix found in obsm. Available keys: {list(train.obsm.keys())}\n"
            "Expected one of: protein_clr, protein_expression, protein_counts, protein"
        )

    print(f"Using protein obsm key: '{protein_key}'")
    y_train = np.array(train.obsm[protein_key])
    y_test = np.array(test.obsm[protein_key])
    return y_train, y_test


def evaluate_and_save(model_name, y_pred, y_test, results_dir):
    """Evaluate predictions and save results."""
    from src.evaluation.metrics import evaluate_crossmodal

    results = evaluate_crossmodal(y_pred, y_test)
    results["model"] = model_name

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "crossmodal_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  {model_name}: PearsonR={results['mean_pearson_r']:.4f}, "
          f"RMSE={results['rmse']:.4f}")
    return results


def run_pca_ridge_baseline(train, test, y_train, y_test):
    """PCA (50 components) + ridge regression baseline."""
    print("\n[Baseline] PCA + Ridge...")
    model_dir = os.path.join(RESULTS_DIR, "pca_ridge")

    X_train = train.X.toarray() if issparse(train.X) else np.array(train.X)
    X_test = test.X.toarray() if issparse(test.X) else np.array(test.X)

    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_pca, y_train)
    y_pred = ridge.predict(X_test_pca)

    return evaluate_and_save("PCA + ridge", y_pred, y_test, model_dir)


def run_sclinear_baseline(train, test, y_train, y_test):
    """scLinear: linear regression on full RNA features (Hanhart et al. 2024)."""
    print("\n[Baseline] scLinear (full RNA features)...")
    model_dir = os.path.join(RESULTS_DIR, "sclinear")

    X_train = train.X.toarray() if issparse(train.X) else np.array(train.X)
    X_test = test.X.toarray() if issparse(test.X) else np.array(test.X)

    # scLinear is equivalent to ridge regression on all RNA features
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    return evaluate_and_save("scLinear", y_pred, y_test, model_dir)


def run_mean_celltype_baseline(train, test, y_train, y_test):
    """Mean protein expression per cell type (trivial baseline)."""
    print("\n[Baseline] Mean protein per cell type...")
    model_dir = os.path.join(RESULTS_DIR, "mean_celltype")

    if "cell_type" not in train.obs.columns:
        print("  SKIP: No cell_type column in train data")
        return None

    # Compute mean protein per cell type from train
    cell_types = train.obs["cell_type"].values
    unique_types = np.unique(cell_types)
    type_means = {}
    for ct in unique_types:
        mask = cell_types == ct
        type_means[ct] = y_train[mask].mean(axis=0)

    global_mean = y_train.mean(axis=0)

    # Predict test
    test_types = test.obs["cell_type"].values
    y_pred = np.stack([
        type_means.get(ct, global_mean) for ct in test_types
    ])

    return evaluate_and_save("Mean celltype", y_pred, y_test, model_dir)


def run_fm_probe(model_name, y_train, y_test, n_train, n_test):
    """Run ridge probe on pre-extracted foundation model embeddings."""
    model_dir = os.path.join(RESULTS_DIR, model_name)

    train_emb_path = os.path.join(model_dir, "cite_train_embeddings.npy")
    test_emb_path = os.path.join(model_dir, "cite_test_embeddings.npy")

    if not os.path.exists(train_emb_path) or not os.path.exists(test_emb_path):
        print(f"  SKIP {model_name}: embeddings not found")
        print(f"    Expected: {train_emb_path}")
        return None

    train_emb = np.load(train_emb_path)
    test_emb = np.load(test_emb_path)

    if train_emb.shape[0] != n_train:
        print(f"  SKIP {model_name}: train embedding mismatch "
              f"({train_emb.shape[0]} vs {n_train})")
        return None
    if test_emb.shape[0] != n_test:
        print(f"  SKIP {model_name}: test embedding mismatch "
              f"({test_emb.shape[0]} vs {n_test})")
        return None

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_emb, y_train)
    y_pred = ridge.predict(test_emb)

    return evaluate_and_save(model_name, y_pred, y_test, model_dir)


def main():
    print("=== Dimension D: Cross-Modal RNA→Protein Prediction ===\n")

    # Always re-run: FM embeddings may be added between runs
    train, test = load_cite_data()
    y_train, y_test = get_protein_matrices(train, test)
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Proteins: {y_train.shape[1]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    # Baselines (CPU-only)
    result = run_pca_ridge_baseline(train, test, y_train, y_test)
    if result:
        all_results["PCA + ridge"] = result

    result = run_sclinear_baseline(train, test, y_train, y_test)
    if result:
        all_results["scLinear"] = result

    result = run_mean_celltype_baseline(train, test, y_train, y_test)
    if result:
        all_results["Mean celltype"] = result

    # Foundation model probes
    print("\n--- Foundation Model Probes ---")
    for model_name in FM_MODELS:
        print(f"\n{model_name}:")
        result = run_fm_probe(
            model_name, y_train, y_test, train.n_obs, test.n_obs
        )
        if result:
            all_results[model_name] = result

    # Save combined results
    out_path = os.path.join(RESULTS_DIR, "all_crossmodal_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results: {out_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Model':<25} {'PearsonR':>10} {'RMSE':>10}")
    print("-" * 48)
    for name, r in all_results.items():
        print(f"{name:<25} {r['mean_pearson_r']:>10.4f} {r['rmse']:>10.4f}")


if __name__ == "__main__":
    main()
