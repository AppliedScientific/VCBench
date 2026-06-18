"""
kNN cross-species transfer evaluation for all models.

Loads pre-extracted embeddings and labels, runs kNN evaluation,
and saves results per model per tissue.
"""

import json
import os
import sys

import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

from src.evaluation.metrics import evaluate_cross_species

RESULTS_BASE = os.path.join(PROJECT_DIR, "results", "dim_b")
TISSUES = ["lung", "liver", "heart", "kidney", "brain"]
MODELS = ["geneformer", "scgpt", "uce", "transcriptformer"]

# Expected embedding dimensions per model
EXPECTED_DIMS = {
    "geneformer": 512,
    "scgpt": 512,
    "uce": 1280,
    "transcriptformer": None,  # Variable
}


def evaluate_model(model_name):
    """Evaluate a single model across all tissues."""
    model_dir = os.path.join(RESULTS_BASE, model_name)
    if not os.path.exists(model_dir):
        print(f"  SKIP {model_name}: no results directory")
        return None

    tissue_results = []
    for tissue in TISSUES:
        h_emb_path = os.path.join(model_dir, f"human_{tissue}_embeddings.npy")
        m_emb_path = os.path.join(model_dir, f"mouse_{tissue}_embeddings.npy")
        h_lbl_path = os.path.join(model_dir, f"human_{tissue}_labels.npy")
        m_lbl_path = os.path.join(model_dir, f"mouse_{tissue}_labels.npy")

        if not all(os.path.exists(p) for p in [h_emb_path, m_emb_path, h_lbl_path, m_lbl_path]):
            print(f"  SKIP {model_name}/{tissue}: missing files")
            continue

        h_emb = np.load(h_emb_path)
        m_emb = np.load(m_emb_path)
        h_labels = np.load(h_lbl_path, allow_pickle=True)
        m_labels = np.load(m_lbl_path, allow_pickle=True)

        # Verify embedding dimensions
        expected = EXPECTED_DIMS.get(model_name)
        if expected and h_emb.shape[1] != expected:
            print(f"  WARNING: {model_name} expected dim {expected}, got {h_emb.shape[1]}")

        result = evaluate_cross_species(h_emb, h_labels, m_emb, m_labels, k=5)
        result["tissue"] = tissue
        tissue_results.append(result)
        print(
            f"  {tissue}: macro_f1={result['macro_f1']:.4f}, "
            f"weighted_f1={result['weighted_f1']:.4f}, "
            f"shared_types={result['n_shared_types']}"
        )

    if not tissue_results:
        return None

    # Aggregate across tissues
    avg_macro = np.mean([r["macro_f1"] for r in tissue_results])
    avg_weighted = np.mean([r["weighted_f1"] for r in tissue_results])

    summary = {
        "per_tissue": tissue_results,
        "avg_macro_f1": avg_macro,
        "avg_weighted_f1": avg_weighted,
        "n_tissues": len(tissue_results),
    }

    out_path = os.path.join(model_dir, "crossspecies_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Average: macro_f1={avg_macro:.4f}, weighted_f1={avg_weighted:.4f}")
    return summary


def main():
    print("=== Cross-Species kNN Transfer Evaluation ===\n")

    all_results = {}
    for model in MODELS:
        print(f"\n{model}:")
        result = evaluate_model(model)
        if result:
            all_results[model] = result

    # Save combined results
    out_path = os.path.join(RESULTS_BASE, "all_crossspecies_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved: {out_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Model':<25} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 50)
    for model, r in all_results.items():
        print(f"{model:<25} {r['avg_macro_f1']:>10.4f} {r['avg_weighted_f1']:>12.4f}")


if __name__ == "__main__":
    main()
