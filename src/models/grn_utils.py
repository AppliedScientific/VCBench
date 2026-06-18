"""
Shared utilities for GRN inference evaluation across all models.
"""

import json
import os
import pickle

import pandas as pd


def evaluate_and_save_grn(predicted_edges_path, processed_dir, results_dir):
    """
    Evaluate predicted GRN edges against BEELINE hESC and TRRUST ground truth.

    Loads predicted edges CSV, filters to shared vocabulary with each ground
    truth set, computes AUROC/AUPRC/EPR, and saves JSON results.

    Args:
        predicted_edges_path: Path to CSV with columns [TF, target, score]
        processed_dir: Path to data/processed/ containing grn_ground_truth.pkl
        results_dir: Path to save grn_eval_*.json files
    """
    from src.evaluation.metrics import evaluate_grn

    if not os.path.exists(predicted_edges_path):
        print("No predicted edges found.")
        return {}

    pred_edges = pd.read_csv(predicted_edges_path)
    model_genes = set(pred_edges["TF"]) | set(pred_edges["target"])

    gt_path = os.path.join(processed_dir, "grn_ground_truth.pkl")
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)

    all_results = {}
    for gt_name, gt_edges in [
        ("beeline_hesc", gt.get("beeline_hesc", set())),
        ("trrust", gt.get("trrust", set())),
    ]:
        filtered_gt = {
            (tf, tgt) for tf, tgt in gt_edges
            if tf in model_genes and tgt in model_genes
        }
        tfs_in_gt = {tf for tf, _ in filtered_gt}
        all_pairs = [
            (tf, g) for tf in tfs_in_gt
            for g in model_genes if g != tf
        ]

        if not all_pairs:
            print(f"  {gt_name}: no overlapping vocabulary")
            continue

        result = evaluate_grn(pred_edges, filtered_gt, all_pairs)
        all_results[gt_name] = result
        print(
            f"  {gt_name}: AUROC={result['AUROC']:.3f}, "
            f"AUPRC={result['AUPRC']:.4f}, EPR={result['EPR']:.2f}"
        )

        with open(os.path.join(results_dir, f"grn_eval_{gt_name}.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

    return all_results
