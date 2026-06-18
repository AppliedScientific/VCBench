"""
Final results assembly.

Collects all model evaluation results and fills the
results table. Produces table1_final.csv with all non-N/A cells filled.

Dim A results are reported separately as PP-compositional (Norman)
and PP-held-out (Replogle) per the capability matrix.

Run in: vcbench-analysis environment (no GPU needed)
"""

import json
import os
import sys

import pandas as pd

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")


def load_json_safe(path):
    """Load JSON file, return None if missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# Shared model display name → directory name mapping
FM_MODEL_MAP = {
    "Geneformer V2-316M": "geneformer",
    "scGPT (fine-tuned)": "scgpt",
    "UCE 33-layer": "uce",
    "TranscriptFormer": "transcriptformer",
    "Arc State": "state",
}

# Models participating in Dim A perturbation prediction
DIM_A_MODELS = {
    "Geneformer V2-316M": "geneformer",
    "scGPT (fine-tuned)": "scgpt",
    "Arc State": "state",
    "TranscriptFormer": "transcriptformer",
}

# Dim A datasets: reported separately as PP-compositional / PP-held-out
DIM_A_DATASETS = {
    "norman": "A_norm",   # PP-compositional (combinatorial CRISPR)
    "replogle": "A_repl",  # PP-held-out (genome-scale Perturb-seq)
}


def _extract_scalar(val):
    """Extract scalar from cell-eval JSON values (may be nested like {"0": 0.5})."""
    if isinstance(val, dict):
        # cell-eval wraps scalars in {"0": value}
        return list(val.values())[0]
    return val


# Map table metric names -> cell_eval JSON keys
CELL_EVAL_KEY_MAP = {
    "PDS": "mean_pearson_r_delta",
    "DES": "mean_direction_score",
    "MAE": "mean_mse_delta",
}


def collect_dim_a_results():
    """
    Collect Dimension A (perturbation) results from cell-eval outputs.

    Cell-eval JSONs use keys like "mean_pearson_r_delta" with values
    wrapped as {"0": scalar}. We map these to our table column names.
    """
    results = {}

    for display_name, dir_name in DIM_A_MODELS.items():
        data = load_json_safe(
            os.path.join(RESULTS_DIR, "dim_a", dir_name, "cell_eval_results.json")
        )
        if data:
            for table_key, json_key in CELL_EVAL_KEY_MAP.items():
                val = data.get(json_key)
                if val is not None:
                    results[(display_name, f"A:{table_key}")] = _extract_scalar(val)
            print(f"  Dim A {display_name}: loaded")

    # Also check for baseline results
    for baseline_name in ["additive", "mean", "no_change"]:
        display_map = {
            "additive": "Additive baseline",
            "mean": "Mean baseline",
            "no_change": "No-change baseline",
        }
        data = load_json_safe(
            os.path.join(RESULTS_DIR, "dim_a", "baselines", f"{baseline_name}_results.json")
        )
        if data:
            display = display_map[baseline_name]
            for table_key, json_key in CELL_EVAL_KEY_MAP.items():
                val = data.get(json_key) or data.get(f"{table_key.lower()}_mean")
                if val is not None:
                    results[(display, f"A:{table_key}")] = _extract_scalar(val)
            print(f"  Dim A {display}: loaded")

    return results


def collect_dim_b_results():
    """Collect Dimension B (cross-species) results."""
    results = {}

    # Try combined results first
    all_data = load_json_safe(
        os.path.join(RESULTS_DIR, "dim_b", "all_crossspecies_results.json")
    )
    if all_data:
        model_map = {v: k for k, v in FM_MODEL_MAP.items() if k != "Arc State"}
        for dir_name, display_name in model_map.items():
            data = all_data.get(dir_name)
            if data:
                results[(display_name, "B:MacroF1")] = data.get("avg_macro_f1", data.get("macro_f1"))
                results[(display_name, "B:WeightedF1")] = data.get("avg_weighted_f1", data.get("weighted_f1"))
                print(f"  Dim B {display_name}: loaded")
    else:
        print("  Dim B: no combined results file found")

    # Also check baselines
    baseline_data = load_json_safe(
        os.path.join(RESULTS_DIR, "dim_b", "baselines", "pca_knn_results.json")
    )
    if baseline_data:
        # Average across tissues
        macro_f1s = [v.get("macro_f1", 0) for v in baseline_data.values() if isinstance(v, dict)]
        weighted_f1s = [v.get("weighted_f1", 0) for v in baseline_data.values() if isinstance(v, dict)]
        if macro_f1s:
            import numpy as np
            results[("PCA + kNN", "B:MacroF1")] = float(np.mean(macro_f1s))
            results[("PCA + kNN", "B:WeightedF1")] = float(np.mean(weighted_f1s))
            print(f"  Dim B PCA + kNN baseline: loaded")

    return results


def collect_dim_c_results():
    """Collect Dimension C (GRN) results."""
    results = {}

    # UCE and Arc State are N/A for Dim C
    grn_models = {
        "Geneformer V2-316M": "geneformer",
        "scGPT (fine-tuned)": "scgpt",
        "TranscriptFormer": "transcriptformer",
    }

    for display_name, dir_name in grn_models.items():
        # Prefer TRRUST evaluation, fall back to BEELINE hESC
        for gt_name in ["trrust", "beeline_hesc"]:
            data = load_json_safe(
                os.path.join(RESULTS_DIR, "dim_c", dir_name, f"grn_eval_{gt_name}.json")
            )
            if data and not pd.isna(data.get("AUROC", float("nan"))):
                results[(display_name, "C:AUROC")] = data["AUROC"]
                results[(display_name, "C:AUPRC")] = data["AUPRC"]
                results[(display_name, "C:EPR")] = data["EPR"]
                print(f"  Dim C {display_name}: loaded ({gt_name})")
                break

    return results


def collect_dim_d_results():
    """Collect Dimension D (cross-modal RNA→protein) results."""
    results = {}

    all_data = load_json_safe(
        os.path.join(RESULTS_DIR, "dim_d", "all_crossmodal_results.json")
    )
    if not all_data:
        print("  Dim D: no combined results file found")
        return results

    # Map result keys to display names (JSON uses dir names or display names)
    key_to_display = {v: k for k, v in FM_MODEL_MAP.items()}
    key_to_display.update({
        "PCA + ridge": "PCA + ridge",
        "scLinear": "scLinear",
        "Mean celltype": "Mean baseline",
    })

    for key, data in all_data.items():
        display_name = key_to_display.get(key, key)
        if "mean_pearson_r" in data:
            results[(display_name, "D:PearsonR")] = data["mean_pearson_r"]
        if "rmse" in data:
            results[(display_name, "D:RMSE")] = data["rmse"]
        print(f"  Dim D {display_name}: loaded")

    return results


def collect_dim_e_results():
    """Collect Dimension E (temporal ordering) results."""
    results = {}

    all_data = load_json_safe(
        os.path.join(RESULTS_DIR, "dim_e", "all_temporal_results.json")
    )
    if not all_data:
        print("  Dim E: no combined results file found")
        return results

    key_to_display = {v: k for k, v in FM_MODEL_MAP.items()}
    key_to_display["PCA_DPT"] = "PCA + DPT"

    # Average across datasets for each model
    model_taus = {}
    model_accs = {}

    for key, data in all_data.items():
        model_key = key.split("/")[0]
        display_name = key_to_display.get(model_key, model_key)

        if display_name not in model_taus:
            model_taus[display_name] = []
            model_accs[display_name] = []

        if "kendall_tau" in data:
            model_taus[display_name].append(data["kendall_tau"])
        if "knn_balanced_accuracy" in data:
            model_accs[display_name].append(data["knn_balanced_accuracy"])

    for display_name in model_taus:
        if model_taus[display_name]:
            import numpy as np
            results[(display_name, "E:KendallTau")] = float(
                np.mean(model_taus[display_name])
            )
        if model_accs[display_name]:
            results[(display_name, "E:kNN_BalAcc")] = float(
                np.mean(model_accs[display_name])
            )
        print(f"  Dim E {display_name}: loaded")

    return results


def assemble():
    """Assemble all results into final table."""
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Load template with baselines
    template_path = os.path.join(TABLES_DIR, "table1_with_baselines.csv")
    if not os.path.exists(template_path):
        template_path = os.path.join(TABLES_DIR, "table1_template.csv")
    if not os.path.exists(template_path):
        print("No results template found. Run table_builder.py first.")
        return

    table = pd.read_csv(template_path, index_col=0)

    print("Collecting results...")
    all_results = {}
    all_results.update(collect_dim_a_results())
    all_results.update(collect_dim_b_results())
    all_results.update(collect_dim_c_results())
    all_results.update(collect_dim_d_results())
    all_results.update(collect_dim_e_results())

    # Fill results into table
    filled = 0
    for (model, metric), value in all_results.items():
        if model in table.index and metric in table.columns:
            table.loc[model, metric] = f"{value:.4f}" if isinstance(value, float) else value
            filled += 1

    # Save final table
    final_path = os.path.join(TABLES_DIR, "table1_final.csv")
    table.to_csv(final_path)
    print(f"\nFilled {filled} cells into results table.")
    print(f"Final table: {final_path}")

    # Count unfilled non-N/A cells
    unfilled = 0
    for model in table.index:
        for metric in table.columns:
            val = table.loc[model, metric]
            if pd.isna(val) or val == "":
                unfilled += 1

    if unfilled > 0:
        print(f"WARNING: {unfilled} non-N/A cells still empty")

    # Print table
    print("\n" + "=" * 80)
    print(table.to_markdown() if hasattr(table, "to_markdown") else table.to_string())
    print("=" * 80)

    return table


if __name__ == "__main__":
    assemble()
