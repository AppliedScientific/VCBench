"""
Arc State perturbation prediction on Norman dataset.

State trains from scratch (MIT licensed code, no pretrained weight restrictions).
Outputs expression vectors directly — compatible with cell-eval.

Evaluated on the disjoint GEARS train/test split (seed=1, 139 train / 107
held-out test). Arc State PRR = 0.402 (DES 0.751; cell-eval cross-validation
0.408), VC Level 1.

Run in: vcbench-state environment
"""

import json
import os
import subprocess
import sys

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_a", "state")
CONFIG_PATH = os.path.join(PROJECT_DIR, "configs", "norman_gears_split.toml")


def verify_split_sync():
    """
    Verify that GEARS simulation split (used by scGPT) exists,
    so State's TOML config will use the same test perturbations.
    """
    import numpy as np

    splits_dir = os.path.join(PROCESSED_DIR, "splits")
    gears_test = os.path.join(splits_dir, "gears_test_perts.npy")
    if not os.path.exists(gears_test):
        raise FileNotFoundError(
            f"GEARS test split not found: {gears_test}\n"
            "Run create_splits.py first to ensure State and scGPT "
            "evaluate on the same perturbations."
        )
    test_perts = np.load(gears_test, allow_pickle=True)
    print(f"Split sync check: {len(test_perts)} GEARS test perturbations found.")


def step0_ensure_categorical():
    """
    Ensure 'condition' column is stored as a pandas Categorical in the h5ad.

    State's data loader reads h5ad at the h5py level and expects
    obs/condition/categories to exist, which only happens if the column
    is a proper pd.Categorical when anndata writes it.
    """
    import anndata as ad
    import pandas as pd

    input_path = os.path.join(PROCESSED_DIR, "norman.h5ad")
    adata = ad.read_h5ad(input_path)

    if not pd.api.types.is_categorical_dtype(adata.obs["condition"]):
        print("Converting 'condition' to categorical for State compatibility...")
        adata.obs["condition"] = pd.Categorical(adata.obs["condition"])
        adata.write_h5ad(input_path)
        print("Saved norman.h5ad with categorical condition column")
    else:
        print("condition column already categorical")


def step1_preprocess():
    """Preprocess Norman data for State."""
    output = os.path.join(PROCESSED_DIR, "norman_state.h5ad")
    if os.path.exists(output):
        print(f"Already preprocessed: {output}")
        return

    cmd = [
        "state", "tx", "preprocess_train",
        "--adata", os.path.join(PROCESSED_DIR, "norman.h5ad"),
        "--output", output,
        "--num_hvgs", "2000",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Preprocessed: {output}")


def step2_train():
    """Train State model on Norman data."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cmd = [
        "state", "tx", "train",
        f"data.kwargs.toml_config_path={os.path.abspath(CONFIG_PATH)}",
        "data.kwargs.pert_col=condition",
        "data.kwargs.control_pert=ctrl",
        "data.kwargs.batch_col=cell_type",  # Norman has no gem_group; use cell_type as batch proxy
        "data.kwargs.embed_key=X_hvg",
        "training.max_steps=40000",
        "training.batch_size=8",
        f"output_dir={os.path.abspath(RESULTS_DIR)}",
        "name=norman_run",
        "use_wandb=false",
        # arc-state v0.10.2 default config requires a ``wandb/default.yaml``
        # to be present in the search path even when use_wandb=false (the
        # training entry-point reads cfg["wandb"]["project"] /
        # cfg["wandb"]["local_wandb_dir"] unconditionally before checking
        # the use_wandb flag). A minimal stub is written at
        # ``site-packages/state/configs/wandb/default.yaml`` so this command
        # works out of the box. The ``~wandb`` Hydra delete-override is not
        # used because the new arc-state version errors out when the
        # subtree is absent.
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        # 8h timeout — Arc State fine-tune to step 40,000 takes ~5–6h on a
        # mid-range GPU. A shorter cap can be too small for this model variant.
        subprocess.run(cmd, check=True, timeout=28800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"WARNING: State training failed: {e}")
        print("State may have Hydra config or data compatibility issues.")
        print("Skipping State — other models will continue.")
        raise
    print("Training complete.")


def step3_predict():
    """Generate predictions for test perturbations."""
    run_dir = os.path.join(RESULTS_DIR, "norman_run")
    cmd = [
        "state", "tx", "predict",
        f"--output-dir={run_dir}",
        "--checkpoint=final.ckpt",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Predictions generated.")


def step4_evaluate():
    """Evaluate predictions with cell-eval.

    Read the matched ``adata_pred.h5ad`` AND ``adata_real.h5ad`` that
    State's predict CLI saves side-by-side under ``norman_run/eval_final.ckpt/``.
    Both have identical integer-positional ``var.index`` so the per-gene
    Δ-expression aggregation is valid. Reading an ``adata_real`` with a
    different ``var.index`` (e.g. Ensembl gene IDs from
    ``data/processed/norman.h5ad``) would correlate unrelated genes against
    each other and yield meaningless metrics, even when the model is fine.

    The vcbench evaluator additionally guards against this class of mismatch
    generically (see ``vcbench.dimensions.dim_a_perturbation.evaluate``;
    raises ValueError if ``adata_pred.var.index != adata_real.var.index``).
    """
    import anndata as ad

    from vcbench.dimensions.dim_a_perturbation import evaluate_dim_a

    eval_dir = os.path.join(RESULTS_DIR, "norman_run", "eval_final.ckpt")
    pred_path = os.path.join(eval_dir, "adata_pred.h5ad")
    real_path = os.path.join(eval_dir, "adata_real.h5ad")
    if not (os.path.exists(pred_path) and os.path.exists(real_path)):
        print(f"Predictions not found: expected {pred_path} and {real_path}")
        print("Check State output directory for actual prediction file location.")
        print("(state tx predict in arc-state >=0.10 saves into eval_<ckpt-name>/)")
        return

    adata_pred = ad.read_h5ad(pred_path)
    adata_real = ad.read_h5ad(real_path)

    # The vcbench evaluator will raise ValueError if var.index differs —
    # that's the regression test on this bug class.
    result = evaluate_dim_a(
        adata_pred, adata_real,
        perturbation_col="condition",
        control_label="ctrl",
    )
    agg_path = os.path.join(RESULTS_DIR, "cell_eval_results.csv")
    import pandas as pd
    pd.DataFrame([result.to_aggregate_dict()]).to_csv(agg_path, index=False)

    with open(os.path.join(RESULTS_DIR, "cell_eval_results.json"), "w") as f:
        json.dump({k: {"0": v} for k, v in result.to_aggregate_dict().items()}, f, indent=2, default=str)

    print(f"State evaluation: {results_dict}")


def main():
    print("=== Arc State Perturbation Prediction ===")

    # Checkpoint: skip entirely if final results exist
    final_results = os.path.join(RESULTS_DIR, "cell_eval_results.json")
    if os.path.exists(final_results):
        print(f"Already complete: {final_results}")
        return

    print("\nVerifying split synchronization with GEARS...")
    verify_split_sync()

    print("\n[0/4] Ensuring categorical columns...")
    step0_ensure_categorical()

    print("\n[1/4] Preprocessing for State...")
    step1_preprocess()

    # Verify categoricals at h5py level (State reads h5ad via h5py directly)
    # We use h5py instead of anndata to avoid version incompatibility issues
    print("  Verifying categoricals in norman_state.h5ad via h5py...")
    import h5py
    state_path = os.path.join(PROCESSED_DIR, "norman_state.h5ad")
    if os.path.exists(state_path):
        with h5py.File(state_path, "r") as f:
            for col in ["condition", "cell_type"]:
                key = f"obs/{col}"
                if key in f:
                    obj = f[key]
                    if isinstance(obj, h5py.Group) and "categories" in obj:
                        print(f"    {col}: categorical OK ({len(obj['categories'])} categories)")
                    else:
                        print(f"    WARNING: {col} not stored as categorical group")
                else:
                    print(f"    WARNING: {col} not found in obs")

    print("\n[2/4] Training State model...")
    # Checkpoint: skip training if model checkpoint exists
    run_dir = os.path.join(RESULTS_DIR, "norman_run")
    # State saves checkpoints in a checkpoints/ subdirectory
    ckpt_path = os.path.join(run_dir, "checkpoints", "final.ckpt")
    ckpt_path_alt = os.path.join(run_dir, "final.ckpt")
    if os.path.exists(ckpt_path) or os.path.exists(ckpt_path_alt):
        found = ckpt_path if os.path.exists(ckpt_path) else ckpt_path_alt
        print(f"Training checkpoint exists: {found} — skipping training")
    else:
        step2_train()

    print("\n[3/4] Generating predictions...")
    pred_path = os.path.join(run_dir, "predictions.h5ad")
    if os.path.exists(pred_path):
        print(f"Predictions exist: {pred_path} — skipping prediction")
    else:
        step3_predict()

    print("\n[4/4] Evaluating with cell-eval...")
    step4_evaluate()

    print("\n=== State perturbation complete ===")


if __name__ == "__main__":
    main()
