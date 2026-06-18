"""
Perturbation prediction on Replogle K562 Essential dataset.

Runs scGPT, Geneformer, TranscriptFormer, and Arc State on Replogle data.
Replogle K562 Essential is genome-scale Perturb-seq (~163K cells, single-gene
knockdowns). Tests held-out generalization (PP-held-out).

CRITICAL DESIGN: Each model is trained AND evaluated within Replogle's own
train/test split. Decoders (for Geneformer/TranscriptFormer) are trained on
Replogle training perturbations, NOT reused from Norman. Reusing Norman decoders
would conflate "does the representation generalize across splits?" with
"does a decoder transfer across datasets?".

Per the paper: "perturbations randomly partitioned into 80/20 train/test
across five seeds." A single GEARS simulation split (seed=1) is used here
for consistency with Norman. Multi-seed evaluation can be added later.

Usage:
    python -m src.models.run_replogle_perturbation --model scgpt
    python -m src.models.run_replogle_perturbation --model geneformer
    python -m src.models.run_replogle_perturbation --model transcriptformer
    python -m src.models.run_replogle_perturbation --model state
    python -m src.models.run_replogle_perturbation --model all
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
RESULTS_BASE = os.path.join(PROJECT_DIR, "results", "dim_a")


# ─── scGPT ───────────────────────────────────────────────────────────────────

def run_scgpt_replogle():
    """
    Fine-tune scGPT on Replogle training perturbations, predict held-out.

    Uses GEARS data loader which handles Replogle natively. The entire
    scGPT pipeline (fine-tune + predict) runs within Replogle data.
    """
    from gears import PertData

    results_dir = os.path.join(RESULTS_BASE, "scgpt", "replogle")
    os.makedirs(results_dir, exist_ok=True)

    print("[scGPT-Replogle] Loading data via GEARS...")
    pert_data = PertData(RAW_DIR)
    pert_data.load(data_name="replogle_k562_essential")
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=64, test_batch_size=64)

    from src.models.run_scgpt_perturbation import (
        step2_build_vocab_and_model,
        step3_finetune,
        step4_predict,
    )
    from src.models import run_scgpt_perturbation as scgpt_mod

    orig_results = scgpt_mod.RESULTS_DIR
    scgpt_mod.RESULTS_DIR = results_dir

    print("[scGPT-Replogle] Building vocab and model...")
    model, vocab, gene_ids = step2_build_vocab_and_model(pert_data)
    if model is None:
        scgpt_mod.RESULTS_DIR = orig_results
        return

    print("[scGPT-Replogle] Fine-tuning on Replogle training perts...")
    model = step3_finetune(model, pert_data, gene_ids)

    print("[scGPT-Replogle] Predicting held-out perts...")
    pred_path = step4_predict(model, pert_data, gene_ids)

    print("[scGPT-Replogle] Evaluating...")
    import anndata as ad
    from src.evaluation.metrics import evaluate_perturbation

    adata_pred = ad.read_h5ad(pred_path)
    adata_real = ad.read_h5ad(os.path.join(PROCESSED_DIR, "replogle_k562_essential.h5ad"))
    results, agg = evaluate_perturbation(adata_pred, adata_real)
    agg.to_csv(os.path.join(results_dir, "cell_eval_results.csv"))
    results_dict = agg.to_dict() if hasattr(agg, "to_dict") else {"results": str(agg)}
    with open(os.path.join(results_dir, "cell_eval_results.json"), "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"[scGPT-Replogle] Results: {results_dict}")

    scgpt_mod.RESULTS_DIR = orig_results


# ─── Arc State ────────────────────────────────────────────────────────────────

def run_state_replogle():
    """
    Train Arc State from scratch on Replogle, predict held-out.

    State trains from scratch so there's no decoder transfer issue.
    """
    import subprocess

    results_dir = os.path.join(RESULTS_BASE, "state", "replogle")
    os.makedirs(results_dir, exist_ok=True)

    output = os.path.join(PROCESSED_DIR, "replogle_state.h5ad")
    if not os.path.exists(output):
        cmd = [
            "state", "tx", "preprocess_train",
            "--adata", os.path.join(PROCESSED_DIR, "replogle_k562_essential.h5ad"),
            "--output", output,
            "--num_hvgs", "2000",
        ]
        print(f"[State-Replogle] Preprocessing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    config_path = os.path.join(PROJECT_DIR, "configs", "replogle_fewshot.toml")
    cmd = [
        "state", "tx", "train",
        f"data.kwargs.toml_config_path={config_path}",
        "data.kwargs.pert_col=condition",
        "data.kwargs.embed_key=X_hvg",
        "training.max_steps=40000",
        "training.batch_size=8",
        f"output_dir={results_dir}",
        "name=replogle_run",
    ]
    print(f"[State-Replogle] Training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    run_dir = os.path.join(results_dir, "replogle_run")
    cmd = [
        "state", "tx", "predict",
        f"--output-dir={run_dir}",
        "--checkpoint=final.ckpt",
    ]
    print(f"[State-Replogle] Predicting: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    import anndata as ad
    from src.evaluation.metrics import evaluate_perturbation

    pred_path = os.path.join(run_dir, "predictions.h5ad")
    if os.path.exists(pred_path):
        adata_pred = ad.read_h5ad(pred_path)
        adata_real = ad.read_h5ad(os.path.join(PROCESSED_DIR, "replogle_k562_essential.h5ad"))
        results, agg = evaluate_perturbation(adata_pred, adata_real)
        agg.to_csv(os.path.join(results_dir, "cell_eval_results.csv"))
        results_dict = agg.to_dict() if hasattr(agg, "to_dict") else {"results": str(agg)}
        with open(os.path.join(results_dir, "cell_eval_results.json"), "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"[State-Replogle] Results: {results_dict}")
    else:
        print(f"[State-Replogle] Predictions not found at {pred_path}")


# ─── Geneformer ──────────────────────────────────────────────────────────────

def run_geneformer_replogle():
    """
    Geneformer on Replogle with FRESH decoder trained on Replogle training perts.

    Pipeline:
    1. Tokenize Replogle control cells
    2. Use the Norman-fine-tuned CellClassifier (transfer the encoder, not decoder)
    3. Extract CLS embeddings for all Replogle perturbation conditions
    4. Split into train/test using GEARS simulation split
    5. Train a NEW Ridge decoder on Replogle training perts
    6. Predict Replogle test perts
    7. Evaluate with cell-eval
    """
    import anndata as ad
    import scanpy as sc
    from scipy.sparse import issparse
    from sklearn.linear_model import Ridge

    results_dir = os.path.join(RESULTS_BASE, "geneformer", "replogle")
    os.makedirs(results_dir, exist_ok=True)

    from src.models.run_geneformer_perturbation import (
        _find_finetuned_model,
        _get_perturbed_embedding,
    )

    finetuned_model = _find_finetuned_model()
    print(f"[Geneformer-Replogle] Using fine-tuned encoder: {finetuned_model}")

    # Load Replogle data
    adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, "replogle_k562_essential.h5ad"))
    conditions = adata.obs["condition"].values

    # Load Replogle split (same GEARS simulation split)
    # Use the split from GEARS or create 80/20 random partition
    splits_dir = os.path.join(PROCESSED_DIR, "splits")
    replogle_train_path = os.path.join(splits_dir, "replogle_train_perts.npy")
    replogle_test_path = os.path.join(splits_dir, "replogle_test_perts.npy")

    if os.path.exists(replogle_train_path) and os.path.exists(replogle_test_path):
        train_perts = set(np.load(replogle_train_path, allow_pickle=True))
        test_perts = set(np.load(replogle_test_path, allow_pickle=True))
    else:
        # Create 80/20 split of Replogle perturbations
        all_perts = sorted(set(conditions) - {"ctrl"})
        rng = np.random.default_rng(seed=1)
        rng.shuffle(all_perts)
        split_idx = int(0.8 * len(all_perts))
        train_perts = set(all_perts[:split_idx])
        test_perts = set(all_perts[split_idx:])
        os.makedirs(splits_dir, exist_ok=True)
        np.save(replogle_train_path, np.array(sorted(train_perts)))
        np.save(replogle_test_path, np.array(sorted(test_perts)))
        print(f"[Geneformer-Replogle] Created split: {len(train_perts)} train, {len(test_perts)} test")

    # Tokenize Replogle
    from src.models.data_prep import prepare_for_geneformer
    from geneformer import TranscriptomeTokenizer
    from datasets import load_from_disk

    replogle_gf_path = os.path.join(PROCESSED_DIR, "replogle_geneformer_ready.h5ad")
    prepare_for_geneformer(
        os.path.join(PROCESSED_DIR, "replogle_k562_essential.h5ad"),
        replogle_gf_path,
    )

    tokenized_dir = os.path.join(PROJECT_DIR, "data", "tokenized")
    os.makedirs(tokenized_dir, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        os.symlink(os.path.abspath(replogle_gf_path), os.path.join(tmpdir, "replogle_geneformer_ready.h5ad"))
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"cell_type": "cell_type"},
            use_h5ad_index=True,
        )
        tk.tokenize_data(tmpdir + "/", tokenized_dir + "/", "replogle_geneformer_ready", file_format="h5ad")

    ctrl_dataset = load_from_disk(
        os.path.join(tokenized_dir, "replogle_geneformer_ready.dataset")
    ).filter(lambda x: x["condition"] == "ctrl")

    # Extract embeddings for ALL perturbation conditions (train + test)
    all_perts = sorted(train_perts | test_perts)
    pert_embeddings = {}

    emb_cache = os.path.join(results_dir, "replogle_pert_embeddings.pkl")
    if os.path.exists(emb_cache):
        print(f"[Geneformer-Replogle] Loading cached embeddings...")
        with open(emb_cache, "rb") as f:
            pert_embeddings = pickle.load(f)
    else:
        print(f"[Geneformer-Replogle] Extracting embeddings for {len(all_perts)} perts...")
        for idx, pert in enumerate(all_perts):
            pert_genes = [g.strip() for g in pert.split("+")]
            embs = _get_perturbed_embedding(finetuned_model, ctrl_dataset, pert_genes)
            if embs is not None:
                pert_embeddings[pert] = embs.mean(axis=0)

            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(all_perts)}")

        with open(emb_cache, "wb") as f:
            pickle.dump(pert_embeddings, f)

    # Train FRESH Ridge decoder on Replogle training perturbations
    print("[Geneformer-Replogle] Training decoder on Replogle training perts...")
    X_train, y_train = [], []
    emb_dim = next(iter(pert_embeddings.values())).shape[0]

    # Add control (zero-vector embedding -> mean control expression)
    ctrl_mask = conditions == "ctrl"
    ctrl_expr = adata[ctrl_mask].X
    ctrl_expr = ctrl_expr.toarray() if issparse(ctrl_expr) else np.array(ctrl_expr)
    X_train.append(np.zeros(emb_dim))
    y_train.append(ctrl_expr.mean(axis=0))

    for cond in sorted(train_perts):
        if cond not in pert_embeddings:
            continue
        mask = conditions == cond
        if mask.sum() == 0:
            continue
        expr = adata[mask].X
        expr = expr.toarray() if issparse(expr) else np.array(expr)
        X_train.append(pert_embeddings[cond])
        y_train.append(expr.mean(axis=0))

    X_train = np.stack(X_train)
    y_train = np.stack(y_train)
    print(f"  Decoder: {X_train.shape} -> {y_train.shape}")

    decoder = Ridge(alpha=1.0)
    decoder.fit(X_train, y_train)

    # Predict TEST perturbations
    pred_names = []
    pred_expressions = []

    for cond in sorted(test_perts):
        if cond not in pert_embeddings:
            continue
        mean_emb = pert_embeddings[cond].reshape(1, -1)
        pred_expr = decoder.predict(mean_emb)
        pred_names.append(cond)
        pred_expressions.append(pred_expr[0])

    if not pred_names:
        print("[Geneformer-Replogle] No test predictions generated.")
        return

    pred_matrix = np.stack(pred_expressions)
    adata_pred = ad.AnnData(X=pred_matrix, obs={"condition": pred_names}, var=adata.var.copy())
    adata_pred.obs.index = pred_names
    pred_path = os.path.join(results_dir, "predictions.h5ad")
    adata_pred.write_h5ad(pred_path)
    print(f"[Geneformer-Replogle] Predictions: {adata_pred.shape}")

    from src.evaluation.metrics import evaluate_perturbation
    results, agg = evaluate_perturbation(adata_pred, adata)
    agg.to_csv(os.path.join(results_dir, "cell_eval_results.csv"))
    results_dict = agg.to_dict() if hasattr(agg, "to_dict") else {"results": str(agg)}
    with open(os.path.join(results_dir, "cell_eval_results.json"), "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"[Geneformer-Replogle] Results: {results_dict}")


# ─── TranscriptFormer ────────────────────────────────────────────────────────

def run_transcriptformer_replogle():
    """
    TranscriptFormer on Replogle with FRESH decoder on Replogle training perts.

    Same design as Geneformer: extract embeddings for all conditions,
    train Ridge on Replogle's train split, predict test split.
    """
    import anndata as ad
    import scanpy as sc
    from scipy.sparse import issparse
    from sklearn.linear_model import Ridge

    results_dir = os.path.join(RESULTS_BASE, "transcriptformer", "replogle")
    os.makedirs(results_dir, exist_ok=True)

    from src.models.run_transcriptformer_perturbation import (
        _build_inference_config,
        _get_embeddings,
    )
    from src.models.data_prep import get_symbol_to_ensembl

    # Load and prepare Replogle data
    adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, "replogle_k562_essential.h5ad"))
    if "ensembl_id" not in adata.var.columns:
        symbol_map = get_symbol_to_ensembl()
        adata.var["ensembl_id"] = [symbol_map.get(g, "NA") for g in adata.var_names]
        adata = adata[:, adata.var["ensembl_id"] != "NA"].copy()

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    conditions = adata.obs["condition"].values
    gene_names = list(adata.var_names)
    cfg = _build_inference_config()

    # Load or create Replogle split
    splits_dir = os.path.join(PROCESSED_DIR, "splits")
    replogle_train_path = os.path.join(splits_dir, "replogle_train_perts.npy")
    replogle_test_path = os.path.join(splits_dir, "replogle_test_perts.npy")

    if os.path.exists(replogle_train_path) and os.path.exists(replogle_test_path):
        train_perts = set(np.load(replogle_train_path, allow_pickle=True))
        test_perts = set(np.load(replogle_test_path, allow_pickle=True))
    else:
        all_perts = sorted(set(conditions) - {"ctrl"})
        rng = np.random.default_rng(seed=1)
        rng.shuffle(all_perts)
        split_idx = int(0.8 * len(all_perts))
        train_perts = set(all_perts[:split_idx])
        test_perts = set(all_perts[split_idx:])
        os.makedirs(splits_dir, exist_ok=True)
        np.save(replogle_train_path, np.array(sorted(train_perts)))
        np.save(replogle_test_path, np.array(sorted(test_perts)))

    ctrl_adata = adata[conditions == "ctrl"].copy()
    all_perts = sorted(train_perts | test_perts)

    # Extract embeddings for all conditions
    emb_cache = os.path.join(results_dir, "replogle_pert_embeddings.pkl")
    if os.path.exists(emb_cache):
        print("[TF-Replogle] Loading cached embeddings...")
        with open(emb_cache, "rb") as f:
            pert_embeddings = pickle.load(f)
    else:
        pert_embeddings = {}
        print(f"[TF-Replogle] Extracting embeddings for {len(all_perts)} perts...")
        for idx, pert in enumerate(all_perts):
            pert_genes = [g.strip() for g in pert.split("+")]
            pert_indices = [gene_names.index(g) for g in pert_genes if g in gene_names]
            if not pert_indices:
                continue

            modified = ctrl_adata.copy()
            X = modified.X.toarray() if issparse(modified.X) else np.array(modified.X)
            for gi in pert_indices:
                X[:, gi] = 0.0
            modified.X = X

            try:
                embs = _get_embeddings(cfg, modified)
                pert_embeddings[pert] = np.mean(embs, axis=0)
            except Exception as e:
                print(f"  ERROR {pert}: {e}")

            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(all_perts)}")

        with open(emb_cache, "wb") as f:
            pickle.dump(pert_embeddings, f)

    # Train FRESH Ridge decoder on Replogle training perturbations
    print("[TF-Replogle] Training decoder on Replogle training perts...")
    X_train, y_train = [], []
    emb_dim = next(iter(pert_embeddings.values())).shape[0]

    ctrl_mask = conditions == "ctrl"
    ctrl_expr = adata[ctrl_mask].X
    ctrl_expr = ctrl_expr.toarray() if issparse(ctrl_expr) else np.array(ctrl_expr)
    X_train.append(np.zeros(emb_dim))
    y_train.append(ctrl_expr.mean(axis=0))

    for cond in sorted(train_perts):
        if cond not in pert_embeddings:
            continue
        mask = conditions == cond
        if mask.sum() == 0:
            continue
        expr = adata[mask].X
        expr = expr.toarray() if issparse(expr) else np.array(expr)
        X_train.append(pert_embeddings[cond])
        y_train.append(expr.mean(axis=0))

    X_train = np.stack(X_train)
    y_train = np.stack(y_train)
    print(f"  Decoder: {X_train.shape} -> {y_train.shape}")

    decoder = Ridge(alpha=1.0)
    decoder.fit(X_train, y_train)

    # Predict TEST perturbations
    pred_names = []
    pred_expressions = []

    for cond in sorted(test_perts):
        if cond not in pert_embeddings:
            continue
        mean_emb = pert_embeddings[cond].reshape(1, -1)
        pred_expr = decoder.predict(mean_emb)
        pred_names.append(cond)
        pred_expressions.append(pred_expr[0])

    if not pred_names:
        print("[TF-Replogle] No test predictions generated.")
        return

    pred_matrix = np.stack(pred_expressions)
    adata_pred = ad.AnnData(X=pred_matrix, obs={"condition": pred_names}, var=adata.var.copy())
    adata_pred.obs.index = pred_names
    pred_path = os.path.join(results_dir, "predictions.h5ad")
    adata_pred.write_h5ad(pred_path)
    print(f"[TF-Replogle] Predictions: {adata_pred.shape}")

    from src.evaluation.metrics import evaluate_perturbation
    results, agg = evaluate_perturbation(adata_pred, adata)
    agg.to_csv(os.path.join(results_dir, "cell_eval_results.csv"))
    results_dict = agg.to_dict() if hasattr(agg, "to_dict") else {"results": str(agg)}
    with open(os.path.join(results_dir, "cell_eval_results.json"), "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"[TF-Replogle] Results: {results_dict}")


RUNNERS = {
    "scgpt": run_scgpt_replogle,
    "geneformer": run_geneformer_replogle,
    "transcriptformer": run_transcriptformer_replogle,
    "state": run_state_replogle,
}


def main():
    parser = argparse.ArgumentParser(description="Run perturbation prediction on Replogle")
    parser.add_argument("--model", choices=list(RUNNERS.keys()) + ["all"], required=True)
    args = parser.parse_args()

    if args.model == "all":
        for name, runner in RUNNERS.items():
            print(f"\n{'='*60}")
            print(f"  Running {name} on Replogle K562 Essential")
            print(f"{'='*60}")
            runner()
    else:
        RUNNERS[args.model]()


if __name__ == "__main__":
    main()
