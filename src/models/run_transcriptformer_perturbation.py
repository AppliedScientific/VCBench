"""
TranscriptFormer perturbation prediction on Norman dataset.

TranscriptFormer (CZI Virtual Cells) uses rank-value encoding and outputs
cell embeddings — NOT expression predictions directly. For perturbation
prediction, we use an embedding-based approach:

  1. Prepare data with ensembl_id and raw counts
  2. Get control cell embeddings via run_inference()
  3. For each perturbation: zero out target gene(s), get perturbed embeddings
  4. Train Ridge decoder: embedding -> expression (on train perturbations)
  5. Apply decoder to test perturbation embeddings
  6. Evaluate with cell-eval

PERTURBATION APPROACH NOTE:
  TranscriptFormer has NO native perturbation prediction API (unlike scGPT's
  pred_perturb() or Geneformer's token deletion). We use an embedding-based
  proxy: zero out target gene counts, re-run inference, decode via Ridge.
  This is weaker than token deletion (Geneformer) or a trained perturbation
  module (scGPT/State) because zeroing creates unnatural input distributions.
  Results evaluate TranscriptFormer's embedding quality for perturbation tasks,
  not a native perturbation capability.

API (from pip package source + official notebooks):
  - transcriptformer.model.inference.run_inference(cfg, data_files=[adata])
  - Config via OmegaConf, not constructor args
  - Checkpoint downloaded via: transcriptformer download tf-sapiens
  - Expects raw counts, ensembl_id in adata.var

Run in: vcbench-pt25 environment (Python >= 3.11, torch <= 2.5.1)
GPU: ~4-8 GB VRAM
"""

import json
import os
import pickle
import sys

import anndata as ad
import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_a", "transcriptformer")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "tf_sapiens")


def step1_prepare_data():
    """
    Ensure Norman data has ensembl_id and RAW counts for TranscriptFormer.

    TranscriptFormer uses rank-value encoding which requires raw unnormalized
    integer counts (not log-normalized or scaled values).
    """
    import scanpy as sc

    input_path = os.path.join(PROCESSED_DIR, "norman.h5ad")
    output_path = os.path.join(PROCESSED_DIR, "norman_tf_ready.h5ad")

    if os.path.exists(output_path):
        print(f"Already prepared: {output_path}")
        return output_path

    adata = sc.read_h5ad(input_path)

    # Add ensembl_id if missing
    if "ensembl_id" not in adata.var.columns:
        # Check if var_names are already Ensembl IDs (ENSG...)
        if adata.var_names[0].startswith("ENSG"):
            # var_names are already Ensembl IDs — use directly
            adata.var["ensembl_id"] = adata.var_names.values
            print(f"var_names are already Ensembl IDs, using directly ({adata.n_vars} genes)")
        else:
            from src.models.data_prep import get_symbol_to_ensembl

            symbol_map = get_symbol_to_ensembl()
            adata.var["ensembl_id"] = [
                symbol_map.get(g, "NA") for g in adata.var_names
            ]
            n_before = adata.n_vars
            adata = adata[:, adata.var["ensembl_id"] != "NA"].copy()
            print(f"Ensembl mapping: {adata.n_vars}/{n_before} genes mapped")

    # Ensure raw counts are in X (TranscriptFormer expects unnormalized counts)
    if "counts" in adata.layers:
        adata.layers["normalized"] = adata.X.copy()
        adata.X = adata.layers["counts"].copy()
        print("Using raw counts from 'counts' layer")
    else:
        print("WARNING: No 'counts' layer. TranscriptFormer expects raw counts.")
        print("Proceeding with current X — results may be suboptimal.")

    # Add assay column (aux_vocab tokenizer requires it)
    if "assay" not in adata.obs.columns:
        adata.obs["assay"] = "single-cell RNA sequencing"

    adata.write_h5ad(output_path)
    print(f"TranscriptFormer-ready: {adata.shape} -> {output_path}")
    return output_path


def _build_inference_config():
    """
    Build OmegaConf config for TranscriptFormer inference.

    The real API uses run_inference(cfg, data_files) where cfg is an
    OmegaConf DictConfig built from the checkpoint's config.json merged
    with inference settings.
    """
    from omegaconf import OmegaConf

    # Load model config from checkpoint
    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"TranscriptFormer config not found: {config_path}\n"
            "Download checkpoint: transcriptformer download tf-sapiens\n"
            f"Expected at: {CHECKPOINT_DIR}"
        )

    with open(config_path) as f:
        config_dict = json.load(f)

    model_cfg = OmegaConf.create(config_dict)

    # Inference-specific overrides
    # The checkpoint config.json has many data_config keys already but is
    # missing several that inference.py accesses directly (use_raw,
    # remove_duplicate_genes, etc.). We must supply ALL keys that
    # inference.py references to avoid OmegaConf MissingKey errors.
    all_emb_dir = os.path.join(os.path.dirname(CHECKPOINT_DIR), "all_embeddings")
    pretrained_emb_path = os.path.join(all_emb_dir, "homo_sapiens_gene.h5")
    inference_overrides = OmegaConf.create({
        "model": {
            "checkpoint_path": CHECKPOINT_DIR,
            "model_config": {
                "compile_block_mask": False,
            },
            "inference_config": {
                "load_checkpoint": os.path.join(CHECKPOINT_DIR, "model_weights.pt"),
                "pretrained_embedding": pretrained_emb_path,
                "obs_keys": ["condition"],
                "batch_size": 8,
                "output_keys": ["embeddings"],
                "emb_type": "cell",
                "precision": "16-mixed",
                "num_gpus": 1,
                "device": "auto",
            },
            "data_config": {
                # Keys already in checkpoint config but we override paths:
                "gene_col_name": "ensembl_id",
                "aux_vocab_path": os.path.join(CHECKPOINT_DIR, "vocabs"),
                "esm2_mappings_path": os.path.join(CHECKPOINT_DIR, "vocabs"),
                # Keys NOT in checkpoint config that inference.py accesses:
                "use_raw": False,
                "remove_duplicate_genes": True,
                "clip_counts": 30,
                "max_len": 2048,
                # Override None values from checkpoint that cause TypeError
                # in dataloader code (compared with > 0 or used as int):
                "filter_outliers": 0,
                "filter_to_vocabs": False,
                "normalize_to_scale": 0,
                "randomize_genes": False,
                "sort_genes": False,
                "min_expressed_genes": 0,
                "n_data_workers": 1,
            },
        }
    })

    cfg = OmegaConf.merge(model_cfg, inference_overrides)

    # Recursively disable struct mode on ALL nested DictConfig nodes.
    # OmegaConf.set_struct only sets it on the immediate node. We need
    # to walk the tree so that missing keys at ANY level return None.
    from omegaconf import DictConfig, ListConfig

    def _disable_struct_recursive(node):
        if isinstance(node, DictConfig):
            OmegaConf.set_struct(node, False)
            for key in node:
                try:
                    child = node._get_child(key)
                    _disable_struct_recursive(child)
                except Exception:
                    pass
        elif isinstance(node, ListConfig):
            for item in node:
                _disable_struct_recursive(item)

    _disable_struct_recursive(cfg)

    return cfg


def _get_embeddings(cfg, adata):
    """Run TranscriptFormer inference to get cell embeddings."""
    from transcriptformer.model.inference import run_inference

    result = run_inference(cfg, data_files=[adata])

    # Embeddings stored in obsm
    if "embeddings" in result.obsm:
        return result.obsm["embeddings"]
    # Fallback: check other obsm keys
    for key in result.obsm:
        if "emb" in key.lower():
            return result.obsm[key]

    raise ValueError(
        f"No embeddings found in inference output. Available obsm keys: {list(result.obsm.keys())}"
    )


def step2_get_all_embeddings():
    """
    Get embeddings for all cells (control + perturbed) in Norman dataset.

    Returns (embeddings, adata) where embeddings is (n_cells, emb_dim).
    """
    import scanpy as sc

    adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, "norman_tf_ready.h5ad"))
    cfg = _build_inference_config()

    emb_path = os.path.join(RESULTS_DIR, "all_embeddings.npy")
    if os.path.exists(emb_path):
        print(f"Loading cached embeddings: {emb_path}")
        embeddings = np.load(emb_path)
        return embeddings, adata

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Running TranscriptFormer inference on {adata.n_obs} cells...")
    embeddings = _get_embeddings(cfg, adata)
    embeddings = np.array(embeddings)

    np.save(emb_path, embeddings)
    print(f"Embeddings: {embeddings.shape} -> {emb_path}")
    return embeddings, adata


def step3_get_perturbed_embeddings(adata):
    """
    For each test perturbation, create a modified AnnData with target gene(s)
    zeroed out, run inference, and collect the mean embedding.

    This simulates "what would cells look like if these genes were knocked out"
    by setting their counts to zero in the input.
    """
    import scanpy as sc
    from scipy.sparse import issparse

    cfg = _build_inference_config()

    splits_dir = os.path.join(PROCESSED_DIR, "splits")
    test_perts = set(np.load(
        os.path.join(splits_dir, "gears_test_perts.npy"), allow_pickle=True
    ))
    train_perts = set(np.load(
        os.path.join(splits_dir, "gears_train_perts.npy"), allow_pickle=True
    ))

    # var_names may be Ensembl IDs after data prep; perturbation names use
    # gene symbols. Build a symbol -> index map from the gene_name column.
    if "gene_name" in adata.var.columns:
        symbol_list = list(adata.var["gene_name"])
        symbol_to_idx = {g: i for i, g in enumerate(symbol_list)}
        print(f"Using gene_name column for symbol lookup ({len(symbol_to_idx)} genes)")
    else:
        symbol_list = list(adata.var_names)
        symbol_to_idx = {g: i for i, g in enumerate(symbol_list)}
        print(f"Using var_names for symbol lookup ({len(symbol_to_idx)} genes)")

    ctrl_mask = adata.obs["condition"] == "ctrl"
    ctrl_adata = adata[ctrl_mask].copy()

    # Subsample control cells to make per-perturbation inference feasible.
    # Full ctrl set (~7400 cells) takes ~37 min per perturbation × 246 = 150+ hours.
    # 300 cells → ~1.2 min per pert × 246 = ~5 hours (still robust mean embedding).
    MAX_CTRL_CELLS = 300
    if ctrl_adata.n_obs > MAX_CTRL_CELLS:
        rng = np.random.RandomState(42)
        idx = rng.choice(ctrl_adata.n_obs, MAX_CTRL_CELLS, replace=False)
        ctrl_adata = ctrl_adata[idx].copy()
        print(f"Subsampled control cells: {MAX_CTRL_CELLS} (from {adata[ctrl_mask].n_obs})")

    # Cache path
    pert_emb_path = os.path.join(RESULTS_DIR, "perturbed_embeddings.pkl")
    if os.path.exists(pert_emb_path):
        with open(pert_emb_path, "rb") as f:
            cached = pickle.load(f)
        if len(cached) > 0:
            print(f"Loading cached perturbed embeddings: {pert_emb_path} ({len(cached)} perts)")
            return cached
        else:
            print(f"Cached perturbed embeddings empty, recomputing...")

    all_perts = sorted(test_perts | train_perts)
    pert_embeddings = {}

    # GPU memory hygiene: TranscriptFormer's run_inference() instantiates a
    # new Lightning Trainer + model on each call, and PyTorch + Lightning
    # don't release VRAM between calls without an explicit gc + cache clear.
    # On smaller-VRAM GPUs this can cause OOM after a few dozen perts, so we
    # force a release after every pert.
    import gc
    try:
        import torch as _torch
        _has_torch_cuda = _torch.cuda.is_available()
    except Exception:
        _torch = None
        _has_torch_cuda = False

    # Resume support: if a partial pkl exists, skip already-done perts
    if os.path.exists(pert_emb_path):
        try:
            with open(pert_emb_path, "rb") as f:
                pert_embeddings = pickle.load(f)
            if pert_embeddings:
                print(f"  Resuming with {len(pert_embeddings)} cached perts")
        except Exception:
            pert_embeddings = {}

    print(f"Computing perturbed embeddings for {len(all_perts)} perturbations...")
    for idx, pert in enumerate(all_perts):
        if pert in pert_embeddings:
            continue  # already done in prior partial run

        pert_genes = [g.strip() for g in pert.split("+")]
        pert_gene_indices = [
            symbol_to_idx[g] for g in pert_genes if g in symbol_to_idx
        ]

        if not pert_gene_indices:
            print(f"  Skipping {pert}: genes not in vocabulary")
            continue

        # Create modified control cells with perturbed genes zeroed
        modified = ctrl_adata.copy()
        X = modified.X.toarray() if issparse(modified.X) else np.array(modified.X)
        for gi in pert_gene_indices:
            X[:, gi] = 0.0
        modified.X = X

        # Get embeddings for modified cells
        try:
            embs = _get_embeddings(cfg, modified)
            pert_embeddings[pert] = np.mean(embs, axis=0)
            del embs
        except Exception as e:
            print(f"  ERROR on {pert}: {e}")
            del modified
            gc.collect()
            if _has_torch_cuda:
                _torch.cuda.empty_cache()
            continue

        # Free the per-pert allocations (model, dataloader, intermediate tensors)
        del modified
        gc.collect()
        if _has_torch_cuda:
            _torch.cuda.empty_cache()

        # Checkpoint every 20 perts so a crash doesn't lose all progress
        if (idx + 1) % 20 == 0:
            with open(pert_emb_path, "wb") as f:
                pickle.dump(pert_embeddings, f)
            print(f"  Processed {idx + 1}/{len(all_perts)} perturbations "
                  f"(checkpointed {len(pert_embeddings)} embs)")

    with open(pert_emb_path, "wb") as f:
        pickle.dump(pert_embeddings, f)
    print(f"Perturbed embeddings: {len(pert_embeddings)} conditions -> {pert_emb_path}")
    return pert_embeddings


def step4_train_decoder(embeddings, adata):
    """
    Train Ridge decoder: embedding -> expression (on train perturbations).

    For each train perturbation, pairs the mean observed embedding with
    the mean observed expression.
    """
    from scipy.sparse import issparse
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    splits_dir = os.path.join(PROCESSED_DIR, "splits")
    train_perts = set(np.load(
        os.path.join(splits_dir, "gears_train_perts.npy"), allow_pickle=True
    ))

    conditions = adata.obs["condition"].values
    X_train, y_train = [], []

    for cond in sorted(train_perts):
        mask = conditions == cond
        if mask.sum() == 0:
            continue

        # Mean embedding for this condition
        cond_embs = embeddings[mask]
        mean_emb = cond_embs.mean(axis=0)

        # Mean expression for this condition
        expr = adata[mask].X
        expr = expr.toarray() if issparse(expr) else np.array(expr)
        mean_expr = expr.mean(axis=0)

        X_train.append(mean_emb)
        y_train.append(mean_expr)

    if not X_train:
        print("ERROR: No training pairs. Check embeddings alignment with conditions.")
        return None

    X_train = np.stack(X_train)
    y_train = np.stack(y_train)
    print(f"Decoder training: {X_train.shape} embeddings -> {y_train.shape} expressions")

    decoder = Ridge(alpha=1.0)
    scores = cross_val_score(
        decoder, X_train, y_train, cv=min(5, len(X_train)), scoring="r2"
    )
    print(f"Decoder cross-val R2: {scores.mean():.4f} +/- {scores.std():.4f}")

    decoder.fit(X_train, y_train)

    decoder_path = os.path.join(RESULTS_DIR, "expression_decoder.pkl")
    with open(decoder_path, "wb") as f:
        pickle.dump(decoder, f)
    print(f"Decoder saved: {decoder_path}")
    return decoder


def step5_predict_and_evaluate(decoder, pert_embeddings, adata):
    """
    Apply decoder to test perturbation embeddings and evaluate with cell-eval.
    """
    from src.evaluation.metrics import evaluate_perturbation

    if decoder is None:
        print("ERROR: No decoder available.")
        return

    splits_dir = os.path.join(PROCESSED_DIR, "splits")
    test_perts = set(np.load(
        os.path.join(splits_dir, "gears_test_perts.npy"), allow_pickle=True
    ))

    pred_names = []
    pred_expressions = []

    for cond in sorted(test_perts):
        if cond not in pert_embeddings:
            print(f"  Skipping {cond}: no perturbed embedding")
            continue

        mean_emb = pert_embeddings[cond].reshape(1, -1)
        pred_expr = decoder.predict(mean_emb)
        pred_names.append(cond)
        pred_expressions.append(pred_expr[0])

    if not pred_names:
        print("ERROR: No test predictions generated.")
        return

    pred_matrix = np.stack(pred_expressions)

    # The Ridge decoder was trained on log1p-normalized targets (norman.h5ad .X,
    # range [0, ~9]) but can extrapolate wildly for out-of-distribution perturbed
    # embeddings. Clip predictions to the valid log-normalized range:
    # - Floor at 0 (gene expression cannot be negative in log1p space)
    # - Ceiling at a generous bound (max observed in GT is ~9, allow some headroom)
    # Load ground truth to determine valid range for clipping
    adata_real = ad.read_h5ad(os.path.join(PROCESSED_DIR, "norman.h5ad"))
    from scipy.sparse import issparse as _issparse
    X_real = adata_real.X.toarray() if _issparse(adata_real.X) else np.array(adata_real.X)
    gt_max = float(X_real.max())

    pred_range_before = (pred_matrix.min(), pred_matrix.max())
    pred_matrix = np.clip(pred_matrix, 0, gt_max)
    pred_range_after = (pred_matrix.min(), pred_matrix.max())
    print(f"Prediction range: [{pred_range_before[0]:.3f}, {pred_range_before[1]:.3f}]"
          f" -> clipped to [{pred_range_after[0]:.3f}, {pred_range_after[1]:.3f}]"
          f" (GT max: {gt_max:.3f})")

    adata_pred = ad.AnnData(
        X=pred_matrix,
        obs={"condition": pred_names},
        var=adata.var.copy(),
    )
    adata_pred.obs.index = pred_names

    pred_path = os.path.join(RESULTS_DIR, "predictions.h5ad")
    adata_pred.write_h5ad(pred_path)
    print(f"Predictions saved: {adata_pred.shape} -> {pred_path}")

    # Evaluate: predictions are test-set only (107 perts), real has all 283
    results, agg = evaluate_perturbation(adata_pred, adata_real)
    agg.to_csv(os.path.join(RESULTS_DIR, "cell_eval_results.csv"))

    results_dict = agg.to_dict() if hasattr(agg, "to_dict") else {"results": str(agg)}
    with open(os.path.join(RESULTS_DIR, "cell_eval_results.json"), "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"TranscriptFormer evaluation: {results_dict}")


def main():
    print("=== TranscriptFormer Perturbation Prediction ===")

    # Checkpoint: skip entirely if final results exist
    final_results = os.path.join(RESULTS_DIR, "cell_eval_results.json")
    if os.path.exists(final_results):
        print(f"Already complete: {final_results}")
        return

    print("\n[1/5] Preparing data...")
    step1_prepare_data()

    print("\n[2/5] Getting cell embeddings...")
    embeddings, adata = step2_get_all_embeddings()

    print("\n[3/5] Getting perturbed embeddings...")
    pert_embeddings = step3_get_perturbed_embeddings(adata)

    print("\n[4/5] Training expression decoder...")
    decoder_path = os.path.join(RESULTS_DIR, "expression_decoder.pkl")
    if os.path.exists(decoder_path):
        print(f"Loading cached decoder: {decoder_path}")
        with open(decoder_path, "rb") as f:
            decoder = pickle.load(f)
    else:
        decoder = step4_train_decoder(embeddings, adata)

    print("\n[5/5] Predicting + evaluating...")
    step5_predict_and_evaluate(decoder, pert_embeddings, adata)

    print("\n=== TranscriptFormer perturbation complete ===")


if __name__ == "__main__":
    main()
