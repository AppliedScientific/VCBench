"""
scGPT perturbation prediction on Norman dataset.

Follows bowang-lab/scGPT tutorials/Tutorial_Perturbation.ipynb exactly.

Key implementation details (from tutorial source):
  - Model: TransformerGenerator (NOT TransformerModel)
  - Weight loading: selective, only ["encoder", "value_encoder", "transformer_encoder"]
  - Gene IDs: map_raw_id_to_vocab_id() maps GEARS indices to scGPT vocab
  - Training: masked_mse_loss, GradScaler, include_zero_gene="all"
  - Inference: model.pred_perturb(batch, include_zero_gene, gene_ids)
  - Novel perts: create_cell_graph_dataset_for_prediction()
  - CLS=False, CCE=False, MVC=False, ECS=False

Run in: vcbench-pt212 environment (scgpt==0.1.7, flash-attn<1.0.5)
GPU: ~8-12 GB VRAM. Fine-tuning ~1-2 hours on A100, inference minutes.
"""

import copy
import json
import os
import sys
import time

import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_a", "scgpt")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "scgpt", "scGPT_human")

# Hyperparameters (from scGPT perturbation tutorial)
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4
D_MODEL = 512
NHEAD = 8
D_HID = 512
NLAYERS = 12
NLAYERS_CLS = 3
DROPOUT = 0.0
SCHEDULE_INTERVAL = 1
SCHEDULE_GAMMA = 0.9
EARLY_STOP = 10
MAX_SEQ_LEN = 1536
INCLUDE_ZERO_GENE = "all"
AMP = True


def step1_load_data():
    """Load Norman data via GEARS with standard simulation split."""
    from gears import PertData

    pert_data = PertData(RAW_DIR)
    pert_data.load(data_name="norman")
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE)
    return pert_data


def step2_build_vocab_and_model(pert_data):
    """
    Build gene vocabulary mapping and initialize TransformerGenerator.

    Returns (model, vocab, gene_ids) where gene_ids maps GEARS gene
    indices to scGPT vocab token IDs.
    """
    import torch
    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import set_seed

    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    vocab_path = os.path.join(MODEL_DIR, "vocab.json")
    model_path = os.path.join(MODEL_DIR, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"ERROR: scGPT checkpoint not found at {model_path}")
        print("Download from bowang-lab/scGPT GitHub README")
        return None, None, None

    # Build vocabulary with special tokens
    vocab = GeneVocab.from_file(vocab_path)
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)

    # Map GEARS gene names to scGPT vocab IDs
    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=int
    )
    n_mapped = np.sum(gene_ids != vocab["<pad>"])
    print(f"Gene vocab mapping: {n_mapped}/{len(genes)} genes in scGPT vocab")

    # Initialize model
    model = TransformerGenerator(
        ntoken=len(vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=NLAYERS,
        nlayers_cls=NLAYERS_CLS,
        n_cls=1,
        vocab=vocab,
        dropout=DROPOUT,
        pad_token="<pad>",
        pad_value=0,
        pert_pad_id=2,
        use_fast_transformer=True,
    )

    # Selective weight loading — only encoder prefixes
    LOAD_PREFIXES = ["encoder", "value_encoder", "transformer_encoder"]
    pretrained_dict = torch.load(model_path, map_location="cpu")
    model_dict = model.state_dict()
    filtered = {
        k: v for k, v in pretrained_dict.items()
        if any(k.startswith(p) for p in LOAD_PREFIXES)
        and k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(filtered)}/{len(pretrained_dict)} pretrained weights "
          f"(prefixes: {LOAD_PREFIXES})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, vocab, gene_ids


def step3_finetune(model, pert_data, gene_ids):
    """
    Fine-tune scGPT on Norman perturbation data.

    Uses the exact training loop from Tutorial_Perturbation.ipynb:
    masked_mse_loss, GradScaler, map_raw_id_to_vocab_id, early stopping.
    """
    import torch
    from torch.cuda.amp import GradScaler, autocast
    from scgpt.loss import masked_mse_loss
    from scgpt.utils import map_raw_id_to_vocab_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_genes = pert_data.adata.n_vars

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, SCHEDULE_INTERVAL, gamma=SCHEDULE_GAMMA
    )
    scaler = GradScaler(enabled=AMP)

    train_loader = pert_data.dataloader["train_loader"]
    val_loader = pert_data.dataloader["val_loader"]

    best_val_loss = float("inf")
    best_model = None
    patience = 0
    ckpt_path = os.path.join(RESULTS_DIR, "scgpt_norman_finetuned.pt")

    print(f"Fine-tuning scGPT: {EPOCHS} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(1, EPOCHS + 1):
        # === TRAIN ===
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_data in train_loader:
            batch_size = len(batch_data.y)
            batch_data.to(device)

            # GEARS >=0.1: x is (batch*n_genes, 1), pert_idx is list of lists
            ori_gene_values = batch_data.x[:, 0].view(batch_size, n_genes)
            pert_flags = torch.zeros(batch_size, n_genes, dtype=torch.long, device=device)
            for i, pidx in enumerate(batch_data.pert_idx):
                for idx in pidx:
                    if idx < n_genes:
                        pert_flags[i, idx] = 1
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            # Gene selection
            if INCLUDE_ZERO_GENE == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )

            if len(input_gene_ids) > MAX_SEQ_LEN:
                input_gene_ids = torch.randperm(
                    len(input_gene_ids), device=device
                )[:MAX_SEQ_LEN]

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_ids = mapped_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

            with autocast(enabled=AMP):
                output_dict = model(
                    mapped_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False, CCE=False, MVC=False, ECS=False,
                )
                output_values = output_dict["mlm_output"]
                masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                loss = masked_mse_loss(output_values, target_values, masked_positions)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=False
            )
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # === VALIDATE ===
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_size = len(batch_data.y)
                batch_data.to(device)
                ori_gene_values = batch_data.x[:, 0].view(batch_size, n_genes)
                pert_flags = torch.zeros(batch_size, n_genes, dtype=torch.long, device=device)
                for i, pidx in enumerate(batch_data.pert_idx):
                    for idx in pidx:
                        if idx < n_genes:
                            pert_flags[i, idx] = 1
                target_gene_values = batch_data.y

                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]
                mapped_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_ids = mapped_ids.repeat(batch_size, 1)
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=device
                )

                with autocast(enabled=AMP):
                    output_dict = model(
                        mapped_ids, input_values, input_pert_flags,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=False, CCE=False, MVC=False, ECS=False,
                    )
                    loss = masked_mse_loss(
                        output_dict["mlm_output"], target_values,
                        torch.ones_like(input_values, dtype=torch.bool),
                    )
                val_loss += loss.item()
                val_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch}/{EPOCHS}: train_loss={avg_train:.4f}, "
              f"val_loss={avg_val:.4f}, time={elapsed:.1f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model = copy.deepcopy(model)
            patience = 0
            torch.save(best_model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"Best val loss: {best_val_loss:.4f} -> {ckpt_path}")
    return best_model if best_model is not None else model


def step4_predict(model, pert_data, gene_ids):
    """
    Generate predictions for test perturbations.

    Custom prediction loop (not model.pred_perturb) to handle GEARS >=0.1
    batch format where x is (N*genes, 1) and pert_idx is a list of lists.
    """
    import anndata as ad
    import torch
    from torch.cuda.amp import autocast

    from scgpt.loss import masked_mse_loss
    from scgpt.utils import map_raw_id_to_vocab_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    n_genes = pert_data.adata.n_vars
    test_loader = pert_data.dataloader["test_loader"]

    pert_cats = []
    all_preds = []
    all_truths = []

    print("Generating predictions for test perturbations...")
    with torch.no_grad():
        for batch_data in test_loader:
            batch_size = len(batch_data.y)
            batch_data.to(device)
            pert_cats.extend(batch_data.pert)

            # Parse batch (GEARS >=0.1 format)
            ori_gene_values = batch_data.x[:, 0].view(batch_size, n_genes)
            pert_flags = torch.zeros(batch_size, n_genes, dtype=torch.long, device=device)
            for i, pidx in enumerate(batch_data.pert_idx):
                for idx in pidx:
                    if idx < n_genes:
                        pert_flags[i, idx] = 1

            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            input_values = ori_gene_values
            input_pert_flags = pert_flags
            mapped_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_ids = mapped_ids.repeat(batch_size, 1)
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

            with autocast(enabled=AMP):
                output_dict = model(
                    mapped_ids, input_values, input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False, CCE=False, MVC=False, ECS=False,
                )
                pred = output_dict["mlm_output"]

            all_preds.append(pred.cpu())
            all_truths.append(batch_data.y.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truths = torch.cat(all_truths, dim=0).numpy()
    pert_cats = np.array(pert_cats)

    # Average predictions per perturbation condition
    unique_perts = sorted(set(pert_cats))
    pred_names = []
    pred_matrix = []

    for pert in unique_perts:
        mask = pert_cats == pert
        mean_pred = all_preds[mask].mean(axis=0)
        pred_names.append(pert)
        pred_matrix.append(mean_pred)

    pred_matrix = np.stack(pred_matrix)

    # Build AnnData for cell-eval
    adata_pred = ad.AnnData(
        X=pred_matrix,
        obs={"condition": pred_names},
        var=pert_data.adata.var.copy(),
    )
    adata_pred.obs.index = pred_names

    pred_path = os.path.join(RESULTS_DIR, "predictions.h5ad")
    adata_pred.write_h5ad(pred_path)
    print(f"Predictions: {adata_pred.shape} ({len(unique_perts)} perturbations) -> {pred_path}")

    # Also save raw per-cell results for debugging
    raw_path = os.path.join(RESULTS_DIR, "raw_predictions.npz")
    np.savez(raw_path, preds=all_preds, truths=all_truths, pert_cats=pert_cats)
    print(f"Raw results: {raw_path}")

    return pred_path


def step5_evaluate(pred_path):
    """Evaluate predictions with cell-eval."""
    import anndata as ad

    from src.evaluation.metrics import evaluate_perturbation

    adata_pred = ad.read_h5ad(pred_path)
    adata_real = ad.read_h5ad(os.path.join(PROCESSED_DIR, "norman.h5ad"))

    results, agg = evaluate_perturbation(adata_pred, adata_real)
    agg.to_csv(os.path.join(RESULTS_DIR, "cell_eval_results.csv"))

    results_dict = agg.to_dict() if hasattr(agg, "to_dict") else {"results": str(agg)}
    with open(os.path.join(RESULTS_DIR, "cell_eval_results.json"), "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"scGPT evaluation: {results_dict}")
    return results_dict


def main():
    print("=== scGPT Perturbation Prediction ===")

    # Check for existing results (skip if complete)
    final_results = os.path.join(RESULTS_DIR, "cell_eval_results.json")
    if os.path.exists(final_results):
        print(f"Results already exist: {final_results}. Skipping.")
        return

    print("\n[1/5] Loading data with GEARS split...")
    pert_data = step1_load_data()

    print("\n[2/5] Building vocab and model...")
    model, vocab, gene_ids = step2_build_vocab_and_model(pert_data)
    if model is None:
        return

    # Skip training if checkpoint exists
    ckpt_path = os.path.join(RESULTS_DIR, "scgpt_norman_finetuned.pt")
    if os.path.exists(ckpt_path):
        import torch
        print(f"\n[3/5] Loading saved checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    else:
        print("\n[3/5] Fine-tuning scGPT...")
        model = step3_finetune(model, pert_data, gene_ids)

    # Skip prediction if predictions exist
    pred_path = os.path.join(RESULTS_DIR, "predictions.h5ad")
    if os.path.exists(pred_path):
        print(f"\n[4/5] Predictions already exist: {pred_path}")
    else:
        print("\n[4/5] Generating predictions...")
        pred_path = step4_predict(model, pert_data, gene_ids)

    print("\n[5/5] Evaluating with cell-eval...")
    step5_evaluate(pred_path)

    print("\n=== scGPT perturbation complete ===")


if __name__ == "__main__":
    main()
