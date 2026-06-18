"""
Geneformer perturbation prediction on Norman dataset.

Follows Ahlmann-Eltze et al. (Nature Methods 2025) exactly:
  1. Prepare data with ensembl_id + n_counts
  2. Tokenize for Geneformer
  3. Fine-tune CellClassifier on perturbation condition labels
  4. Extract control CLS embeddings via EmbExtractor
  5. Extract perturbed CLS embeddings via custom get_perturbed_embedding()
     (ISP.perturb_data() only saves cosine sims, NOT raw embeddings)
  6. Train Ridge decoder: mean CLS embedding -> mean expression (train perts)
  7. Predict test perturbation expression + evaluate with cell-eval

CRITICAL: InSilicoPerturber.perturb_data() discards raw embeddings and only
writes cosine similarity scalars to disk. To get raw perturbed embeddings,
we must call geneformer.emb_extractor.get_embs() directly on perturbed
token sequences, bypassing the ISP output pipeline.
"""

import json
import os
import pickle
import sys

import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
TOKENIZED_DIR = os.path.join(PROJECT_DIR, "data", "tokenized")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_a", "geneformer")
MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "geneformer", "Geneformer", "Geneformer-V2-316M"
)
FINETUNED_DIR = os.path.join(RESULTS_DIR, "finetuned_classifier")


def step1_prepare_data():
    """Add ensembl_id and n_counts required for tokenization."""
    from src.models.data_prep import prepare_for_geneformer

    return prepare_for_geneformer(
        os.path.join(PROCESSED_DIR, "norman.h5ad"),
        os.path.join(PROCESSED_DIR, "norman_geneformer_ready.h5ad"),
    )


def step2_tokenize():
    """Tokenize Norman dataset for Geneformer.

    IMPORTANT: Geneformer's tokenizer globs ALL .h5ad files in the input
    directory. We must isolate the target file via a symlink in a temp dir
    to avoid tokenizing unrelated files (census, weinreb, etc.).
    """
    from geneformer import TranscriptomeTokenizer

    token_path = os.path.join(TOKENIZED_DIR, "norman_geneformer_ready.dataset")
    if os.path.exists(token_path):
        print(f"Tokenized dataset already exists: {token_path}")
        return

    os.makedirs(TOKENIZED_DIR, exist_ok=True)

    # Isolate target file to prevent tokenizer from globbing other h5ad files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(PROCESSED_DIR, "norman_geneformer_ready.h5ad")
        dst = os.path.join(tmpdir, "norman_geneformer_ready.h5ad")
        os.symlink(os.path.abspath(src), dst)

        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={
                "cell_type": "cell_type",
                "condition": "condition",
            },
            use_h5ad_index=True,
        )
        tk.tokenize_data(
            tmpdir + "/",
            TOKENIZED_DIR + "/",
            "norman_geneformer_ready",
            file_format="h5ad",
        )
    print(f"Tokenized dataset saved to {TOKENIZED_DIR}")


def step3_finetune_classifier():
    """
    Fine-tune CellClassifier on perturbation condition labels.

    Ahlmann-Eltze froze 2 layers and used state_key="condition".
    The CLS token from this fine-tuned model captures perturbation-relevant
    information needed for the downstream decoder.
    """
    from geneformer import Classifier

    os.makedirs(FINETUNED_DIR, exist_ok=True)

    # Geneformer V2 uses training_args dict instead of individual kwargs
    training_args = {
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 12,
        "per_device_eval_batch_size": 12,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "save_strategy": "epoch",
    }
    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "condition", "states": "all"},
        nproc=4,
        forward_batch_size=100,
        training_args=training_args,
        freeze_layers=2,
    )

    cc.prepare_data(
        input_data_file=os.path.join(TOKENIZED_DIR, "norman_geneformer_ready.dataset"),
        output_directory=FINETUNED_DIR,
        output_prefix="norman",
    )

    # Geneformer V2 Classifier uses train_classifier (not train)
    # prepare_data outputs: norman_labeled_train.dataset, norman_labeled_test.dataset,
    # norman_id_class_dict.pkl
    import pickle
    # Try both with and without .pkl extension
    id_class_dict_path = os.path.join(FINETUNED_DIR, "norman_id_class_dict.pkl")
    if not os.path.exists(id_class_dict_path):
        id_class_dict_path = os.path.join(FINETUNED_DIR, "norman_id_class_dict")
    with open(id_class_dict_path, "rb") as f:
        id_class_dict = pickle.load(f)
    num_classes = len(set(id_class_dict.values()))
    print(f"  Fine-tuning with {num_classes} classes")

    # train_classifier expects loaded Dataset objects, not path strings.
    # V2 API: validate_and_clean_cols calls train_data.column_names,
    # so we must pass datasets.Dataset instances.
    from datasets import load_from_disk

    train_data_path = os.path.join(FINETUNED_DIR, "norman_labeled_train.dataset")
    if not os.path.exists(train_data_path):
        train_data_path = os.path.join(FINETUNED_DIR, "norman_labeled_train")
    eval_data_path = os.path.join(FINETUNED_DIR, "norman_labeled_test.dataset")
    if not os.path.exists(eval_data_path):
        eval_data_path = os.path.join(FINETUNED_DIR, "norman_labeled_test")

    train_dataset = load_from_disk(train_data_path)
    eval_dataset = load_from_disk(eval_data_path)
    print(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    cc.train_classifier(
        model_directory=MODEL_PATH,
        num_classes=num_classes,
        train_data=train_dataset,
        eval_data=eval_dataset,
        output_directory=FINETUNED_DIR,
        predict=False,
    )

    print(f"CellClassifier fine-tuned -> {FINETUNED_DIR}")
    return _find_finetuned_model()


def _find_finetuned_model():
    """Locate the fine-tuned classifier model directory.

    The model files (model.safetensors, config.json) live directly in
    FINETUNED_DIR alongside dataset directories and checkpoints.
    Return FINETUNED_DIR if it contains model.safetensors.
    """
    if not os.path.exists(FINETUNED_DIR):
        return None
    # Model weights are at the root of FINETUNED_DIR
    if os.path.exists(os.path.join(FINETUNED_DIR, "model.safetensors")):
        return FINETUNED_DIR
    if os.path.exists(os.path.join(FINETUNED_DIR, "pytorch_model.bin")):
        return FINETUNED_DIR
    # Check subdirectories for checkpoints with model weights
    for name in sorted(os.listdir(FINETUNED_DIR), reverse=True):
        candidate = os.path.join(FINETUNED_DIR, name)
        if os.path.isdir(candidate) and name.startswith("checkpoint-"):
            if os.path.exists(os.path.join(candidate, "model.safetensors")):
                return candidate
    return None


def step4_extract_control_embeddings(finetuned_model):
    """
    Extract CLS embeddings for control cells using EmbExtractor.

    EmbExtractor returns raw embedding vectors (unlike ISP which only
    saves cosine similarities).
    """
    from geneformer import EmbExtractor

    emb_path = os.path.join(RESULTS_DIR, "ctrl_cls_embeddings.npy")
    if os.path.exists(emb_path):
        print(f"Loading cached control embeddings: {emb_path}")
        return np.load(emb_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Filter tokenized dataset to control cells only
    from datasets import load_from_disk
    dataset = load_from_disk(
        os.path.join(TOKENIZED_DIR, "norman_geneformer_ready.dataset")
    )
    ctrl_dataset = dataset.filter(lambda x: x["condition"] == "ctrl")
    ctrl_path = os.path.join(TOKENIZED_DIR, "norman_ctrl_only.dataset")
    ctrl_dataset.save_to_disk(ctrl_path)
    print(f"Control cells: {len(ctrl_dataset)} cells")

    # num_classes must match the finetuned CellClassifier (284 conditions in Norman)
    embex = EmbExtractor(
        model_type="CellClassifier",
        num_classes=284,
        emb_mode="cls",
        emb_layer=0,
        forward_batch_size=100,
        nproc=4,
    )

    embs = embex.extract_embs(
        model_directory=finetuned_model,
        input_data_file=ctrl_path,
        output_directory=RESULTS_DIR + "/",
        output_prefix="ctrl",
        output_torch_embs=False,
    )

    # embs is a DataFrame with embedding columns
    if hasattr(embs, "values"):
        # DataFrame: drop non-numeric columns if present
        numeric_cols = embs.select_dtypes(include=[np.number]).columns
        ctrl_emb = embs[numeric_cols].values
    else:
        ctrl_emb = np.array(embs)

    np.save(emb_path, ctrl_emb)
    print(f"Control CLS embeddings: {ctrl_emb.shape} -> {emb_path}")
    return ctrl_emb


def _get_perturbed_embedding(finetuned_model, ctrl_dataset, genes_to_perturb):
    """
    Extract raw CLS embeddings from control cells with specified genes deleted.

    This bypasses ISP's cosine-similarity-only output pipeline by calling
    get_embs() directly on the perturbed token sequences.

    Follows Ahlmann-Eltze et al. custom get_perturbed_embedding() pattern.
    """
    import torch
    from transformers import BertForSequenceClassification
    import geneformer.perturber_utils as pu
    from geneformer.emb_extractor import get_embs

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        finetuned_model, output_hidden_states=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get model parameters
    max_len = pu.get_model_input_size(model)
    layer_to_quant = pu.quant_layers(model) + 0  # emb_layer=0 (Ahlmann-Eltze)

    # Load token dictionary (Ensembl ID -> token_id) and symbol->Ensembl mapping
    import geneformer
    gf_dir = os.path.dirname(geneformer.__file__)
    token_dict_path = os.path.join(gf_dir, "token_dictionary_gc104M.pkl")
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)  # Ensembl ID -> token_id

    # Load symbol -> Ensembl mapping for gene name lookup
    ensembl_map_path = os.path.join(gf_dir, "ensembl_mapping_dict_gc104M.pkl")
    with open(ensembl_map_path, "rb") as f:
        symbol_to_ensembl = pickle.load(f)  # gene symbol -> Ensembl ID

    # Build symbol -> token_id mapping (symbol -> Ensembl -> token_id)
    symbol_to_token = {}
    for symbol, ensembl_id in symbol_to_ensembl.items():
        if ensembl_id in token_dict:
            symbol_to_token[symbol] = token_dict[ensembl_id]

    # Get token IDs for genes to perturb
    perturb_token_ids = []
    for gene in genes_to_perturb:
        if gene in symbol_to_token:
            perturb_token_ids.append(symbol_to_token[gene])
        elif gene in token_dict:
            # Direct Ensembl ID lookup as fallback
            perturb_token_ids.append(token_dict[gene])

    if not perturb_token_ids:
        return None

    # Apply perturbation: delete perturbed gene tokens from each cell's sequence
    def delete_genes(example):
        input_ids = example["input_ids"]
        # Remove tokens corresponding to perturbed genes
        filtered = [t for t in input_ids if t not in perturb_token_ids]
        # Pad to original length
        pad_id = 0  # Geneformer pad token
        filtered = filtered[:max_len]
        filtered = filtered + [pad_id] * (max_len - len(filtered))
        example["input_ids"] = filtered
        return example

    perturbed_data = ctrl_dataset.map(delete_genes)

    # Extract CLS embeddings from perturbed sequences
    # Build token->gene dict for get_embs (Ensembl ID -> token_id, inverted)
    token_gene_dict = {v: k for k, v in token_dict.items()}
    pad_token_id = 0
    embs = get_embs(
        model,
        perturbed_data,
        "cls",
        layer_to_quant,
        pad_token_id,
        100,  # forward_batch_size
        token_gene_dict,
    )

    return embs.cpu().detach().numpy()


def step5_extract_perturbed_embeddings(finetuned_model):
    """
    For each perturbation condition, delete the perturbed gene token(s)
    from control cells and extract CLS embeddings.

    Returns dict: condition_name -> mean CLS embedding vector
    """
    emb_path = os.path.join(RESULTS_DIR, "perturbed_embeddings.pkl")
    if os.path.exists(emb_path):
        print(f"Loading cached perturbed embeddings: {emb_path}")
        with open(emb_path, "rb") as f:
            return pickle.load(f)

    from datasets import load_from_disk

    # Load control cells dataset
    ctrl_path = os.path.join(TOKENIZED_DIR, "norman_ctrl_only.dataset")
    if not os.path.exists(ctrl_path):
        dataset = load_from_disk(
            os.path.join(TOKENIZED_DIR, "norman_geneformer_ready.dataset")
        )
        ctrl_dataset = dataset.filter(lambda x: x["condition"] == "ctrl")
        ctrl_dataset.save_to_disk(ctrl_path)
    else:
        ctrl_dataset = load_from_disk(ctrl_path)

    # Load all perturbation conditions
    splits_dir = os.path.join(PROCESSED_DIR, "splits")
    test_perts = set(np.load(
        os.path.join(splits_dir, "gears_test_perts.npy"), allow_pickle=True
    ))
    train_perts = set(np.load(
        os.path.join(splits_dir, "gears_train_perts.npy"), allow_pickle=True
    ))
    all_perts = sorted(test_perts | train_perts)

    # Load partial checkpoint if one exists (incremental saving)
    partial_path = emb_path + ".partial"
    if os.path.exists(partial_path):
        with open(partial_path, "rb") as f:
            pert_embeddings = pickle.load(f)
        print(f"Resuming from partial checkpoint: {len(pert_embeddings)} conditions done")
    else:
        pert_embeddings = {}
    print(f"Extracting perturbed embeddings for {len(all_perts)} conditions...")

    for idx, pert in enumerate(all_perts):
        if pert in pert_embeddings:
            continue  # Already computed in partial checkpoint

        pert_genes = [g.strip() for g in pert.split("+")]

        embs = _get_perturbed_embedding(finetuned_model, ctrl_dataset, pert_genes)
        if embs is not None:
            pert_embeddings[pert] = embs.mean(axis=0)  # Mean across cells
        else:
            print(f"  Skipping {pert}: genes not in token dictionary")

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(all_perts)} perturbations")
            # Incremental checkpoint every 20 conditions
            with open(partial_path, "wb") as f:
                pickle.dump(pert_embeddings, f)

    with open(emb_path, "wb") as f:
        pickle.dump(pert_embeddings, f)
    # Clean up partial checkpoint
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"Perturbed embeddings: {len(pert_embeddings)} conditions -> {emb_path}")
    return pert_embeddings


def step6_train_decoder(ctrl_emb, pert_embeddings):
    """
    Train Ridge decoder: mean CLS embedding -> mean expression.

    Trained on TRAIN perturbations only. For each train perturbation,
    pairs the mean perturbed CLS embedding with the mean observed expression.
    The control condition maps to zero-vector embedding -> mean control expression.
    """
    import scanpy as sc
    from scipy.sparse import issparse
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    adata = sc.read_h5ad(os.path.join(PROCESSED_DIR, "norman.h5ad"))
    conditions = adata.obs["condition"].values

    splits_dir = os.path.join(PROCESSED_DIR, "splits")
    train_perts = set(np.load(
        os.path.join(splits_dir, "gears_train_perts.npy"), allow_pickle=True
    ))

    X_train, y_train = [], []

    # Add control: zero-vector embedding -> mean control expression
    # (Ahlmann-Eltze uses np.zeros for control embedding)
    emb_dim = ctrl_emb.shape[1]
    ctrl_mask = conditions == "ctrl"
    ctrl_expr = adata[ctrl_mask].X
    ctrl_expr = ctrl_expr.toarray() if issparse(ctrl_expr) else np.array(ctrl_expr)
    X_train.append(np.zeros(emb_dim))
    y_train.append(ctrl_expr.mean(axis=0))

    # Add train perturbation conditions
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
    print(f"Decoder training: {X_train.shape} embeddings -> {y_train.shape} expressions")
    print(f"  ({len(X_train)} conditions: 1 ctrl + {len(X_train)-1} train perts)")

    decoder = Ridge(alpha=1.0)
    n_cv = min(5, len(X_train))
    if n_cv >= 2:
        scores = cross_val_score(decoder, X_train, y_train, cv=n_cv, scoring="r2")
        print(f"Decoder cross-val R2: {scores.mean():.4f} +/- {scores.std():.4f}")

    decoder.fit(X_train, y_train)

    decoder_path = os.path.join(RESULTS_DIR, "expression_decoder.pkl")
    with open(decoder_path, "wb") as f:
        pickle.dump(decoder, f)
    print(f"Decoder saved: {decoder_path}")
    return decoder


def step7_predict_and_evaluate(decoder, pert_embeddings):
    """
    Apply decoder to test perturbation embeddings -> predicted expression.
    Build AnnData and evaluate with cell-eval.
    """
    import anndata as ad
    import scanpy as sc

    from src.evaluation.metrics import evaluate_perturbation

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

    # Build predicted AnnData
    adata_real = sc.read_h5ad(os.path.join(PROCESSED_DIR, "norman.h5ad"))
    pred_matrix = np.stack(pred_expressions)
    adata_pred = ad.AnnData(
        X=pred_matrix,
        obs={"condition": pred_names},
        var=adata_real.var.copy(),
    )
    adata_pred.obs.index = pred_names

    pred_path = os.path.join(RESULTS_DIR, "predictions.h5ad")
    adata_pred.write_h5ad(pred_path)
    print(f"Predictions saved: {adata_pred.shape} -> {pred_path}")

    # Evaluate
    results, agg = evaluate_perturbation(adata_pred, adata_real)
    agg.to_csv(os.path.join(RESULTS_DIR, "cell_eval_results.csv"))

    results_dict = agg.to_dict() if hasattr(agg, "to_dict") else {"results": str(agg)}
    with open(os.path.join(RESULTS_DIR, "cell_eval_results.json"), "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"Geneformer evaluation: {results_dict}")


def main():
    print("=== Geneformer Perturbation Prediction ===")

    # Skip if results exist
    final_results = os.path.join(RESULTS_DIR, "cell_eval_results.json")
    if os.path.exists(final_results):
        print(f"Results already exist: {final_results}. Skipping.")
        return

    print("\n[1/7] Preparing data...")
    step1_prepare_data()

    print("\n[2/7] Tokenizing...")
    step2_tokenize()

    print("\n[3/7] Fine-tuning CellClassifier...")
    existing_model = _find_finetuned_model()
    if existing_model and os.path.exists(os.path.join(existing_model, "config.json")):
        print(f"Finetuned model already exists: {existing_model}")
        finetuned_model = existing_model
    else:
        finetuned_model = step3_finetune_classifier()

    print("\n[4/7] Extracting control CLS embeddings...")
    ctrl_emb = step4_extract_control_embeddings(finetuned_model)

    print("\n[5/7] Extracting perturbed CLS embeddings...")
    pert_embeddings = step5_extract_perturbed_embeddings(finetuned_model)

    print("\n[6/7] Training expression decoder...")
    decoder = step6_train_decoder(ctrl_emb, pert_embeddings)

    print("\n[7/7] Predicting test perturbations + evaluating...")
    step7_predict_and_evaluate(decoder, pert_embeddings)

    print("\n=== Geneformer perturbation complete ===")


if __name__ == "__main__":
    main()
