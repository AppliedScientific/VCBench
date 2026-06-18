"""
TranscriptFormer cross-species embedding extraction.

TranscriptFormer handles cross-species natively via species-specific gene
embeddings. No ortholog remapping needed.

Uses Python API (run_inference) instead of CLI to control config properly.
The CLI fails because the tokenizer expects 'assay' in obs, which census
data doesn't have. By using the API we can set obs_keys=[] to skip this.

Run in: vcbench-pt25 environment
"""

import os
import sys

import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_b", "transcriptformer")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "tf_sapiens")

TISSUES = ["lung", "liver", "heart", "kidney", "brain"]

SPECIES_EMBEDDING_MAP = {
    "homo_sapiens": "homo_sapiens_gene.h5",
    "mus_musculus": "mus_musculus_gene.h5",
}


def _build_crossspecies_config(species_key):
    """Build OmegaConf config for TranscriptFormer cross-species inference."""
    from src.models.run_transcriptformer_perturbation import _build_inference_config

    cfg = _build_inference_config()

    # Override pretrained_embedding for the target species
    emb_file = SPECIES_EMBEDDING_MAP.get(species_key, "homo_sapiens_gene.h5")
    emb_path = os.path.join(
        PROJECT_DIR, "checkpoints", "all_embeddings", emb_file
    )
    cfg.model.inference_config.pretrained_embedding = emb_path

    # CRITICAL: obs_keys must include at least one column so that
    # run_inference creates an AnnData with the correct number of obs rows.
    # With obs_keys=[], the obs DataFrame has 0 rows and obsm assignment
    # fails with shape mismatch (embeddings have N rows but obs has 0).
    cfg.model.inference_config.obs_keys = ["cell_type"]

    return cfg


def ensure_ensembl_ids(adata_path, output_path, species="homo_sapiens"):
    """Ensure AnnData has ensembl_id column in .var and 'assay' in .obs."""
    import scanpy as sc

    if os.path.exists(output_path):
        return output_path

    adata = sc.read_h5ad(adata_path)
    if "ensembl_id" not in adata.var.columns:
        from src.models.data_prep import get_symbol_to_ensembl

        symbol_map = get_symbol_to_ensembl()
        adata.var["ensembl_id"] = [
            symbol_map.get(g, "NA") for g in adata.var_names
        ]
        adata = adata[:, adata.var["ensembl_id"] != "NA"].copy()

    # Add assay column if missing (tokenizer may need it for aux_vocab)
    if "assay" not in adata.obs.columns:
        adata.obs["assay"] = "single-cell RNA sequencing"

    adata.write_h5ad(output_path)
    return output_path


def run_transcriptformer(adata_path, species_key, output_dir, batch_size=32):
    """Run TranscriptFormer inference via Python API."""
    import scanpy as sc

    # Checkpoint: skip if embeddings npy already exists
    emb_npy = os.path.join(output_dir, "embeddings.npy")
    if os.path.exists(emb_npy):
        print(f"  Already extracted: {emb_npy}")
        return np.load(emb_npy)

    os.makedirs(output_dir, exist_ok=True)

    cfg = _build_crossspecies_config(species_key)
    cfg.model.inference_config.batch_size = batch_size

    adata = sc.read_h5ad(adata_path)

    # Ensure raw counts
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    from transcriptformer.model.inference import run_inference

    result = run_inference(cfg, data_files=[adata])

    # Extract embeddings from result
    embeddings = None
    if "embeddings" in result.obsm:
        embeddings = np.array(result.obsm["embeddings"])
    else:
        for key in result.obsm:
            if "emb" in key.lower():
                embeddings = np.array(result.obsm[key])
                break

    if embeddings is not None:
        np.save(emb_npy, embeddings)
        print(f"  Embeddings: {embeddings.shape} -> {emb_npy}")

    return embeddings


def _run_single_tissue(tissue, species_key, species_label):
    """Run a single tissue+species combination. Returns (label, embeddings) or None."""
    adata_path = os.path.join(
        PROCESSED_DIR, f"census_{species_key}_{tissue}.h5ad"
    )
    if not os.path.exists(adata_path):
        print(f"  Skipping {species_label} {tissue} (not found)")
        return None

    # Ensure ensembl_id column
    prepped_path = os.path.join(
        PROCESSED_DIR, f"census_{species_key}_{tissue}_tf.h5ad"
    )
    ensure_ensembl_ids(adata_path, prepped_path, species_key)

    # Run inference
    output_dir = os.path.join(RESULTS_DIR, f"{species_label}_{tissue}")

    try:
        embs = run_transcriptformer(prepped_path, species_key, output_dir)
    except Exception as e:
        print(f"  ERROR {species_label} {tissue}: {e}")
        return None

    if embs is not None:
        np.save(
            os.path.join(
                RESULTS_DIR,
                f"{species_label}_{tissue}_embeddings.npy",
            ),
            embs,
        )
        label_path = os.path.join(RESULTS_DIR, f"{species_label}_{tissue}_labels.npy")
        if not os.path.exists(label_path):
            import scanpy as sc
            orig = sc.read_h5ad(adata_path)
            labels = orig.obs["cell_type"].values if "cell_type" in orig.obs.columns else np.arange(orig.n_obs)
            np.save(label_path, labels)

    return embs


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== TranscriptFormer Cross-Species Embeddings ===")

    for tissue in TISSUES:
        print(f"\n--- {tissue} ---")

        # Check if both species already done for this tissue
        human_done = os.path.exists(os.path.join(RESULTS_DIR, f"human_{tissue}", "embeddings.npy"))
        mouse_done = os.path.exists(os.path.join(RESULTS_DIR, f"mouse_{tissue}", "embeddings.npy"))

        if human_done and mouse_done:
            print(f"  Both species done for {tissue}, skipping")
            continue

        # Run sequentially with larger batch size (32 uses ~40GB)
        for species_key, species_label in [
            ("homo_sapiens", "human"),
            ("mus_musculus", "mouse"),
        ]:
            _run_single_tissue(tissue, species_key, species_label)

    print("\n=== TranscriptFormer cross-species embeddings complete ===")


if __name__ == "__main__":
    main()
