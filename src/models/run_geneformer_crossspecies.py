"""
Geneformer cross-species embedding extraction.

For each tissue: embed human cells directly, remap mouse genes to human
orthologs then embed. kNN evaluation done in the evaluate script.

Run in: vcbench-geneformer environment
GPU: ~2-4 GB VRAM
"""

import os
import sys

import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_DIR)

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
TOKENIZED_DIR = os.path.join(PROJECT_DIR, "data", "tokenized")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "dim_b", "geneformer")
MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "geneformer", "Geneformer", "Geneformer-V2-316M"
)

TISSUES = ["lung", "liver", "heart", "kidney", "brain"]


def extract_embeddings(adata_path, output_prefix, dataset_name):
    """Tokenize and extract Geneformer cell embeddings."""
    # Checkpoint: skip if embeddings already exist
    out_path = os.path.join(RESULTS_DIR, f"{output_prefix}_embeddings.npy")
    if os.path.exists(out_path):
        print(f"  Already extracted: {out_path}")
        return np.load(out_path)

    import tempfile
    from geneformer import EmbExtractor, TranscriptomeTokenizer

    # Tokenize — isolate target file to prevent tokenizer from globbing other h5ads
    basename = os.path.basename(adata_path).replace(".h5ad", "")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.symlink(os.path.abspath(adata_path), os.path.join(tmpdir, os.path.basename(adata_path)))
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"cell_type": "cell_type"},
            use_h5ad_index=True,
        )
        tk.tokenize_data(
            tmpdir + "/",
            TOKENIZED_DIR + "/",
            basename,
            file_format="h5ad",
        )

    # Extract embeddings using mean pooling
    emb_extractor = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        emb_layer=-1,
        forward_batch_size=100,
        nproc=8,
    )

    dataset_path = os.path.join(
        TOKENIZED_DIR,
        os.path.basename(adata_path).replace(".h5ad", ".dataset"),
    )

    embs = emb_extractor.extract_embs(
        MODEL_PATH,
        dataset_path,
        RESULTS_DIR + "/",
        dataset_name,
    )

    emb_array = np.array(embs)
    out_path = os.path.join(RESULTS_DIR, f"{output_prefix}_embeddings.npy")
    np.save(out_path, emb_array)
    print(f"  Embeddings: {emb_array.shape} -> {out_path}")
    return emb_array


def main():
    from src.models.data_prep import prepare_for_geneformer, remap_mouse_to_human

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TOKENIZED_DIR, exist_ok=True)

    print("=== Geneformer Cross-Species Embeddings ===")

    for tissue in TISSUES:
        print(f"\n--- {tissue} ---")

        # Human: prepare and embed
        human_in = os.path.join(PROCESSED_DIR, f"census_homo_sapiens_{tissue}.h5ad")
        human_gf = os.path.join(PROCESSED_DIR, f"census_homo_sapiens_{tissue}_gf.h5ad")
        if os.path.exists(human_in):
            prepare_for_geneformer(human_in, human_gf)
            extract_embeddings(human_gf, f"human_{tissue}", f"human_{tissue}")

            # Save labels
            import scanpy as sc
            adata = sc.read_h5ad(human_in)
            np.save(
                os.path.join(RESULTS_DIR, f"human_{tissue}_labels.npy"),
                adata.obs["cell_type"].values,
            )

        # Mouse: remap to human orthologs, prepare, embed
        mouse_in = os.path.join(PROCESSED_DIR, f"census_mus_musculus_{tissue}.h5ad")
        mouse_remapped = os.path.join(PROCESSED_DIR, f"census_mus_musculus_{tissue}_human_orthologs.h5ad")
        mouse_gf = os.path.join(PROCESSED_DIR, f"census_mus_musculus_{tissue}_gf.h5ad")
        if os.path.exists(mouse_in):
            remap_mouse_to_human(mouse_in, mouse_remapped)
            prepare_for_geneformer(mouse_remapped, mouse_gf)
            extract_embeddings(mouse_gf, f"mouse_{tissue}", f"mouse_{tissue}")

            import scanpy as sc
            adata = sc.read_h5ad(mouse_in)
            np.save(
                os.path.join(RESULTS_DIR, f"mouse_{tissue}_labels.npy"),
                adata.obs["cell_type"].values,
            )

    print("\n=== Geneformer cross-species embeddings complete ===")


if __name__ == "__main__":
    main()
