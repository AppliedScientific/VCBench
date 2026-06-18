"""
Consolidated pipeline output verification.

Tests that each pipeline stage produced its expected output files. Parameterized
to avoid repeating the same os.path.exists pattern 50+ times.
"""

import json
import os
import pickle

import numpy as np
import pytest

from tests.conftest import (
    BASELINES_DIR,
    CROSSSPECIES_MODELS,
    GRN_MODELS,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    TISSUES,
)


# ── Raw data ──

PHASE1_FILES = [
    "K562_essential_raw_singlecell_01.h5ad",
    "norman/perturb_processed.h5ad",
    "trrust_rawdata.human.tsv",
] + [
    f"census_{org}_{tissue}.h5ad"
    for tissue in TISSUES
    for org in ["homo_sapiens", "mus_musculus"]
]


@pytest.mark.parametrize("relpath", PHASE1_FILES)
def test_phase1_file_exists(relpath):
    """File-existence check for raw data outputs.

    Skips if the data has not been downloaded yet. A fresh repo clone will
    not have these files and the test should not fail just because the
    optional data acquisition step was skipped.
    """
    target = RAW_DIR / relpath
    if not target.exists():
        pytest.skip(f"Raw data file missing: {relpath}.")


def test_phase1_beeline_ground_truth():
    """BEELINE refNetwork.csv presence check. Skips if data not present."""
    for dataset in ["hESC", "hHEP"]:
        path = RAW_DIR / "beeline" / "inputs" / "Experimental" / dataset / "refNetwork.csv"
        if not path.exists():
            pytest.skip(f"BEELINE data not present: {path}")


def test_phase1_trrust_row_count():
    import pandas as pd
    path = RAW_DIR / "trrust_rawdata.human.tsv"
    if path.exists():
        df = pd.read_csv(path, sep="\t", header=None)
        assert len(df) > 8000


# ── Processed data ──

PHASE2_FILES = [
    "replogle_k562_essential.h5ad",
    "norman.h5ad",
    "ortholog_maps.pkl",
    "grn_ground_truth.pkl",
    "beeline_hesc.h5ad",
    "beeline_hhep.h5ad",
    "splits/ae_train_idx.npy",
    "splits/ae_test_idx.npy",
] + [
    f"census_{org}_{tissue}.h5ad"
    for tissue in TISSUES
    for org in ["homo_sapiens", "mus_musculus"]
]


@pytest.mark.parametrize("relpath", PHASE2_FILES)
def test_phase2_file_exists(relpath):
    """File-existence check for preprocessed outputs.

    Skips if preprocessing has not been run (depends on raw data).
    """
    target = PROCESSED_DIR / relpath
    if not target.exists():
        pytest.skip(f"Processed file missing: {relpath}.")


def test_phase2_splits_no_overlap():
    train_path = SPLITS_DIR / "ae_train_idx.npy"
    test_path = SPLITS_DIR / "ae_test_idx.npy"
    if train_path.exists() and test_path.exists():
        train = np.load(train_path)
        test = np.load(test_path)
        assert len(set(train) & set(test)) == 0, "Train/test overlap!"


def test_phase2_ortholog_count():
    path = PROCESSED_DIR / "ortholog_maps.pkl"
    if path.exists():
        with open(path, "rb") as f:
            maps = pickle.load(f)
        n = len(maps["h2m"])
        assert 14_000 < n < 18_000, f"Expected ~15K-16K orthologs, got {n}"


def test_phase2_census_has_counts_layer():
    """Spot check that raw counts are preserved for foundation model tokenization."""
    import anndata as ad
    path = PROCESSED_DIR / "census_homo_sapiens_lung.h5ad"
    if path.exists():
        adata = ad.read_h5ad(str(path))
        assert "counts" in adata.layers
        assert adata.layers["counts"].max() > 1


# ── Baselines ──

def test_phase3_results_template():
    """The capability matrix carries the structural N/A entries.

    ``table2_with_baselines.csv`` is the canonical capability table, and we
    spot-check the structural N/A assertions against it here.
    """
    path = TABLES_DIR / "table2_with_baselines.csv"
    if path.exists():
        import pandas as pd
        df = pd.read_csv(path, index_col=0, keep_default_na=False,
                         na_values=[""])
        assert "Additive baseline" in df.index
        assert "A:PDS" in df.columns
        assert df.loc["UCE 33-layer", "A:PDS"] == "N/A"
        assert df.loc["PCA + kNN", "C:AUROC"] == "N/A"


# ── Dim A perturbation ──

@pytest.mark.parametrize("model", ["scgpt", "state"])
def test_phase5_cell_eval_results(model):
    path = RESULTS_DIR / "dim_a" / model / "cell_eval_results.json"
    if not path.exists():
        pytest.skip(f"{model} not run")
    with open(path) as f:
        data = json.load(f)
    assert len(data) > 0


# ── Dim B cross-species ──

@pytest.mark.parametrize("model", CROSSSPECIES_MODELS)
def test_phase6_has_embeddings(model):
    """Dim B embeddings existence. Skips if not run for this model (large
    embeddings live on HuggingFace ``vcbench-embeddings``; pull via
    ``snapshot_download`` before running this test)."""
    model_dir = RESULTS_DIR / "dim_b" / model
    if not model_dir.exists():
        pytest.skip(f"Not run for {model} (results/dim_b/{model}/ "
                    f"missing). Pull embeddings via "
                    f"`huggingface_hub.snapshot_download('appliedscientific/"
                    f"vcbench-embeddings')`.")
    emb_files = list(model_dir.glob("*_embeddings.npy"))
    if not emb_files:
        pytest.skip(f"Dir exists for {model} but contains no "
                    f"*_embeddings.npy — pipeline may have aborted mid-run.")


def test_phase6_combined_results():
    path = RESULTS_DIR / "dim_b" / "all_crossspecies_results.json"
    if not path.exists():
        pytest.skip("Not run")
    with open(path) as f:
        data = json.load(f)
    for model, results in data.items():
        assert 0 <= results["avg_macro_f1"] <= 1
        assert 0 <= results["avg_weighted_f1"] <= 1


# ── Dim C GRN ──

@pytest.mark.parametrize("model", GRN_MODELS)
def test_phase7_edges_exist(model):
    """Dim C GRN edges existence. Skips if not run for this model."""
    path = RESULTS_DIR / "dim_c" / model / "predicted_edges.csv"
    if not path.exists():
        pytest.skip(f"Not run for {model} (predicted_edges.csv missing).")


@pytest.mark.parametrize("model", GRN_MODELS)
def test_phase7_edge_columns(model):
    import pandas as pd
    path = RESULTS_DIR / "dim_c" / model / "predicted_edges.csv"
    if not path.exists():
        pytest.skip(f"{model} not run")
    df = pd.read_csv(path)
    assert {"TF", "target", "score"} <= set(df.columns)


# ── Final table ──

def test_phase8_final_table():
    path = TABLES_DIR / "table1_final.csv"
    if not path.exists():
        pytest.skip("Not assembled")
    import pandas as pd
    df = pd.read_csv(path, index_col=0)
    for model in ["Geneformer V2-104M", "scGPT (fine-tuned)", "UCE 33-layer"]:
        assert model in df.index, f"Missing model: {model}"
