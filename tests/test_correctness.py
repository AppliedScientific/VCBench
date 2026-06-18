"""
Algorithmic correctness spot-checks.

Tests actual computation logic, not just file existence.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import BASELINES_DIR, PROCESSED_DIR, RESULTS_DIR


class TestGRNEdgeRanking:
    """Verify GRN edges are properly sorted by score."""

    @pytest.mark.parametrize("source", [
        BASELINES_DIR / "coexpr_hesc_edges.csv",
        RESULTS_DIR / "dim_c" / "geneformer" / "predicted_edges.csv",
        RESULTS_DIR / "dim_c" / "scgpt" / "predicted_edges.csv",
    ])
    def test_scores_sorted_descending(self, source):
        if not source.exists():
            pytest.skip(f"Not found: {source}")
        df = pd.read_csv(source)
        scores = df["score"].values
        assert np.all(scores[:-1] >= scores[1:]), "Scores not sorted descending"


class TestGRNEvalMetrics:
    """Verify GRN eval metrics are in valid ranges."""

    @pytest.mark.parametrize("model", ["geneformer", "scgpt", "transcriptformer"])
    def test_auroc_valid_range(self, model):
        for gt in ["beeline_hesc", "trrust"]:
            path = RESULTS_DIR / "dim_c" / model / f"grn_eval_{gt}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                auroc = data.get("AUROC")
                if auroc is not None and auroc == auroc:  # not NaN
                    assert 0.3 <= auroc <= 1.0, f"AUROC={auroc:.3f} — likely bug if <0.4"
                return
        pytest.skip(f"{model}: no eval results")


class TestPerturbationBaselines:
    """Verify baseline predictions are numerically valid."""

    def test_additive_no_nan(self):
        path = BASELINES_DIR / "additive_norman.h5ad"
        if not path.exists():
            pytest.skip("Not run")
        import anndata as ad
        adata = ad.read_h5ad(str(path))
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        assert not np.any(np.isnan(X)), "NaN in additive predictions"
        assert not np.any(np.isinf(X)), "Inf in additive predictions"

    def test_additive_has_predictions(self):
        path = BASELINES_DIR / "additive_norman.h5ad"
        if not path.exists():
            pytest.skip("Not run")
        import anndata as ad
        adata = ad.read_h5ad(str(path))
        assert adata.n_obs > 100, "Expected >100 double perturbation predictions"


class TestCrossSpeciesEmbeddings:
    """Verify embedding dimensions and scale."""

    EXPECTED_DIMS = {"geneformer": 512, "scgpt": 512, "uce": 1280}

    @pytest.mark.parametrize("model,dim", EXPECTED_DIMS.items())
    def test_embedding_dimensions(self, model, dim):
        model_dir = RESULTS_DIR / "dim_b" / model
        if not model_dir.exists():
            pytest.skip(f"{model} not run")
        for f in model_dir.glob("*_embeddings.npy"):
            emb = np.load(f)
            assert emb.shape[1] == dim, f"{f.name}: expected dim {dim}, got {emb.shape[1]}"
            assert emb.std() < 100, f"{f.name}: embedding scale too large"
            break
        else:
            pytest.skip("No embeddings found")
