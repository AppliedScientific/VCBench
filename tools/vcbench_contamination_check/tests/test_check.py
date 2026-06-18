"""Unit tests for the contamination validator.

Covers all five verdict paths:
  * confirmed
  * likely
  * unlikely
  * unknown
  * schema_incomplete

Uses tmp_path fixtures and tiny in-memory AnnData files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from vcbench_contamination_check import check_contamination


# ── Fixtures ────────────────────────────────────────────────────────────

def _write_md5_parquet(path: Path, barcodes: list[str]) -> None:
    df = pd.DataFrame({
        "barcode_md5":     [hashlib.md5(b.encode("utf-8")).hexdigest() for b in barcodes],
        "source_accession": ["TEST_GSE"] * len(barcodes),
        "corpus_version":   ["test-2026-01"] * len(barcodes),
    })
    df.to_parquet(path)


def _write_manifest(path: Path, sources: list[dict]) -> None:
    path.write_text(yaml.safe_dump({
        "corpus_version": "test-2026-01",
        "model_name":     "test-model",
        "sources":        sources,
    }))


def _make_eval_h5ad(path: Path, barcodes: list[str], accession: str | None) -> None:
    a = ad.AnnData(X=np.zeros((len(barcodes), 2), dtype=np.float32))
    a.obs_names = barcodes
    if accession is not None:
        a.uns["source_accession"] = accession
    a.write_h5ad(path)


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model_release"
    d.mkdir()
    return d


# ── Tests ───────────────────────────────────────────────────────────────

def test_confirmed_verdict(tmp_path, release_dir):
    """Overlap between pretraining barcodes and eval barcodes → confirmed."""
    shared = ["AAAA", "CCCC"]
    _write_md5_parquet(release_dir / "training_cells.md5.parquet", shared + ["GGGG"])
    _write_manifest(release_dir / "pretraining_manifest.yaml", [
        {"accession": "TEST_GSE", "included": True, "cells_used": 3, "notes": "test"},
    ])
    eval_path = tmp_path / "eval.h5ad"
    _make_eval_h5ad(eval_path, shared + ["TTTT"], accession="OTHER_GSE")

    out = check_contamination(release_dir, eval_path)
    assert out["verdict"] == "confirmed"
    assert out["overlap_cells"] == 2
    assert 0 < out["overlap_fraction"] <= 1


def test_likely_verdict(tmp_path, release_dir):
    """No overlap but accession marked included → likely."""
    _write_md5_parquet(release_dir / "training_cells.md5.parquet", ["GGGG", "TTTT"])
    _write_manifest(release_dir / "pretraining_manifest.yaml", [
        {"accession": "TEST_GSE", "included": True, "cells_used": 100, "notes": "used in pretrain"},
    ])
    eval_path = tmp_path / "eval.h5ad"
    _make_eval_h5ad(eval_path, ["AAAA", "CCCC"], accession="TEST_GSE")

    out = check_contamination(release_dir, eval_path)
    assert out["verdict"] == "likely"
    assert out["accession"] == "TEST_GSE"


def test_unlikely_verdict(tmp_path, release_dir):
    """No overlap, accession marked excluded → unlikely."""
    _write_md5_parquet(release_dir / "training_cells.md5.parquet", ["GGGG", "TTTT"])
    _write_manifest(release_dir / "pretraining_manifest.yaml", [
        {
            "accession":   "TEST_GSE",
            "included":    False,
            "cells_used":  0,
            "exclusions":  ["perturbation_assay"],
            "notes":       "excluded per Census cell-culture filter",
        },
    ])
    eval_path = tmp_path / "eval.h5ad"
    _make_eval_h5ad(eval_path, ["AAAA", "CCCC"], accession="TEST_GSE")

    out = check_contamination(release_dir, eval_path)
    assert out["verdict"] == "unlikely"
    assert "perturbation_assay" in out["exclusions"]


def test_unknown_verdict_accession_not_listed(tmp_path, release_dir):
    _write_md5_parquet(release_dir / "training_cells.md5.parquet", ["GGGG"])
    _write_manifest(release_dir / "pretraining_manifest.yaml", [
        {"accession": "OTHER_GSE", "included": True, "cells_used": 100},
    ])
    eval_path = tmp_path / "eval.h5ad"
    _make_eval_h5ad(eval_path, ["AAAA"], accession="TEST_GSE")

    out = check_contamination(release_dir, eval_path)
    assert out["verdict"] == "unknown"
    assert "not listed" in out["reason"]


def test_unknown_verdict_no_accession_in_uns(tmp_path, release_dir):
    _write_md5_parquet(release_dir / "training_cells.md5.parquet", ["GGGG"])
    _write_manifest(release_dir / "pretraining_manifest.yaml", [
        {"accession": "TEST_GSE", "included": True, "cells_used": 100},
    ])
    eval_path = tmp_path / "eval.h5ad"
    _make_eval_h5ad(eval_path, ["AAAA"], accession=None)  # no source_accession

    out = check_contamination(release_dir, eval_path)
    assert out["verdict"] == "unknown"
    assert "source_accession" in out["reason"]


def test_schema_incomplete_missing_md5(tmp_path, release_dir):
    _write_manifest(release_dir / "pretraining_manifest.yaml", [
        {"accession": "TEST_GSE", "included": True, "cells_used": 10},
    ])
    eval_path = tmp_path / "eval.h5ad"
    _make_eval_h5ad(eval_path, ["AAAA"], accession="TEST_GSE")

    out = check_contamination(release_dir, eval_path)
    assert out["verdict"] == "schema_incomplete"
    assert any("training_cells.md5" in m for m in out["missing"])


def test_schema_incomplete_missing_manifest(tmp_path, release_dir):
    _write_md5_parquet(release_dir / "training_cells.md5.parquet", ["GGGG"])
    eval_path = tmp_path / "eval.h5ad"
    _make_eval_h5ad(eval_path, ["AAAA"], accession="TEST_GSE")

    out = check_contamination(release_dir, eval_path)
    assert out["verdict"] == "schema_incomplete"
    assert any("pretraining_manifest.yaml" in m for m in out["missing"])
