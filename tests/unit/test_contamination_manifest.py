"""Unit tests for vcbench.contamination."""

from __future__ import annotations

from pathlib import Path

import pytest

from vcbench.contamination import (
    ContaminationManifest,
    ManifestValidationError,
    ValidationSummary,
    validate_manifest,
)


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "src" / "vcbench" / "contamination" / "examples"


def _minimal_dict(**overrides):
    base = {
        "schema_version": "vcbench-contamination-v1",
        "model": "geneformer-v2-316m",
        "pretraining_corpus": "Genecorpus-30M",
    }
    base.update(overrides)
    return base


def test_round_trip_minimal_manifest():
    summary = validate_manifest(_minimal_dict())
    assert isinstance(summary, ValidationSummary)
    assert summary.model == "geneformer-v2-316m"
    assert summary.n_included == 0
    assert summary.n_excluded == 0
    assert summary.has_md5 is False
    # Two warnings: missing MD5, and empty included+excluded lists.
    assert any("md5" in w.lower() for w in summary.warnings)
    assert any("included" in w.lower() and "excluded" in w.lower() for w in summary.warnings)


def test_full_manifest_no_warnings():
    summary = validate_manifest(
        _minimal_dict(
            pretraining_corpus_md5="0123456789abcdef" * 2,
            excluded_datasets=["GSE133344"],
        )
    )
    assert summary.has_md5 is True
    assert summary.n_excluded == 1
    assert summary.warnings == []


def test_wrong_schema_version_raises():
    with pytest.raises(ManifestValidationError) as ei:
        validate_manifest(_minimal_dict(schema_version="vcbench-contamination-v0"))
    assert any("schema_version" in e for e in ei.value.errors)


def test_extra_field_rejected():
    with pytest.raises(ManifestValidationError):
        validate_manifest(_minimal_dict(typo_field="oops"))


def test_load_from_yaml_path(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "schema_version: vcbench-contamination-v1\n"
        "model: scgpt\n"
        "pretraining_corpus: CELLxGENE Census\n"
        "excluded_datasets: [GSE133344]\n"
    )
    summary = validate_manifest(p)
    assert summary.model == "scgpt"
    assert summary.n_excluded == 1


def test_load_from_json_path(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(
        '{"schema_version": "vcbench-contamination-v1", '
        '"model": "uce", "pretraining_corpus": "Census"}'
    )
    summary = validate_manifest(p)
    assert summary.model == "uce"


def test_pydantic_model_constructible_directly():
    """The Pydantic model should be importable and instantiable on its own
    (so downstream consumers can compose it without going through validate_manifest)."""
    m = ContaminationManifest(
        schema_version="vcbench-contamination-v1",
        model="arc-state-transition",
        pretraining_corpus="Adduri 2025 100M perturbed obs",
        included_datasets=["10.25452/figshare.plus.20029387"],
    )
    assert m.included_datasets == ["10.25452/figshare.plus.20029387"]


@pytest.mark.parametrize("name", [
    "geneformer_genecorpus30m.yaml",
    "scgpt_cellxgene33m.yaml",
    "uce_cellxgene.yaml",
    "transcriptformer_112m.yaml",
    "arc_state_transition.yaml",
])
def test_shipped_examples_validate(name):
    """Every example manifest under contamination/examples/ must validate clean."""
    summary = validate_manifest(EXAMPLES_DIR / name)
    assert summary.schema_version == "vcbench-contamination-v1"


def test_arc_state_example_records_replogle_inclusion():
    """The Arc State example must declare Replogle as a known training inclusion —
    the load-bearing detail behind VCBench's decision not to evaluate Replogle."""
    summary = validate_manifest(EXAMPLES_DIR / "arc_state_transition.yaml")
    assert summary.n_included >= 1
