"""Tests for vcbench.models — regime-declaration audits + train/test-overlap guard.

The load-bearing tests in this file are:

1. **Regime declarations match §I.4**: every wrapper's
   ``per_dimension_regime`` dict must reproduce the §I.4 capability matrix
   row for that model. Drift here means the capability matrix is no longer
   directly auditable from code.

2. **Arc State train/test-overlap guard**: the ArcState wrapper must refuse
   to load any config with nonzero train/test perturbation overlap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vcbench.models import (
    SCGPT,
    UCE,
    ArcState,
    FoundationModel,
    Geneformer,
    TranscriptFormer,
)
from vcbench.models.arc_state import (
    DEFAULT_NORMAN_GEARS_CONFIG,
    TrainTestOverlapError,
)


# ---- Regime declarations match §I.4 ---------------------------------------

EXPECTED_REGIMES_FROM_SECTION_I_4 = {
    Geneformer:        {"A": "FT+D", "B": "ZS",  "C": "IE",  "D": "ZS+D", "E": "ZS"},
    SCGPT:             {"A": "FT",   "B": "ZS",  "C": "IE",  "D": "ZS+D", "E": "ZS"},
    UCE:               {"A": "N/A",  "B": "ZS",  "C": "N/A", "D": "ZS+D", "E": "ZS"},
    TranscriptFormer:  {"A": "ZS+D", "B": "ZS",  "C": "N/A", "D": "ZS+D", "E": "ZS+D"},
    ArcState:          {"A": "FT",   "B": "N/A", "C": "N/A", "D": "DNR",  "E": "DNR"},
}


@pytest.mark.parametrize(
    "model_cls,expected", list(EXPECTED_REGIMES_FROM_SECTION_I_4.items())
)
def test_regime_declarations_match_section_i_4(model_cls, expected):
    """Each wrapper's per_dimension_regime must reproduce §I.4 verbatim.

    Drift here breaks the manuscript-to-code traceability that reviewers
    use to verify N/A and DNR designations.
    """
    instance = model_cls()
    assert instance.per_dimension_regime == expected, (
        f"{model_cls.__name__} regime declaration drifted from §I.4: "
        f"got {instance.per_dimension_regime}, expected {expected}"
    )


@pytest.mark.parametrize("model_cls", list(EXPECTED_REGIMES_FROM_SECTION_I_4))
def test_each_wrapper_subclasses_foundation_model(model_cls):
    assert issubclass(model_cls, FoundationModel)


@pytest.mark.parametrize("model_cls", list(EXPECTED_REGIMES_FROM_SECTION_I_4))
def test_each_wrapper_has_name(model_cls):
    assert isinstance(model_cls.name, str) and len(model_cls.name) > 0


# ---- is_dimension_supported reflects N/A and DNR -------------------------


def test_uce_dim_a_and_dim_c_marked_unsupported():
    uce = UCE()
    assert uce.is_dimension_supported("A") is False
    assert uce.is_dimension_supported("C") is False
    assert uce.is_dimension_supported("B") is True
    assert uce.is_dimension_supported("E") is True


def test_arc_state_dim_d_and_dim_e_marked_dnr():
    arc = ArcState()
    assert arc.is_dimension_supported("D") is False    # DNR
    assert arc.is_dimension_supported("E") is False    # DNR
    assert arc.is_dimension_supported("A") is True     # FT


def test_uce_predict_perturbation_raises_with_design_scope_message():
    """The N/A justification must be surfaced in the error message so the
    'why' is recoverable from the exception alone."""
    uce = UCE()
    with pytest.raises(NotImplementedError, match="N/A"):
        uce.predict_perturbation([])


def test_uce_extract_gene_attention_raises_with_design_scope_message():
    uce = UCE()
    with pytest.raises(NotImplementedError, match="N/A"):
        uce.extract_gene_attention()


def test_transcriptformer_extract_gene_attention_documents_dim_c_na():
    """TF Dim C is N/A per Supp Note 2 §S2.1 (implementation gap, not model
    failure). The error message must distinguish that."""
    tf = TranscriptFormer()
    with pytest.raises(NotImplementedError, match="S2.1"):
        tf.extract_gene_attention()


# ---- Arc State train/test-overlap guard (CRITICAL) ------------------------


def test_arc_state_default_config_path_is_gears_split():
    """The default config path must be the Norman GEARS-split TOML."""
    arc = ArcState()
    arc.load_pretrained("dummy_checkpoint_path")
    assert arc.config_path == DEFAULT_NORMAN_GEARS_CONFIG
    assert "gears_split" in str(arc.config_path)


def test_arc_state_raises_on_overlapping_train_test_perts(tmp_path):
    """If a custom config has nonzero overlap between train and test
    perturbation IDs, the wrapper must refuse to load."""
    overlapping = tmp_path / "overlapping.toml"
    overlapping.write_text(
        '[train]\n'
        'perturbations = ["AHR+KLF1", "BCL2L11+ctrl", "CEBPE+RUNX1T1"]\n'
        '[test]\n'
        'perturbations = ["AHR+KLF1", "FOXA1+ctrl"]\n'   # AHR+KLF1 in both
    )
    arc = ArcState()
    with pytest.raises(TrainTestOverlapError, match="AHR\\+KLF1|perturbations in both"):
        arc.load_pretrained("dummy_checkpoint", config_path=overlapping)


def test_arc_state_clean_config_loads_without_error(tmp_path):
    clean = tmp_path / "clean.toml"
    clean.write_text(
        '[train]\nperturbations = ["A+B", "C+D"]\n'
        '[test]\nperturbations  = ["E+F", "G+H"]\n'
    )
    arc = ArcState()
    arc.load_pretrained("dummy_checkpoint", config_path=clean)
    assert arc._train_test_overlap == 0


def test_arc_state_default_config_file_exists_in_repo():
    """If the spec'd configs/dim_a/arc_state_norman_gears_split.toml is
    missing the wrapper can't run the overlap guard. Verifies the layout."""
    assert DEFAULT_NORMAN_GEARS_CONFIG.exists(), (
        f"Spec-mandated default config missing: {DEFAULT_NORMAN_GEARS_CONFIG}"
    )


# ---- pre_registration.yaml --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_REG_PATH = REPO_ROOT / "configs" / "pre_registration.yaml"


def test_pre_registration_file_exists():
    assert PRE_REG_PATH.exists()


def test_pre_registration_expected_assignments_match_section_i_5():
    """The pre-registered VC Level assignments must reproduce §I.5 exactly."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(PRE_REG_PATH.read_text())
    expected = {
        "geneformer_v2_316m": 1,
        "scgpt_fine_tuned":   1,
        "uce_33_layer":       1,
        "transcriptformer":   2,
        "arc_state":          1,
    }
    assert cfg["expected_assignments"] == expected
