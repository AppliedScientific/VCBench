"""Unit tests for vcbench.dimensions.dim_b_cross_species.protocols."""

from __future__ import annotations

import numpy as np
import pytest

from vcbench.dimensions.dim_b_cross_species.protocols import (
    score_native,
    score_under_common_set,
)


def test_score_native_perfect_classification():
    y = np.array(["A", "B", "C", "A", "B"])
    score = score_native(y, y)
    assert score.macro_f1 == pytest.approx(1.0)
    assert score.weighted_f1 == pytest.approx(1.0)
    assert score.n_classes == 3
    assert score.n_eval_cells == 5


def test_score_native_n_classes_counts_only_y_true():
    """n_classes reflects ground-truth vocabulary; predicted-only labels don't count."""
    y_true = np.array(["A", "A", "A"])
    y_pred = np.array(["A", "B", "C"])  # B, C predicted but not in true
    score = score_native(y_true, y_pred)
    assert score.n_classes == 1


def test_common_set_filters_to_intersection():
    """Cells whose true label is outside the common set are dropped from the score."""
    y_true = np.array(["A", "B", "C", "A", "B", "D"])
    y_pred = np.array(["A", "B", "X", "A", "B", "Y"])
    common = {"A", "B"}
    score = score_under_common_set(y_true, y_pred, common)
    # 4 cells survived (the two A's and two B's). All correct → F1=1.
    assert score.n_eval_cells == 4
    assert score.n_classes == 2
    assert score.macro_f1 == pytest.approx(1.0)


def test_common_set_keeps_predicted_outside_labels_as_wrong():
    """A predicted label outside common_classes is *not* dropped — it counts as wrong."""
    y_true = np.array(["A", "B", "A", "B"])
    y_pred = np.array(["A", "OTHER", "A", "B"])
    common = {"A", "B"}
    score = score_under_common_set(y_true, y_pred, common)
    assert score.n_eval_cells == 4   # all true labels survived (A, B in common)
    # 3 of 4 predictions correct (A,A,B), 1 wrong (B→OTHER) → not perfect F1
    assert score.macro_f1 < 1.0


def test_common_set_class_count_confound_correction():
    """The whole point of the protocol: a method scored on a 2-class subset
    of a 5-class true vocab gets a different macro F1 than under native."""
    y_true = np.array(["A", "B", "C", "D", "E"] * 100)
    # Predictions: perfect on A and B, random across C/D/E
    rng = np.random.default_rng(0)
    y_pred = y_true.copy()
    cde_mask = np.isin(y_true, ["C", "D", "E"])
    y_pred[cde_mask] = rng.choice(["C", "D", "E"], size=cde_mask.sum())

    native = score_native(y_true, y_pred)
    common = score_under_common_set(y_true, y_pred, {"A", "B"})
    # Restricting to A+B (perfect cells only) → macro F1 = 1.0
    assert common.macro_f1 == pytest.approx(1.0)
    # Native macro F1 dragged down by random C/D/E
    assert native.macro_f1 < common.macro_f1 - 0.1


def test_common_set_empty_intersection_raises():
    y_true = np.array(["A", "B"])
    y_pred = np.array(["A", "B"])
    with pytest.raises(ValueError, match="no cells"):
        score_under_common_set(y_true, y_pred, {"X", "Y"})


def test_common_set_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        score_under_common_set(["A"], ["A", "B"], {"A"})
