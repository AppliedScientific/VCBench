"""Unit tests for vcbench.dimensions.dim_b_cross_species.metrics (Eq. 4)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from vcbench.dimensions.dim_b_cross_species.metrics import macro_f1, weighted_f1


def test_macro_f1_perfect_classification():
    y = ["A", "B", "C", "A", "B"]
    assert macro_f1(y, y) == pytest.approx(1.0)


def test_macro_f1_all_wrong():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["B", "A", "B", "A"]
    assert macro_f1(y_true, y_pred) == pytest.approx(0.0)


def test_weighted_f1_perfect_classification():
    y = ["A", "B", "C", "A", "B"]
    assert weighted_f1(y, y) == pytest.approx(1.0)


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 5, n)
    y_pred = rng.integers(0, 5, n)
    expected = f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert macro_f1(y_true, y_pred) == pytest.approx(expected, abs=1e-12)


def test_weighted_f1_matches_sklearn():
    rng = np.random.default_rng(1)
    n = 500
    y_true = rng.integers(0, 5, n)
    y_pred = rng.integers(0, 5, n)
    expected = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    assert weighted_f1(y_true, y_pred) == pytest.approx(expected, abs=1e-12)


def test_class_count_confound_demonstrated():
    """Macro F1 mechanically inflates when the label set shrinks — this is
    the exact phenomenon the common-label-set protocol exists to correct."""
    # 5-class problem: random predictions → low macro F1
    rng = np.random.default_rng(2)
    n = 1000
    five_class_true = rng.integers(0, 5, n)
    five_class_pred = rng.integers(0, 5, n)
    macro_5class = macro_f1(five_class_true, five_class_pred)

    # 2-class problem: random predictions on the SAME cells but collapsed to
    # binary {0,1} — chance level here is much higher (≈0.5 vs ≈0.2).
    two_class_true = (five_class_true >= 3).astype(int)
    two_class_pred = (five_class_pred >= 3).astype(int)
    macro_2class = macro_f1(two_class_true, two_class_pred)

    assert macro_2class > macro_5class + 0.15  # comfortably bigger


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        macro_f1(["A"], ["A", "B"])
    with pytest.raises(ValueError):
        weighted_f1(["A"], ["A", "B"])


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        macro_f1([], [])
    with pytest.raises(ValueError):
        weighted_f1([], [])
