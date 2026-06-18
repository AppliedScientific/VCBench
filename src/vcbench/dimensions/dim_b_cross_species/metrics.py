"""Dim B metrics — macro F1 and weighted F1 (Eq. 4 from VCBench (2026)).

::

    F1_macro    = (1 / |C|) * sum_c F1_c
    F1_weighted = (1 / N)   * sum_c n_c * F1_c

where ``n_c`` is the support of class ``c``, ``N = sum_c n_c``, and ``C`` is
the label set being evaluated. The label set ``C`` depends on the protocol —
under the native protocol it's the per-method-admitted vocabulary; under the
common-label-set protocol (Eq. 5) it's the per-tissue intersection.

Class-count confound: macro F1 depends mechanically on |C|, so the
common-label-set protocol is the **binding** scoring protocol for
VC-Level-defining comparisons. Native-protocol values are reported only
for cross-paper comparability.

Both functions are thin wrappers around ``sklearn.metrics.f1_score`` —
exposed in this package so the Dim B API surface doesn't expose sklearn to
downstream code.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import f1_score


def _validate_inputs(y_true: Sequence, y_pred: Sequence) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true (n={len(y_true)}) and y_pred (n={len(y_pred)}) must be the same length"
        )
    if len(y_true) == 0:
        raise ValueError("y_true is empty; F1 is undefined")


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """Macro-averaged F1 (Eq. 4, first form).

    Equal weight per class regardless of class support. Sensitive to rare
    classes — this is the property that makes |C| the dominant factor in
    cross-method comparisons under the native protocol.

    Parameters
    ----------
    y_true, y_pred : sequence
        Ground-truth and predicted class labels (any hashable type).

    Returns
    -------
    float
        Macro-averaged F1 in [0, 1]. Uses ``zero_division=0`` so classes with
        zero support contribute 0 rather than raising.
    """
    _validate_inputs(y_true, y_pred)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """Support-weighted F1 (Eq. 4, second form).

    Per-class F1 averaged with weights proportional to class support. Less
    sensitive than macro F1 to rare classes; tracks overall accuracy more
    closely on imbalanced label sets.

    Parameters
    ----------
    y_true, y_pred : sequence
        Ground-truth and predicted class labels.

    Returns
    -------
    float
        Support-weighted F1 in [0, 1].
    """
    _validate_inputs(y_true, y_pred)
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
