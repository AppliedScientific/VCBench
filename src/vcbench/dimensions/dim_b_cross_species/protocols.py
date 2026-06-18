"""Native vs common-label-set scoring protocols for Dim B (Eq. 5).

Both protocols collapse a (y_true, y_pred) pair to one (macro F1, weighted F1)
score per tissue; they differ in which cells enter the score.

* **Native protocol**: every cell with a non-null cell-type label enters the
  score. Per-method label vocabularies vary (Geneformer drops cells whose
  in-vocabulary gene count is below the rank-based filter), so methods are
  scored on different label sets — the class-count confound.

* **Common-label-set protocol** (Eq. 5): cells are restricted to the
  per-tissue intersection of the cell-type vocabularies admitted by every
  evaluated method. This removes the class-count confound at the cost of
  reducing the cell pool.

The intersection itself is computed by
:func:`vcbench.protocols.common_label_set` from this same package.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from vcbench.dimensions.dim_b_cross_species.metrics import macro_f1, weighted_f1


@dataclass(frozen=True)
class TissueScore:
    """Macro F1 / weighted F1 / class count for one (model, tissue) cell."""

    macro_f1: float
    weighted_f1: float
    n_classes: int
    n_eval_cells: int


def score_native(y_true: Sequence, y_pred: Sequence) -> TissueScore:
    """Native-protocol score: macro / weighted F1 over every (true, pred) pair as-is.

    The label set ``C`` is the union of distinct values that appear in
    ``y_true`` (sklearn's default behaviour for ``f1_score(average='macro')``).
    No restriction to a common vocabulary.

    Parameters
    ----------
    y_true, y_pred : sequence
        Ground-truth and predicted cell-type labels.

    Returns
    -------
    TissueScore
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = int(len(set(y_true)))
    return TissueScore(
        macro_f1=macro_f1(y_true, y_pred),
        weighted_f1=weighted_f1(y_true, y_pred),
        n_classes=n_classes,
        n_eval_cells=len(y_true),
    )


def score_under_common_set(
    y_true: Sequence,
    y_pred: Sequence,
    common_classes: Iterable[str],
) -> TissueScore:
    """Common-label-set scored macro / weighted F1 (Eq. 5 application).

    Restricts to cells whose **true** label lies in ``common_classes``, then
    computes macro/weighted F1 on that subset.

    Predicted labels outside ``common_classes`` are kept as-is — they simply
    register as wrong predictions for whichever class they fall under in the
    confusion matrix. This matches the existing pipeline behaviour persisted
    in ``results/dim_b/common_label_macroF1.csv``.

    Parameters
    ----------
    y_true, y_pred : sequence
        Ground-truth and predicted cell-type labels.
    common_classes : iterable of str
        Per-tissue intersection from
        :func:`vcbench.protocols.common_label_set`.

    Returns
    -------
    TissueScore
        Counts and scores reflect the post-filter cell subset.

    Raises
    ------
    ValueError
        If the resulting subset is empty (no true labels in common_classes).
    """
    common = frozenset(common_classes)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true (n={len(y_true)}) and y_pred (n={len(y_pred)}) length mismatch"
        )
    mask = np.array([t in common for t in y_true])
    if not mask.any():
        raise ValueError(
            "no cells with true label in common_classes — common-set "
            "protocol cannot score this (model, tissue) pair"
        )
    yt = y_true[mask]
    yp = y_pred[mask]
    return TissueScore(
        macro_f1=macro_f1(yt, yp),
        weighted_f1=weighted_f1(yt, yp),
        n_classes=len(common),
        n_eval_cells=int(mask.sum()),
    )
