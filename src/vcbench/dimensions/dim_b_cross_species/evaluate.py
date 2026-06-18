"""Dim B end-to-end evaluator — per-tissue dual-protocol scoring.

Aggregates a (model, tissue, y_true, y_pred) collection into:

* per-tissue native and common-set TissueScore values
* a per-tissue intersection vocabulary derived from
  :func:`vcbench.protocols.common_label_set`
* aggregate macro F1 (mean over tissues) under each protocol

The evaluator is intentionally sklearn-free at the surface: it accepts
already-predicted labels rather than running PCA+kNN itself, so it composes
cleanly over either baseline predictions or foundation-model embedding
predictions. To run the binding PCA+kNN baseline as the predictions, fit
:func:`vcbench.dimensions.dim_b_cross_species.baselines.pca_knn_classifier`
upstream and feed the predictions in here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from vcbench.dimensions.dim_b_cross_species.protocols import (
    TissueScore,
    score_native,
    score_under_common_set,
)
from vcbench.protocols import common_label_set


@dataclass(frozen=True)
class DimBPerTissueResult:
    """Per-tissue dual-protocol scores for one (model, tissue) cell."""

    tissue: str
    native: TissueScore
    common_set: TissueScore | None  # None if tissue lacks a common-set vocabulary


@dataclass(frozen=True)
class DimBResult:
    """Aggregate Dim B result for one model across multiple tissues.

    Attributes
    ----------
    model : str
        Display name for the evaluated model.
    per_tissue : list[DimBPerTissueResult]
        One entry per tissue evaluated.
    aggregate_native_macro_f1 : float
        Mean native macro F1 across tissues. Reported in §I.4 Table 2.
    aggregate_common_set_macro_f1 : float | None
        Mean common-set macro F1 across tissues that have a common-set score.
        ``None`` if no tissue had a common-set score.
    """

    model: str
    per_tissue: list[DimBPerTissueResult] = field(default_factory=list)
    aggregate_native_macro_f1: float = 0.0
    aggregate_common_set_macro_f1: float | None = None


def evaluate_dim_b(
    model: str,
    per_tissue_predictions: Mapping[str, dict],
    method_label_vocabularies: Mapping[str, Mapping[str, set[str]]] | None = None,
    *,
    common_label_sets: Mapping[str, set[str]] | None = None,
) -> DimBResult:
    """Run Dim B per-tissue dual-protocol evaluation for one model.

    Parameters
    ----------
    model : str
        Display name for the model under evaluation (used only in the
        returned ``DimBResult.model`` field).
    per_tissue_predictions : mapping[str, dict]
        ``{tissue: {"y_true": [...], "y_pred": [...]}}``. The model's
        predictions for each tissue, already aligned to ground-truth labels.
    method_label_vocabularies : mapping[str, mapping[str, set[str]]] | None
        ``{method_id: {tissue: set_of_labels}}`` covering every method whose
        per-tissue vocabulary should enter the common-label-set intersection.
        Required if ``common_label_sets`` is not provided.
    common_label_sets : mapping[str, set[str]] | None
        Pre-computed per-tissue intersections. If provided, takes precedence
        over ``method_label_vocabularies`` (the latter then only needs to be
        ``None``). Useful when the intersection has been computed once and
        cached, e.g. ``results/dim_b/common_label_set_per_tissue.json``.

    Returns
    -------
    DimBResult

    Raises
    ------
    ValueError
        If neither ``method_label_vocabularies`` nor ``common_label_sets``
        is supplied, or if a tissue's prediction dict lacks ``y_true`` /
        ``y_pred`` keys.
    """
    if common_label_sets is None and method_label_vocabularies is None:
        raise ValueError(
            "must supply either common_label_sets or method_label_vocabularies"
        )
    if common_label_sets is None:
        common_label_sets = common_label_set(dict(method_label_vocabularies))

    rows: list[DimBPerTissueResult] = []
    native_macro_vals: list[float] = []
    common_macro_vals: list[float] = []
    for tissue, payload in per_tissue_predictions.items():
        if "y_true" not in payload or "y_pred" not in payload:
            raise ValueError(
                f"per_tissue_predictions[{tissue!r}] must have 'y_true' and 'y_pred' keys"
            )
        nat = score_native(payload["y_true"], payload["y_pred"])
        native_macro_vals.append(nat.macro_f1)

        common = common_label_sets.get(tissue, set())
        common_score: TissueScore | None
        if common:
            try:
                common_score = score_under_common_set(
                    payload["y_true"], payload["y_pred"], common
                )
                common_macro_vals.append(common_score.macro_f1)
            except ValueError:
                # Tissue's common-set vocabulary doesn't intersect this
                # model's true labels — score is undefined for this cell.
                common_score = None
        else:
            common_score = None

        rows.append(DimBPerTissueResult(tissue=tissue, native=nat, common_set=common_score))

    agg_common: float | None
    agg_common = (
        sum(common_macro_vals) / len(common_macro_vals)
        if common_macro_vals
        else None
    )
    return DimBResult(
        model=model,
        per_tissue=rows,
        aggregate_native_macro_f1=(
            sum(native_macro_vals) / len(native_macro_vals)
            if native_macro_vals
            else 0.0
        ),
        aggregate_common_set_macro_f1=agg_common,
    )
