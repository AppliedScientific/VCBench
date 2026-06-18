"""Dim E end-to-end evaluator: per-dataset τ-b plus across-dataset aggregate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from vcbench.dimensions.dim_e_temporal.aggregation import aggregate_across_datasets


@dataclass(frozen=True)
class DimEPerDatasetResult:
    dataset: str
    kendall_tau_b: float
    n_cells: int = 0
    note: str = ""


@dataclass(frozen=True)
class DimEResult:
    """Aggregate Dim E result across multiple datasets for one model."""

    model: str
    per_dataset: list[DimEPerDatasetResult] = field(default_factory=list)
    aggregate_kendall_tau_b: float = 0.0


def evaluate_dim_e(
    model: str,
    per_dataset_taus: Mapping[str, float],
    *,
    per_dataset_n_cells: Mapping[str, int] | None = None,
    notes: Mapping[str, str] | None = None,
) -> DimEResult:
    """Aggregate per-dataset τ-b values into a Dim E result.

    Parameters
    ----------
    model : str
        Display name for the evaluated model.
    per_dataset_taus : mapping[str, float]
        ``{dataset: kendall_tau_b}``. NaN values are skipped in the aggregate.
    per_dataset_n_cells : mapping[str, int] | None
        Optional cell counts per dataset, propagated into the per-dataset rows.
    notes : mapping[str, str] | None
        Optional per-dataset annotations (e.g. "bootstrap of 10×5K subsamples").

    Returns
    -------
    DimEResult

    Notes
    -----
    Aggregation is **unweighted arithmetic mean** across datasets. See the
    package docstring for the rationale (preserve per-dataset signal,
    don't let cell-count-imbalanced datasets dominate).
    """
    notes = notes or {}
    per_dataset_n_cells = per_dataset_n_cells or {}
    rows = [
        DimEPerDatasetResult(
            dataset=ds,
            kendall_tau_b=float(tau),
            n_cells=int(per_dataset_n_cells.get(ds, 0)),
            note=str(notes.get(ds, "")),
        )
        for ds, tau in per_dataset_taus.items()
    ]
    return DimEResult(
        model=model,
        per_dataset=rows,
        aggregate_kendall_tau_b=aggregate_across_datasets(per_dataset_taus),
    )
