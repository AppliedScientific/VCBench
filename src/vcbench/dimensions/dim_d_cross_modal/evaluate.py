"""Dim D end-to-end evaluator: per-protein Pearson + RMSE for one (model, test) pair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vcbench.dimensions.dim_d_cross_modal.metrics import (
    mean_pearson_per_protein,
    median_pearson_per_protein,
    pearson_per_protein,
    rmse,
)


@dataclass(frozen=True)
class DimDResult:
    """Aggregate Dim D scores for one (model, evaluation-set) pair.

    Schema mirrors ``results/dim_d/<model>/crossmodal_results.json``.
    """

    mean_pearson_r: float
    median_pearson_r: float
    rmse: float
    n_proteins: int
    n_valid_proteins: int
    model: str = ""

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "mean_pearson_r": self.mean_pearson_r,
            "median_pearson_r": self.median_pearson_r,
            "rmse": self.rmse,
            "n_proteins": self.n_proteins,
            "n_valid_proteins": self.n_valid_proteins,
            "model": self.model,
        }


def evaluate_dim_d(
    predictions: np.ndarray,
    observed: np.ndarray,
    model_name: str = "",
) -> DimDResult:
    """Score one (model, test-set) pair on Dim D's full metric triple.

    Parameters
    ----------
    predictions, observed : np.ndarray
        Shape (n_test_cells, n_proteins). Both must match.
    model_name : str
        Display label propagated to the result.

    Returns
    -------
    DimDResult
    """
    rs = pearson_per_protein(predictions, observed)
    n_proteins = len(rs)
    n_valid = int(np.sum(~np.isnan(rs)))
    return DimDResult(
        mean_pearson_r=mean_pearson_per_protein(predictions, observed),
        median_pearson_r=median_pearson_per_protein(predictions, observed),
        rmse=rmse(predictions, observed),
        n_proteins=n_proteins,
        n_valid_proteins=n_valid,
        model=model_name,
    )
