"""Cross-dataset aggregation + bootstrap-subsample helpers for Dim E.

Two operations isolated here:

1. **aggregate_across_datasets** — unweighted arithmetic mean of per-dataset
   τ-b values (Eq. 9 aggregation rule). VCBench Dim E reports the across-
   dataset mean over sci-fate + Weinreb LARRY. Cell-count-weighted averaging
   is intentionally NOT used because it would let Weinreb (49K cells)
   dominate sci-fate (6.5K cells) and dilute the dataset-level signal — most
   notably, scGPT's clean -0.103 inversion on Weinreb specifically would
   average down to noise.

2. **bootstrap_subsample** — for the TranscriptFormer Weinreb special case
   (ARPACK non-convergent on the 49K-cell graph). Reports mean ± std across
   N random subsamples of size M (default N=10, M=5,000 — the manuscript
   protocol). Returns the bootstrap point estimate and standard deviation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def aggregate_across_datasets(per_dataset_taus: Mapping[str, float] | Iterable[float]) -> float:
    """Unweighted arithmetic mean of per-dataset τ-b values.

    NaN values are skipped (e.g. when a dataset failed to produce a valid τ-b).

    Parameters
    ----------
    per_dataset_taus : mapping[str, float] or iterable of float
        Either ``{dataset: tau_b}`` or a flat iterable of τ-b values.

    Returns
    -------
    float
        Unweighted mean. Returns NaN if every input is NaN.

    Raises
    ------
    ValueError
        If the input is empty.
    """
    if isinstance(per_dataset_taus, Mapping):
        vals = list(per_dataset_taus.values())
    else:
        vals = list(per_dataset_taus)
    if not vals:
        raise ValueError("per_dataset_taus is empty; aggregate is undefined")
    arr = np.asarray(vals, dtype=float)
    finite = arr[~np.isnan(arr)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap mean + std for a per-dataset τ-b under the TF Weinreb protocol."""

    mean: float
    std: float
    n_bootstrap: int
    subsample_size: int
    per_iteration: tuple[float, ...]


def bootstrap_subsample(
    full_predicted: Sequence,
    full_ground_truth: Sequence,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 10,
    subsample_size: int = 5000,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap a metric over random subsamples of the full cell array.

    Used for the TranscriptFormer Weinreb special case where ARPACK
    non-convergence on the full 49,008-cell graph forced a 10×5K subsample
    protocol. Returns mean ± std across iterations.

    Parameters
    ----------
    full_predicted, full_ground_truth : sequence
        Full per-cell predicted ordering and ground-truth label arrays.
    metric_fn : callable
        ``f(predicted, ground_truth) -> float``, typically
        :func:`vcbench.dimensions.dim_e_temporal.metrics.kendall_tau_b`.
    n_bootstrap : int, default 10
        Number of subsample iterations (manuscript value).
    subsample_size : int, default 5000
        Cells per subsample (manuscript value).
    seed : int | None
        RNG seed.

    Returns
    -------
    BootstrapResult
    """
    p = np.asarray(full_predicted, dtype=float)
    g = np.asarray(full_ground_truth)
    if p.shape != g.shape:
        raise ValueError(
            f"shape mismatch: predicted {p.shape} vs ground_truth {g.shape}"
        )
    if subsample_size > len(p):
        raise ValueError(
            f"subsample_size {subsample_size} > population size {len(p)}"
        )
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    rng = np.random.default_rng(seed)

    vals: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(p), size=subsample_size, replace=False)
        v = metric_fn(p[idx], g[idx])
        if not np.isnan(v):
            vals.append(float(v))
    if not vals:
        raise ValueError("every bootstrap iteration produced NaN — check inputs")
    return BootstrapResult(
        mean=float(np.mean(vals)),
        std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        n_bootstrap=len(vals),
        subsample_size=subsample_size,
        per_iteration=tuple(vals),
    )
