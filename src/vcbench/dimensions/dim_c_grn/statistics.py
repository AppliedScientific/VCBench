"""Bootstrap CI + paired test for Dim C (Eq. 7 from VCBench (2026)).

Per the manuscript convention, the bootstrap iteration index is ``t`` (to
avoid collision with the baseline subscript ``b``)::

    For t = 1..B:
        Sample edge indices with replacement from the evaluation set
        Compute AUPRC_m^(t) and AUPRC_b^(t) on the SAME resampled edges

    95% CI on AUPRC_m: [2.5th, 97.5th] percentile of {AUPRC_m^(t)}
    Paired one-sided p: (1/B) * sum_t 1[AUPRC_m^(t) <= AUPRC_b^(t)]

BH FDR correction is then applied across the family of (model, baseline)
pairwise tests within each metric.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Empirical bootstrap percentile CI for a single (model, metric) cell."""

    mean: float
    ci_lower: float
    ci_upper: float
    se: float
    n_bootstrap: int


@dataclass(frozen=True)
class PairedBootstrapResult:
    """One-sided paired bootstrap test of model vs baseline (Eq. 7)."""

    delta_mean: float
    delta_ci_lower: float
    delta_ci_upper: float
    p_value_one_sided: float
    n_bootstrap: int


def bootstrap_ci(
    y_true: Sequence,
    y_score: Sequence,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapCI:
    """Empirical-percentile bootstrap CI for a single metric.

    Resamples (y_true, y_score) edge indices with replacement ``n_bootstrap``
    times, evaluates ``metric_fn`` on each resample, and returns the mean
    plus the ``[lower, upper]`` percentile interval at the requested
    confidence level.

    Parameters
    ----------
    y_true, y_score : sequence
        Ground truth and predicted scores aligned by edge index.
    metric_fn : callable
        Function ``f(y_true, y_score) -> float`` (e.g. ``auprc``, ``epr``).
    n_bootstrap : int, default 1000
        Number of resamples. Matches the existing pipeline's
        ``bootstrap_cis.csv`` value.
    confidence : float in (0, 1), default 0.95
        Two-sided confidence level for the percentile CI.
    seed : int | None, default None
        RNG seed for reproducibility.

    Returns
    -------
    BootstrapCI
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be >= 2, got {n_bootstrap}")
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    samples = np.empty(n_bootstrap, dtype=float)
    for t in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[t] = metric_fn(y_true[idx], y_score[idx])
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        raise ValueError("every bootstrap iteration produced NaN — check inputs")

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(samples, 100 * alpha))
    hi = float(np.percentile(samples, 100 * (1 - alpha)))
    return BootstrapCI(
        mean=float(np.mean(samples)),
        ci_lower=lo,
        ci_upper=hi,
        se=float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        n_bootstrap=int(len(samples)),
    )


def paired_bootstrap_test(
    y_true: Sequence,
    y_score_model: Sequence,
    y_score_baseline: Sequence,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> PairedBootstrapResult:
    """Paired bootstrap test of model vs baseline on the same metric (Eq. 7).

    Both ``y_score_model`` and ``y_score_baseline`` are evaluated on the SAME
    resampled edge indices each iteration (the 'paired' part — controls for
    edge-set sampling noise). Returns the mean Δ = metric(model) - metric(baseline),
    its percentile CI, and the one-sided p-value
    p = (1/B) * sum 1[Δ_t <= 0] (the rate at which baseline ties or beats model).

    Parameters
    ----------
    y_true, y_score_model, y_score_baseline : sequence
        All three same length; aligned by edge index.

    Returns
    -------
    PairedBootstrapResult
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    y_true = np.asarray(y_true)
    y_score_model = np.asarray(y_score_model, dtype=float)
    y_score_baseline = np.asarray(y_score_baseline, dtype=float)
    if not (len(y_true) == len(y_score_model) == len(y_score_baseline)):
        raise ValueError("y_true / y_score_model / y_score_baseline length mismatch")
    n = len(y_true)
    rng = np.random.default_rng(seed)

    deltas = np.empty(n_bootstrap, dtype=float)
    for t in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = metric_fn(y_true[idx], y_score_model[idx])
        b = metric_fn(y_true[idx], y_score_baseline[idx])
        deltas[t] = m - b
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        raise ValueError("every bootstrap iteration produced NaN — check inputs")

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(deltas, 100 * alpha))
    hi = float(np.percentile(deltas, 100 * (1 - alpha)))
    p_one_sided = float(np.mean(deltas <= 0.0))
    return PairedBootstrapResult(
        delta_mean=float(np.mean(deltas)),
        delta_ci_lower=lo,
        delta_ci_upper=hi,
        p_value_one_sided=p_one_sided,
        n_bootstrap=int(len(deltas)),
    )


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns adjusted q-values in the same order as the input p-values.

    This implementation sorts by p-value and matches the values in
    ``results/dim_c/pairwise_delta_pvalues.csv``.

    Parameters
    ----------
    p_values : sequence of float
        Raw two-sided or one-sided p-values.

    Returns
    -------
    np.ndarray
        BH-adjusted q-values, length equals input.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"p_values must be 1-D, got shape {p.shape}")
    n = len(p)
    if n == 0:
        return p
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must be in [0, 1]")
    order = np.argsort(p)            # ascending by p
    ranks = np.arange(1, n + 1)
    q_sorted = p[order] * n / ranks
    # Enforce monotonicity: q[i] = min(q[i], q[i+1], ...) reading right-to-left
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    out = np.empty_like(q_sorted)
    out[order] = q_sorted
    return out
