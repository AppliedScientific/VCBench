"""Unit tests for vcbench.dimensions.dim_e_temporal.aggregation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vcbench.dimensions.dim_e_temporal.aggregation import (
    BootstrapResult,
    aggregate_across_datasets,
    bootstrap_subsample,
)
from vcbench.dimensions.dim_e_temporal.metrics import kendall_tau_b


# ---- aggregate_across_datasets --------------------------------------------


def test_aggregate_unweighted_mean_dict():
    out = aggregate_across_datasets({"sci_fate": 0.225, "weinreb": 0.153})
    assert out == pytest.approx((0.225 + 0.153) / 2, abs=1e-12)


def test_aggregate_unweighted_mean_iterable():
    out = aggregate_across_datasets([0.5, -0.5])
    assert out == pytest.approx(0.0, abs=1e-12)


def test_aggregate_skips_nan():
    out = aggregate_across_datasets({"a": 0.4, "b": float("nan"), "c": 0.6})
    assert out == pytest.approx(0.5, abs=1e-12)


def test_aggregate_all_nan_returns_nan():
    out = aggregate_across_datasets([float("nan"), float("nan")])
    assert math.isnan(out)


def test_aggregate_empty_input_raises():
    with pytest.raises(ValueError):
        aggregate_across_datasets({})


def test_aggregate_unweighted_preserves_dataset_signal():
    """The cell-count-weighted vs unweighted distinction matters in practice:
    a 49K-cell dataset shouldn't drown out a 6.5K-cell signal. Unweighted
    averaging gives both equal weight. Demonstrate with synthetic numbers
    matching the order of magnitude of the manuscript's scGPT case."""
    # scGPT: sci_fate τ ≈ 0.011, Weinreb τ ≈ -0.103
    # Unweighted mean = -0.046 → preserved as 'consistent inversion signal'
    # Weighted by cell count (6567 vs 49008) would give ≈ -0.090 — still
    # negative, but if scGPT had been positive on sci_fate the weighting
    # would have hidden it.
    unweighted = aggregate_across_datasets({"sci_fate": 0.011, "weinreb": -0.103})
    assert unweighted < 0.0   # mean still negative in this case
    assert unweighted == pytest.approx((0.011 - 0.103) / 2, abs=1e-12)


# ---- bootstrap_subsample --------------------------------------------------


def test_bootstrap_returns_dataclass_with_finite_fields():
    rng = np.random.default_rng(0)
    n = 8000
    pred = rng.normal(0, 1, n)
    truth = rng.integers(0, 5, n)
    result = bootstrap_subsample(
        pred, truth, kendall_tau_b,
        n_bootstrap=5, subsample_size=2000, seed=42,
    )
    assert isinstance(result, BootstrapResult)
    assert result.n_bootstrap == 5
    assert result.subsample_size == 2000
    assert len(result.per_iteration) == 5
    assert result.std >= 0.0


def test_bootstrap_seed_is_deterministic():
    rng = np.random.default_rng(0)
    n = 5000
    pred = rng.normal(0, 1, n)
    truth = rng.integers(0, 5, n)
    a = bootstrap_subsample(pred, truth, kendall_tau_b,
                            n_bootstrap=5, subsample_size=1000, seed=7)
    b = bootstrap_subsample(pred, truth, kendall_tau_b,
                            n_bootstrap=5, subsample_size=1000, seed=7)
    assert a.mean == b.mean
    assert a.std == b.std
    assert a.per_iteration == b.per_iteration


def test_bootstrap_subsample_too_large_raises():
    pred = np.zeros(100)
    truth = np.zeros(100)
    with pytest.raises(ValueError):
        bootstrap_subsample(pred, truth, kendall_tau_b,
                            subsample_size=200)


def test_bootstrap_n_zero_raises():
    pred = np.zeros(100)
    truth = np.zeros(100)
    with pytest.raises(ValueError):
        bootstrap_subsample(pred, truth, kendall_tau_b,
                            n_bootstrap=0, subsample_size=10)


def test_bootstrap_shape_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_subsample(np.zeros(10), np.zeros(11), kendall_tau_b,
                            n_bootstrap=5, subsample_size=5)


def test_bootstrap_default_protocol_matches_manuscript():
    """Defaults n_bootstrap=10, subsample_size=5000 match the TF Weinreb protocol."""
    import inspect
    sig = inspect.signature(bootstrap_subsample)
    assert sig.parameters["n_bootstrap"].default == 10
    assert sig.parameters["subsample_size"].default == 5000
