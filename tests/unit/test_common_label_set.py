"""Unit tests for vcbench.protocols.common_label_set (Eq. 5)."""

from __future__ import annotations

import pytest

from vcbench.protocols import common_label_set


def test_basic_two_method_intersection():
    methods = {
        "geneformer": {"lung": {"T cell", "B cell", "NK cell"},
                       "liver": {"hepatocyte"}},
        "pca_knn":    {"lung": {"T cell", "B cell", "NK cell", "monocyte"},
                       "liver": {"hepatocyte", "stellate"}},
    }
    out = common_label_set(methods)
    assert out["lung"] == {"T cell", "B cell", "NK cell"}
    assert out["liver"] == {"hepatocyte"}


def test_single_method_passthrough():
    methods = {"only": {"lung": {"a", "b", "c"}}}
    out = common_label_set(methods)
    assert out == {"lung": {"a", "b", "c"}}


def test_no_overlap_returns_empty():
    methods = {
        "m1": {"lung": {"a", "b"}},
        "m2": {"lung": {"c", "d"}},
    }
    out = common_label_set(methods)
    assert out == {"lung": set()}


def test_all_overlap_returns_full_set():
    s = {"x", "y", "z"}
    methods = {"m1": {"t": s}, "m2": {"t": s}, "m3": {"t": s}}
    assert common_label_set(methods) == {"t": s}


def test_partial_tissue_coverage_yields_empty_intersection():
    """If a method is missing a tissue entirely, the intersection becomes empty."""
    methods = {
        "m1": {"lung": {"a", "b"}, "liver": {"x"}},
        "m2": {"lung": {"a", "b"}},                # liver missing
    }
    out = common_label_set(methods)
    assert out["lung"] == {"a", "b"}
    assert out["liver"] == set()


def test_empty_dict_raises():
    with pytest.raises(ValueError):
        common_label_set({})


def test_reproduces_vcbench_supp_table_2_label_set_intersections():
    """Smoke check against the per-tissue intersections persisted in
    results/dim_b/common_label_set_per_tissue.json (VCBench (2026) Supp Table 2).

    Synthetic three-method case sized to the same intersection cardinalities
    (lung=7, liver=6, heart=2, kidney=11, brain=5).
    """
    methods = {
        "geneformer": {
            "lung":  set(f"lung-{i}"   for i in range(10)),
            "liver": set(f"liver-{i}"  for i in range(8)),
            "heart": set(f"heart-{i}"  for i in range(4)),
            "kidney":set(f"kidney-{i}" for i in range(15)),
            "brain": set(f"brain-{i}"  for i in range(7)),
        },
        "scgpt": {
            "lung":  set(f"lung-{i}"   for i in range(7)),
            "liver": set(f"liver-{i}"  for i in range(6)),
            "heart": set(f"heart-{i}"  for i in range(3)),
            "kidney":set(f"kidney-{i}" for i in range(12)),
            "brain": set(f"brain-{i}"  for i in range(6)),
        },
        "pca_knn": {
            "lung":  set(f"lung-{i}"   for i in range(20)),
            "liver": set(f"liver-{i}"  for i in range(10)),
            "heart": set(f"heart-{i}"  for i in range(2)),
            "kidney":set(f"kidney-{i}" for i in range(11)),
            "brain": set(f"brain-{i}"  for i in range(5)),
        },
    }
    out = common_label_set(methods)
    assert len(out["lung"]) == 7
    assert len(out["liver"]) == 6
    assert len(out["heart"]) == 2
    assert len(out["kidney"]) == 11
    assert len(out["brain"]) == 5
