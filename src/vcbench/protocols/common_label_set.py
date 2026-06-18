"""Common-label-set protocol — Eq. 5 from VCBench (2026).

Per-tissue cell-type vocabulary intersection across evaluated methods.

Implements the binding protocol for VC Level assignments and Table 2 bolding
decisions on Dimension B (cross-species). Restricting all methods to the
shared label set per tissue removes the class-count confound that otherwise
inflates native-protocol macro F1 for methods with shrunken vocabularies.
"""

from __future__ import annotations


def common_label_set(
    methods: dict[str, dict[str, set[str]]],
) -> dict[str, set[str]]:
    """Compute the per-tissue cell-type vocabulary intersection across methods.

    Implements Eq. 5 from VCBench (2026) (VCBench)::

        C_t = intersection over m in M of C^m_t

    where ``C^m_t`` is the cell-type vocabulary admitted by method ``m`` in
    tissue ``t``, and ``M`` is the full set of evaluated methods.

    Parameters
    ----------
    methods : dict[str, dict[str, set[str]]]
        Outer key: method id. Inner key: tissue id. Inner value: set of
        cell-type labels admitted by that method in that tissue.

        Tissues that are absent from a given method's inner dict are
        treated as that method admitting the empty set for that tissue
        — and therefore the per-tissue intersection becomes empty too.

    Returns
    -------
    dict[str, set[str]]
        Per-tissue intersection set. Keys are the union of all tissue ids
        observed across the input methods.

    Raises
    ------
    ValueError
        If ``methods`` is empty (the intersection is undefined).

    Examples
    --------
    >>> methods = {
    ...     "geneformer": {"lung": {"T cell", "B cell", "NK cell"},
    ...                    "liver": {"hepatocyte"}},
    ...     "pca_knn":    {"lung": {"T cell", "B cell", "NK cell", "monocyte"},
    ...                    "liver": {"hepatocyte", "stellate"}},
    ... }
    >>> result = common_label_set(methods)
    >>> sorted(result["lung"])
    ['B cell', 'NK cell', 'T cell']
    >>> sorted(result["liver"])
    ['hepatocyte']

    Notes
    -----
    The reference implementation that produced VCBench's Supplementary Table 2
    common-set rows is bit-identical to this function applied to the per-tissue
    label sets in ``results/dim_b/common_label_set_per_tissue.json``.
    """
    if not methods:
        raise ValueError(
            "common_label_set requires at least one method; received empty dict."
        )

    method_dicts = list(methods.values())
    all_tissues: set[str] = set()
    for d in method_dicts:
        all_tissues.update(d.keys())

    out: dict[str, set[str]] = {}
    for tissue in all_tissues:
        per_method_sets = [d.get(tissue, set()) for d in method_dicts]
        out[tissue] = set.intersection(*per_method_sets) if per_method_sets else set()
    return out
