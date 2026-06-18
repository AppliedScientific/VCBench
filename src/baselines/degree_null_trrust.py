"""
Degree-null (random-ranking) baseline for Dimension C on TRRUST.

Generates the expected-metric distribution for a GRN predictor that
assigns random scores while preserving the TF out-degree cardinality of
the candidate-pair universe. For the TRRUST evaluation geometry used by
every foundation model in VCBench, this collapses analytically to:

* AUROC  = 0.5  (pairwise-ranking expectation over random scores)
* AUPRC  = edge_density  (positive-class prior)
* EPR    = 1.0  (early precision / edge_density)

We still run a Monte Carlo to report the empirical distribution
and the associated standard errors — matching the schema used by
``src.models.grn_utils.evaluate_and_save_grn`` for every FM JSON.

Reads ``n_true_edges``, ``n_total_pairs``, and ``edge_density`` from an
existing FM JSON (Geneformer by default) to lock the evaluation universe
to the same TRRUST TF × target pair graph the FMs were scored against.

Writes:
    results/dim_c/degree_null/grn_eval_trrust.json

Run::

    python -m src.baselines.degree_null_trrust
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_JSON = REPO_ROOT / "results" / "dim_c" / "geneformer" / "grn_eval_trrust.json"
OUT_DIR = REPO_ROOT / "results" / "dim_c" / "degree_null"


def _load_reference_geometry(path: Path) -> dict:
    """Pick up (n_true_edges, n_total_pairs, edge_density) from any FM JSON.

    Every foundation-model TRRUST evaluation uses the exact same candidate
    pair universe (filtered by shared vocabulary), so reusing a single
    reference JSON guarantees apples-to-apples numerics.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Reference FM JSON not found: {path}. Run a foundation model "
            "Dim C eval first, or point this script at any existing "
            "grn_eval_trrust.json."
        )
    with path.open() as fh:
        payload = json.load(fh)
    for key in ("n_true_edges", "n_total_pairs", "edge_density"):
        if key not in payload:
            raise KeyError(f"Reference JSON missing {key!r}: {path}")
    return payload


def _simulate_once(n_total: int, n_pos: int, rng: np.random.Generator
                   ) -> tuple[float, float, float]:
    """Single Monte Carlo draw: random scores, compute AUROC / AUPRC / EPR."""
    y_true = np.zeros(n_total, dtype=np.int8)
    pos_idx = rng.choice(n_total, size=n_pos, replace=False)
    y_true[pos_idx] = 1
    y_score = rng.random(n_total)

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    sorted_idx = np.argsort(-y_score)
    early_precision = y_true[sorted_idx[:n_pos]].sum() / n_pos
    edge_density = n_pos / n_total
    epr = float(early_precision / edge_density)
    return auroc, auprc, epr


def run(n_shuffles: int = 100, seed: int = 42,
        reference_json: Path = REFERENCE_JSON,
        out_dir: Path = OUT_DIR) -> dict:
    ref = _load_reference_geometry(reference_json)
    n_total = int(ref["n_total_pairs"])
    n_pos = int(ref["n_true_edges"])
    edge_density = float(ref["edge_density"])

    rng = np.random.default_rng(seed)
    rows = np.array([_simulate_once(n_total, n_pos, rng)
                     for _ in range(n_shuffles)])
    auroc_mean, auprc_mean, epr_mean = rows.mean(axis=0).tolist()
    auroc_std, auprc_std, epr_std = rows.std(axis=0, ddof=1).tolist()

    result = {
        "AUROC":          round(auroc_mean, 4),
        "AUPRC":          round(auprc_mean, 6),
        "AUPRC_ratio":    round(auprc_mean / edge_density, 4),
        "EPR":            round(epr_mean, 4),
        "edge_density":   edge_density,
        "n_true_edges":   n_pos,
        "n_total_pairs":  n_total,
        # MC-specific diagnostics — the null's empirical spread.
        # The FM JSONs don't carry these, hence the extra keys.
        "mc_n_shuffles":  n_shuffles,
        "mc_seed":        seed,
        "AUROC_std":      round(auroc_std, 4),
        "AUPRC_std":      round(auprc_std, 6),
        "EPR_std":        round(epr_std, 4),
        "baseline_kind":  "degree-null (random ranking MC)",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grn_eval_trrust.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Degree-null baseline ({n_shuffles} shuffles, seed={seed})")
    print(f"  AUROC = {auroc_mean:.4f} ± {auroc_std:.4f}")
    print(f"  AUPRC = {auprc_mean:.6f} ± {auprc_std:.6f} "
          f"(edge density = {edge_density:.6f})")
    print(f"  EPR   = {epr_mean:.4f} ± {epr_std:.4f}")
    print(f"  -> {out_path}")
    return result


if __name__ == "__main__":
    run()
