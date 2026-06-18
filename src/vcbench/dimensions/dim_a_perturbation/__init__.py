"""Dim A — perturbation prediction metrics, baselines, and evaluator.

Implements:

* Eq. 1 — PRR (Perturbation Response Recovery)
* Eq. 2 — Additive baseline (Ahlmann-Eltze)
* Eq. 3 — DES (Direction score, top-K sign-agreement)

Reference values for unit tests (VCBench (2026), §I.4):

* Geneformer V2-316M FT+D, additive-evaluable (n=71): PRR = 0.627
* Geneformer V2-316M FT+D, full (n=106):              PRR = 0.6267, DES = 0.8778
* scGPT FT, additive-evaluable (n=71):                PRR = 0.545
* scGPT FT, full (n=107):                             PRR = 0.5025, DES = 0.8439
* scGPT FT, novel (n=36):                             PRR = 0.420
* TranscriptFormer ZS+D, full (n=107):                PRR = -0.174
* Arc State FT, disjoint GEARS split (n=107):         PRR = 0.402
* Additive baseline (n=71):                           PRR = 0.890, DES = 0.999
* Mean-prediction baseline (n=107):                   PRR = 0.579
* No-change baseline (n=107):                         PRR = 0.000
"""

from vcbench.dimensions.dim_a_perturbation.baselines import (
    additive_baseline,
    mean_baseline,
    no_change_baseline,
)
from vcbench.dimensions.dim_a_perturbation.evaluate import evaluate_dim_a
from vcbench.dimensions.dim_a_perturbation.metrics import des, prr

__all__ = [
    "prr",
    "des",
    "additive_baseline",
    "mean_baseline",
    "no_change_baseline",
    "evaluate_dim_a",
]
