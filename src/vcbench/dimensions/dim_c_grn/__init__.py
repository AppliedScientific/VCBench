"""Dim C — gene regulatory network inference metrics, baselines, and statistics.

Implements:

* Eq. 6 — Early Precision Ratio (EPR), plus AUROC and AUPRC
* Eq. 7 — Bootstrap CI and paired test with BH FDR correction

Reference values (VCBench (2026), §I.4 Table 2 + §I.3 Eq. 7):

============= ====== ====== ======
Method        AUROC  AUPRC  EPR
============= ====== ====== ======
Co-expression 0.558  0.004  15.50
Degree-null   0.500  0.0003 1.13
pySCENIC      0.501  0.0011 3.50
Geneformer    0.626  0.001  0.000
scGPT         0.519  0.003  20.05
============= ====== ====== ======

Conjunctive passing rule for Dim C: both AUPRC AND EPR must exceed all three
baselines (co-expression, degree-null, pySCENIC). Originally AUPRC-only;
amended to conjunctive after scGPT EPR=20.05 with AUPRC at noise floor
exposed insufficiency of AUPRC-only rule. Verdict identical under both rules
(no FM passes Dim C).

Bootstrap finding (T1 paired test): Geneformer vs co-expression on AUPRC
returns BH q = 0.692 — the 'overlap with co-expression cannot be rejected'
result that drives the §3.3 Dim C reframing.
"""

from vcbench.dimensions.dim_c_grn.evaluate import DimCResult, evaluate_dim_c
from vcbench.dimensions.dim_c_grn.metrics import auprc, auroc, epr
from vcbench.dimensions.dim_c_grn.statistics import (
    BootstrapCI,
    PairedBootstrapResult,
    benjamini_hochberg,
    bootstrap_ci,
    paired_bootstrap_test,
)

__all__ = [
    "auprc",
    "auroc",
    "epr",
    "evaluate_dim_c",
    "DimCResult",
    "bootstrap_ci",
    "paired_bootstrap_test",
    "benjamini_hochberg",
    "BootstrapCI",
    "PairedBootstrapResult",
]
