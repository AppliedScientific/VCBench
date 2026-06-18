# VCBench Leaderboard

Source-of-record table for every foundation-model submission scored against VCBench. Auto-regenerated from `tests/reference_values.json` + accepted submission memos at every release.

## v1.0 — VCBench

| Model | Submission memo | A: PRR | B: macroF1 (common-set) | C: AUPRC | D: Pearson | E: τ-b | VC Level | Notes |
|---|---|---|---|---|---|---|---|---|
| **Geneformer V2-316M** | (v1 baseline) | 0.627 (FT+D) | 0.171 | 0.001 (IE) | 0.001 (ZS+D) | -0.017 (ZS) | **1** | Census-independent control |
| **scGPT (fine-tuned)** | (v1 baseline) | 0.503 (FT) | 0.123 | 0.003 (IE) | 0.064 (ZS+D) | -0.057 (ZS) | **1** | Weinreb-specific τ-b = -0.103 |
| **UCE 33-layer** | (v1 baseline) | N/A | 0.379 (heart+brain) | N/A | 0.132 (ZS+D) | 0.136 (ZS) | **1** | Outside design scope on Dim D |
| **TranscriptFormer** | (v1 baseline) | -0.174 (ZS+D) | 0.351 (lung+liver) | N/A | **0.232** (ZS+D) | 0.046 † | **2** | Only Level-2 cell — beats mean-celltype on Dim D |
| **Arc State** | (v1 baseline) | **0.402** (FT) | N/A | N/A | DNR | DNR | **1** | Disjoint GEARS train/test split (seed=1, 139 train / 107 held-out test) |

### Baselines (binding for VC Level decisions)

| Baseline | A: PRR | B: macroF1 | C: AUPRC | D: Pearson | E: τ-b |
|---|---|---|---|---|---|
| Additive (Ahlmann-Eltze) | **0.890** | — | — | — | — |
| Mean-prediction | 0.579 | — | — | 0.152 | — |
| No-change | 0.000 | — | — | — | — |
| PCA + kNN (common-set) | — | **0.497** | — | — | — |
| Co-expression | — | — | **0.004** | — | — |
| Mean-celltype | — | — | — | **0.152** | — |
| PCA + DPT | — | — | — | — | **0.190** |

**Bold** = binding strongest non-FM baseline per dimension (§I.5 pre-registration).

### Footnotes

- † TranscriptFormer Dim E = unweighted mean of sci-fate (0.051) and Weinreb (0.041 ± 0.078); the Weinreb component is a bootstrap mean over 10 random 5K-cell subsamples (ARPACK non-convergent on the full 49K-cell graph).
- Arc State Norman is evaluated on the disjoint GEARS train/test split (seed=1, 139 train / 107 held-out test) under `configs/dim_a/arc_state_norman_gears_split.toml`. **Canonical PRR = 0.402** (vcbench `evaluate_dim_a` real-control anchor; cell-eval cross-validation gives 0.408 to 2e-6 absolute under matched anchor). Arc Institute's State preprint does not report a Norman benchmark, so this number is not in tension with any upstream-published Arc number.
- Arc State on Dim B is structurally N/A (no ortholog mapping). Arc State on Dim D / E is DNR (no public API).
- TF / UCE report common-set values only for their covered tissues (TF: lung, liver; UCE: heart, brain); the other tissues are not evaluated under the common-set protocol in this release.

## Future submissions

Add new rows above the baselines section. Each new submission requires:
- A submission memo at `submissions/<YYYY-MM>-<model>.md` (use `submissions/TEMPLATE.md`)
- Wrapper code under `src/vcbench/models/`
- Contamination manifest under `src/vcbench/contamination/examples/`
- Predictions JSONs under `submissions/predictions/<model>/`
- A maintainer-verified reproduction within ±0.001 of the manuscript-reported metric values
