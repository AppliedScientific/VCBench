# VCBench

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.06.18.733146-blue.svg)](https://doi.org/10.64898/2026.06.18.733146)
[![Leaderboard](https://img.shields.io/badge/%F0%9F%8F%86%20leaderboard-HF%20Space-ff9d00.svg)](https://huggingface.co/spaces/appliedscientific/vcbench-leaderboard)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20models%20%26%20data-appliedscientific-ffce1c.svg)](https://huggingface.co/appliedscientific)
[![tests](https://github.com/AppliedScientific/VCBench/actions/workflows/test.yml/badge.svg)](https://github.com/AppliedScientific/VCBench/actions/workflows/test.yml)
[![Applied Scientific Intelligence](https://img.shields.io/badge/lab-appliedscientific.ai-000000.svg)](https://appliedscientific.ai)

VCBench is a capability-stratified benchmark for single-cell foundation models, evaluating five models against pre-registered baselines across five dimensions.

> **First-time visitors — start here.** The v1.0.0 release reconciles VCBench's evaluator with upstream cell-eval to numerical precision via an explicit anchor-convention parameter. The Arc State checkpoint lives publicly at [`huggingface.co/appliedscientific/arc-state-norman-gears-corrected`](https://huggingface.co/appliedscientific/arc-state-norman-gears-corrected) — with paste-able reproduction snippets that recover the headline numbers in &lt;5 min on CPU.

## What it produces

The headline result is a 5-model × 5-dimension capability matrix scored against pre-registered trivial and strongest non-FM baselines, plus three reusable methodological tools that work independently of the specific models evaluated here.

| Model | A: PRR | B: MacroF1 (native) | C: AUROC | C: AUPRC | C: EPR | D: Pearson | E: τ-b |
|---|---|---|---|---|---|---|---|
| Geneformer V2-316M | 0.627 (FT+D) | 0.181 (ZS) | 0.626 (IE) | 0.001 (IE) | 0.000 (IE) | 0.001 (ZS+D) | -0.017 (ZS) |
| scGPT (fine-tuned) | 0.503 (FT) | 0.012 (ZS) | 0.519 (IE) | 0.003 (IE) | 20.05 (IE) | 0.064 (ZS+D) | -0.057 (ZS) |
| UCE 33-layer | N/A | 0.027 (ZS) | N/A | N/A | N/A | 0.132 (ZS+D) | 0.136 (ZS) |
| TranscriptFormer | -0.174 (ZS+D) | 0.156 (ZS) | N/A | N/A | N/A | **0.232** (ZS+D) | 0.046 † |
| Arc State | **0.402** (FT) ‡ | N/A | N/A | N/A | N/A | DNR | DNR |
| **Baselines** | | | | | | | |
| Additive (Ahlmann-Eltze) | 0.890 | — | — | — | — | — | — |
| Mean-prediction | 0.579 | — | — | — | — | 0.152 | — |
| No-change | 0.000 | — | — | — | — | — | — |
| PCA + kNN | — | 0.166 | — | — | — | — | — |
| Co-expression | — | — | 0.558 | 0.004 | 15.50 | — | — |
| pySCENIC | — | — | 0.501 | 0.0011 | 3.50 | — | — |
| scLinear | — | — | — | — | — | 0.129 | — |
| Mean-celltype | — | — | — | — | — | 0.152 | — |
| PCA + DPT | — | — | — | — | — | — | 0.190 |

† TranscriptFormer Dim E = unweighted mean of sci-fate (0.051) and Weinreb (0.041 ± 0.078); the Weinreb component is itself a bootstrap mean over 10 random 5K-cell subsamples (ARPACK non-convergence on the full 49K-cell graph).
‡ Arc State Norman evaluated on the disjoint GEARS train/test split (seed=1); PRR = 0.402, VC Level 1.

**VC Level outcomes** (binding scoring uses the common-label-set protocol on Dim B):
- Geneformer V2-316M, scGPT, UCE, Arc State → **Level 1** (clears trivial baseline on at least one dim)
- TranscriptFormer → **Level 2** (clears mean-celltype on Dim D: 0.232 > 0.152)
- No model achieves Level 3.

## How to run it

```bash
git clone https://github.com/AppliedScientific/VCBench.git
cd VCBench
git checkout v1.0.0  # pin to the release tag
pip install -e .
make tests           # full test suite (unit + integration + correctness), ~3s, no GPU needed
make all             # full reproduction from cached HF embeddings (CPU-OK)
```

The three reusable methodological tools work independently of `make all`:

```python
from vcbench.protocols import common_label_set        # Eq. 5
from vcbench.probes    import spread_error_correlation # Eq. 10
from vcbench.contamination import ContaminationManifest, validate_manifest
```

## Hardware

| Stage | CPU? | Wall-clock on H200 |
|---|---|---|
| Cached-embedding reproduction (`make all`) | ✅ | ~30 min on a Mac |
| Baselines (additive, PCA+kNN, co-expression, mean-celltype, PCA+DPT) | ✅ | ~1 h CPU |
| FM fine-tuning (Geneformer 5.7h + scGPT 2.75h + Arc State ~3h) | GPU | ~12 h |
| FM embedding extraction at scale (TranscriptFormer Dim B) | GPU | ~2 h |
| **Total fresh reproduction** (`make fresh`) | GPU | **~18 h H200** |

## Where next

- **Install paths:** [INSTALL.md](INSTALL.md) (Docker / conda / pip)
- **Reusable methodological tools:** the `vcbench.protocols` / `vcbench.probes` / `vcbench.contamination` packages each have full numpy-style docstrings + worked examples
- **Manuscript ↔ code traceability:** every Eq. (1–9) of VCBench (2026) is implemented in `src/vcbench/dimensions/dim_*/metrics.py` with the equation number and reference values in the module docstring
- **Reference-value drift detection:** `tests/reference_values.json` locks every §I.4 capability-matrix cell; `tests/unit/` contains drift detectors that fire if the on-disk JSONs / CSVs ever diverge

## Note on the two `src/` trees

The repository carries two source trees:

- **`src/vcbench/`** — the canonical Python package, `pip install -e .`-able. This is what model developers and downstream tooling import from. It owns the three reusable methodological tools, the per-dimension evaluation modules, the `FoundationModel` ABC + per-model wrappers (incl. Arc State), the `python -m vcbench` CLI, the contamination schema, and the test suite.
- **`src/{baselines,data,evaluation,models,utils}/`** — the legacy pipeline tree that produced the on-disk reference artefacts (`results/dim_*/`, `results/baselines/`, etc.). Its `step1`–`stepN` runtime functions are the verified path that produced every §I.4 reference value. The new `vcbench.models.*` wrappers compose these `step*` functions in their `run_dim_a()` orchestration so end-to-end execution still flows through proven code while the public ABI lives in the new package.

Import only from `vcbench.*`; treat `src.{baselines,data,evaluation,models,utils}.*` as internal implementation detail.

## Artifacts & reproducibility

Per-dimension reference outputs live under [`results/`](results/); the
published supplementary tables are in [`tables/`](tables/).

Trained model checkpoints and embedding tensors are archived on HuggingFace
Hub — see [`docs/MANIFEST.md`](docs/MANIFEST.md) for the file-by-file index.
Three HF repos:

| Repo | Type | Contents |
|------|------|----------|
| [`arc-state-norman-gears-corrected`](https://huggingface.co/appliedscientific/arc-state-norman-gears-corrected) | model | Arc State checkpoint + eval CSVs + training config |
| [`vcbench-geneformer-perturbation`](https://huggingface.co/appliedscientific/vcbench-geneformer-perturbation) | model | Geneformer V2-316M fine-tuned classifier |
| [`vcbench-embeddings`](https://huggingface.co/datasets/appliedscientific/vcbench-embeddings) | dataset | Cell/gene embeddings across Dim A–E |

All three repos are publicly available on HuggingFace.

```python
from huggingface_hub import snapshot_download

snapshot_download("appliedscientific/arc-state-norman-gears-corrected", repo_type="model")
snapshot_download("appliedscientific/vcbench-geneformer-perturbation", repo_type="model")
snapshot_download("appliedscientific/vcbench-embeddings", repo_type="dataset")
```

## Environments

3 GPU environments grouped by compatible PyTorch versions, plus 2 CPU environments:

| Environment | Python | PyTorch | Models |
|-------------|--------|---------|--------|
| `vcbench-analysis` | 3.10 | - | Baselines, evaluation, probes, assembly |
| `vcbench-scenic` | 3.8 | - | pySCENIC (dependency conflicts) |
| `vcbench-pt118` | 3.10 | CUDA 11.8 | Geneformer, UCE |
| `vcbench-pt212` | 3.9 | 2.1.2 | scGPT |
| `vcbench-pt25` | 3.11 | <=2.5.1 | TranscriptFormer, State |

## Models

| Model | Dim A | Dim B | Dim C | Dim D | Dim E | Environment |
|-------|-------|-------|-------|-------|-------|-------------|
| Geneformer V2-316M | Embedding shift + decoder | Ortholog remap + embed | Attention layer 13 | Embedding probe | DPT probe | vcbench-pt118 |
| scGPT (fine-tuned) | Fine-tune + predict | Ortholog remap + embed | Gene embedding similarity | Embedding probe | DPT probe | vcbench-pt212 |
| UCE 33-layer | N/A | Native cross-species | N/A | Embedding probe | DPT probe | vcbench-pt118 |
| TranscriptFormer | Autoregressive generation | Native cross-species | Gene prompting | Embedding probe | DPT probe | vcbench-pt25 |
| Arc State | Train from scratch | N/A | N/A | Embedding probe | DPT probe | vcbench-pt25 |

## Metrics

| Dimension | Metrics | Baselines |
|-----------|---------|-----------|
| A: Perturbation | PRR (Pearson R on Δ-expression), DES, MAE, Composite | Additive, Mean, No-change |
| B: Cross-Species | Macro F1, Weighted F1 | PCA + kNN |
| C: GRN | AUROC, AUPRC, EPR | Co-expression, pySCENIC |
| D: Cross-Modal | Pearson R, RMSE | PCA + ridge, scLinear, Mean celltype |
| E: Temporal | Kendall tau, kNN balanced accuracy | PCA + DPT |

## Requirements

- **RAM:** 32-64 GB recommended
- **Disk:** 100 GB+ for raw datasets + model weights
- **GPU:** A100 80GB recommended (required for UCE; 40GB sufficient for others)
- **Baseline construction:** CPU-only, no GPU needed
- **Estimated GPU time:** ~75 GPU-hours total

## License

VCBench is released under the [MIT License](LICENSE).

## Citation

If you use VCBench in your research, please cite both the preprint and the
software release. Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

**Preprint** — Weidener, L., Brkić, M., Jovanović, M., Ulgac, E., Meduri, A.
*VCBench: A Multi-Dimensional Benchmark for Single-Cell Foundation Models*,
Applied Scientific Intelligence, Inc. (2026).
[doi:10.64898/2026.06.18.733146](https://doi.org/10.64898/2026.06.18.733146)

```bibtex
@article{weidener2026vcbench,
  title   = {VCBench: A Multi-Dimensional Benchmark for Single-Cell Foundation Models},
  author  = {Weidener, L. and Brki\'{c}, M. and Jovanovi\'{c}, M. and Ulgac, E. and Meduri, A.},
  year    = {2026},
  doi     = {10.64898/2026.06.18.733146},
  url     = {https://doi.org/10.64898/2026.06.18.733146},
  note    = {Preprint}
}
```
