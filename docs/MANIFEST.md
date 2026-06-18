# VCBench v1.0 — Artifact Manifest

This manifest documents every artifact produced by the VCBench benchmark run and
where it lives. All numeric results in the paper trace back to the JSON/CSV files
in this repository; the inputs (model checkpoints) and outputs (embedding
tensors) used to produce them are archived on HuggingFace Hub.

**Release tag:** `v1.0.0`
**Access:** all three repos are publicly available on HuggingFace.

---

## Repository index

| # | HF repo | Type | Size | Contents |
|---|---------|------|-----:|----------|
| 1 | [`appliedscientific/arc-state-norman-gears-corrected`](https://huggingface.co/appliedscientific/arc-state-norman-gears-corrected) | model | ~2.3 GB | Arc State Norman fine-tune on the disjoint GEARS train/test split (final + best ckpts, training config, GEARS-split TOML, eval CSVs, model card). Public. |
| 2 | [`appliedscientific/vcbench-geneformer-perturbation`](https://huggingface.co/appliedscientific/vcbench-geneformer-perturbation) | model | ~1.4 GB | Geneformer V2-316M fine-tuned for Norman perturbation classification. Public. |
| 3 | [`appliedscientific/vcbench-embeddings`](https://huggingface.co/datasets/appliedscientific/vcbench-embeddings) | dataset | ~14 GB | All cell/gene embeddings across Dim A–E (5 FMs × datasets). Public. |

---

## 1. `arc-state-norman-gears-corrected` (model)

Arc State 110M-parameter model fine-tuned on the Norman combinatorial CRISPR
perturbation dataset with the disjoint GEARS train/test split (139 train perts,
107 test perts, zero overlap).

```
final.ckpt               # Final model state at step 40,000 (1.13 GB)
best.ckpt                # Model state at lowest validation loss (1.13 GB)
training_config.yaml     # Resolved Hydra config used by arc-state v0.10.2
data_split.toml          # The GEARS-split TOML — 139 train / 107 test perturbations
eval_aggregate.csv       # Aggregate cell-eval metrics across the 107 test perts
eval_per_perturbation.csv# Per-perturbation cell-eval metrics (107 rows)
```

**Dim A result: PRR = 0.402** (real-control anchor) / 0.408 (cell-eval
cross-validation), VC Level 1, on the disjoint GEARS train/test split
(seed=1, 139 train / 107 held-out test). Per-perturbation and aggregate metrics
are in `eval_per_perturbation.csv` / `eval_aggregate.csv`.

---

## 2. `vcbench-geneformer-perturbation` (model)

Geneformer V2-316M fine-tuned for Norman perturbation classification, trained
via `BertForSequenceClassification` over the Norman perturbation classes.

```
model.safetensors                        # Fine-tuned classifier weights (1.27 GB)
config.json                              # HF model config
training_args.bin                        # Training argument state
norman_id_class_dict.pkl                 # perturbation ID → class index mapping
norman_labeled_train.dataset/            # Tokenized training split (123 MB)
norman_labeled_test.dataset/             # Tokenized held-out test split (14 MB)
```

**Load via:**
```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(
    "appliedscientific/vcbench-geneformer-perturbation",
)
```

---

## 3. `vcbench-embeddings` (dataset, ~14 GB)

All cell and gene embedding tensors produced by the five foundation models
across the five benchmark dimensions, organized by dimension:

```
dim_a/            # Dim A: Perturbation prediction (Norman ctrl set + predictions)
dim_b/            # Dim B: Cross-species cell-type transfer (CELLxGENE Census)
  # NB: UCE only covers heart+brain; TranscriptFormer only covers lung+liver
dim_c/            # Dim C: GRN inference (BEELINE + TRRUST)
dim_d/            # Dim D: Cross-modal RNA→Protein (CITE-seq)
dim_e/            # Dim E: Temporal ordering (sci-fate + Weinreb/LARRY)
```

**Key methodology notes:**
- **All Dim E probes use PCA(50) → DPT** to match the baseline path. Every Dim E
  `temporal_results.json` carries `embedding_dim_pca: 50`.
- **TF Weinreb** is the bootstrap mean of 10 random 5K subsamples due to ARPACK
  non-convergence on the full 49K graph (τ = 0.041 ± 0.078, BalAcc = 0.351 ± 0.024).

**Download:**
```python
from huggingface_hub import snapshot_download
path = snapshot_download("appliedscientific/vcbench-embeddings", repo_type="dataset")
```

---

## Reproducibility contract

Given this repo at tag `v1.0.0` + the three HF repos, any reviewer can:

1. **Reproduce the v1.0.0 Arc State Dim A scores.** The per-perturbation and
   aggregate metrics ship in `arc-state-norman-gears-corrected`
   (`eval_per_perturbation.csv` / `eval_aggregate.csv`); the wrapper
   `vcbench.models.ArcState` re-runs the full pipeline to confirm PRR = 0.402
   (real anchor) / 0.408 (cell-eval cross-validation) on the disjoint GEARS
   train/test split (seed=1, 139 train / 107 held-out test).
2. **Reproduce Dim A Geneformer perturbation** — load
   `vcbench-geneformer-perturbation` via
   `BertForSequenceClassification.from_pretrained(...)` and re-run inference on
   the bundled `norman_labeled_test.dataset`.
3. **Reproduce Dim B–E downstream metrics** — load any embedding `.npy` from
   `vcbench-embeddings`, re-run the evaluator in `src/evaluation/metrics.py`,
   and confirm the matching JSON result under `results/`.

For the source data (never uploaded, always re-downloadable):
- **Norman combinatorial** — `src/data/download_norman.py` (GEARS API)
- **Replogle K562 essential** — `src/data/download_replogle.py`
- **CELLxGENE Census** — `src/data/download_census.py` (5 tissues × 2 species)
- **BEELINE + TRRUST v2** — `src/data/download_beeline.py`, `download_trrust.py`
- **NeurIPS 2021 CITE-seq** — `src/data/download_cite_seq.py`
- **sci-fate, Weinreb/LARRY** — `src/data/download_temporal.py`
