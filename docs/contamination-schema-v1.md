<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: VCBench contributors -->

# VCBench Contamination Reporting Schema v1

A minimal machine-checkable standard for single-cell foundation-model
training-data provenance, released alongside the VCBench benchmark. The
goal is to make it possible for a benchmark author to run:

```
vcbench-check-contamination MODEL_RELEASE_DIR/ EVAL_DATASET.h5ad
```

and receive a definitive `confirmed` / `likely` / `unlikely` / `unknown`
verdict in under one second, replacing platform-inference audits with
direct overlap detection.

## Scope and motivation

Single-cell foundation-model papers (Geneformer, scGPT, UCE,
TranscriptFormer, Arc State) currently publish training-corpus summaries at
the dataset level (for example, "CELLxGENE Census, Jan 2025 build") but not
at the cell level. This makes exhaustive contamination detection between
pretraining and evaluation datasets impossible. As a result, a benchmark
like VCBench can only assert *likely* or *unlikely* based on platform
inclusion/exclusion rules. NLP addressed the analogous problem with n-gram
overlap detection and canonical decontamination protocols; single-cell
biology does not yet have an equivalent standard.

The schema is minimal by design: one manifest file per released model, one
metadata flag per evaluated cell. It does not require authors to withhold
data they did not already withhold, and it does not require retraining. It
requires **disclosure only**.

## Component 1. Cell-barcode MD5 manifest

Every model released alongside a pretraining corpus publishes a file named
`training_cells.md5.txt` or `training_cells.md5.parquet` at the root of the
release. The file contains one row per cell used in pretraining, with the
following columns:

| Column | Type | Description |
|---|---|---|
| `barcode_md5` | string (32 hex chars) | MD5 hash of the cell barcode as it appears in the source dataset |
| `source_accession` | string | GEO, SRA BioProject, Figshare DOI, or CELLxGENE Census dataset ID |
| `corpus_version` | string | Version identifier of the pretraining corpus build (e.g. `"census-2025-01-15"`) |

The MD5 hash is computed on the raw barcode string (no whitespace, no
source-ID prefix), encoded UTF-8. The hash makes privacy-safe publication
straightforward while still permitting exact intersection tests: a
benchmark can MD5-hash its own evaluation-set barcodes and intersect
against the published manifest.

## Component 2. AnnData `.obs` split flag

Models released with held-out evaluation data (any paper that reports a
held-out metric) include an `.obs` column named `vcbench_pretrain_split`
with one of the following values per cell:

- `"train"` — cell was used in pretraining
- `"val"` — cell was held out for validation during pretraining
- `"test"` — cell was held out entirely from pretraining, safe for
  downstream benchmarking
- `"unknown"` — provenance not recorded

The `"unknown"` value is permitted but must not exceed 10% of cells in any
released dataset; otherwise the release does not satisfy the schema. This
forces authors to track provenance for the bulk of released data while
allowing a small unknown-tail for legacy re-releases.

## Component 3. Accession-level manifest

A human-readable file named `pretraining_manifest.yaml` in the release
root, listing every dataset used for pretraining at the accession level:

```yaml
corpus_version: "census-2025-01-15"
model_name: "example-foundation-model-v2"
sources:
  - accession: "GSE194122"
    platform: "GEO"
    included: true
    cells_used: 66175
    inclusion_criteria: "bone marrow CITE-seq, all four sites"
    exclusions: []
    notes: "RNA component only; ADT component not used"
  - accession: "10.25452/figshare.plus.20029387"
    platform: "Figshare+"
    included: false
    cells_used: 0
    inclusion_criteria: null
    exclusions: ["cell_culture", "perturbation_assay"]
    notes: "Excluded per Census cell-culture filter"
```

The `exclusions` list uses canonical reasons from a short controlled
vocabulary: `cell_culture`, `perturbation_assay`, `non_human`,
`insufficient_metadata`, `cohort_restricted`, `other`. Accessions that were
considered and rejected must be listed; silent exclusions do not satisfy
the schema.

## Validator behavior

A compliant validator (`vcbench-check-contamination`) takes a model release
directory and one or more evaluation datasets and returns, for each
(model, evaluation-dataset) pair:

- `confirmed` — non-empty MD5 intersection between pretraining manifest
  and evaluation barcodes
- `likely` — no MD5 intersection, but evaluation-dataset accession is
  listed as included in `pretraining_manifest.yaml`
- `unlikely` — no MD5 intersection, and evaluation-dataset accession is
  listed as excluded
- `unknown` — no MD5 intersection and evaluation-dataset accession not
  mentioned in manifest

The validator also emits warnings when the schema is partially satisfied
(for example, manifest present but MD5 file missing), so users can
distinguish "released model does not satisfy schema" from "no
contamination detected."

## Reference implementation

Ships in this repo at `tools/vcbench_contamination_check/`. The validator
core is approximately 200 lines of Python in production form; dependencies
are `pyyaml`, `pandas`, `pyarrow`, `click`, and `anndata`. CLI entry point:
`vcbench-check-contamination` (installed via `pip install
tools/vcbench_contamination_check`).

## Adoption pathway

The schema is deliberately sized so that adoption costs are dominated by
authorship, not computation. MD5-hashing a pretraining corpus of 100
million cell barcodes takes under ten minutes on a single CPU. Writing the
accession-level manifest takes an hour of an author's time for a typical
foundation-model paper. The `vcbench_pretrain_split` column is one line of
`adata.obs` assignment.

Future schema versions may add:

- (v2) Cryptographic commitment hashes that permit third-party verification
  without exposing barcode-level data
- (v3) Canonical per-dataset hashes for public datasets, so that "same
  dataset, different versions" can be distinguished
- (v3) Integration with CELLxGENE Census versioning to permit automated
  inclusion/exclusion lookups

## Citation

If the schema is adopted, please cite as:

> VCBench contributors. VCBench Contamination Reporting Schema v1. Released
> with the VCBench benchmark.
> GitHub: https://github.com/AppliedScientific/VCBench

## License

CC-BY-4.0 for the specification text. MIT for the reference validator
implementation.
