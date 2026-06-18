# VCBench documentation index

Reference documentation for VCBench v1.0.0. The user-facing entry points stay at the repo root (`README.md`, `INSTALL.md`, `CONTRIBUTING.md`); everything else lives here.

## Public-facing references

| File | What it covers |
|---|---|
| [`MANIFEST.md`](MANIFEST.md) | HuggingFace artifact manifest — the checkpoints and embedding tensors archived for the v1.0.0 release, with per-repo file inventories. |
| [`LEADERBOARD.md`](LEADERBOARD.md) | Source-of-record per-model VC Level table for every accepted submission. Auto-regenerated from `tests/reference_values.json` + accepted submission memos at every release. |
| [`contamination-schema-v1.md`](contamination-schema-v1.md) | Specification of the contamination-manifest format that every Dim A model wrapper must declare (Eq. 11 of the manuscript). |

## Where else to look

- **Per-dimension reference outputs**: `results/` (on-disk JSON/CSV per dimension)
- **Pre-registration (frozen)**: `configs/pre_registration.yaml`
- **Reference values for drift detection**: `tests/reference_values.json`
- **Per-package API documentation**: numpy-style docstrings on every public symbol in `src/vcbench/`
