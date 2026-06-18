# INSTALL

Three installation paths, recommended in order. The Docker path is fastest if you only want to verify reproducibility; pip is fastest if you already have a Python 3.10+ environment and only want the three reusable methodological tools.

> **Fast reproducibility check.** The Arc State checkpoint is publicly available at [`huggingface.co/appliedscientific/arc-state-norman-gears-corrected`](https://huggingface.co/appliedscientific/arc-state-norman-gears-corrected). After Option B or C below, you can recover the headline canonical PRR (0.402) from the bundled adata h5ads in &lt;5 min on CPU.

## Option A (recommended for reproducibility): Docker

Build locally from this checkout:

```bash
docker build -t vcbench:cpu .
docker run --rm -it -v "$(pwd):/work" -w /work vcbench:cpu make all
```

The CPU image is sufficient for `make all` (cached-embedding reproduction) and `make tests`. Fine-tuning + raw embedding extraction requires a CUDA-enabled image — derive from `nvidia/cuda:12.1-runtime-ubuntu22.04` and add the per-model environments under `configs/environments/`.

## Option B: conda

```bash
conda env create -f configs/environments/vcbench-analysis.yml
conda activate vcbench-analysis
pip install -e .
make tests
```

The FM fine-tuning environments (`vcbench-pt118`, `vcbench-pt212`, `vcbench-pt25`, `vcbench-scgpt`) have fragile dependency trees — see `configs/environments/vcbench-scgpt-pinned.txt` for the verified pin set. Install one of those only if you need to re-run a specific model end-to-end.

## Option C: pip

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install scikit-learn anndata    # for Dim B/D baselines + Dim A evaluator
make tests
```

Minimal install — covers the three reusable APIs and every metric module. To run a full `make all` reproduction you'll additionally need the heavy ML stack (`torch`, `transformers`, plus the per-model packages).

## GPU requirements (only for `make fresh`)

| Step | GPU memory needed |
|---|---|
| Geneformer V2-316M fine-tune | ~24 GB |
| scGPT fine-tune | ~16 GB |
| Arc State fine-tune (8 layers) | ~16 GB |
| UCE 33-layer embedding extraction | ~80 GB (A100 required) |
| TranscriptFormer embedding extraction at scale | ~24 GB (H200 used in v1) |

CPU-only execution is fully supported for every baseline, the contamination check, the spread-error probe, and `make all` against cached embeddings.

## Smoke test

After install, the three reusable APIs must import without optional deps:

```bash
python -c "from vcbench.protocols import common_label_set"
python -c "from vcbench.probes    import spread_error_correlation"
python -c "from vcbench.contamination import ContaminationManifest, validate_manifest"
```

If those three lines run without import errors and `make tests` passes (301/301 collected: 219 unit + 20 integration + 51 output-regression + 11 correctness), the install is complete. To run the full suite:

```bash
pytest tests/ -v   # ~3 s on a Mac for the unit slice; integration tests need ~30 s
```

Some integration tests skip without Norman on disk (download via `python src/data/download_norman.py`); the remainder run unconditionally on CPU (no GPU required).
