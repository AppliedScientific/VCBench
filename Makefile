# VCBench reproduction harness.
#
# `make all`   — reproduce every Table 2 cell from cached embeddings.
# `make fresh` — reproduce every Table 2 cell from raw model checkpoints.
# `make tests` — run the full test suite (no GPU, ~3s on a Mac).
#
# Wall-clock targets on a single H200:
#   Dim A (Geneformer FT 5.7h + scGPT FT 2.75h + Arc State FT ~3h)   ~12 h
#   Dim B (embedding + kNN)                                           ~2 h
#   Dim C (attention/cosine + scoring)                                 ~30 min
#   Dim D (embedding + ridge regression)                              ~1 h
#   Dim E (embedding + DPT)                                           ~3 h
#   ----------------------------------------------------------------
#   Total                                                             ~18 h
#
# CPU-only runs are possible for every baseline plus inference on cached
# embeddings; GPU is required for fine-tuning Geneformer/scGPT/Arc State and
# for embedding extraction on TranscriptFormer at scale.

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.PHONY: all fresh tests dim_a dim_b dim_c dim_d dim_e \
        contamination spread_error clean install lint

# Default: run the full test suite — fast feedback for local development.
.DEFAULT_GOAL := tests

install:
	$(PYTHON) -m pip install -e .[dev]

tests:
	$(PYTEST) tests/ -q

lint:
	$(PYTHON) -m ruff check src/ tests/

# Reproduction targets. Each `make dim_X` dispatches through
# src/vcbench/dimensions/dim_X/evaluate.py to produce the relevant
# results/dim_X/ artefacts.

dim_a:
	$(PYTHON) -m src.vcbench.dimensions.dim_a.evaluate

dim_b:
	$(PYTHON) -m src.vcbench.dimensions.dim_b.evaluate

dim_c:
	$(PYTHON) -m src.vcbench.dimensions.dim_c.evaluate

dim_d:
	$(PYTHON) -m src.vcbench.dimensions.dim_d.evaluate

dim_e:
	$(PYTHON) -m src.vcbench.dimensions.dim_e.evaluate

contamination:
	$(PYTHON) -m src.evaluation.contamination_audit

spread_error:
	$(PYTHON) -m src.evaluation.calibration_probe

all: dim_a dim_b dim_c dim_d dim_e contamination spread_error
	@echo "VCBench reproduction complete. Inspect results/."

# `make fresh` runs the full pipeline from raw model checkpoints —
# expensive, ~18h H200. Same target list as `all`; difference is that
# fresh skips the cached-embedding shortcut.
fresh: clean all

clean:
	rm -rf results/dim_a results/dim_b results/dim_c results/dim_d results/dim_e
	rm -rf results/baselines
	@echo "Reproduction artefacts removed. Cached HuggingFace artefacts preserved."
