# VCBench reproducibility image — CPU base.
#
# This image installs vcbench + dev deps + the lighter scientific stack
# (numpy, pandas, scipy, scikit-learn, anndata, matplotlib, pyyaml,
# pydantic). It does NOT include CUDA, PyTorch, or any of the heavy
# foundation-model runtimes (transformers, scgpt, geneformer, transcriptformer,
# state). Those are needed only for `make fresh` reproduction from raw
# checkpoints; `make all` runs against cached embeddings and the dependencies
# this image carries are sufficient for that.
#
# To produce a CUDA-enabled image for `make fresh`, derive from
# nvidia/cuda:12.1-runtime-ubuntu22.04 and add `pip install torch
# transformers ...` per configs/environments/.
#
# Build: docker build -t vcbench:cpu .
# Run:   docker run --rm -it -v $(pwd):/work -w /work vcbench:cpu make tests

FROM python:3.12-slim AS base

# System deps that scientific Python wheels expect at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /vcbench

# Copy package metadata first so Docker layer-caches dependency install.
COPY pyproject.toml ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip && \
    python -m pip install -e ".[dev]" && \
    python -m pip install scikit-learn anndata matplotlib seaborn

# Copy the rest of the repo (configs, tests, results,
# pre-registration etc.). .dockerignore should exclude data/raw/ and
# any large model checkpoints to keep the image small.
COPY . .

# Smoke check — the three reusable APIs must import on every build.
RUN python -c "from vcbench.protocols import common_label_set; \
                from vcbench.probes import spread_error_correlation; \
                from vcbench.contamination import validate_manifest; \
                print('vcbench reusable APIs OK')"

CMD ["python", "-m", "pytest", "tests/unit/", "-q"]
