"""Shared test fixtures and paths."""

import os
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
RESULTS_DIR = PROJECT_DIR / "results"
BASELINES_DIR = RESULTS_DIR / "baselines"
TABLES_DIR = RESULTS_DIR / "tables"

TISSUES = ["lung", "liver", "heart", "kidney", "brain"]
GRN_MODELS = ["geneformer", "scgpt", "transcriptformer"]
CROSSSPECIES_MODELS = ["geneformer", "scgpt", "uce", "transcriptformer"]
