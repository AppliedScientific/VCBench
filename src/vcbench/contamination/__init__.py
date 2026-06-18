"""Pretraining-overlap manifest schema and validator.

Used to declare what corpora a foundation model was trained on so that
benchmark consumers can flag (model, evaluation-dataset) pairs that risk
training-set contamination.
"""

from vcbench.contamination.manifest import (
    ContaminationManifest,
    ManifestValidationError,
    ValidationSummary,
    validate_manifest,
)

__all__ = [
    "ContaminationManifest",
    "ManifestValidationError",
    "ValidationSummary",
    "validate_manifest",
]
