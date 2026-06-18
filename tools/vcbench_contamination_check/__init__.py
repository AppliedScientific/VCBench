"""VCBench contamination reporting schema v1 — reference validator.

Public API:
    check_contamination(model_release_dir, eval_dataset) -> dict

See docs/contamination-schema-v1.md for the full specification.
"""

from .check import check_contamination

__all__ = ["check_contamination"]
__version__ = "1.0.0"
