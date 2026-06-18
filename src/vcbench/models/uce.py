"""UCE (Universal Cell Embedding) 33-layer wrapper.

Per §I.4: A=N/A, B=ZS (heart and brain only), C=N/A, D=ZS+D (outside design
scope), E=ZS.

N/A justifications (§I.2):
- Dim A: produces embeddings but not expression vectors; no native
  fine-tuning interface for perturbation-conditioned generation.
- Dim C: gene tokens are static ESM-2 protein-language-model features
  (sequence-similarity-based, not learned regulatory features).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vcbench.models.base import FoundationModel

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from anndata import AnnData


class UCE(FoundationModel):
    name = "UCE 33-layer"
    per_dimension_regime = {
        "A": "N/A",
        "B": "ZS",
        "C": "N/A",
        "D": "ZS+D",
        "E": "ZS",
    }

    def __init__(self) -> None:
        self.checkpoint_path: str | None = None

    def load_pretrained(self, checkpoint_path: str) -> None:
        self.checkpoint_path = str(checkpoint_path)

    def embed(self, adata: "AnnData") -> "np.ndarray":  # pragma: no cover
        raise NotImplementedError(
            "UCE embed() — runtime port not yet inlined into this wrapper.\n\n"
            "The wrapper exposes the ABI surface and the regime declaration; "
            "actual embedding extraction lives in the legacy "
            "src/models/run_uce_*.py and configs/environments/vcbench-pt118.\n\n"
            "UCE's 33-layer 656M-parameter architecture requires an A100 80GB "
            "to embed at the 50K-cell-per-tissue scale used for Dim B, and the "
            "existing legacy runtime is verified against the on-disk "
            "results/dim_b/uce/ artefacts."
        )

    def predict_perturbation(self, perturbations: list[str]) -> "AnnData":
        raise NotImplementedError(
            "UCE Dim A is N/A (per §I.2): UCE produces cell embeddings but "
            "not expression vectors, and has no native fine-tuning interface "
            "for perturbation-conditioned generation. This is a structural "
            "N/A — predict_perturbation is not in scope for any UCE wrapper "
            "version."
        )

    def extract_gene_attention(self) -> "np.ndarray":
        raise NotImplementedError(
            "UCE Dim C is N/A (per §I.2): UCE gene tokens are static ESM-2 "
            "protein-language-model features (sequence similarity, not "
            "learned regulatory features). This is a structural N/A — "
            "extract_gene_attention is not in scope for any UCE wrapper "
            "version."
        )
